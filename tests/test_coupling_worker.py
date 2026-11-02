"""Tests for the framework-agnostic AMUSE worker (Torch/AMUSE coupling, Phase 4).

Covers the hydro-grid community-code surface: lifecycle, the conservative
grid-state exchange with the **total↔internal energy seam**, index↔position
geometry, and `evolve_model`. No `amuse` dependency. See
`docs/TORCH.md` (Phase 4).
"""

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")


from tests.test_regressions import _build_cfg_for_test  # noqa: E402
from amuse_luna.worker import LunaWorker  # noqa: E402


def _cfg(N=8):
    return _build_cfg_for_test(
        test_problem="Soundwave-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0, gamma=1.4,
        ORDER=1, LIMITER=1,
    )


def _init_worker(N=8):
    w = LunaWorker(_cfg(N))
    w.initialize_code()
    w.commit_parameters()
    return w


def test_grid_range_matches_config():
    w = _init_worker(8)
    assert w.get_grid_range() == (0, 7, 0, 7, 0, 7)


def test_position_index_roundtrip():
    w = _init_worker(8)
    i, j, k = w._all_indices()
    x, y, z = w.get_position_of_index(i, j, k)
    i2, j2, k2 = w.get_index_of_position(x, y, z)
    # cell centres must map back to the originating cell
    np.testing.assert_array_equal(i2, i)
    np.testing.assert_array_equal(j2, j)
    np.testing.assert_array_equal(k2, k)


def test_grid_state_roundtrip_preserves_internal_energy():
    """set_grid_state(internal e) then get_grid_state must return the same
    internal energy density even with non-zero momentum — i.e. the
    total↔internal conversion is consistent."""
    w = _init_worker(8)
    i, j, k = w._all_indices()
    rng = np.random.default_rng(0)
    shape = i.shape
    rho = 1.0 + 0.5 * rng.random(shape)
    rhovx = rng.standard_normal(shape)
    rhovy = rng.standard_normal(shape)
    rhovz = rng.standard_normal(shape)
    rhoen = 1.0 + 0.5 * rng.random(shape)   # internal energy density > 0

    w.set_grid_state(i, j, k, rho, rhovx, rhovy, rhovz, rhoen)
    g_rho, g_mx, g_my, g_mz, g_eint = w.get_grid_state(i, j, k)

    np.testing.assert_allclose(g_rho, rho, rtol=0, atol=1e-12)
    np.testing.assert_allclose(g_mx, rhovx, rtol=0, atol=1e-12)
    np.testing.assert_allclose(g_my, rhovy, rtol=0, atol=1e-12)
    np.testing.assert_allclose(g_mz, rhovz, rtol=0, atol=1e-12)
    np.testing.assert_allclose(g_eint, rhoen, rtol=0, atol=1e-10)


def test_total_energy_stored_includes_kinetic():
    """The worker must store TOTAL energy internally (internal + kinetic)."""
    w = _init_worker(8)          # divisible by the suite's 8 virtual devices
    i, j, k = w._all_indices()
    rho = np.full(i.shape, 2.0)
    rhovx = np.full(i.shape, 4.0)   # vx = 2
    rhovy = np.zeros(i.shape)
    rhovz = np.zeros(i.shape)
    rhoen = np.full(i.shape, 3.0)   # internal energy density
    w.set_grid_state(i, j, k, rho, rhovx, rhovy, rhovz, rhoen)
    # E_total = rhoen + 0.5*|mom|^2/rho = 3 + 0.5*16/2 = 7
    E = np.asarray(w.state.U[4])
    np.testing.assert_allclose(E, 7.0, rtol=0, atol=1e-12)


def test_evolve_model_advances_time_and_conserves_mass():
    """A uniform medium evolved a short time stays finite and conserves total
    mass (periodic box, no sources)."""
    # 32, not 8: on 8 devices a 8^3 grid is ONE cell per shard in x, which is
    # thinner than the MUSCL stencil's two ghost cells.
    w = _init_worker(32)
    i, j, k = w._all_indices()
    rho = np.ones(i.shape)
    zeros = np.zeros(i.shape)
    rhoen = np.full(i.shape, 1.0 / (w.gamma - 1.0))  # P = 1
    w.set_grid_state(i, j, k, rho, zeros, zeros, zeros, rhoen)

    m0 = float(np.sum(np.asarray(w.state.U[0])))
    assert w.get_time() == 0.0
    dt = w.get_timestep()
    assert dt > 0 and np.isfinite(dt)

    w.evolve_model(0.05)
    assert w.get_time() == pytest.approx(0.05)
    U = np.asarray(w.state.U)
    assert np.all(np.isfinite(U))
    m1 = float(np.sum(U[0]))
    np.testing.assert_allclose(m1, m0, rtol=1e-10, atol=0)  # mass conserved


def _uniform_worker(t_end):
    """A 32^3 uniform medium at rest (P = 1) that ends at `t_end`."""
    w = LunaWorker(_build_cfg_for_test(
        test_problem="Soundwave-3D", SPATIAL_DIMENSION=3, BOUNDARY=0, Nx=32, Ny=32, Nz=32,
        Lx=1.0, Ly=1.0, Lz=1.0, gamma=1.4, ORDER=1, LIMITER=0, CFL=0.4, t_end=t_end))
    w.initialize_code()
    w.commit_parameters()
    i, j, k = w._all_indices()
    zeros = np.zeros(i.shape)
    w.set_grid_state(i, j, k, np.ones(i.shape), zeros, zeros, zeros,
                     np.full(i.shape, 1.0 / (w.gamma - 1.0)))
    return w


def test_evolve_model_lands_on_t_end_a_few_ulps_short():
    """A t_end target a few ulps short (a unit round trip) still ends at t == t_end exactly."""
    t_end = 0.05
    w = _uniform_worker(t_end)
    short = t_end - 3 * np.spacing(t_end)
    assert short != t_end
    assert w.evolve_model(short) == 0
    assert w.get_time() == w.get_end_time() == t_end


def test_evolve_model_to_a_past_target_is_a_no_op(capsys):
    """A target at or behind t changes nothing (bitwise) and says so; no zero-length step."""
    w = _uniform_worker(0.05)
    w.evolve_model(0.02)
    U0, t0, step0 = np.asarray(w.state.U), w.get_time(), w.get_current_step()
    capsys.readouterr()
    for target in (t0, 0.5 * t0):
        assert w.evolve_model(target) == 0
        assert "nothing to do" in capsys.readouterr().out
    assert (w.get_time(), w.get_current_step()) == (t0, step0)
    np.testing.assert_array_equal(np.asarray(w.state.U), U0)


def test_requires_initialization():
    # N must be divisible by the device count: conftest gives 8
    w = LunaWorker(_cfg(8))
    with pytest.raises(RuntimeError):
        w.get_time()
