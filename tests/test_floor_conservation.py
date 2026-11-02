"""The density floor (cosmo_rho_floor) takes the mass it adds from the floored cells' 3x3x3
neighbours, any shortfall from the whole grid: mass conserved to round-off on one device and
on a (2,2,2) mesh; a state the floor does not touch comes back bitwise."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import jax
import jax.numpy as jnp  # noqa: E402

from luna.setup.assemble import build_projections  # noqa: E402
from luna.setup.clock import make_clock  # noqa: E402
from luna.setup.run_kind import GAS  # noqa: E402
from luna.sources.cosmo_floors import CosmoFloors  # noqa: E402
from tests._helpers import _build_cfg_for_test  # noqa: E402

GAMMA = 5.0 / 3.0
N = 8
RO = 1e-14          # round-off, of the total mass


def _uniform(rho=1.0, v=(0.3, -0.2, 0.1), P=1.0):
    U = np.zeros((5, N, N, N))
    U[0] = rho
    for d in range(3):
        U[1 + d] = rho * v[d]
    U[4] = P / (GAMMA - 1.0) + 0.5 * rho * sum(c * c for c in v)
    return U


def _floors(periodic=True, mesh=None, floor_frac=1e-3, buffer=1.0, rho_bar=1.0):
    return CosmoFloors(gamma=GAMMA, coeff_fac=1.0, dx=1.0 / N, rho_bar=rho_bar, p_floor=0.0,
                       jeans_ncell=0.0, T_floor=10.0, T_max=0.0, v_max=0.0, mu=2.3,
                       rho_floor_frac=floor_frac, floor_buffer=buffer, mesh=mesh,
                       periodic=periodic)


def _mesh8():
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh
    return Mesh(mesh_utils.create_device_mesh((2, 2, 2), jax.devices()[:8]), ("x", "y", "z"))


def _on(mesh, U):
    from jax.sharding import NamedSharding, PartitionSpec as P
    return jax.device_put(jnp.asarray(U), NamedSharding(mesh, P(None, "x", "y", "z")))


def test_the_neighbours_pay_for_a_floored_cell():
    """One vacuum cell: rho_floor - rho leaves its 26 neighbours, at their own velocity and specific energy."""
    U = _uniform()
    U[0, 4, 4, 4] = -2e-3
    U[1:4, 4, 4, 4] = 0.7
    fl = _floors()
    V = np.asarray(fl(jnp.asarray(U), 1.0))
    floor = fl.rho_floor
    assert V[0, 4, 4, 4] == floor and np.all(V[1:4, 4, 4, 4] == 0.0)
    np.testing.assert_allclose(V[0].sum(), U[0].sum(), rtol=RO)
    nb = np.zeros((N, N, N), bool)
    nb[3:6, 3:6, 3:6] = True
    nb[4, 4, 4] = False
    give = (floor + 2e-3) / 26.0
    np.testing.assert_allclose(U[0][nb] - V[0][nb], give, rtol=1e-12)
    for k in range(1, 5):                     # velocity and specific energy kept
        np.testing.assert_allclose(V[k][nb] / V[0][nb], U[k][nb] / U[0][nb], rtol=1e-14)
    far = ~nb
    far[4, 4, 4] = False
    np.testing.assert_array_equal(V[:, far], U[:, far])


def test_a_hole_too_deep_for_its_neighbours_borrows_from_the_grid():
    """A 4^3 block of vacuum: its inner cells have no donors; the grid covers them, no cell under the floor."""
    U = _uniform(rho=0.05)
    U[0, 2:6, 2:6, 2:6] = 1e-9
    fl = _floors(floor_frac=1.0, rho_bar=0.04)        # the floor 0.04 of a 0.05 box
    V = np.asarray(fl(jnp.asarray(U), 1.0))
    np.testing.assert_allclose(V[0].sum(), U[0].sum(), rtol=RO)
    assert V[0].min() >= fl.rho_floor * (1.0 - 1e-14)
    np.testing.assert_array_equal(V[0, 2:6, 2:6, 2:6], fl.rho_floor)
    # every donor gave: the hole needs 64 x 0.04, the 26-neighbourhoods hold only part of it
    hole = np.zeros((N, N, N), bool)
    hole[2:6, 2:6, 2:6] = True
    assert np.all(V[0][~hole] < U[0][~hole])


def test_no_donation_through_a_wall():
    """Non-periodic: a cell on the x_lo face takes nothing from the x_hi face."""
    U = _uniform()
    U[0, 0, 4, 4] = 1e-9
    V = np.asarray(_floors(periodic=False)(jnp.asarray(U), 1.0))
    np.testing.assert_array_equal(V[:, N - 1], U[:, N - 1])
    np.testing.assert_allclose(V[0].sum(), U[0].sum(), rtol=RO)
    assert np.all(V[0, 0:2, 3:6, 3:6] < 1.0)                 # 17 neighbours inside the box gave
    W = np.asarray(_floors(periodic=True)(jnp.asarray(U), 1.0))
    assert np.all(W[0, N - 1, 3:6, 3:6] < 1.0)               # periodic: the far face gives


def test_untouched_state_is_bitwise():
    """No cell under the floor: the stack's output equals the stack without a density floor."""
    U = jnp.asarray(_uniform())
    a = np.asarray(_floors()(U, 1.0))
    b = np.asarray(_floors(floor_frac=0.0)(U, 1.0))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[:4], np.asarray(U)[:4])


@pytest.mark.skipif(jax.device_count() < 8, reason="a (2,2,2) mesh needs 8 devices")
@pytest.mark.parametrize("periodic", [True, False])
def test_sharded_is_one_device(periodic):
    """Floored cells on shard seams and corners: the (2,2,2) mesh gives the one-device grid."""
    rng = np.random.default_rng(3)
    U = _uniform()
    U[0] *= 1.0 + 0.5 * rng.random((N, N, N))
    for c in [(3, 3, 3), (4, 4, 4), (0, 0, 0), (7, 3, 4), (3, 7, 0), (1, 2, 5)]:
        U[0][c] = -1e-4
    mesh = _mesh8()
    one = np.asarray(_floors(periodic)(jnp.asarray(U), 1.0))
    many = _floors(periodic, mesh=mesh)(_on(mesh, U), 1.0)
    np.testing.assert_allclose(np.asarray(many), one, rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(one[0].sum(), U[0].sum(), rtol=RO)


def _sedov(**kw):
    cfg = _build_cfg_for_test(
        test_problem="Sedov-Blast-3D", SPATIAL_DIMENSION=3, BOUNDARY=0, Nx=16, Ny=16, Nz=16,
        Lx=1.0, Ly=1.0, Lz=1.0, gamma=GAMMA, ORDER=2, LIMITER=0, enable_x64=True,
        write_snapshots=False, t_end=1.0, cosmo_rho_floor=0.3, cosmo_T_floor=1e-6, cosmo_mu=1.0)
    return replace(cfg, **kw)


def _run(cfg, steps=40):
    """(initial mass, final mass, floor, lowest density seen, cell-steps on the floor)."""
    from amuse_luna.stepper import build
    st = build(cfg)
    dV = (cfg.Lx / cfg.Nx) ** 3
    m0 = float(jnp.sum(st.U[0])) * dV
    floor = cfg.cosmo_rho_floor * m0 / (cfg.Lx * cfg.Ly * cfg.Lz)
    n_hit, low = 0, np.inf
    for _ in range(steps):
        st.step_once(st.compute_dt())
        rho = np.asarray(st.U[0])
        n_hit += int(np.sum(np.isclose(rho, floor, rtol=1e-12)))
        low = min(low, float(rho.min()))
    return m0, float(jnp.sum(st.U[0])) * dV, floor, low, n_hit


@pytest.mark.skipif(jax.device_count() < 8, reason="a (2,2,2) mesh needs 8 devices")
def test_a_blast_on_the_floor_conserves_mass():
    """Sedov in a periodic box on a (2,2,2) mesh, the floor at 0.3 of the mean: the interior sits on it, no mass appears."""
    m0, m1, floor, low, n_hit = _run(_sedov(MESH_SHAPE=(2, 2, 2)))
    assert low >= floor * (1.0 - 1e-12)                        # the floor holds
    assert n_hit > 100                                         # cell-steps at the floor
    assert abs(m1 - m0) <= RO * m0, (m1 - m0) / m0


def test_the_projection_built_for_a_run_is_conservative():
    """build_projections passes the run's boundary: a wall run gets the wall floor."""
    cfg = _sedov(BOUNDARY=1)
    U = _uniform()
    U[0, 0, 0, 0] = 1e-9
    proj = build_projections(cfg, make_clock(cfg), None, GAS, SimpleNamespace(U=jnp.asarray(U)),
                             None, is_main=False)
    (fl,) = [p for p in proj if isinstance(p, CosmoFloors)]
    V = np.asarray(fl(jnp.asarray(U), 1.0))
    np.testing.assert_allclose(V[0].sum(), U[0].sum(), rtol=RO)
    np.testing.assert_array_equal(V[:, N - 1], U[:, N - 1])
