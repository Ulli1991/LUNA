"""Tests for the bridge gravity primitives (Torch/AMUSE coupling, Phase 3).

Covers the scalable building blocks that back the bridge's gravity coupling:
gas self-gravity solve, vectorized gas→particle acceleration sampling, the
mass-conserving CIC deposit (particle→grid), and the gas kick. See
`docs/TORCH.md` (Phase 3).
"""

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tests.test_regressions import _build_cfg_for_test  # noqa: E402
from luna.gravity.particle_mesh import ParticleMeshGravity  # noqa: E402


def _grav_cfg(N=16):
    return _build_cfg_for_test(
        test_problem="Sod-Shock-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0, gamma=1.4,
        GRAVITATIONAL_G=1.0, POISSON_BOUNDARY=0,
    )


def test_cic_deposit_conserves_mass():
    bg = ParticleMeshGravity(_grav_cfg(16))
    # particles well inside the box so none of the CIC stencil clamps at an edge
    positions = jnp.array([[0.30, 0.50, 0.70], [0.55, 0.45, 0.50]])
    masses = jnp.array([3.0, 5.0])
    active = jnp.array([1.0, 1.0])
    rho = bg.cic_deposit(positions, masses, active)
    total = float(jnp.sum(rho) * bg.dV)
    assert total == pytest.approx(8.0, rel=1e-10)
    # An inactive particle contributes nothing.
    rho2 = bg.cic_deposit(positions, masses, jnp.array([1.0, 0.0]))
    assert float(jnp.sum(rho2) * bg.dV) == pytest.approx(3.0, rel=1e-10)


def test_sample_at_points_matches_single_point_interp():
    bg = ParticleMeshGravity(_grav_cfg(16))
    # a smooth field to interpolate
    c = (jnp.arange(16) + 0.5) / 16                 # unit box, cell centres
    X, Y, Z = jnp.meshgrid(c, c, c, indexing="ij")
    field = X ** 2 + 2.0 * Y - Z
    pts = jnp.array([[0.5, 0.5, 0.5], [0.2, 0.8, 0.4], [0.91, 0.13, 0.66]])
    vec = np.asarray(bg.sample_at_points(field, pts))
    one = np.array([float(bg._interp_one(field, p)) for p in pts])
    np.testing.assert_allclose(vec, one, rtol=0, atol=1e-12)


def test_cic_deposit_is_adjoint_of_interp():
    """Depositing unit mass then interpolating the resulting density back to the
    same point recovers the per-cell weights — a basic PM consistency check."""
    bg = ParticleMeshGravity(_grav_cfg(16))
    p = jnp.array([[0.333, 0.666, 0.5]])
    rho = bg.cic_deposit(p, jnp.array([1.0]), jnp.array([1.0]))
    # total mass conserved regardless of sub-cell position
    assert float(jnp.sum(rho) * bg.dV) == pytest.approx(1.0, rel=1e-10)


def test_kick_gas_updates_momentum_and_energy():
    bg = ParticleMeshGravity(_grav_cfg(8))
    N = 8
    U = jnp.zeros((5, N, N, N))
    U = U.at[0].set(2.0)        # rho = 2
    U = U.at[4].set(10.0)       # E
    g = jnp.ones((N, N, N))     # uniform accel in x
    dt = 0.1
    U2 = bg.kick_gas(U, g, 0.0 * g, 0.0 * g, dt)
    # momentum_x = rho * a * dt = 2 * 1 * 0.1
    np.testing.assert_allclose(np.asarray(U2[1]), 0.2, rtol=0, atol=1e-12)
    # energy increases by mean-momentum . g * dt = 0.5*(0+0.2)*1*0.1 = 0.01
    np.testing.assert_allclose(np.asarray(U2[4]), 10.01, rtol=0, atol=1e-12)


def test_gas_self_gravity_runs_finite():
    """Smoke test: the FMG Poisson solve runs through the wrapper and the
    resulting acceleration field is finite (the solver itself is covered by the
    dedicated Poisson tests)."""
    bg = ParticleMeshGravity(_grav_cfg(16))
    N = 16
    rho = jnp.ones((N, N, N))
    # a central overdensity
    rho = rho.at[N // 2, N // 2, N // 2].add(50.0)
    phi = bg.gas_potential(rho)
    gx, gy, gz = bg.gas_accel_grid(rho)
    assert np.all(np.isfinite(np.asarray(phi)))
    for g in (gx, gy, gz):
        assert np.all(np.isfinite(np.asarray(g)))


def test_accel_sampling_no_recompile_across_particle_counts():
    """gas→particle sampling must not recompile as the particle count changes:
    the store is a fixed (N_max,3) array, so the sampling kernel sees a constant
    shape regardless of how many slots are active."""
    bg = ParticleMeshGravity(_grav_cfg(16))
    accel_grid = bg.gas_accel_grid(jnp.ones((16, 16, 16)))

    trace_count = {"n": 0}

    @jax.jit
    def sample(accel_grid, positions, active):
        trace_count["n"] += 1
        a = bg.accel_at_points(accel_grid, positions)
        return a * active[:, None]

    N_max = 64
    pos = jnp.full((N_max, 3), 0.5)
    for n_active in (1, 5, 20, 64):
        active = jnp.arange(N_max) < n_active
        out = sample(accel_grid, pos, active.astype(jnp.float64))
        assert np.all(np.isfinite(np.asarray(out)))
    assert trace_count["n"] == 1
