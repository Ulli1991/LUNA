"""Tests for LUNA's sink-moment exposure to Torch (Phase 5 / #3b).

Injects a sink into a known gas flow and checks `get_sink_gas_moments` returns
the right mean sound speed, mean velocity, and angular momentum — the quantities
Torch's `make_stars_from_sinks` reads to seed formed-star kinematics.
"""
import os

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp  # noqa: E402

from tests.test_regressions import _build_cfg_for_test  # noqa: E402
from amuse_luna.worker import LunaWorker  # noqa: E402


def _worker(N=16):
    cfg = _build_cfg_for_test(
        test_problem="Sod-Shock-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0, gamma=1.4,
        ORDER=1, LIMITER=1, GRAVITATIONAL_G=1.0, POISSON_BOUNDARY=0,
    )
    w = LunaWorker(cfg)
    w.initialize_code()
    w.commit_parameters()
    return w


def _inject_sink(w, p):
    """Put a single active sink at position p.

    Sinks are a SPECIES on current main -- `state.species[Sinks.name]`, a
    `Particles` -- not the source carry they were before assemble/evolve.
    `extra` is left None on purpose: that is a sink with no accretion history,
    which is exactly when `get_sink_gas_moments` must fall back to the
    local-gas estimate these tests measure."""
    from luna.setup.state import Particles
    from luna.species.sinks import Sinks
    N = int(getattr(w.cfg, "sink_max_count", 4))
    pos = np.zeros((N, 3)); pos[0] = p
    mass = np.zeros(N); mass[0] = 1.0
    active = np.zeros(N); active[0] = 1.0
    w.state.species[Sinks.name] = Particles(
        jnp.asarray(pos), jnp.zeros((N, 3)), jnp.asarray(mass),
        jnp.asarray(active))


def test_no_sinks_returns_empty():
    w = _worker(8)
    assert w.get_number_of_sinks() == 0
    m = w.get_sink_gas_moments()
    assert m["mean_cs"].shape == (0,)


def test_sink_moments_in_rotating_flow():
    w = _worker(16)
    N = 16
    gamma = w.gamma
    # gas: rho=1, solid-body rotation about z through the box centre, uniform P
    c = (np.arange(N) + 0.5) / N                     # unit box, cell centres
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    cx = cy = cz = 0.5
    Omega = 2.0
    rho = np.ones((N, N, N))
    vx = -Omega * (Y - cy)
    vy = Omega * (X - cx)
    vz = np.zeros_like(rho)
    P = 0.6
    cs_expected = np.sqrt(P / 1.0)              # isothermal: Torch's convention
    E = P / (gamma - 1.0) + 0.5 * rho * (vx**2 + vy**2 + vz**2)
    U = np.stack([rho, rho * vx, rho * vy, rho * vz, E])
    w.state.U = jnp.asarray(U)

    _inject_sink(w, [cx, cy, cz])
    assert w.get_number_of_sinks() == 1

    m = w.get_sink_gas_moments(r_acc=0.2)
    # mean sound speed of the (uniform-P, uniform-rho) gas
    np.testing.assert_allclose(m["mean_cs"][0], cs_expected, rtol=1e-6)
    # solid-body rotation about the sink at the centre -> ~zero net velocity
    assert np.linalg.norm(m["mean_velocity"][0]) < 1e-6
    # angular momentum about +z must be positive (counter-clockwise rotation)
    Lx, Ly, Lz = m["angular_momentum"][0]
    assert Lz > 0.0
    assert abs(Lx) < 1e-6 and abs(Ly) < 1e-6
    # rotation gives nonzero velocity dispersion in x and y, ~0 in z
    vvar = m["var_velocity"][0]
    assert vvar[0] > 0 and vvar[1] > 0
    assert vvar[2] < 1e-12


def test_sink_mean_velocity_tracks_bulk_flow():
    """A uniform bulk flow -> sink sees that mean velocity, ~zero dispersion."""
    w = _worker(16)
    N = 16
    gamma = w.gamma
    rho = np.ones((N, N, N))
    vbulk = np.array([0.3, -0.2, 0.1])
    vx = np.full_like(rho, vbulk[0]); vy = np.full_like(rho, vbulk[1]); vz = np.full_like(rho, vbulk[2])
    P = 0.6
    E = P / (gamma - 1.0) + 0.5 * rho * (vx**2 + vy**2 + vz**2)
    w.state.U = jnp.asarray(np.stack([rho, rho * vx, rho * vy, rho * vz, E]))
    _inject_sink(w, [0.5, 0.5, 0.5])
    m = w.get_sink_gas_moments(r_acc=0.2)
    np.testing.assert_allclose(m["mean_velocity"][0], vbulk, rtol=1e-6)
    assert np.all(m["var_velocity"][0] < 1e-12)        # uniform flow -> no dispersion
