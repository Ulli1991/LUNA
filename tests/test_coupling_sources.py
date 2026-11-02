"""The star -> radiation handoff at the coupling boundary.

Torch owns the stars and, through SeBa, their ionizing photon rates. LUNA owns
the radiation field. `LunaWorker.set_ionizing_sources` is the seam, and the
thing to check is that it is a seam and not a leak: the photons Torch asks for
are the photons that go in, the field survives a step, and turning a star on
does not retrace the jitted stage.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from tests.test_regressions import _build_cfg_for_test        # noqa: E402
from amuse_luna.worker import LunaWorker                      # noqa: E402
from tests._helpers import check                              # noqa: E402

pytestmark = pytest.mark.fast

CHAT = 30.0
HELD_MIN = 0.9999       # measured 0.999977 (CPU, A100): one step, few recombinations


def _cfg(N=16):
    """A small ionizing run: radiation on, ionization on, one group."""
    from dataclasses import replace
    cfg = _build_cfg_for_test(
        test_problem="Soundwave-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0, gamma=5.0 / 3.0,
        ORDER=1, LIMITER=0, enable_x64=True)
    return replace(
        cfg, radiation_enabled=True, rad_ionization=True, rad_groups=1,
        rad_chat=CHAT, rad_E0=0.0, rad_substeps=1,
        rad_sigma=1.0, rad_alpha_B=1.0e-3, rad_m_H=1.0,
        rad_photon_rate=0.0,            # no built-in source; the stars are it
        rad_cs2_neutral=1.0, rad_cs2_ionized=4.0, rad_x0=0.0)


def _worker(N=16):
    """A worker with gas in it. `initialize_code` leaves the grid zero-filled
    -- the AMUSE pattern is that the orchestrator fills it -- and a zero
    density makes the very first dt non-finite."""
    w = LunaWorker(_cfg(N))
    w.initialize_code()
    i, j, k = w._all_indices()
    ones = np.ones(i.shape)
    zeros = np.zeros(i.shape)
    w.set_grid_state(i, j, k, ones, zeros, zeros, zeros,
                     np.full(i.shape, 1.0 / (w.gamma - 1.0)))   # P = 1
    return w


def test_the_photons_torch_asks_for_are_the_photons_that_go_in():
    """The seam's one job. If the total drifts, the ionization front is wrong
    and nothing else will say so."""
    w = _worker()
    pos = np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5], [0.5, 0.25, 0.5]])
    rates = np.array([2.0, 3.0, 5.0])
    w.set_ionizing_sources(pos, rates)
    assert abs(w.get_ionizing_photon_rate() - 10.0) < 1e-10


def test_inactive_stars_inject_nothing():
    """The star array is fixed-capacity and mostly empty."""
    w = _worker()
    pos = np.array([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5], [0.5, 0.25, 0.5]])
    rates = np.array([2.0, 3.0, 5.0])
    w.set_ionizing_sources(pos, rates, active=np.array([1.0, 0.0, 1.0]))
    assert abs(w.get_ionizing_photon_rate() - 7.0) < 1e-10


def test_the_sources_survive_a_step():
    """The field lives in the carry, so the stage must hand it back unchanged
    -- otherwise the stars stop shining after one step."""
    w = _worker()
    w.set_ionizing_sources(np.array([[0.5, 0.5, 0.5]]), np.array([4.0]))
    before = w.get_ionizing_photon_rate()
    w.evolve_model(w.get_timestep())
    assert abs(w.get_ionizing_photon_rate() - before) < 1e-10


def test_radiation_energy_grows_once_the_stars_are_on():
    """End to end: no sources, nothing happens; sources, photons appear."""
    import jax.numpy as jnp
    w = _worker()
    dt = w.get_timestep()
    w.evolve_model(dt)
    quiet = float(jnp.sum(w.state.sources["radiation"][0]))
    assert quiet == 0.0                      # rad_E0 = 0 and no source yet

    w.set_ionizing_sources(np.array([[0.5, 0.5, 0.5]]), np.array([4.0]))
    U0, t0 = np.asarray(w.state.U), w.get_time()
    w.evolve_model(w.get_time() + dt)
    assert float(jnp.sum(w.state.sources["radiation"][0])) > 0.0
    # ionisation touches neither rho nor momentum (no force on the ionising path): periodic hydro keeps both;
    # gas energy is not conserved by design (two_temperature sets e_int)
    U = np.asarray(w.state.U)
    scale = np.sqrt(U0[0].sum() * U0[4].sum())
    check("ionising worker |dM|/M", abs(U[0].sum() / U0[0].sum() - 1.0), 1e-13)
    check("ionising worker |dP|/sqrt(M E)", float(np.abs(U[1:4].sum((1, 2, 3)) - U0[1:4].sum((1, 2, 3))).max()) / scale,
          1e-13)
    # photons: stored + ionisations <= injected (alpha_B > 0 recombines the rest; nothing leaves a periodic box)
    R, x = w.state.sources["radiation"][:2]
    dV = 1.0 / U.shape[1] ** 3
    held = float(jnp.sum(R[0, 0]) + jnp.sum(U[0] * x)) * dV / (4.0 * (w.get_time() - t0))
    check("ionising worker (stored + ionised) / injected", held, 1.0 + 1e-12)
    check("ionising worker (stored + ionised) / injected", held, HELD_MIN, lower=True)


def test_moving_the_stars_is_data_not_recompilation():
    """A cluster moves its sources every bridge step. This checks the thing
    that makes that affordable: setting them does not rebuild the assembled
    physics, and the totals stay right as they move. The compile count itself
    is asserted where it belongs, on `deposit_sources` in
    test_radiation_sources.py -- claiming it here would need the stage's jit
    cache, which assemble does not expose."""
    w = _worker()
    rng = np.random.default_rng(0)
    physics_before = w.state.physics
    for _ in range(4):
        pos = rng.uniform(0.2, 0.8, size=(5, 3))
        w.set_ionizing_sources(pos, np.full(5, 1.0))
        w.evolve_model(w.get_time() + w.get_timestep())
        assert abs(w.get_ionizing_photon_rate() - 5.0) < 1e-10
    assert w.state.physics is physics_before
