"""The resident stepper an AMUSE worker drives.

What this file used to test was that the stepper's own copy of the per-step
pipeline reproduced the trusted bare loop bit-for-bit. That copy is gone --
`amuse_luna.stepper` now goes through `setup.assemble` + `setup.evolve`, the
same two functions production runs through -- so the physics is no longer
this module's to get wrong.

What IS still its own is the loop: substep on the CFL dt, land exactly on the
target, do nothing when the target is in the past, and survive being called
again and again with advancing targets, which is how a bridge drives it. So
that is what is tested, against the evolve primitives it is built from.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from amuse_luna.stepper import build                          # noqa: E402
from luna.setup.evolve import step as evolve_step             # noqa: E402
from luna.setup.evolve import timestep                        # noqa: E402


def _sod_cfg():
    """A small 3D problem. NOT 1D: `assemble` builds the sharding from the
    device mesh, and a rank-2 state cannot take a P(None,'x','y','z') spec on
    the suite's 8 virtual devices. The worker is 3D-only anyway."""
    from dataclasses import replace
    from luna.template_config import Config
    return replace(
        Config(), test_problem="Soundwave-3D", SPATIAL_DIMENSION=3,
        Nx=16, Ny=16, Nz=16, Lx=1.0, Ly=1.0, Lz=1.0, BOUNDARY=0,
        gamma=1.4, ORDER=1, LIMITER=0, t_end=0.02, enable_x64=True,
    )


def test_evolve_to_matches_the_primitives_it_is_built_from():
    """The adapter's loop must do exactly what an explicit timestep/step loop
    does -- same dt sequence, same state, bit for bit. If it differs, the
    clamping or the termination is wrong."""
    cfg = _sod_cfg()

    ref = build(cfg)
    s = ref.state
    while s.t < cfg.t_end:
        dt, t_new = timestep(s, ref.physics, cfg.t_end)
        s = evolve_step(s, dt, t_new, ref.physics)
    U_ref = np.asarray(s.U)

    got = build(cfg)
    got.evolve_to(cfg.t_end)

    np.testing.assert_array_equal(np.asarray(got.U), U_ref)


def test_evolve_to_lands_exactly_on_target():
    """No float drift: the last substep is clamped, not overshot."""
    cfg = _sod_cfg()
    state = build(cfg)
    state.evolve_to(cfg.t_end)
    assert state.t == cfg.t_end


def test_evolve_to_incremental_targets():
    """Repeated calls with advancing targets -- the bridge calling pattern."""
    cfg = _sod_cfg()
    state = build(cfg)

    for tgt in np.linspace(cfg.t_end / 4, cfg.t_end, 4):
        state.evolve_to(float(tgt))
        assert state.t == float(tgt)
        assert np.all(np.isfinite(np.asarray(state.U)))

    # a target in the past advances nothing and must not raise
    before = state.step
    assert state.evolve_to(cfg.t_end / 2) == 0
    assert state.step == before


def test_a_dt_below_the_clock_resolution_raises_instead_of_spinning():
    """Late in a run t + dt can round to t; the loop must stop loudly, not step forever at the same t."""
    state = build(_sod_cfg())
    t0 = 1.0e20                              # ulp 16384, far above the CFL dt
    state._s = state._s.update(t=t0)
    assert t0 + state.compute_dt() == t0
    with pytest.raises(RuntimeError, match="below the clock's resolution"):
        state.evolve_to(t0 + 1.0e6, max_steps=5)


def test_the_worker_surface_is_present():
    """The five attributes the AMUSE worker reads off the state, and the two
    it calls. This is the contract; breaking it breaks the bridge."""
    state = build(_sod_cfg())
    for name in ("U", "t", "step", "sources", "species"):
        assert hasattr(state, name), name
    assert callable(state.compute_dt) and callable(state.evolve_to)
    assert float(state.compute_dt()) > 0.0
    # U is writable -- set_grid_state and the gravity kicks assign to it
    state.U = state.U
