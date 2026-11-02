"""The wind's injection bounds the timestep it is applied with.

Strang-splitting the CC85 injection puts half the mass and energy into the
injection cells BEFORE the hydro step, while dt was computed on the state
before any injection. In a cold box the sound speed those cells reach over
h = dt/2 then violates the CFL condition within the very same step: the M82
setup at 32 cells (10 kpc box, 300 pc sphere) went to a negative pressure
after step 1, on the old dense runner too (bisected source by source,
2026-09-20). The wind stage now declares the hydro rule's dt at rest at the
sound speed it creates, dt <= (CFL / sum_d (1/dx_d) sqrt(2 rho / (gamma
(gamma-1) e_dot)))^(2/3), and the loop takes the minimum.

Gate: the M82 configuration at 32 cells, adaptive dt, runs 12 finite steps
through main.run, and its first step is well below the hydro's own CFL dt.
Teeth: with the wind's limit removed (the stage's dt_limit deleted on the
assembled physics) the same run is non-finite within the first steps. GPU.
"""

import numpy as np
import pytest

import jax



pytestmark = pytest.mark.slow

from scripts.examples.autodiff.m82_autodiff_run import config as m82_config   # noqa: E402

N_STEPS = 12


def _cfg(tmp_path):
    from dataclasses import replace
    cfg = m82_config(nx=32, selfgrav=False)
    return replace(cfg, autodiff_mode=False, max_steps=N_STEPS, write_snapshots=False,
                   enable_diagnostics=False, output_cadence=1,
                   output_dir=str(tmp_path / "out"))


def _run(cfg, with_limit):
    """main.run's path on ONE device (the wind physics does not depend on the
    device count, and a two-device dense run of this setup inside pytest
    stalls at the loop's first CFL call on ROCm, the async-pileup class),
    with the wind's limit optionally removed after assembly (the wind stage
    replaced by a plain CellLocal stage with the same apply)."""
    from jax.sharding import Mesh
    from luna.setup.assemble import assemble
    from luna.setup.ics import gas_grid
    from luna.setup.evolve import evolve
    from luna.setup.ics import Initial
    from luna.setup.mesh import DENSE
    from luna.setup.run_kind import GAS
    from luna.setup.state import State
    from luna.sources.stages import CellLocal
    cfg.validate()
    kind, layout = GAS, DENSE
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1), ("x", "y", "z"))
    U0, _, _ = gas_grid(cfg, mesh)
    init = Initial(cfg=cfg, U=U0)
    physics = assemble(init.cfg, mesh, kind, layout, init, is_main=False)
    i = next(k for k, s in enumerate(physics.stages) if s.name == "cc85")
    wind = physics.stages[i]
    cfl_dt = float(physics.hydro.compute_dt(init.U))
    limit_dt = float(wind.dt_limit(State(U=init.U)))   # a setup-time constant
    if not with_limit:
        physics.stages[i] = CellLocal("cc85", wind._apply)
    try:
        state = evolve(init.cfg, init, physics, mesh, is_main=False)
        return state, cfl_dt, limit_dt, None
    except RuntimeError as e:                    # the loop's non-finite guard
        return None, cfl_dt, limit_dt, e


def test_wind_limit_keeps_the_cold_box_finite(tmp_path):
    state, cfl_dt, limit_dt, err = _run(_cfg(tmp_path), with_limit=True)
    print(f"[wind] hydro CFL dt={cfl_dt:.3e}  wind limit dt={limit_dt:.3e}  "
          f"ratio={limit_dt / cfl_dt:.3e}")
    assert err is None, f"finite run expected, got: {err}"
    assert state.step == N_STEPS
    assert np.isfinite(np.asarray(state.U)).all()
    assert limit_dt < 0.1 * cfl_dt          # the limit, not the CFL, set the first step


def test_without_the_limit_the_first_steps_blow_up(tmp_path):
    state, cfl_dt, limit_dt, err = _run(_cfg(tmp_path), with_limit=False)
    assert err is not None, "the run without the wind's limit stayed finite: no teeth"
    assert "non-finite" in str(err)
