"""cfg.timing_every: a timed step is the same step part by part, and timing off is the old step.

Bitwise: a driven self-gravitating box (main loop), the Tier-1 cloud with radiation (AMUSE stepper)."""
import json
import os
from dataclasses import replace

import jax
import numpy as np
import pytest


N_STEPS = 5


def _driven_cfg(out, **kw):
    from luna.hydro.turbulence import OUDriverConfig
    from luna.template_config import Config
    ou = OUDriverConfig(k_min=1.0, k_max=2.0, solenoidal_weight=1.0, autocorrelation_time=1.3,
                        rms_acceleration=0.25, seed=7, spectrum="parabolic")
    return replace(Config(), test_problem="Driven-Turbulence-3D", SPATIAL_DIMENSION=3,
                   BOUNDARY=0, Nx=32, Ny=32, Nz=32, gamma=5.0 / 3.0, ORDER=1, LIMITER=0,
                   t_end=10.0, max_steps=N_STEPS, ou_config=ou, ou_seed=7,
                   isothermal_cs2=1.0, SELF_GRAVITY=True, GRAVITATIONAL_G=1.0,
                   POISSON_BOUNDARY=0, write_snapshots=False, PLOT_AFTER_SIM=False,
                   enable_checkpointing=False, enable_diagnostics=True, diag_interval=2,
                   output_cadence=1, enable_x64=True, output_dir=str(out), **kw)


def _physics(cfg):
    from luna.setup.assemble import assemble
    from luna.setup.ics import read_ics
    from luna.setup.mesh import build_mesh, decomposition
    from luna.setup.run_kind import run_kind
    kind = run_kind(cfg)
    layout = decomposition(cfg, kind)
    mesh, _ = build_mesh(cfg, layout, kind)
    init = read_ics(cfg, mesh, kind, layout, is_main=False)
    return init, assemble(cfg, mesh, kind, layout, init, is_main=False), mesh


def _run_loop(cfg):
    from luna.setup.evolve import evolve
    init, physics, mesh = _physics(cfg)
    return evolve(cfg, init, physics, mesh, is_main=jax.process_index() == 0)


def _host(state):
    """Every array the state carries, on the host, by path."""
    leaves = jax.tree_util.tree_leaves_with_path(
        {"U": state.U, "psi": state.psi, "g": state.g_gas, "src": state.sources,
         "sp": state.species})
    return {jax.tree_util.keystr(k): np.asarray(v) for k, v in leaves if v is not None}


def _differ(a, b):
    """The paths whose bytes differ between two host states."""
    assert a.keys() == b.keys()
    return [k for k in a if a[k].tobytes() != b[k].tobytes()]


@pytest.fixture(scope="module")
def driven(tmp_path_factory):
    """The driven box untimed, then timed on every step and every 2nd step."""
    base = tmp_path_factory.mktemp("timing")
    runs = {}
    for every in (0, 1, 2):
        out = base / f"every{every}"
        state = _run_loop(_driven_cfg(out, timing_every=every))
        runs[every] = (_host(state), out, state)
    return runs


def test_timed_steps_are_bitwise_the_untimed_steps(driven):
    ref = driven[0][0]
    assert driven[0][2].step == N_STEPS
    for every in (1, 2):
        assert _differ(ref, driven[every][0]) == [], f"timing_every={every} changed the state"


def test_timing_writes_one_record_per_timed_step_with_every_part(driven):
    assert not os.path.exists(driven[0][1] / "step_timing.jsonl")
    for every, steps in ((1, [1, 2, 3, 4, 5]), (2, [2, 4])):
        recs = [json.loads(line) for line in open(driven[every][1] / "step_timing.jsonl")]
        assert [r["step"] for r in recs] == steps
        parts = recs[-1]["parts_ms"]
        for name in ("dt.hydro", "dt.gravity", "forcing.before", "gravity.before", "hydro",
                     "gravity.after", "forcing.after", "project.isothermal", "gravity.check",
                     "outputs", "gravity.before/gravity.kick_gas",
                     "gravity.after/gravity.solve", "gravity.after/gravity.kick_gas",
                     "gravity.after/gravity.source"):
            assert name in parts, f"{name} missing from {sorted(parts)}"
        top = sum(v[1] for k, v in parts.items() if "/" not in k)
        assert 0.0 < top <= recs[-1]["total_ms"][1] * 1.001
        assert all(lo <= hi for lo, hi in parts.values())
    assert "untimed_ms_per_step" in recs[-1]


def _main_step(state, dt, t_new, physics):
    """The step as it was before the timing (origin/main), the reference for timing off."""
    from luna.setup.evolve import BUDGET
    h = 0.5 * dt
    for stage in physics.stages:
        state = stage.before(state, h, dt)
    if physics.hydro is not None:
        U0 = state.U
        state = state.update(U=physics.hydro.step(U0, dt))
        face_mass = getattr(physics.hydro, "face_mass", None)
        if face_mass is not None:
            state = state.with_source(BUDGET, face_mass(U0, dt, state.sources[BUDGET]))
    state = state.update(t=t_new, a=physics.clock.scale(t_new))
    for stage in reversed(physics.stages):
        state = stage.after(state, h, dt)
    U = state.U
    for project in physics.projections:
        U = project(U, state.a)
    return state.update(U=U, step=state.step + 1)


def test_timing_off_is_the_old_step(driven, tmp_path):
    from luna.setup.evolve import initial_state, release_initial, timestep
    cfg = _driven_cfg(tmp_path)
    init, physics, mesh = _physics(cfg)
    state = initial_state(init, physics)
    release_initial(init, physics)
    state = physics.gravity.init(state)
    for _ in range(N_STEPS):
        dt, t_new = timestep(state, physics, float("inf"))
        state = _main_step(state, dt, t_new, physics)
    assert _differ(_host(state), driven[0][0]) == []


def test_the_comparison_catches_parts_out_of_order(driven, tmp_path, monkeypatch):
    """Teeth: a timed path that runs the after halves in the before order must differ."""
    import luna.setup.evolve as ev
    good = ev.step

    def misordered(state, dt, t_new, physics, timer=None):
        if timer is None:
            return good(state, dt, t_new, physics)
        swapped = replace(physics, stages=list(physics.stages))
        swapped.stages.reverse()
        return good(state, dt, t_new, swapped, timer)
    monkeypatch.setattr(ev, "step", misordered)
    bad = _host(_run_loop(_driven_cfg(tmp_path, timing_every=1)))
    assert _differ(driven[0][0], bad), "a misordered timed step went unnoticed"


def test_coupled_stepper_timed_is_bitwise(tmp_path):
    """The Tier-1 cloud with radiation and sinks through the AMUSE stepper, timed and not."""
    import runpy
    from amuse_luna.stepper import build
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = runpy.run_path(os.path.join(here, "scripts/coupling/tier1/luna_cloud_rad.py"))["Config"]
    states = {}
    for every in (0, 1):
        cfg = replace(base, Nx=32, Ny=32, Nz=32, rad_substeps=2, timing_every=every,
                      output_dir=str(tmp_path / f"cloud{every}"), write_snapshots=False)
        luna = build(cfg)
        for _ in range(3):
            luna.step_once(luna.compute_dt())
        luna.evolve_to(luna.t + 2.5 * luna.compute_dt())
        states[every] = _host(luna.state)
    assert _differ(states[0], states[1]) == []
    recs = [json.loads(line) for line in open(tmp_path / "cloud1" / "step_timing.jsonl")]
    assert recs and {"radiation.before", "radiation.after", "gravity.after/gravity.solve",
                     "hydro"} <= set(recs[0]["parts_ms"])


def test_table_flags_parts_slower_than_the_hydro():
    from luna.setup.step_timing import format_table, over_hydro, tree_order
    names = ["a/x", "a", "hydro", "b/y/z", "b/y", "b"]
    assert tree_order(names) == ["a", "a/x", "hydro", "b", "b/y", "b/y/z"]
    parts = {"hydro": [50.0, 55.0], "forcing.after": [100.0, 103.0], "dt.hydro": [1.0, 1.0],
             "forcing.after/kick": [90.0, 99.0]}
    assert over_hydro(parts) == ["forcing.after"]
    rec = {"step": 3, "ranks": 2, "total_ms": [150.0, 170.0], "parts_ms": parts}
    table = format_table(rec)
    assert "> hydro" in table.split("forcing.after")[1].split("\n")[0]
    assert "step total" in table


def test_breakdown_median_counts_a_skipped_part_as_zero_and_lists_spikes():
    from tools.step_breakdown import report, summarize
    recs = [{"step": s, "ranks": 1, "total_ms": [100.0 + (900.0 if s == 5 else 0.0)] * 2,
             "parts_ms": {"hydro": [50.0, 50.0], "gravity.before": [10.0, 10.0],
                          **({"gravity.before/gravity.exchange": [900.0, 900.0]} if s == 5 else {})}}
            for s in range(1, 8)]
    summary = summarize(recs, warmup=2)
    assert summary["steps"] == [3, 4, 5, 6, 7]
    assert summary["parts_ms"]["gravity.before/gravity.exchange"] == [0.0, 0.0]
    assert summary["spikes"] == {"gravity.before/gravity.exchange": {"steps": [5], "max_ms": 900.0}}
    assert summary["between_ms"] == 40.0
    assert "900.00 ms at most, above twice its median on steps [5]" in report(summary)
