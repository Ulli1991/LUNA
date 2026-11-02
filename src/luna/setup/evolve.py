"""The time loop, one for every kind of run: evolve(cfg, init, physics, mesh) -> State.
One step is a palindrome around the hydro update; the loop and the step in docs/ARCHITECTURE.md.
"""
import os

from luna.setup.run_report import require_finite
from luna.setup.step_timing import StepTiming, label_methods, run_part
from luna.setup.state import State

BUDGET = "boundary_mass"   # (2, 6) mass in / out through each face since the start


def timestep(state, physics, target, timer=None):
    """(dt, t_new): the smallest limit of the physics and the clock, landing exactly on `target`."""
    dt = float("inf")
    if physics.hydro is not None:
        dt = run_part("dt.hydro", lambda: float(physics.hydro.compute_dt(state.U)), timer)
    for stage in physics.stages:
        if hasattr(stage, "dt_limit"):
            lim = run_part(f"dt.{stage.name}", lambda: float(stage.dt_limit(state)), timer)
            dt = min(dt, lim)
    dt = physics.clock.cap(dt, state.t)
    require_finite(dt, what="dt", step=state.step, t=state.t,
                   hint="last good snapshot is the newest in the output dir; lower CFL "
                        "or check the source terms.")
    remaining = target - state.t
    if dt >= remaining:
        return remaining, target            # snap exactly, avoid float roundoff
    t_new = state.t + dt
    if not t_new > state.t and remaining < float("inf"):       # an infinite target only asks for dt
        raise RuntimeError(f"dt {dt:.3e} at step {state.step} (t={state.t:.6e}) is below the "
                           "clock's resolution: t + dt == t, the run cannot advance.")
    return dt, t_new


def step(state, dt, t_new, physics, timer=None, reports=None):
    """One step from state.t to t_new; pure, and with `timer` each part waited for and timed; reports: a list the
    hydro's face report is appended to (setup/face_report.py), else it is dropped."""
    h = 0.5 * dt
    hyd = physics.hydro
    rstep = getattr(hyd, "step_report", None) if reports is not None else None
    rflux = getattr(hyd, "step_flux_report", None) if reports is not None else None
    for stage in physics.stages:
        state = run_part(f"{stage.name}.before", lambda: stage.before(state, h, dt), timer)
    if physics.hydro is not None:
        U0 = state.U
        if getattr(physics.hydro, "scalars", None):
            carry = state.sources[physics.hydro.scalars]
            if rstep is not None:
                U1, S1, rep = run_part("hydro", lambda: rstep(U0, carry["S"], dt), timer)
                reports.append(rep)
            else:
                U1, S1 = run_part("hydro", lambda: physics.hydro.step(U0, carry["S"], dt), timer)
            state = state.update(U=U1).with_source(physics.hydro.scalars, {**carry, "S": S1})
        elif getattr(physics.hydro, "step_flux", None) is not None:
            # gravity_energy='flux': the gravity stage's closing half-kick works with these fluxes
            if rflux is not None:
                U1, F, rep = run_part("hydro", lambda: rflux(U0, dt), timer)
                reports.append(rep)
            else:
                U1, F = run_part("hydro", lambda: physics.hydro.step_flux(U0, dt), timer)
            state = state.update(U=U1, mass_flux=F)
        elif rstep is not None:
            U1, rep = run_part("hydro", lambda: rstep(U0, dt), timer)
            reports.append(rep)
            state = state.update(U=U1)
        else:
            state = state.update(U=run_part("hydro", lambda: physics.hydro.step(U0, dt), timer))
        face_mass = getattr(physics.hydro, "face_mass", None)
        if face_mass is not None:
            state = state.with_source(BUDGET, run_part(
                "boundary_mass", lambda: face_mass(U0, dt, state.sources[BUDGET]), timer))
    state = state.update(t=t_new, a=physics.clock.scale(t_new))
    for stage in reversed(physics.stages):
        state = run_part(f"{stage.name}.after", lambda: stage.after(state, h, dt), timer)
    U = state.U
    for project in physics.projections:
        name = getattr(project, "name", None) or type(project).__name__.lower()
        U = run_part(f"project.{name}", lambda: project(U, state.a), timer)
    # the face mass fluxes and the opening kick work belong to this step: any stage may read them until here
    return state.update(U=U, step=state.step + 1, mass_flux=None, kick_work=None)


def strong(U):
    """U without JAX's weak type (jnp.full with a Python float makes one): weak FP64 times FP32 is FP32."""
    if U is None or not getattr(U, "weak_type", False):
        return U
    from jax._src.lax.lax import _convert_element_type
    return _convert_element_type(U, U.dtype, weak_type=False)


def initial_state(init, physics):
    """The state at t_start: gas grid, each source's carry, the species' buffers (gravity after a restore)."""
    clock = physics.clock
    t0 = init.t_start if not clock.comoving else clock.t_start
    init.U = strong(init.U)                 # in place: the state owns the one array (test_release_initial)
    state = State(U=init.U, t=t0, a=clock.scale(t0), step=init.step_start)
    for stage in physics.stages:
        if hasattr(stage, "carry0"):
            if stage.carry0 is None:
                raise RuntimeError(f"{stage.name}: its initial carry was released "
                                   "(release_initial); the run has already started")
            state = state.with_source(stage.name, stage.carry0)
    if getattr(physics.hydro, "face_mass", None) is not None:
        import numpy as np
        state = state.with_source(BUDGET, np.zeros((2, 6), init.U.dtype))
    if physics.gravity is not None:
        state = physics.gravity.assemble(state, init)
    return state


def release_initial(init, physics):
    """Let go of the initial gas grid and the stages' initial carries once the state owns them."""
    init.U = None
    for stage in physics.stages:
        if hasattr(stage, "carry0"):
            stage.carry0 = None


def evolve(cfg, init, physics, mesh, is_main=True):
    """The run from the initial state to the end of the clock (or max_steps)."""
    from luna.setup.outputs import Outputs
    state = initial_state(init, physics)
    out = Outputs(cfg, physics, mesh, init, is_main)
    state = out.restore(state)
    release_initial(init, physics)
    if physics.gravity is not None:
        state = physics.gravity.init(state)
    if physics.hydro is not None:
        if os.environ.get("LUNA_MEM_REPORT"):
            # what the device holds before the first compiled step, and its
            # limit: the number a "does it fit" run is after, from every rank
            import jax
            m = jax.local_devices()[0].memory_stats() or {}
            GiB = 2.0 ** 30
            print(f"[mem] rank {jax.process_index()}: in use "
                  f"{m.get('bytes_in_use', 0) / GiB:.2f} GiB of limit "
                  f"{m.get('bytes_limit', 0) / GiB:.2f} GiB before the first step", flush=True)
        physics.hydro.warmup(state.U, is_main)
    out.start(state)
    label_methods(out, "outputs", ("diagnostics", "snapshot", "checkpoint", "progress"))
    every = int(getattr(cfg, "timing_every", 0) or 0)
    timing = (StepTiming(every, cfg.output_dir, is_main, fresh=not cfg.restart)
              if every > 0 else None)
    max_steps = int(cfg.max_steps or 0)
    if physics.hydro is not None and getattr(physics.hydro, "step_report", None) is not None:
        from luna.setup.face_report import FaceLog
        out.face_log = FaceLog(cfg, physics.hydro.grid, is_main)
    while not physics.clock.finished(state.t) and (max_steps <= 0 or state.step < max_steps):
        timer = timing.timer(state.step + 1) if timing is not None else None
        if timer is None:
            state, dt = loop_step(state, physics, out)
        else:
            with timer:
                state, dt = loop_step(state, physics, out, timer)
            timing.record(timer, state, dt)
    if getattr(out, "face_log", None) is not None:
        out.face_log.finish()
    return out.finish(state)


def loop_step(state, physics, out, timer=None):
    """One iteration of the loop: dt, the previous step's face report, the step, the gravity check, the outputs."""
    log = getattr(out, "face_log", None)
    try:
        dt, t_new = timestep(state, physics, out.target(state.t), timer)
    except RuntimeError:
        if log is not None:
            log.flush()             # refused faces (NaN) explain a non-finite dt: name them first
        raise
    reports = None
    if log is not None:
        log.flush()                 # read a step late: the dt above already waited for it
        reports = []
    state = step(state, dt, t_new, physics, timer, reports)
    if reports:
        log.push(reports[0], state.step, float(t_new))
    if physics.gravity is not None:
        run_part("gravity.check", lambda: physics.gravity.check(state), timer)
    run_part("outputs", lambda: out.after_step(state, dt), timer)
    return state, dt
