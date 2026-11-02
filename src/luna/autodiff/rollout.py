"""The differentiable rollout: the production step scanned over a fixed number of steps.

Take every gradient of `simulate` under jax.jit. See docs/AUTODIFF.md.
"""
import numpy as np
import jax
import jax.numpy as jnp


def snapshots_for(cfg, state_shape):
    """Revolve snapshot count: cfg.autodiff_snapshots, or what fits in cfg.autodiff_snapshot_gib per device."""
    gib = float(cfg.autodiff_snapshot_gib)
    if gib <= 0.0:
        return int(cfg.autodiff_snapshots)
    shards = int(np.prod(cfg.MESH_SHAPE or (1,)))
    per_dev = sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(state_shape)) / shards
    return max(1, int(gib * 2 ** 30 // per_dev) - 1)


def build_simulate(init, cfg, physics, is_main_process=True, forcing_theta=False):
    """Return (simulate, step_once, carry0, dt_info, eta_true); simulate(theta, U0) -> final U.

    theta are the CC85 loadings (eta_M, eta_E); with forcing_theta the OU driving (sigma_1..sigma_M, zeta),
    with forcing_theta="tac" also the correlation time T_ac.
    """
    from dataclasses import replace as _replace

    from luna.setup.evolve import initial_state, step
    from luna.sources.stages import CellLocal

    fixed_dt = float(cfg.autodiff_fixed_dt)
    adaptive = fixed_dt <= 0.0
    dt_mode = cfg.autodiff_dt_mode
    if dt_mode not in ("stopgrad", "diff"):
        raise ValueError(f"autodiff_dt_mode={dt_mode!r}: 'stopgrad' or 'diff'")
    n_steps = int(cfg.autodiff_n_steps)
    t_end = float(cfg.t_end) if adaptive else 0.0
    U0_ref = init.U
    cc85 = next((s for s in physics.stages if getattr(s, "name", "") == "cc85"), None)
    eta_true = (jnp.array([cfg.cc85_eta_M, cfg.cc85_eta_E], dtype=U0_ref.dtype)
                if cc85 is not None else jnp.zeros(2, dtype=U0_ref.dtype))

    if cfg.sinks_enabled or cfg.particles_enabled:
        raise NotImplementedError(
            "autodiff_mode does not support sinks_enabled/particles_enabled "
            "(their per-step state is not threaded through the scan carry)")

    forcing = next((s for s in physics.stages if getattr(s, "name", "") == "forcing"), None)
    if forcing_theta:
        if forcing is None or cc85 is not None:
            raise ValueError("forcing_theta needs ou_config and no CC85 stage")
        from luna.sources.stages import ForcingTheta
        eta_true = jnp.concatenate([jnp.asarray(forcing.ou["driver"].weight, U0_ref.dtype),
                                    jnp.asarray([cfg.ou_config.solenoidal_weight], U0_ref.dtype)])
        if forcing_theta == "tac":
            eta_true = jnp.concatenate([eta_true, jnp.asarray([cfg.ou_config.autocorrelation_time],
                                                              U0_ref.dtype)])

    def physics_for(theta):
        """The production physics with the CC85 (or, forcing_theta, the OU) stage parameterised by theta."""
        if forcing_theta:
            stages = [ForcingTheta(forcing.ou, theta, tac=forcing_theta == "tac") if s is forcing else s
                      for s in physics.stages]
            return _replace(physics, stages=stages)
        if cc85 is None:
            return physics
        stages = [CellLocal("cc85", lambda U, h: cc85.apply_eta(U, h, theta)) if s is cc85 else s
                  for s in physics.stages]
        return _replace(physics, stages=stages)

    def carry0(U0):
        st = initial_state(_replace(init, U=U0), physics_for(eta_true) if forcing_theta else physics)
        if physics.gravity is not None:
            st = physics.gravity.init(st)
        return st.update(t=jnp.asarray(0.0, U0.dtype), a=jnp.asarray(1.0, U0.dtype),
                         step=jnp.asarray(0, jnp.int32))

    def step_once(state, theta):
        if adaptive:
            # "stopgrad": no tangent may reach the CFL min collectives (no AD rule).
            U_dt = state.U if dt_mode == "diff" else jax.lax.stop_gradient(state.U)
            dt = physics.hydro.compute_dt(U_dt)
            for stage in physics.stages:
                if hasattr(stage, "dt_limit"):
                    dt = jnp.minimum(dt, stage.dt_limit(state.update(U=U_dt)))
            dt = jnp.minimum(dt, jnp.maximum(t_end - state.t, 0.0))
            active = state.t < t_end
        else:
            dt = fixed_dt
            active = jnp.asarray(True)
        new = step(state, dt, state.t + dt, physics_for(theta))
        return jax.tree.map(lambda n, o: jnp.where(active, n, o), new, state)

    remat = cfg.autodiff_checkpoint
    if remat == "revolve" and cfg.autodiff_grad_mode == "forward":
        remat = "step"                  # a custom_vjp has no forward mode
    if remat == "revolve":
        from luna.time_integration.revolve import revolve_rollout
        snaps = snapshots_for(cfg, jax.eval_shape(carry0, U0_ref))
        rollout = revolve_rollout(step_once, n_steps, snaps)

        def simulate(theta, U0):
            return rollout(theta, carry0(U0)).U
    else:
        def simulate(theta, U0):
            @jax.checkpoint
            def body(carry, _):
                return step_once(carry, theta), None
            final, _ = jax.lax.scan(body, carry0(U0), None, length=n_steps)
            return final.U

    dt_info = f"adaptive({dt_mode})" if adaptive else f"fixed {fixed_dt:.3e}"
    if remat == "revolve":
        dt_info += f", revolve {rollout.snapshots} snapshots"
    return simulate, step_once, carry0, dt_info, eta_true


def example_task(name):
    """The `task` of scripts/examples/autodiff/<name>.py; needs the repository root on sys.path."""
    import importlib
    module = f"scripts.examples.autodiff.{name}"
    try:
        return importlib.import_module(module).task
    except ModuleNotFoundError as e:
        if e.name is None or not module.startswith(e.name):
            raise
        raise ModuleNotFoundError(
            f"autodiff_task={name!r} is the example scripts/examples/autodiff/{name}.py, "
            "which is not importable from here: run from the repository root, or "
            "pass the task yourself, luna.main.run(cfg, task)") from e
