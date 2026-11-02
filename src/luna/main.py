"""LUNA's entry point. `main()` is the command line:

    python main.py                          config.py in the cwd (`luna` = `python main.py`)
    python main.py my_run.py Nx=512 t_end=0.5 that file's Config, then overrides
    python main.py runs/kh_run.py nx=512    a run script: its config(nx=...)
    python main.py ad.py n_iter=40          autodiff_mode: the task's options too
    python main.py stellar --help          experimental FP64 stellar 1D runner

It reads the config (luna/setup/cli.py), starts JAX (luna/setup/jax_init.py)
and calls `run(cfg)`, which prepares the run and calls, in order:

    mesh    = build_mesh(cfg)             how the data lies on devices   setup/mesh.py
    init    = read_ics(cfg, mesh)         ICs, or the restart's grid     setup/ics.py
    physics = assemble(cfg, mesh, init)   step, clock, gravity, sources  setup/assemble.py
    state   = evolve(cfg, init, physics)  the loop, outputs, the end     setup/evolve.py

Refinement (cfg.AMR) takes the same mesh and initial state into its own scan.
With cfg.autodiff_mode the prepared run goes to an adjoint task instead of
the loop: `task(init, cfg, physics, is_main)`, by default the example named
by cfg.autodiff_task (scripts/examples/autodiff/). Every path returns a
RunResult.
"""


def run(cfg, task=None, U0=None, ic=None):
    """Run cfg; U0 (conserved, whole grid) or ic(x, y, z, cfg) replace the registered IC (ic_generation/user_ic.py)."""
    import jax
    from pathlib import Path
    from luna.file_io.checkpoint_utils import write_run_json
    from luna.ic_generation.ic import is_cosmological, resolve
    from luna.setup.assemble import assemble
    from luna.setup.evolve import evolve
    from luna.setup.ics import read_ics
    from luna.setup.mesh import build_mesh, decomposition
    from luna.setup.run_kind import run_kind
    from luna.setup.run_report import RunResult, result_of
    from luna.setup.output_dir import prepare_output_dir

    cfg.validate()
    if is_cosmological(cfg.test_problem) and not cfg.CosmologicalIntegration:
        raise ValueError(f"test_problem '{cfg.test_problem}' requires comoving integration, "
                         "but CosmologicalIntegration=False")
    cfg = resolve(cfg)                   # the grid some problems derive (Sod-3D's tube)
    from luna.ic_generation.user_ic import user_ic
    user = user_ic(cfg, U0, ic)          # refused before anything is written
    is_main = jax.process_index() == 0
    # parents[2] is the repo root from src/luna/main.py; it is a guard
    # for prepare_output_dir, which refuses to wipe it
    prepare_output_dir(cfg, Path(__file__).resolve().parents[2])
    write_run_json(cfg.output_dir, cfg, is_main_process=is_main)

    kind = run_kind(cfg)
    layout = decomposition(cfg, kind)
    mesh, _ = build_mesh(cfg, layout, kind)
    init = read_ics(cfg, mesh, kind, layout, is_main, user)
    cfg = init.cfg

    if cfg.AMR:
        from luna.setup.runner_amr import run_simulation_amr
        t, nstep, out = run_simulation_amr(cfg, init, is_main_process=is_main)
        return RunResult(t=t, n_steps=nstep, output_dir=out)

    physics = assemble(cfg, mesh, kind, layout, init, is_main)

    if cfg.autodiff_mode:
        from luna.autodiff.rollout import example_task
        res = (task or example_task(cfg.autodiff_task))(init, cfg, physics, is_main)
        return RunResult(output_dir=cfg.output_dir,
                         extra=res if isinstance(res, dict) else {"result": res})

    state = evolve(cfg, init, physics, mesh, is_main)
    return result_of(state, cfg, physics)


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "stellar":
        from luna.stellar import main as stellar_main
        return stellar_main(argv[1:])
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return None
    import jax
    from luna.setup.cli import configure_file, own_main
    from luna.setup.jax_init import initialize_jax_distributed
    from luna.utils.logo import print_LUNA_banner
    entry = own_main(argv)
    if entry is not None:                # a run script with its own main()
        return entry()
    cfg, task, report = configure_file(argv)
    initialize_jax_distributed(enable_x64=cfg.enable_x64)
    if jax.process_index() == 0:
        print_LUNA_banner()
    res = run(cfg, task)
    if report is not None and jax.process_index() == 0:
        report(res)
    return res


def cli():
    """The `luna` console script: `main()` with exit status 0 once the run returns."""
    main()
    return 0


if __name__ == "__main__":
    main()
