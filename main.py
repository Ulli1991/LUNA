"""
Multi-node distributed JAX hydro solver.

This is the main entry point for distributed multi-node simulations using
JAX's automatic sharding with NamedSharding.
"""
from config import Config
from utils.grid_params import GridParams
from utils.logo import print_LUNA_banner
from setup import (
    initialize_jax_distributed,
    create_distributed_mesh,
    initialize_simulation_distributed,
    create_solver_distributed,
    run_simulation_distributed,
    finalize_simulation_distributed
)


def run(cfg):
    """Setup -> distributed runner -> finalize, for a fully-built cfg.

    Shared by `python main.py` and the autodiff run scripts so both go through
    the SAME path. The runner is selected by cfg.autodiff_mode: the standard
    HPC time loop (run_simulation_distributed) or the differentiable scan loop
    (run_simulation_autodiff) — same setup, mesh, step_fn, and sharding, just a
    different runner_distributed.
    """
    # Cosmological problems must be integrated in comoving coordinates. Abort
    # early (before any output side effects) if the comoving integrator is off.
    from ic_generation.ic import is_cosmological
    if is_cosmological(cfg.test_problem) and not getattr(
        cfg, "CosmologicalIntegration", False
    ):
        raise ValueError(
            f"test_problem '{cfg.test_problem}' requires comoving integration, "
            f"but CosmologicalIntegration=False. Set CosmologicalIntegration=True "
            f"to run cosmological problems."
        )

    import jax
    from pathlib import Path

    # DM-only cosmological N-body: collisionless particles, no gas grid ->
    # dispatch before the gas setup (mesh / U / step_fn are never built).
    if getattr(cfg, "CosmoDM", False):
        from setup.runner_cosmo_dm import run_simulation_cosmo_dm
        return run_simulation_cosmo_dm(cfg, is_main_process=(jax.process_index() == 0))

    # Coupled gas + DM cosmological box: gas meshed from MUSIC baryon IC, DM as
    # particles, evolved under one joint potential. Builds its own grid/U/step_fn.
    if getattr(cfg, "CosmoHydro", False):
        from setup.runner_cosmo_hydro import run_simulation_cosmo_hydro_music
        return run_simulation_cosmo_hydro_music(
            cfg, is_main_process=(jax.process_index() == 0))

    # Opt-in fresh-run wipe (validation analysis globs every snapdir, so stale
    # output from a prior run would corrupt it). Default-off keeps production /
    # checkpoint-restart safe. Only process 0 deletes; others wait below.
    if getattr(cfg, "clear_output_dir", False) and jax.process_index() == 0:
        import shutil
        shutil.rmtree(cfg.output_dir, ignore_errors=True)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Block-AMR multi-level hydro: builds its own (refined) state and runs its
    # own multi-level time loop, writing composited snapshots through the same
    # HDF5 writer. Dispatch before the dense gas setup (mesh/U/step_fn unused).
    if getattr(cfg, "AMR", False):
        from setup.runner_amr import run_simulation_amr
        return run_simulation_amr(cfg, is_main_process=(jax.process_index() == 0))

    mesh, grid_shape = create_distributed_mesh(cfg)
    U_sharded, t_start, step_start = initialize_simulation_distributed(cfg, mesh)
    grid = GridParams.from_config(cfg)
    compute_dt_fn, step_fn, integrator_name, sharding = create_solver_distributed(cfg, grid, mesh)

    # === Autodiff run mode: alternative distributed runner ===
    if getattr(cfg, "autodiff_mode", False):
        from setup.runner_autodiff import run_simulation_autodiff
        return run_simulation_autodiff(
            U_sharded, cfg, grid, step_fn, compute_dt_fn, sharding,
            is_main_process=(jax.process_index() == 0),
        )

    # === Cosmological (comoving) run mode: super-comoving runner ===
    if getattr(cfg, "CosmologicalIntegration", False):
        from setup.runner_cosmo import run_simulation_cosmo
        return run_simulation_cosmo(
            U_sharded, cfg, grid, step_fn, compute_dt_fn, sharding,
            is_main_process=(jax.process_index() == 0),
        )

    # === Standard HPC run ===
    U_sharded, t_final, step_final, start_time = run_simulation_distributed(
        U_sharded, t_start, step_start, cfg, grid,
        compute_dt_fn, step_fn, sharding
    )
    finalize_simulation_distributed(
        U_sharded, t_final, step_final, start_time, cfg, grid, grid_shape
    )
    return t_final, step_final, cfg.output_dir


def main():
    """Main entry point for distributed simulation."""
    initialize_jax_distributed()
    import jax
    if jax.process_index() == 0:
        print_LUNA_banner()
    run(Config())


if __name__ == "__main__":
    main()
