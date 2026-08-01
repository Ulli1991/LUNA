"""
Multi-node distributed JAX hydro solver.

This is the main entry point for distributed multi-node simulations using
JAX's automatic sharding with NamedSharding.
"""
from dataclasses import dataclass, field

try:
    from config import Config
except ImportError as e:
    raise ImportError(
        "no config.py found — create one with: cp template_config.py "
        "config.py  (then edit it; config.py is gitignored)") from e
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


@dataclass
class RunResult:
    """Uniform return of `main.run` across every run mode.

    t          final simulation time (the scale factor `a` for cosmological
               runs; None for autodiff tasks, whose product is `extra`)
    n_steps    steps taken (None where not meaningful)
    output_dir where snapshots / products were written
    extra      mode-specific payload (final states, autodiff task results)

    Iterates as ``(t, n_steps, output_dir)`` so existing tuple unpacking keeps
    working on every path.
    """
    t: float = None
    n_steps: int = None
    output_dir: str = None
    extra: dict = field(default_factory=dict)

    def __iter__(self):
        return iter((self.t, self.n_steps, self.output_dir))


def run(cfg):
    """Setup -> runner -> finalize, for a fully-built cfg. Returns RunResult.

    Shared by `python main.py` and every run script so all modes go through
    the SAME path. Dispatch order (first match wins): cfg.CosmoDM (particle
    N-body), cfg.CosmoHydro (coupled gas+DM box), cfg.AMR (block-AMR),
    cfg.autodiff_mode (differentiable scan loop), cfg.CosmologicalIntegration
    (super-comoving gas), else the standard dense HPC loop
    (run_simulation_distributed). Every path returns a RunResult.
    """
    if hasattr(cfg, "validate"):
        cfg.validate()
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

    # Opt-in fresh-run wipe (validation analysis globs every snapdir, so stale
    # output from a prior run would corrupt it). Default-off keeps production /
    # checkpoint-restart safe. Only process 0 deletes; others wait below.
    # Runs BEFORE any runner dispatch so all run modes get the same treatment.
    if getattr(cfg, "clear_output_dir", False) and jax.process_index() == 0:
        import shutil
        shutil.rmtree(cfg.output_dir, ignore_errors=True)
    try:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass   # multi-process NFS race: exist_ok can still raise when many
               # ranks create the tree simultaneously; the dir exists -- fine.

    # DM-only cosmological N-body: collisionless particles, no gas grid ->
    # dispatch before the gas setup (mesh / U / step_fn are never built).
    if getattr(cfg, "CosmoDM", False):
        from setup.runner_cosmo_dm import run_simulation_cosmo_dm
        res = run_simulation_cosmo_dm(cfg, is_main_process=(jax.process_index() == 0))
        return RunResult(t=res.get("a"), n_steps=res.get("n_steps"),
                         output_dir=cfg.output_dir, extra=res)

    # Coupled gas + DM cosmological box: gas meshed from MUSIC baryon IC, DM as
    # particles, evolved under one joint potential. Builds its own grid/U/step_fn.
    # With cfg.AMR the coupled box goes through the block-AMR runner instead
    # (quasi-Lagrangian refinement; DM via amr/particles.py).
    if getattr(cfg, "CosmoHydro", False) and not getattr(cfg, "AMR", False):
        from setup.runner_cosmo_hydro import run_simulation_cosmo_hydro_music
        U, dm_pos, dm_p, a, step = run_simulation_cosmo_hydro_music(
            cfg, is_main_process=(jax.process_index() == 0))
        return RunResult(t=a, n_steps=step, output_dir=cfg.output_dir,
                         extra={"U": U, "dm_pos": dm_pos, "dm_p": dm_p})

    # Block-AMR multi-level hydro: builds its own (refined) state and runs its
    # own multi-level time loop, writing composited snapshots through the same
    # HDF5 writer. Dispatch before the dense gas setup (mesh/U/step_fn unused).
    if getattr(cfg, "AMR", False):
        from setup.runner_amr import run_simulation_amr
        t, nstep, out = run_simulation_amr(cfg, is_main_process=(jax.process_index() == 0))
        return RunResult(t=t, n_steps=nstep, output_dir=out)

    mesh, grid_shape = create_distributed_mesh(cfg)
    U_sharded, t_start, step_start = initialize_simulation_distributed(cfg, mesh)
    grid = GridParams.from_config(cfg)
    compute_dt_fn, step_fn, sharding = create_solver_distributed(cfg, grid, mesh)

    # === Autodiff run mode: alternative distributed runner ===
    if getattr(cfg, "autodiff_mode", False):
        from setup.runner_autodiff import run_simulation_autodiff
        res = run_simulation_autodiff(
            U_sharded, cfg, step_fn, compute_dt_fn, sharding,
            is_main_process=(jax.process_index() == 0),
        )
        return RunResult(output_dir=cfg.output_dir,
                         extra=res if isinstance(res, dict) else {"result": res})

    # === Cosmological (comoving) run mode: super-comoving runner ===
    if getattr(cfg, "CosmologicalIntegration", False):
        from setup.runner_cosmo import run_simulation_cosmo
        U, a, step = run_simulation_cosmo(
            U_sharded, cfg, step_fn, compute_dt_fn,
            is_main_process=(jax.process_index() == 0),
        )
        return RunResult(t=a, n_steps=step, output_dir=cfg.output_dir,
                         extra={"U": U})

    # === Standard HPC run ===
    U_sharded, t_final, step_final, start_time = run_simulation_distributed(
        U_sharded, t_start, step_start, cfg, grid,
        compute_dt_fn, step_fn, sharding
    )
    finalize_simulation_distributed(
        U_sharded, t_final, step_final, start_time, cfg, grid, grid_shape
    )
    return RunResult(t=t_final, n_steps=step_final, output_dir=cfg.output_dir)


def main():
    """Main entry point for distributed simulation."""
    # Build the config first (pure dataclass, no JAX arrays yet) so the JAX
    # backend can be initialized with the right precision. jax_enable_x64 must
    # be set before any array is created, which initialize_jax_distributed does.
    cfg = Config()
    initialize_jax_distributed(enable_x64=getattr(cfg, "enable_x64", False))
    import jax
    if jax.process_index() == 0:
        print_LUNA_banner()
    run(cfg)


if __name__ == "__main__":
    main()
