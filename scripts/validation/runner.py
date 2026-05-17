"""
Programmatic LUNA runner for validation sweeps.

Drives the same setup → run → finalize sequence as main.py but with a Config
object built in Python (via dataclasses.replace on template_config.Config), so
per-test plot scripts can sweep over Nx / t_end / test_problem / etc. without
shell scripting.

JAX-distributed init is one-shot per process — call run_one() with
jax_already_initialized=True on subsequent invocations inside the same job.
"""
from dataclasses import replace
from pathlib import Path

from template_config import Config
from setup import (
    initialize_jax_distributed,
    create_distributed_mesh,
    initialize_simulation_distributed,
    create_solver_distributed,
    run_simulation_distributed,
    finalize_simulation_distributed,
)
from utils.grid_params import GridParams


def run_one(overrides, *, jax_already_initialized=False):
    """Run a single LUNA simulation with the given Config overrides.

    overrides : dict
        Keyword overrides for the Config dataclass.
        e.g. dict(test_problem="Sod-Shock", Nx=256, t_end=0.245,
                  output_dir="snapshots/sod_256")
    jax_already_initialized : bool
        True on second and subsequent calls within one process.

    Returns (t_final, step_final, output_dir).
    """
    if not jax_already_initialized:
        initialize_jax_distributed()

    cfg = replace(Config(), **overrides)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    mesh, grid_shape = create_distributed_mesh(cfg)
    U_sharded, t_start, step_start = initialize_simulation_distributed(cfg, mesh)
    grid = GridParams.from_config(cfg)
    compute_dt_fn, step_fn, _, sharding = create_solver_distributed(cfg, grid, mesh)

    U_sharded, t_final, step_final, start_time = run_simulation_distributed(
        U_sharded, t_start, step_start, cfg, grid,
        compute_dt_fn, step_fn, sharding,
    )

    finalize_simulation_distributed(
        U_sharded, t_final, step_final, start_time, cfg, grid, grid_shape,
    )
    return t_final, step_final, cfg.output_dir


def sweep(base_overrides, scan_field, scan_values, output_dir_template):
    """Run a sequence of simulations varying a single field.

    base_overrides : dict
        Overrides applied to every run.
    scan_field : str
        Config field name to vary (e.g. "Nx").
    scan_values : iterable
        Values of scan_field to run.
    output_dir_template : str
        Format string with one {} placeholder for the scan value.
        e.g. "snapshots/sod_{}"

    Returns a list of (value, output_dir) pairs.
    """
    results = []
    for i, val in enumerate(scan_values):
        overrides = dict(base_overrides)
        overrides[scan_field] = val
        # For 2D/3D scans we usually want Nx=Ny=Nz=val unless explicitly set.
        if scan_field == "Nx":
            overrides.setdefault("Ny", val)
            overrides.setdefault("Nz", val)
        overrides["output_dir"] = output_dir_template.format(val)
        _, _, out = run_one(overrides, jax_already_initialized=(i > 0))
        results.append((val, out))
    return results
