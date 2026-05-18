"""Diagnostic: square advection on 4 GPUs at Nx=1024, step-by-step state dump."""
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import replace

from template_config import Config
from setup import (
    initialize_jax_distributed,
    create_distributed_mesh,
    initialize_simulation_distributed,
    create_solver_distributed,
)
from utils.grid_params import GridParams


def main():
    initialize_jax_distributed(enable_x64=True)

    cfg = replace(
        Config(),
        test_problem="Square-Advected",
        SPATIAL_DIMENSION=2,
        BOUNDARY=0,
        Nx=1024, Ny=1024,
        Lx=1.0, Ly=1.0,
        gamma=5.0 / 3.0,
        LIMITER=1, ORDER=1, TIME_SOLVER=1,
        CFL=0.8,
        t_end=0.1,
        time_bet_snaps=1.0,
        output_dir="snapshots/square_dbg",
        production_mode=True,
        PLOT_AFTER_SIM=False,
    )

    mesh, grid_shape = create_distributed_mesh(cfg)
    U_sharded, t_start, step_start = initialize_simulation_distributed(cfg, mesh)
    grid = GridParams.from_config(cfg)
    compute_dt_fn, step_fn, _, sharding = create_solver_distributed(cfg, grid, mesh)

    print(f"[dbg] U_sharded shape={U_sharded.shape}", flush=True)

    # Warmup
    dt0 = float(compute_dt_fn(U_sharded))
    U_warm = step_fn(U_sharded, dt0)
    jax.block_until_ready(U_warm)
    print(f"[dbg] warmup done. dt0={dt0:.4e}", flush=True)

    t = 0.0
    for step in range(1, 121):
        dt = float(compute_dt_fn(U_sharded))
        U_sharded = step_fn(U_sharded, dt)
        jax.block_until_ready(U_sharded)
        t += dt
        rho_min = float(jnp.min(U_sharded[0]))
        rho_max = float(jnp.max(U_sharded[0]))
        p = (cfg.gamma - 1.0) * (U_sharded[3] - 0.5 * (U_sharded[1]**2 + U_sharded[2]**2) / U_sharded[0])
        p_min = float(jnp.min(p))
        p_max = float(jnp.max(p))
        nan_count = int(jnp.sum(jnp.isnan(U_sharded)))
        print(f"[dbg] step {step:4d}: t={t:.5f} dt={dt:.4e} "
              f"rho [{rho_min:.6f},{rho_max:.6f}] p [{p_min:.6f},{p_max:.6f}] nan_cells={nan_count}",
              flush=True)
        if nan_count > 0 or rho_min <= 0 or p_min <= 0:
            print(f"[dbg] >>> BAD STATE at step {step} <<<", flush=True)
            break

    print("[dbg] done.", flush=True)


if __name__ == "__main__":
    main()
