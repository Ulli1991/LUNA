"""
Simplified profiling script that avoids interfering with simulation logic.

This version times complete timesteps without wrapping internal operations,
which prevents synchronization issues that can cause numerical instability.

Usage:
    python utils/profile_simulation.py
    (from the project root directory)
"""

import os
import sys
import time

# Add parent directory to path so we can import from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp

# JAX configuration
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "high")
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

from functools import partial
from config import Config
from ic_generation.ic import initial_conditions
from time_integration.integrators import rk2_step, rk4_step, muscl_hancock_step
from time_integration.cfl import compute_dt_and_diagnostics
from hydro.physics_checks import validate_state, validate_from_diagnostics
from hydro.divergence import flux_divergence


def main():
    # --- Load configuration ---
    cfg = Config()

    print("="*60)
    print("SIMPLE PROFILING MODE")
    print("="*60)
    print(f"Will profile ~200 timesteps")
    print("No internal wrapping - clean timing")
    print()

    # --- Device initialization ---
    print("="*60)
    print("DEVICE INITIALIZATION")
    print("="*60)
    device_type = cfg.DEVICE
    if device_type == 0:
        print("Using CPU")
        jax.config.update("jax_platform_name", "cpu")
    else:
        print("Using GPU")
        try:
            jax.config.update("jax_platform_name", "rocm")
            _ = jax.devices()[0]
            print("✓ ROCm GPU")
        except Exception:
            try:
                jax.config.update("jax_platform_name", "cuda")
                _ = jax.devices()[0]
                print("✓ CUDA GPU")
            except Exception:
                sys.exit("Error: No GPU available.")
    print()

    # --- Initialize fields ---
    print("="*60)
    print("INITIAL CONDITIONS")
    print("="*60)
    U = initial_conditions(cfg)
    print(f"Grid: {cfg.Nx} x {cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else 'N/A'} x {cfg.Nz if cfg.SPATIAL_DIMENSION == 3 else 'N/A'}")

    # Validate
    validate_state(U, cfg.gamma, step=0, time=cfg.t_start)
    print("✓ Initial conditions validated")
    print()

    # --- Setup ---
    dx = cfg.Lx / cfg.Nx
    dy = cfg.Ly / cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else None
    dz = cfg.Lz / cfg.Nz if cfg.SPATIAL_DIMENSION == 3 else None
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy if dy is not None else None
    inv_dz = 1.0 / dz if dz is not None else None

    flux_fn = partial(
        flux_divergence,
        gamma=cfg.gamma,
        order=cfg.ORDER,
        limiter=cfg.LIMITER,
        boundary=cfg.BOUNDARY,
        dx=dx, dy=dy, dz=dz,
        inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz
    )

    time_solver_map = {0: rk2_step, 1: muscl_hancock_step, 2: rk4_step}
    time_step_fn = partial(time_solver_map[cfg.TIME_SOLVER], flux_fn=flux_fn)

    # --- Warmup ---
    print("="*60)
    print("WARMUP (JIT COMPILATION)")
    print("="*60)
    print("Running 5 warmup steps...")

    t = cfg.t_start
    step = 0

    for _ in range(5):
        dt, min_rho, min_p, has_nan, has_inf = compute_dt_and_diagnostics(
            U, cfg.gamma, cfg.CFL, t, cfg.t_end, dx, dy, dz
        )
        U = time_step_fn(U, dt)
        t += dt
        step += 1

    jax.block_until_ready(U)
    print("✓ Warmup complete")
    print()

    # --- PROFILED RUN ---
    print("="*60)
    print("PROFILING (200 STEPS)")
    print("="*60)

    cfl_times = []
    timestep_times = []
    validation_times = []

    target_steps = 200
    profiled_steps = 0

    while profiled_steps < target_steps:
        # === Time CFL computation ===
        jax.block_until_ready(U)
        t_cfl_start = time.time()

        dt, min_rho, min_p, has_nan, has_inf = compute_dt_and_diagnostics(
            U, cfg.gamma, cfg.CFL, t, cfg.t_end, dx, dy, dz
        )

        jax.block_until_ready(dt)
        cfl_time = (time.time() - t_cfl_start) * 1000
        cfl_times.append(cfl_time)

        # === Time timestep ===
        jax.block_until_ready(U)
        t_step_start = time.time()

        U = time_step_fn(U, dt)

        jax.block_until_ready(U)
        timestep_time = (time.time() - t_step_start) * 1000
        timestep_times.append(timestep_time)

        t += dt
        step += 1
        profiled_steps += 1

        # === Validation (every 50 steps) ===
        if profiled_steps % 50 == 0:
            jax.block_until_ready(U)
            t_val_start = time.time()

            validate_from_diagnostics(min_rho, min_p, has_nan, has_inf, step=step, time=t)

            jax.block_until_ready(U)
            val_time = (time.time() - t_val_start) * 1000
            validation_times.append(val_time)

            print(f"Step {profiled_steps:3d}/200: CFL={cfl_time:.2f}ms, Timestep={timestep_time:.2f}ms")

    # --- REPORT ---
    print()
    print("="*60)
    print("PROFILING RESULTS")
    print("="*60)

    # Calculate statistics (skip first 5 for warmup)
    cfl_avg = sum(cfl_times[5:]) / len(cfl_times[5:])
    timestep_avg = sum(timestep_times[5:]) / len(timestep_times[5:])
    validation_avg = sum(validation_times) / len(validation_times) if validation_times else 0

    total_per_step = cfl_avg + timestep_avg + (validation_avg / 50)  # Validation every 50 steps

    print()
    print(f"{'Component':<30} {'Avg (ms)':<12} {'% Total':<10}")
    print("-"*60)
    print(f"{'CFL computation':<30} {cfl_avg:>10.2f}  {(cfl_avg/total_per_step*100):>8.1f}%")
    print(f"{'Time step':<30} {timestep_avg:>10.2f}  {(timestep_avg/total_per_step*100):>8.1f}%")
    print(f"{'Validation (per step)':<30} {(validation_avg/50):>10.2f}  {((validation_avg/50)/total_per_step*100):>8.1f}%")
    print("-"*60)
    print(f"{'TOTAL per timestep':<30} {total_per_step:>10.2f}  {'100.0%':>10}")
    print()

    # Detailed stats
    print("Detailed Statistics:")
    print(f"  CFL:      avg={cfl_avg:.2f}ms, min={min(cfl_times[5:]):.2f}ms, max={max(cfl_times[5:]):.2f}ms")
    print(f"  Timestep: avg={timestep_avg:.2f}ms, min={min(timestep_times[5:]):.2f}ms, max={max(timestep_times[5:]):.2f}ms")
    print()

    # === BOTTLENECK ANALYSIS ===
    print("="*60)
    print("BOTTLENECK ANALYSIS")
    print("="*60)

    if timestep_avg / total_per_step > 0.75:
        print("✓ Time stepping dominates (75%+) - this is expected")
        print("  → Time stepping includes: flux_divergence, slopes, padding, HLLC")
        print("  → Run 'python profile_detailed.py' to drill deeper")

    if cfl_avg / total_per_step > 0.2:
        print("⚠️  CFL computation is >20% of runtime")
        print("  → Consider computing CFL less frequently")
        print("  → CFL computation already fused with diagnostics")

    print()

    # === PERFORMANCE ESTIMATE ===
    print("="*60)
    print("PERFORMANCE ESTIMATE")
    print("="*60)

    time_solver_name = ['RK2', 'MUSCL-Hancock', 'RK4'][cfg.TIME_SOLVER]
    rk_multiplier = {'RK2': 2, 'MUSCL-Hancock': 2, 'RK4': 4}[time_solver_name]

    print(f"Time solver: {time_solver_name}")
    print(f"Flux divergence calls per step: {rk_multiplier}")

    flux_div_time = timestep_avg / rk_multiplier
    print(f"Estimated flux_divergence time: ~{flux_div_time:.2f} ms")
    print()

    print("For comparison:")
    print(f"  HLLC flux benchmark (4096²): 2.52 ms")
    print(f"  Your flux_divergence (~{cfg.Nx}x{cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else cfg.Nx}): ~{flux_div_time:.2f} ms")
    print()

    print("="*60)
    print("NEXT STEPS")
    print("="*60)
    print("To drill deeper into flux_divergence components, run:")
    print("  python profile_detailed.py")
    print()


if __name__ == "__main__":
    main()
