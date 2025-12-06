import os
import sys
import time
import jax
import jax.numpy as jnp

# ============================================================================
# JAX PERFORMANCE FLAGS - Set BEFORE any JAX operations!
# ============================================================================
# Use float32 instead of float64 for 2x speed and 2x memory savings
jax.config.update("jax_enable_x64", False)
# Use high precision for matmuls (enables tensor cores on modern GPUs)
jax.config.update("jax_default_matmul_precision", "high")
# Enable optimizations (usually on by default, but explicit is better)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")  # Cache compiled kernels
print("JAX Performance Settings:")
print(f"  - Precision: {'float64' if jax.config.jax_enable_x64 else 'float32'}")
print(f"  - Matmul precision: {jax.config.jax_default_matmul_precision}")
print(f"  - Platform: {jax.default_backend()}")
print()

# ============================================================================
from functools import partial
from config import Config
from utils.memory_utils import print_memory_usage
from ic_generation.ic import initial_conditions
from time_integration.integrators import rk2_step, rk4_step, muscl_hancock_step
from time_integration.cfl import compute_dt_and_diagnostics
from file_io.write_snapshot import write_snapshot
from hydro.physics_checks import validate_state, validate_from_diagnostics
from hydro.divergence import flux_divergence
from utils.logo import print_KRATOS_banner 

def main():

    # --- Print logo ---
    print_KRATOS_banner()

    # --- Load configuration (pure Python object) ---
    cfg = Config()

    print("="*60)
    print("DEVICE INITIALIZATION")
    print("="*60)
    device_type = cfg.DEVICE
    if device_type == 0:
        print("Initializing Run on CPU...")
        jax.config.update("jax_platform_name", "cpu")
    else:
        print("Initializing Run on GPU...")
        # Try ROCm first
        try:
            jax.config.update("jax_platform_name", "rocm")
            _ = jax.devices()[0]  # Trigger device query to verify availability
            print("✓ Using ROCm GPU")
        except Exception:
            try:
                jax.config.update("jax_platform_name", "cuda")
                _ = jax.devices()[0]
                print("✓ Using CUDA GPU")
            except Exception:
                sys.exit("Error: No GPU available. Cannot proceed with GPU run.")
    print()

    # --- Initialize fields (must be done before computing grid spacing) ---
    # Note: Some problems (e.g., Sod-Shock-3D) may modify cfg.Nx, cfg.Ny, cfg.Nz, cfg.Lx, cfg.Ly, cfg.Lz
    # to ensure proper grid spacing, so we initialize first
    print("="*60)
    print("INITIAL CONDITIONS")
    print("="*60)
    U = initial_conditions(cfg)

    # Validate initial conditions
    print("Validating initial conditions...")
    validate_state(U, cfg.gamma, step=0, time=cfg.t_start)
    print("✓ Initial conditions validated successfully")
    print()

    # Print initial memory usage after all arrays are allocated
    print_memory_usage("After initialization: ")
    print()

    # --- Precompute constants to pass explicitly ---
    # These are computed AFTER initial_conditions so that any modifications to cfg are reflected
    dx = cfg.Lx / cfg.Nx
    dy = cfg.Ly / cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else None
    dz = cfg.Lz / cfg.Nz if cfg.SPATIAL_DIMENSION == 3 else None

    # Precompute inverse grid spacings (multiplication is faster than division on GPU)
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy if dy is not None else None
    inv_dz = 1.0 / dz if dz is not None else None

    # --- Create specialized functions using partial application ---
    # These functions have all constant parameters baked in, eliminating parameter threading
    print("="*60)
    print("CREATING SPECIALIZED SIMULATION FUNCTIONS")
    print("="*60)
    print("Baking constant parameters into functions:")
    dy_str = f"{dy:.6f}" if dy is not None else "None"
    dz_str = f"{dz:.6f}" if dz is not None else "None"
    print(f"  - Grid spacing: dx={dx:.6f}, dy={dy_str}, dz={dz_str}")
    print(f"  - Physics: gamma={cfg.gamma:.3f}")
    print(f"  - Numerics: order={cfg.ORDER}, limiter={cfg.LIMITER}, boundary={cfg.BOUNDARY}")
    print(f"  - Time solver: {['RK2', 'MUSCL-Hancock', 'RK4'][cfg.TIME_SOLVER]}")
    print()

    # Create specialized flux_divergence with all parameters baked in
    flux_fn = partial(
        flux_divergence,
        gamma=cfg.gamma,
        order=cfg.ORDER,
        limiter=cfg.LIMITER,
        boundary=cfg.BOUNDARY,
        dx=dx, dy=dy, dz=dz,
        inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz
    )

    # Select and specialize the time integration method
    time_solver_map = {
        0: rk2_step,
        1: muscl_hancock_step,
        2: rk4_step
    }
    time_step_fn = partial(time_solver_map[cfg.TIME_SOLVER], flux_fn=flux_fn)

    # --- Set output frequency ---
    # Compute number of snapshots based on time between snapshots and total simulation time
    n_snapshots = int(cfg.t_end / cfg.time_bet_snaps)  # Number of snapshots based on desired interval
    snapshot_times = jnp.linspace(cfg.t_start, cfg.t_end, n_snapshots)

    # If the final time t_end is not exactly included, add it
    if snapshot_times[-1] < cfg.t_end:
        snapshot_times = jnp.append(snapshot_times, cfg.t_end)

    snapshot_idx = 0

    # --- Output directory ---
    os.makedirs(cfg.output_dir, exist_ok=True)

    # --- Time loop variables ---
    t = cfg.t_start
    step = 0

    print("="*60)
    print("SIMULATION START")
    print("="*60)
    print(f"Time range: {cfg.t_start:.3f} → {cfg.t_end:.3f}")
    print(f"Grid: {cfg.Nx} x {cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else 'N/A'} x {cfg.Nz if cfg.SPATIAL_DIMENSION == 3 else 'N/A'}")
    print()

    # Start the timer for the total runtime
    start_time = time.time()  # Record the start time of the simulation

    # Flag to print memory after first few steps (to measure peak during computation)
    printed_peak_memory = cfg.check_memory

    # --- Time integration loop ---
    while t < cfg.t_end:

        # Compute dt AND physics diagnostics in one pass (fused to avoid redundant computation)
        t_snapshot = snapshot_times[snapshot_idx]
        dt, min_rho, min_p, has_nan, has_inf = compute_dt_and_diagnostics(
            U, cfg.gamma, cfg.CFL, t, t_snapshot, dx, dy, dz
        )
        if t + dt > cfg.t_end:
            dt = cfg.t_end - t  # Ensure we do not overshoot the final time

        # Time stepping with specialized function (all parameters baked in!)
        U = time_step_fn(U, dt)
        t += dt
        step += 1

        # Validate state less frequently to avoid GPU synchronization overhead
        # Checking every step causes GPU to wait for CPU - very slow!
        # Only validate every 100 steps (or first step) - assumes simulation is stable
        if step % cfg.output_cadence == 0 or step == 1:
            validate_from_diagnostics(min_rho, min_p, has_nan, has_inf, step=step, time=t)

        # Print progress less frequently to avoid GPU synchronization overhead
        # Printing every step causes GPU to wait for CPU - very slow!
        if step % cfg.output_cadence == 0 or step == 1:  # Print every 10 steps (or first step)
            print(f"Step {step:5d}, t = {t:.5f}, dt = {dt:.5e}")

        # Print peak memory after first 3 steps (after JIT compilation and warmup)
        if not printed_peak_memory and step == 3:
            print_memory_usage("After 3 steps (peak during computation): ")
            printed_peak_memory = True

        # Save snapshot if the current time matches the snapshot time
        if t >= t_snapshot:
            print(f"\n{'─'*60}")
            print(f"💾 Saving snapshot #{snapshot_idx} at t = {t:.5f}")
            print(f"{'─'*60}")
            # Call the updated write_snapshot function to save the full state to HDF5
            write_snapshot(U, cfg, t, snapshot_idx)
            snapshot_idx += 1

            # Ensure we don't overshoot the snapshot time
            if snapshot_idx < len(snapshot_times):
                t_snapshot = snapshot_times[snapshot_idx]
            else:
                t_snapshot = cfg.t_end  # Once all snapshots are saved, set t_snapshot to the end time

            print(f"✓ Snapshot saved successfully\n")

    # End the timer and calculate the total runtime
    end_time = time.time()  # Record the end time of the simulation
    total_time = end_time - start_time  # Calculate the total runtime

    print("\n" + "="*60)
    print("SIMULATION COMPLETED")
    print("="*60)
    print(f"✓ Total runtime: {total_time:.2f} seconds")
    print(f"✓ Total steps: {step}")
    print(f"✓ Final time: {t:.5f}")
    print()

    # Print final memory usage
    print_memory_usage("Final memory usage: ")
    print()

    # Plotting phase
    if cfg.PLOT_AFTER_SIM:
        print("="*60)
        print("GENERATING PLOTS")
        print("="*60)
        if cfg.SPATIAL_DIMENSION == 1 or cfg.SPATIAL_DIMENSION == 3:
            print("Generating 1D fluid quantity plots...")
            from utils.plot_1d_fluid_quants import plot_1d_fluid_quantities
            plot_1d_fluid_quantities(cfg.output_dir, cfg.base_name)
            print("✓ Plots generated successfully")
        elif cfg.SPATIAL_DIMENSION == 2:
            print("Generating density plots from all HDF5 snapshots...")
            from utils.plot_density2d import plot_density2d
            plot_density2d(cfg.output_dir, cfg.base_name)
            print("✓ Plots generated successfully")
        print()

    print("="*60)
    print("Good Bye and Good Day!")
    print("="*60)

if __name__ == "__main__":
    main()

