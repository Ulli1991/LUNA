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
from file_io.write_checkpoint import write_checkpoint
from file_io.read_checkpoint import read_checkpoint, find_latest_checkpoint
from file_io.checkpoint_utils import manage_checkpoints, validate_checkpoint_compatibility
from hydro.physics_checks import validate_state, validate_from_diagnostics
from hydro.divergence import flux_divergence_primitive
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

    # --- Initialize fields OR restart from checkpoint ---
    # Note: Some problems (e.g., Sod-Shock-3D) may modify cfg.Nx, cfg.Ny, cfg.Nz, cfg.Lx, cfg.Ly, cfg.Lz
    # to ensure proper grid spacing, so we initialize first
    print("="*60)

    if cfg.restart:
        print("RESTART FROM CHECKPOINT")
        print("="*60)

        # Find checkpoint file
        if cfg.restart_file is not None:
            checkpoint_file = cfg.restart_file
        else:
            checkpoint_file = find_latest_checkpoint(cfg.checkpoint_dir)
            if checkpoint_file is None:
                sys.exit(f"ERROR: No checkpoint found in {cfg.checkpoint_dir}")

        # Load checkpoint
        checkpoint_data = read_checkpoint(checkpoint_file)

        # Validate compatibility
        validate_checkpoint_compatibility(checkpoint_data, cfg)

        # Extract state
        U = checkpoint_data["U"]
        t_start = checkpoint_data["time"]
        step_start = checkpoint_data["step"]

        print(f"✓ Restarting from t = {t_start:.5f}, step = {step_start}")
        print()

    else:
        print("INITIAL CONDITIONS")
        print("="*60)
        U = initial_conditions(cfg)
        t_start = cfg.t_start
        step_start = 0

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

    # Create specialized flux_divergence with all parameters baked in
    # Using PRIMITIVE variable reconstruction (Athena++ style)
    # Note: flux_fn signature is flux_fn(U, dt)
    # - For MUSCL-Hancock (TIME_SOLVER=1): Pass dt to enable predictor step
    # - For RK2/RK4 (TIME_SOLVER=0,2): Pass dt=None (no predictor, RK handles time)
    print("  - Using PRIMITIVE variable reconstruction (Athena++ style)")

    if cfg.TIME_SOLVER == 1:  # MUSCL-Hancock
        print("  - MUSCL-Hancock: Predictor step enabled (dt passed to flux function)")
        def flux_fn(U, dt):
            return flux_divergence_primitive(
                U, cfg.gamma, cfg.ORDER, cfg.LIMITER, cfg.BOUNDARY,
                dx, dy, dz, inv_dx, inv_dy, inv_dz, dt=dt
            )
    else:  # RK2 or RK4
        print("  - RK2/RK4: No predictor (dt=None, temporal integration external)")
        def flux_fn(U, dt):
            return flux_divergence_primitive(
                U, cfg.gamma, cfg.ORDER, cfg.LIMITER, cfg.BOUNDARY,
                dx, dy, dz, inv_dx, inv_dy, inv_dz, dt=None
            )
    print()

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

    # Find the first snapshot index after restart time
    snapshot_idx = 0
    if cfg.restart:
        snapshot_idx = jnp.searchsorted(snapshot_times, t_start)
        if snapshot_idx >= len(snapshot_times):
            snapshot_idx = len(snapshot_times) - 1

    # --- Checkpoint timing (wall-clock based) ---
    last_checkpoint_time = None  # Track when we last checkpointed (wall-clock seconds)

    # --- Output directory ---
    os.makedirs(cfg.output_dir, exist_ok=True)
    if cfg.enable_checkpointing:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # --- Time loop variables ---
    t = t_start
    step = step_start

    print("="*60)
    print("SIMULATION START")
    print("="*60)
    print(f"Time range: {cfg.t_start:.3f} → {cfg.t_end:.3f}")
    print(f"Grid: {cfg.Nx} x {cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else 'N/A'} x {cfg.Nz if cfg.SPATIAL_DIMENSION == 3 else 'N/A'}")
    print()

    # Start the timer for the total runtime
    start_time = time.time()  # Record the start time of the simulation
    last_perf_time = start_time  # For computing cell updates/second

    # Flag to print memory after first few steps (to measure peak during computation)
    printed_peak_memory = cfg.check_memory

    # Total number of cells in the grid
    if cfg.SPATIAL_DIMENSION == 1:
        total_cells = cfg.Nx
    elif cfg.SPATIAL_DIMENSION == 2:
        total_cells = cfg.Nx * cfg.Ny
    else:  # 3D
        total_cells = cfg.Nx * cfg.Ny * cfg.Nz

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
        if step % cfg.output_cadence == 0 or step == 1:  # Print every output_cadence steps (or first step)
            # Compute performance metric: cell updates per second
            current_time = time.time()
            elapsed_since_last = current_time - last_perf_time
            if step == 1:
                # First step includes JIT compilation overhead, just report step info
                print(f"Step {step:5d}, t = {t:.5f}, dt = {dt:.5e}")
            else:
                # Compute cell updates/second over the last interval
                steps_since_last = cfg.output_cadence if step > cfg.output_cadence else step - 1
                cell_updates = total_cells * steps_since_last
                cells_per_sec = cell_updates / elapsed_since_last
                mcells_per_sec = cells_per_sec / 1e6  # Convert to millions
                print(f"Step {step:5d}, t = {t:.5f}, dt = {dt:.5e}, perf = {mcells_per_sec:.2f} Mcells/s")
            last_perf_time = current_time

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

        # Save checkpoint based on wall-clock time
        if cfg.enable_checkpointing and cfg.checkpoint_wall_time_interval > 0:
            current_wall_time = time.time()

            # Check if it's time to checkpoint
            should_checkpoint = False
            if last_checkpoint_time is None:
                # First checkpoint - save immediately after first step
                should_checkpoint = (step > 0)
            else:
                # Check if enough wall-clock time has elapsed
                elapsed_wall_time = current_wall_time - last_checkpoint_time
                should_checkpoint = (elapsed_wall_time >= cfg.checkpoint_wall_time_interval)

            if should_checkpoint:
                print(f"\n{'─'*60}")
                if last_checkpoint_time is not None:
                    elapsed = current_wall_time - last_checkpoint_time
                    print(f"💾 Writing checkpoint (wall-time elapsed: {elapsed:.1f}s)")
                else:
                    print(f"💾 Writing first checkpoint")
                print(f"   Simulation: t = {t:.5f}, step = {step}")
                print(f"{'─'*60}")
                write_checkpoint(U, cfg, t, step, cfg.checkpoint_dir)
                manage_checkpoints(cfg.checkpoint_dir, cfg.max_checkpoints)
                last_checkpoint_time = current_wall_time
                print()

    # End the timer and calculate the total runtime
    end_time = time.time()  # Record the end time of the simulation
    total_time = end_time - start_time  # Calculate the total runtime

    print("\n" + "="*60)
    print("SIMULATION COMPLETED")
    print("="*60)
    print(f"✓ Total runtime: {total_time:.2f} seconds")
    print(f"✓ Total steps: {step}")
    print(f"✓ Final time: {t:.5f}")

    # Compute overall performance (excluding first step with JIT overhead)
    if step > 1:
        total_cell_updates = total_cells * (step - 1)
        avg_cells_per_sec = total_cell_updates / total_time
        avg_mcells_per_sec = avg_cells_per_sec / 1e6  # Convert to millions
        print(f"✓ Average performance: {avg_mcells_per_sec:.2f} Mcells/s")
        print(f"✓ Time per step: {total_time/(step-1)*1000:.2f} ms (excluding first step)")
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

