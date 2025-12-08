"""
Multi-GPU hydro simulation using JAX pmap for single-node parallelism.

This version achieves TRUE speedup by:
1. Each GPU holds only its local subdomain
2. Computation runs independently on each GPU
3. Only halo exchange requires communication

Expected speedup: ~1.8x on 2 GPUs, ~3.5x on 4 GPUs
"""
import os
import sys
import time

# Set JAX to use all available GPUs
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3')

import jax
import jax.numpy as jnp

print("="*60)
print("MULTI-GPU PMAP MODE")
print("="*60)

# Configure JAX
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "high")
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

print(f"Platform: {jax.default_backend()}")
print(f"Devices: {len(jax.devices())}")
for i, dev in enumerate(jax.devices()):
    print(f"  Device {i}: {dev.device_kind}")
print()

from config import Config
from utils.domain_decomp import (compute_decomposition, get_local_domain_info,
                                  extract_local_subdomain, reconstruct_global_array)
from utils.halo_exchange import pad_with_halo_exchange
from utils.memory_utils import print_memory_usage
from ic_generation.ic import initial_conditions
from time_integration.integrators import (create_rk2_pmap_step, create_rk4_pmap_step,
                                          create_muscl_hancock_pmap_step)
from time_integration.cfl import create_compute_local_dt_pmap
from hydro.divergence import flux_divergence_from_padded
from hydro.physics_checks import validate_state, validate_from_diagnostics
from utils.logo import print_KRATOS_banner
from utils.halo_pmap import create_pad_halo
from file_io.write_snapshot import write_snapshot
from file_io.write_checkpoint import write_checkpoint
from file_io.read_checkpoint import read_checkpoint, find_latest_checkpoint
from file_io.checkpoint_utils import manage_checkpoints, validate_checkpoint_compatibility


def main():
    print_KRATOS_banner()

    cfg = config()

    # Check device count
    n_devices = len(jax.local_devices())
    if n_devices < 2:
        print(f"ERROR: Need at least 2 GPUs for multi-GPU mode. Found {n_devices}")
        sys.exit(1)

    print("="*60)
    print(f"DOMAIN DECOMPOSITION ({n_devices} GPUs)")
    print("="*60)

    # Compute decomposition
    grid_shape = compute_decomposition(cfg.Nx, cfg.Ny, n_devices, cfg.SPATIAL_DIMENSION)
    print(f"Process grid: {grid_shape}")

    # Create global initial conditions OR restart from checkpoint
    print("\n" + "="*60)

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
        U_global = checkpoint_data["U"]
        t_start = checkpoint_data["time"]
        step_start = checkpoint_data["step"]

        print(f"✓ Restarting from t = {t_start:.5f}, step = {step_start}")
        print()

    else:
        print("INITIAL CONDITIONS")
        print("="*60)
        print("Generating global initial conditions...")
        U_global = initial_conditions(cfg)
        print(f"Global shape: {U_global.shape}")
        t_start = cfg.t_start
        step_start = 0

        # Validate initial conditions
        print("Validating initial conditions...")
        validate_state(U_global, cfg.gamma, step=0, time=cfg.t_start)
        print("✓ Initial conditions validated successfully")
        print()

    # Split into local subdomains
    print("\nCreating local subdomains...")
    U_locals = []
    domain_infos = []

    for device_id in range(n_devices):
        domain_info = get_local_domain_info(
            device_id, cfg.Nx, cfg.Ny, cfg.Nz,
            grid_shape, cfg.SPATIAL_DIMENSION
        )
        domain_infos.append(domain_info)

        U_local = extract_local_subdomain(U_global, domain_info, cfg.SPATIAL_DIMENSION)
        U_locals.append(U_local)

        print(f"  Device {device_id}: local shape = {U_local.shape}, "
              f"coords = {domain_info['proc_coords']}")

    # Stack into a single array with leading device dimension
    # Shape: (n_devices, Nvar, nx_local, ...)
    U_locals = jnp.stack(U_locals, axis=0)
    print(f"\nStacked local arrays shape: {U_locals.shape}")

    # Print initial memory usage after all arrays are allocated
    print()
    print_memory_usage("After initialization: ")
    print()

    # Precompute grid spacing
    dx = cfg.Lx / cfg.Nx
    dy = cfg.Ly / cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else None
    dz = cfg.Lz / cfg.Nz if cfg.SPATIAL_DIMENSION == 3 else None
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy if dy is not None else None
    inv_dz = 1.0 / dz if dz is not None else None

    dy_str = f"{dy:.6f}" if dy is not None else "None"
    dz_str = f"{dz:.6f}" if dz is not None else "None"
    print(f"\nGrid spacing: dx={dx:.6f}, dy={dy_str}, dz={dz_str}")
    print()

    # ==================================================================
    # CREATE PMAPPED FUNCTIONS WITH PROPER HALO EXCHANGE
    # ==================================================================

    # Create halo padding function
    pad_halo = create_pad_halo(cfg.SPATIAL_DIMENSION, cfg.BOUNDARY, grid_shape)

    # Create time stepper
    if cfg.TIME_SOLVER == 0:
        print("Using RK2 time integration")
        time_step_fn = create_rk2_pmap_step(
            pad_halo, flux_divergence_from_padded,
            cfg.gamma, cfg.ORDER, cfg.LIMITER,
            dx, dy, dz, inv_dx, inv_dy, inv_dz
        )
    elif cfg.TIME_SOLVER == 1:
        print("Using MUSCL-Hancock predictor-corrector time integration")
        time_step_fn = create_muscl_hancock_pmap_step(
            pad_halo, flux_divergence_from_padded,
            cfg.gamma, cfg.ORDER, cfg.LIMITER,
            dx, dy, dz, inv_dx, inv_dy, inv_dz
        )
    elif cfg.TIME_SOLVER == 2:
        print("Using RK4 time integration")
        time_step_fn = create_rk4_pmap_step(
            pad_halo, flux_divergence_from_padded,
            cfg.gamma, cfg.ORDER, cfg.LIMITER,
            dx, dy, dz, inv_dx, inv_dy, inv_dz
        )
    else:
        print(f"ERROR: TIME_SOLVER={cfg.TIME_SOLVER} not supported in pmap mode")
        sys.exit(1)

    print("="*60)
    print("COMPILING PMAPPED FUNCTIONS")
    print("="*60)
    print("Running one step to trigger JIT compilation...")

    # Initial dt
    dt = 0.0001

    # Warmup
    U_locals = time_step_fn(U_locals, dt)
    U_locals = jax.block_until_ready(U_locals)
    print("✓ Compilation complete\n")

    # ==================================================================
    # SNAPSHOTS AND OUTPUT
    # ==================================================================
    # Set output frequency
    n_snapshots = int(cfg.t_end / cfg.time_bet_snaps)
    snapshot_times = jnp.linspace(cfg.t_start, cfg.t_end, n_snapshots)
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

    # Output directory
    os.makedirs(cfg.output_dir, exist_ok=True)
    if cfg.enable_checkpointing:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # ==================================================================
    # TIME LOOP
    # ==================================================================
    print("="*60)
    print("SIMULATION START")
    print("="*60)
    print(f"Time range: {t_start:.3f} → {cfg.t_end:.3f}")
    print(f"Global grid: {cfg.Nx} x {cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else 'N/A'}")
    print(f"Local grid per GPU: {U_locals.shape[2:]} (excluding Nvar dimension)")
    print()

    t = t_start
    step = step_start
    start_time = time.time()
    last_perf_time = start_time  # For computing cell updates/second
    printed_peak_memory = not cfg.check_memory  # Only print if check_memory is True

    # Total number of cells in the GLOBAL grid
    if cfg.SPATIAL_DIMENSION == 1:
        total_cells = cfg.Nx
    elif cfg.SPATIAL_DIMENSION == 2:
        total_cells = cfg.Nx * cfg.Ny
    else:  # 3D
        total_cells = cfg.Nx * cfg.Ny * cfg.Nz

    # Create compute_local_dt function outside the loop
    compute_local_dt = create_compute_local_dt_pmap(cfg.gamma, cfg.CFL, dx, dy, dz)

    while t < cfg.t_end:
        # Compute dt AND physics diagnostics from all local subdomains
        # We need to compute dt from ALL devices and take the minimum
        t_snapshot_local = snapshot_times[snapshot_idx]

        # Compute on all devices (need to broadcast t and t_snapshot to all devices)
        t_array = jnp.full(n_devices, t)
        t_snapshot_array = jnp.full(n_devices, t_snapshot_local)
        dt_vals, min_rho_vals, min_p_vals, has_nan_vals, has_inf_vals = compute_local_dt(
            U_locals, t_array, t_snapshot_array
        )

        # Take minimum dt across all devices
        dt = float(jnp.min(dt_vals))
        min_rho = float(jnp.min(min_rho_vals))
        min_p = float(jnp.min(min_p_vals))
        has_nan = bool(jnp.any(has_nan_vals))
        has_inf = bool(jnp.any(has_inf_vals))

        # Ensure we don't overshoot t_end
        if t + dt > cfg.t_end:
            dt = cfg.t_end - t

        # Time step (runs independently on each GPU!)
        U_locals = time_step_fn(U_locals, dt)

        # Force synchronization (wait for all GPUs)
        U_locals = jax.block_until_ready(U_locals)

        t += dt
        step += 1

        # Validation and progress printing
        if step % cfg.output_cadence == 0 or step == 1:
            validate_from_diagnostics(min_rho, min_p, has_nan, has_inf, step=step, time=t)

            # Compute performance metric: cell updates per second
            current_time = time.time()
            elapsed_since_last = current_time - last_perf_time
            if step == 1:
                # First step includes JIT compilation overhead
                print(f"Step {step:5d}, t = {t:.5f}, dt = {dt:.5e}")
            else:
                # Compute cell updates/second over the last interval (global cells)
                steps_since_last = cfg.output_cadence if step > cfg.output_cadence else step - 1
                cell_updates = total_cells * steps_since_last
                cells_per_sec = cell_updates / elapsed_since_last
                mcells_per_sec = cells_per_sec / 1e6  # Convert to millions
                print(f"Step {step:5d}, t = {t:.5f}, dt = {dt:.5e}, perf = {mcells_per_sec:.2f} Mcells/s")
            last_perf_time = current_time

        # Print peak memory after first 3 steps
        if not printed_peak_memory and step == 3:
            print_memory_usage("After 3 steps (peak during computation): ")
            printed_peak_memory = True

        # Save snapshot if time matches
        if t >= snapshot_times[snapshot_idx]:
            # Gather all local subdomains to reconstruct full array
            print(f"\n{'─'*60}")
            print(f"💾 Saving snapshot #{snapshot_idx} at t = {t:.5f}")
            print(f"{'─'*60}")

            # Reconstruct global array from local subdomains
            U_global = reconstruct_global_array(
                U_locals, domain_infos, cfg.Nx, cfg.Ny, cfg.Nz, cfg.SPATIAL_DIMENSION
            )

            # Write snapshot
            write_snapshot(U_global, cfg, t, snapshot_idx)

            snapshot_idx += 1
            if snapshot_idx >= len(snapshot_times):
                snapshot_idx = len(snapshot_times) - 1

            print(f"✓ Snapshot saved successfully\n")

        # Save checkpoint based on wall-clock time
        if cfg.enable_checkpointing and cfg.checkpoint_wall_time_interval > 0:
            current_wall_time = time.time()

            # Check if it's time to checkpoint
            should_checkpoint = False
            if last_checkpoint_time is None:
                # First checkpoint - save immediately after first step
                should_checkpoint = (step > step_start)
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

                # Reconstruct global array for checkpoint
                U_global = reconstruct_global_array(
                    U_locals, domain_infos, cfg.Nx, cfg.Ny, cfg.Nz, cfg.SPATIAL_DIMENSION
                )

                write_checkpoint(U_global, cfg, t, step, cfg.checkpoint_dir)
                manage_checkpoints(cfg.checkpoint_dir, cfg.max_checkpoints)
                last_checkpoint_time = current_wall_time
                print()

    end_time = time.time()
    total_time = end_time - start_time

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
        print(f"✓ Average performance: {avg_mcells_per_sec:.2f} Mcells/s (global grid)")
        print(f"✓ Time per step: {total_time/(step-1)*1000:.2f} ms (excluding first step)")
    print()

    # Estimate speedup vs single GPU
    print(f"Number of devices: {n_devices}")
    print(f"Expected speedup vs 1 GPU: ~{n_devices * 0.85:.1f}x")
    print(f"\nTo measure actual speedup: run main.py (single GPU) and compare cells/s.")
    print(f"Global grid: {cfg.Nx}x{cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else cfg.Nx} = {total_cells} cells")
    print(f"Each of {n_devices} GPUs processes {U_locals.shape[2]}x{U_locals.shape[3] if cfg.SPATIAL_DIMENSION >= 2 else U_locals.shape[2]} = {total_cells//n_devices} cells in PARALLEL")
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
