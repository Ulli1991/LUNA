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
from time_integration.integrators import (create_rk2_pmap_step, create_rk4_pmap_step)
from time_integration.cfl import create_compute_local_dt_pmap
from hydro.divergence import flux_divergence_from_padded
from hydro.physics_checks import validate_state, validate_from_diagnostics
from utils.logo import print_KRATOS_banner
from utils.halo_pmap import create_pad_halo
from file_io.write_snapshot import write_snapshot


def main():
    print_KRATOS_banner()

    cfg = Config()

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

    # Create global initial conditions (only once)
    print("\n" + "="*60)
    print("INITIAL CONDITIONS")
    print("="*60)
    print("Generating global initial conditions...")
    U_global = initial_conditions(cfg)
    print(f"Global shape: {U_global.shape}")

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
        time_step_fn = create_rk2_pmap_step(
            pad_halo, flux_divergence_from_padded,
            cfg.gamma, cfg.ORDER, cfg.LIMITER,
            dx, dy, dz, inv_dx, inv_dy, inv_dz
        )
    elif cfg.TIME_SOLVER == 2:
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
    snapshot_idx = 0

    # Output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ==================================================================
    # TIME LOOP
    # ==================================================================
    print("="*60)
    print("SIMULATION START")
    print("="*60)
    print(f"Time range: {cfg.t_start:.3f} → {cfg.t_end:.3f}")
    print(f"Global grid: {cfg.Nx} x {cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else 'N/A'}")
    print(f"Local grid per GPU: {U_locals.shape[2:]} (excluding Nvar dimension)")
    print()

    t = cfg.t_start
    step = 0
    start_time = time.time()
    printed_peak_memory = not cfg.check_memory  # Only print if check_memory is True

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
            print(f"Step {step:5d}, t = {t:.5f}, dt = {dt:.5e}")

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

    end_time = time.time()
    total_time = end_time - start_time

    print("\n" + "="*60)
    print("SIMULATION COMPLETED")
    print("="*60)
    print(f"✓ Total runtime: {total_time:.2f} seconds")
    print(f"✓ Total steps: {step}")
    print(f"✓ Final time: {t:.5f}")
    print(f"✓ Time per step: {total_time/step:.4f} seconds")
    print()

    # Estimate speedup vs single GPU
    print(f"Number of devices: {n_devices}")
    print(f"Expected speedup vs 1 GPU: ~{n_devices * 0.85:.1f}x")
    print(f"\nTo measure actual speedup: run main.py (single GPU) and compare.")
    print(f"Single GPU would do {cfg.Nx}x{cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else cfg.Nx} cells")
    print(f"Each of {n_devices} GPUs does {U_locals.shape[2]}x{U_locals.shape[3] if cfg.SPATIAL_DIMENSION >= 2 else U_locals.shape[2]} cells in PARALLEL")
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
