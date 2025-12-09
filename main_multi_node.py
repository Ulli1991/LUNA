import os
import sys
import time

print("="*60)
print("MULTI-NODE SHARDING MODE (1 GPU per process)")
print("="*60)

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental import multihost_utils
from jax import shard_map

try:
    # Let JAX auto-detect everything from SLURM environment
    # It will use SLURM_PROCID and SLURM_NTASKS
    jax.distributed.initialize(
        coordinator_address=os.environ.get('JAX_COORDINATOR_ADDRESS'),
        num_processes=int(os.environ.get('SLURM_NTASKS', 1)),
        process_id=int(os.environ.get('SLURM_PROCID', 0)),
    )
    print(f"✓ JAX distributed initialized")
    print(f"  Process: {jax.process_index()} / {jax.process_count()}")
except (RuntimeError, ValueError) as e:
    # If not in a distributed environment, run in single-process mode
    print(f"  Running in single-process mode (use SLURM/MPI for multi-node)")
    print(f"  Error: {e}")

# Configure JAX
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "high")
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

print(f"Platform: {jax.default_backend()}")
print(f"Total devices: {len(jax.devices())}")
print(f"Local devices: {len(jax.local_devices())}")

# Show device topology
for i, dev in enumerate(jax.devices()):
    print(f"  Device {i}: {dev.device_kind} (process {dev.process_index})")
print()

from config import Config
from utils.memory_utils import print_memory_usage
from ic_generation.ic import initial_conditions
from hydro.divergence import flux_divergence_from_padded
from hydro.physics_checks import validate_state, validate_from_diagnostics
from utils.logo import print_KRATOS_banner
from file_io.write_snapshot_distributed import write_snapshot_distributed
from file_io.write_checkpoint import write_checkpoint
from file_io.read_checkpoint import read_checkpoint, find_latest_checkpoint
from file_io.checkpoint_utils import manage_checkpoints, validate_checkpoint_compatibility
from utils.halo_sharding_simple import pad_with_halo_2d_sharded, pad_with_halo_1d_sharded
from utils.collectives import global_min, global_any
from time_integration.cfl import compute_dt_and_diagnostics
from utils.distributed import create_device_mesh, shard_array_to_devices, gather_array_from_devices
from time_integration.sharded_integrators import select_time_stepper


def main():
    # Only print from process 0 to avoid cluttered output
    if jax.process_index() == 0:
        print_KRATOS_banner()

    cfg = Config()

    n_devices = len(jax.devices())
    if n_devices < 1:
        print(f"ERROR: No devices available!")
        sys.exit(1)

    if jax.process_index() == 0:
        print("="*60)
        print(f"DOMAIN DECOMPOSITION ({n_devices} devices)")
        print("="*60)

    # Create device mesh
    mesh, grid_shape = create_device_mesh(n_devices, cfg.SPATIAL_DIMENSION)

    if jax.process_index() == 0:
        print(f"Process grid: {grid_shape}")
        print()

    # ==================================================================
    # INITIAL CONDITIONS OR RESTART
    # ==================================================================
    if jax.process_index() == 0:
        print("="*60)

    # All processes need to create/load the data for proper sharding
    # In JAX distributed, each process needs the global array to properly shard it
    if cfg.restart:
        if jax.process_index() == 0:
            print("RESTART FROM CHECKPOINT")
            print("="*60)

        # Find checkpoint file
        if cfg.restart_file is not None:
            checkpoint_file = cfg.restart_file
        else:
            checkpoint_file = find_latest_checkpoint(cfg.checkpoint_dir)
            if checkpoint_file is None:
                sys.exit(f"ERROR: No checkpoint found in {cfg.checkpoint_dir}")

        # Load checkpoint (all processes load it)
        checkpoint_data = read_checkpoint(checkpoint_file)

        # Validate compatibility
        validate_checkpoint_compatibility(checkpoint_data, cfg)

        # Extract state
        U_global = checkpoint_data["U"]
        t_start = checkpoint_data["time"]
        step_start = checkpoint_data["step"]

        if jax.process_index() == 0:
            print(f"✓ Restarting from t = {t_start:.5f}, step = {step_start}")
            print()

    else:
        if jax.process_index() == 0:
            print("INITIAL CONDITIONS")
            print("="*60)
            print("Generating global initial conditions...")

        # All processes generate initial conditions
        # JAX sharding requires all processes to have the global array
        U_global = initial_conditions(cfg)

        if jax.process_index() == 0:
            print(f"Global shape: {U_global.shape}")

        t_start = cfg.t_start
        step_start = 0

        # Validate initial conditions (only on process 0 for output)
        if jax.process_index() == 0:
            print("Validating initial conditions...")
            validate_state(U_global, cfg.gamma, step=0, time=cfg.t_start)
            print("✓ Initial conditions validated successfully")
            print()

    # Shard the global array to all devices
    # Each process will only keep its local portion after sharding
    if jax.process_index() == 0:
        print("Sharding global array to all devices...")

    U_sharded = shard_array_to_devices(U_global, mesh, cfg.SPATIAL_DIMENSION)

    if jax.process_index() == 0:
        print(f"✓ Array sharded across {n_devices} devices")
        print(f"  Each process keeps only its local shard")
        print()

    # Print memory usage
    if jax.process_index() == 0:
        print_memory_usage("After initialization: ")
        print()

    # Precompute grid spacing
    dx = cfg.Lx / cfg.Nx
    dy = cfg.Ly / cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else None
    dz = cfg.Lz / cfg.Nz if cfg.SPATIAL_DIMENSION == 3 else None
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy if dy is not None else None
    inv_dz = 1.0 / dz if dz is not None else None

    if jax.process_index() == 0:
        dy_str = f"{dy:.6f}" if dy is not None else "None"
        dz_str = f"{dz:.6f}" if dz is not None else "None"
        print(f"\nGrid spacing: dx={dx:.6f}, dy={dy_str}, dz={dz_str}")
        print()

    # ==================================================================
    # CREATE AUTOMATIC SHARDING (credit: Philip Mocz)
    # ==================================================================

    # Create NamedSharding for automatic sharding
    if cfg.SPATIAL_DIMENSION == 1:
        sharding = NamedSharding(mesh, P(None, 'x'))
    elif cfg.SPATIAL_DIMENSION == 2:
        sharding = NamedSharding(mesh, P(None, 'x', 'y'))
    else:
        raise NotImplementedError("3D not yet implemented")

    # Apply sharding constraint to initial array
    U_sharded = jax.lax.with_sharding_constraint(U_sharded, sharding)

    if jax.process_index() == 0:
        print(f"Using automatic sharding with NamedSharding")
        print(f"Sharding spec: {sharding.spec}")

    # Import halo exchange that works with automatic sharding
    from utils.halo_exchange_auto import pad_with_halo_auto

    # Create single JIT-compiled time step function (like reference)
    # This replaces the shard_map approach
    @jax.jit
    def time_step_fn(U, dt):
        """Single time step using MUSCL-Hancock with automatic sharding."""
        # Halo exchange - JAX auto-handles communication
        U_padded = pad_with_halo_auto(U, cfg.BOUNDARY, cfg.SPATIAL_DIMENSION)

        # Compute flux divergence
        dUdt = flux_divergence_from_padded(
            U_padded, cfg.gamma, cfg.ORDER, cfg.LIMITER,
            dx, dy, dz, inv_dx, inv_dy, inv_dz
        )

        # Time integration
        U_new = U + dt * dUdt

        return U_new

    integrator_name = "MUSCL-Hancock (automatic sharding)"
    if jax.process_index() == 0:
        print(f"Using {integrator_name} time integration")

    # Distributed dt computation using automatic sharding
    # JAX automatically handles the global minimum across sharded arrays
    if cfg.enable_diagnostics:
        @jax.jit
        def compute_dt_auto(U, t_val, t_snapshot_val):
            """Compute dt and diagnostics with automatic sharding."""
            dt_local, min_rho_local, min_p_local, has_nan_local, has_inf_local = compute_dt_and_diagnostics(
                U, cfg.gamma, cfg.CFL, t_val, t_snapshot_val, dx, dy, dz
            )
            dt = jnp.min(dt_local)
            min_rho = jnp.min(min_rho_local)
            min_p = jnp.min(min_p_local)
            has_nan = jnp.any(has_nan_local)
            has_inf = jnp.any(has_inf_local)
            return dt, min_rho, min_p, has_nan, has_inf
    else:
        # Fast path: dt only, no diagnostics
        from time_integration.cfl import compute_dt
        @jax.jit
        def compute_dt_auto(U, t_val, t_snapshot_val):
            """Compute dt only (no diagnostics) - maximum performance."""
            dt_local = compute_dt(U, cfg.gamma, cfg.CFL, t_val, t_snapshot_val, dx, dy, dz)
            dt = jnp.min(dt_local)
            return dt, 0.0, 0.0, False, False  # Dummy values for diagnostics

    # ==================================================================
    # WARMUP COMPILATION
    # ==================================================================
    if jax.process_index() == 0:
        print("="*60)
        print("COMPILING SHARDED FUNCTIONS")
        print("="*60)
        print("Running one step to trigger JIT compilation...")

    # Initial dt for warmup
    dt = 0.0001

    # Warmup
    U_sharded = time_step_fn(U_sharded, dt)
    U_sharded = jax.block_until_ready(U_sharded)

    if jax.process_index() == 0:
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
    if jax.process_index() == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        if cfg.enable_checkpointing:
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # ==================================================================
    # TIME LOOP
    # ==================================================================
    if jax.process_index() == 0:
        print("="*60)
        print("SIMULATION START")
        print("="*60)
        print(f"Time range: {t_start:.3f} → {cfg.t_end:.3f}")
        print(f"Global grid: {cfg.Nx} x {cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else 'N/A'}")
        print(f"Process grid: {grid_shape}")
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
    else:
        total_cells = cfg.Nx * cfg.Ny * cfg.Nz

    # Main time loop
    while t < cfg.t_end:
        # Compute dt using automatic sharding (JAX handles collectives)
        t_snapshot_local = snapshot_times[snapshot_idx]
        dt, min_rho, min_p, has_nan, has_inf = compute_dt_auto(
            U_sharded, t, t_snapshot_local
        )

        # Ensure dt doesn't exceed t_end
        if t + dt > cfg.t_end:
            dt = cfg.t_end - t

        # Time step (runs on all devices in parallel!)
        U_sharded = time_step_fn(U_sharded, dt)
        U_sharded = jax.block_until_ready(U_sharded)

        t += dt
        step += 1

        # Validation and progress (only on process 0)
        if jax.process_index() == 0:
            if step % cfg.output_cadence == 0 or step == 1:
                if cfg.enable_diagnostics:
                    validate_from_diagnostics(min_rho, min_p, has_nan, has_inf, step=step, time=t)

                current_time = time.time()
                elapsed_since_last = current_time - last_perf_time
                if step == 1:
                    print(f"Step {step:5d}, t = {t:.5f}, dt = {dt:.5e}")
                else:
                    steps_since_last = cfg.output_cadence if step > cfg.output_cadence else step - 1
                    cell_updates = total_cells * steps_since_last
                    cells_per_sec = cell_updates / elapsed_since_last
                    mcells_per_sec = cells_per_sec / 1e6
                    print(f"Step {step:5d}, t = {t:.5f}, dt = {dt:.5e}, perf = {mcells_per_sec:.2f} Mcells/s")
                last_perf_time = current_time

            if not printed_peak_memory and step == 3:
                print_memory_usage("After 3 steps (peak during computation): ")
                printed_peak_memory = True

        # Save snapshot (all processes participate)
        if t >= snapshot_times[snapshot_idx]:
            # Print header (only process 0)
            if jax.process_index() == 0:
                print(f"\n{'─'*60}")
                print(f"💾 Saving snapshot #{snapshot_idx} at t = {t:.5f}")
                print(f"{'─'*60}")

            # Each process writes its own shard to separate file
            write_snapshot_distributed(U_sharded, cfg, t, snapshot_idx)

            # Synchronize all processes after snapshot write
            # This ensures all file writes complete before continuing
            multihost_utils.sync_global_devices("snapshot_write")

            # Update snapshot index
            snapshot_idx += 1
            if snapshot_idx >= len(snapshot_times):
                snapshot_idx = len(snapshot_times) - 1

            # Print completion message (only process 0)
            if jax.process_index() == 0:
                print(f"✓ All shards saved successfully\n")

        # Checkpoint (only process 0)
        if cfg.enable_checkpointing and cfg.checkpoint_wall_time_interval > 0:
            if jax.process_index() == 0:
                current_wall_time = time.time()

                should_checkpoint = False
                if last_checkpoint_time is None:
                    should_checkpoint = (step > step_start)
                else:
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

                    # Gather global array
                    U_global = gather_array_from_devices(U_sharded)

                    write_checkpoint(U_global, cfg, t, step, cfg.checkpoint_dir)
                    manage_checkpoints(cfg.checkpoint_dir, cfg.max_checkpoints)
                    last_checkpoint_time = current_wall_time
                    print()

    # ==================================================================
    # FINALIZATION
    # ==================================================================
    if jax.process_index() == 0:
        end_time = time.time()
        total_time = end_time - start_time

        print("\n" + "="*60)
        print("SIMULATION COMPLETED")
        print("="*60)
        print(f"✓ Total runtime: {total_time:.2f} seconds")
        print(f"✓ Total steps: {step}")
        print(f"✓ Final time: {t:.5f}")

        if step > 1:
            total_cell_updates = total_cells * (step - 1)
            avg_cells_per_sec = total_cell_updates / total_time
            avg_mcells_per_sec = avg_cells_per_sec / 1e6
            print(f"✓ Average performance: {avg_mcells_per_sec:.2f} Mcells/s (global grid)")
            print(f"✓ Time per step: {total_time/(step-1)*1000:.2f} ms (excluding first step)")
        print()

        print(f"Number of devices: {n_devices}")
        print(f"Process grid: {grid_shape}")
        print()

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
