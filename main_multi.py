import os
import sys
import time
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental import multihost_utils

# ============================================================================
# JAX PERFORMANCE FLAGS - Set BEFORE any JAX operations!
# ============================================================================
# Use float32 instead of float64 for 2x speed and 2x memory savings
jax.config.update("jax_enable_x64", False)
# Use high precision for matmuls (enables tensor cores on modern GPUs)
jax.config.update("jax_default_matmul_precision", "high")
# Enable optimizations (usually on by default, but explicit is better)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")  # Cache compiled kernels

# ============================================================================
# MULTI-GPU/MULTI-NODE INITIALIZATION
# ============================================================================
# Initialize JAX distributed if running on multiple processes
# This must be done BEFORE any JAX operations!
if "JAX_COORDINATOR_ADDRESS" in os.environ or "SLURM_NTASKS" in os.environ:
    # Multi-node/multi-process mode
    try:
        coordinator = os.environ.get("JAX_COORDINATOR_ADDRESS", "")
        num_processes = int(os.environ.get("SLURM_NTASKS", 1))
        process_id = int(os.environ.get("SLURM_PROCID", 0))

        print(f"Initializing JAX distributed:")
        print(f"  Coordinator: {coordinator}")
        print(f"  Process {process_id} of {num_processes}")

        jax.distributed.initialize(
            coordinator_address=coordinator if coordinator else None,
            num_processes=num_processes,
            process_id=process_id
        )
        print(f"✓ JAX distributed initialized: process {jax.process_index()} of {jax.process_count()}")
    except Exception as e:
        print(f"⚠ JAX distributed initialization failed: {e}")
        print("Falling back to single-process mode")
else:
    # Single process - no initialization needed
    print("Single-process mode (no JAX distributed initialization)")

print("JAX Performance Settings:")
print(f"  - Precision: {'float64' if jax.config.jax_enable_x64 else 'float32'}")
print(f"  - Matmul precision: {jax.config.jax_default_matmul_precision}")
print(f"  - Platform: {jax.default_backend()}")
print(f"  - Devices: {len(jax.devices())} ({jax.devices()[0].platform})")
print(f"  - Process: {jax.process_index()} of {jax.process_count()}")
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


# ============================================================================
# SHARDING UTILITIES
# ============================================================================
def setup_sharding(spatial_dim, devices):
    """
    Setup JAX sharding for multi-GPU/multi-node execution.

    Parameters
    ----------
    spatial_dim : int
        1, 2, or 3
    devices : array-like
        JAX devices to use

    Returns
    -------
    mesh : Mesh
        Device mesh
    sharding : NamedSharding
        Sharding specification for state arrays
    """
    import numpy as np

    devices = np.array(devices)
    n_devices = len(devices)

    if spatial_dim == 1:
        # Shard along x-dimension only
        mesh = Mesh(devices, ('x',))
        sharding = NamedSharding(mesh, P(None, 'x'))  # (n_fields, Nx)
        print(f"  Sharding strategy: 1D along x ({n_devices} devices)")

    elif spatial_dim == 2:
        # Shard along x and y dimensions
        # Try to make mesh as square as possible
        nx = int(n_devices ** 0.5)
        while n_devices % nx != 0:
            nx -= 1
        ny = n_devices // nx

        mesh = Mesh(devices.reshape(nx, ny), ('x', 'y'))
        sharding = NamedSharding(mesh, P(None, 'x', 'y'))  # (n_fields, Nx, Ny)
        print(f"  Sharding strategy: 2D grid {nx}×{ny} devices")

    elif spatial_dim == 3:
        # Shard along all three dimensions
        # Try to factorize n_devices into nx * ny * nz
        nx = int(round(n_devices ** (1/3)))
        while n_devices % nx != 0:
            nx -= 1
        ny = int((n_devices // nx) ** 0.5)
        while (n_devices // nx) % ny != 0:
            ny -= 1
        nz = n_devices // (nx * ny)

        mesh = Mesh(devices.reshape(nx, ny, nz), ('x', 'y', 'z'))
        sharding = NamedSharding(mesh, P(None, 'x', 'y', 'z'))  # (n_fields, Nx, Ny, Nz)
        print(f"  Sharding strategy: 3D grid {nx}×{ny}×{nz} devices")

    return mesh, sharding


def main():

    # --- Print logo ---
    print_KRATOS_banner()

    # --- Load configuration (pure Python object) ---
    cfg = Config()

    print("="*60)
    print("DEVICE INFORMATION")
    print("="*60)
    print(f"Platform: {jax.default_backend()}")
    print(f"Devices available: {len(jax.devices())}")
    for i, device in enumerate(jax.devices()):
        print(f"  Device {i}: {device.device_kind} ({device.platform})")
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

    # =========================================================================
    # MULTI-GPU SHARDING SETUP
    # =========================================================================
    devices = jax.devices()
    n_devices = len(devices)
    use_sharding = n_devices > 1

    if use_sharding:
        print("="*60)
        print("MULTI-GPU SHARDING SETUP")
        print("="*60)
        print(f"Detected {n_devices} devices - enabling JAX sharding")

        mesh, sharding = setup_sharding(cfg.SPATIAL_DIMENSION, devices)

        # Shard the initial conditions across devices
        print("  Distributing data across devices...")
        U = jax.device_put(U, sharding)
        print("✓ Data sharded successfully")
        print()
    else:
        print(f"Single device mode - no sharding needed")
        sharding = None
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
    print(f"  - Grid spacing: dx={dx:.6f}, dy={dy}, dz={dz}")
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
    # Only process 0 creates the directory (avoid race conditions)
    if jax.process_index() == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)

    # Sync all processes before continuing
    if use_sharding:
        jax.block_until_ready(U)  # Implicit sync

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
        if step % cfg.output_cadence == 0 or step == 1:
            validate_from_diagnostics(min_rho, min_p, has_nan, has_inf, step=step, time=t)

        # Print progress less frequently to avoid GPU synchronization overhead
        if step % cfg.output_cadence == 0 or step == 1:
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

            # Gather sharded data if needed
            if use_sharding:
                # Gather from all devices to process 0
                U_gathered = multihost_utils.process_allgather(U, tiled=True)
                # Only process 0 writes
                if jax.process_index() == 0:
                    write_snapshot(U_gathered, cfg, t, snapshot_idx)
            else:
                # Single device - write directly
                write_snapshot(U, cfg, t, snapshot_idx)

            snapshot_idx += 1

            # Ensure we don't overshoot the snapshot time
            if snapshot_idx < len(snapshot_times):
                t_snapshot = snapshot_times[snapshot_idx]
            else:
                t_snapshot = cfg.t_end  # Once all snapshots are saved, set t_snapshot to the end time

            if jax.process_index() == 0:
                print(f"✓ Snapshot saved successfully\n")

    # End the timer and calculate the total runtime
    end_time = time.time()  # Record the end time of the simulation
    total_time = end_time - start_time  # Calculate the total runtime

    # Only process 0 prints final summary
    if jax.process_index() == 0:
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

    # Sync all processes before plotting
    if use_sharding:
        jax.block_until_ready(U)

    # Plotting phase - only on process 0
    if cfg.PLOT_AFTER_SIM and jax.process_index() == 0:
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

    if jax.process_index() == 0:
        print("="*60)
        print("Good Bye and Good Day!")
        print("="*60)

if __name__ == "__main__":
    main()

