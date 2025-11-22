"""
Multi-GPU hydrodynamics simulation using MPI.

This is a pure MPI implementation where:
- One MPI rank per GPU
- Each rank owns one patch
- Explicit MPI communication for halo exchange

Usage:
    # Single node (8 GPUs)
    mpirun -np 8 python main_multigpu_mpi.py

    # Multi-node (4 nodes × 8 GPUs = 32 GPUs)
    mpirun -np 32 --hostfile hosts.txt python main_multigpu_mpi.py

    # SLURM
    srun --nodes=4 --ntasks-per-node=8 python main_multigpu_mpi.py
"""

import os
import sys
import time
from functools import partial

# MPI must be imported before JAX to avoid initialization issues
from mpi4py import MPI

# === CRITICAL: Set GPU visibility BEFORE importing JAX ===
# JAX reads these environment variables during import/initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_ranks = comm.Get_size()

# GPU assignment strategy depends on how we're launched:
# - SLURM with --gpus-per-task: SLURM handles it automatically (don't override!)
# - Direct mpirun: We need to manually assign GPUs
if 'SLURM_STEP_GPUS' in os.environ or 'SLURM_JOB_GPUS' in os.environ or 'SLURM_LOCALID' in os.environ:
    # Running under SLURM with GPU allocation
    if rank == 0:
        print(f"=== Running under SLURM (MPI ranks: {n_ranks}) ===")
        print(f"SLURM_LOCALID: {os.environ.get('SLURM_LOCALID', 'not set')}")
        print(f"SLURM_STEP_GPUS: {os.environ.get('SLURM_STEP_GPUS', 'not set')}")
        print(f"GPU_DEVICE_ORDINAL: {os.environ.get('GPU_DEVICE_ORDINAL', 'not set')}")
        print(f"ROCR_VISIBLE_DEVICES (before): {os.environ.get('ROCR_VISIBLE_DEVICES', 'not set')}")

    # SLURM with --gpus-per-task=1 sets GPU_DEVICE_ORDINAL for ROCm
    # We need to copy it to ROCR_VISIBLE_DEVICES if it's not already set
    if 'GPU_DEVICE_ORDINAL' in os.environ and 'ROCR_VISIBLE_DEVICES' not in os.environ:
        os.environ['ROCR_VISIBLE_DEVICES'] = os.environ['GPU_DEVICE_ORDINAL']
        if rank == 0:
            print(f"Set ROCR_VISIBLE_DEVICES from GPU_DEVICE_ORDINAL: {os.environ['GPU_DEVICE_ORDINAL']}")

    # Alternatively, use SLURM_LOCALID (local rank on the node)
    if 'ROCR_VISIBLE_DEVICES' not in os.environ and 'SLURM_LOCALID' in os.environ:
        local_id = os.environ['SLURM_LOCALID']
        os.environ['ROCR_VISIBLE_DEVICES'] = local_id
        os.environ['HIP_VISIBLE_DEVICES'] = local_id
        if rank == 0:
            print(f"Set ROCR_VISIBLE_DEVICES from SLURM_LOCALID: {local_id}")
else:
    # Manual GPU assignment for direct mpirun
    if rank == 0:
        print(f"=== Running with direct mpirun (MPI ranks: {n_ranks}) ===")
        print("Manually assigning GPUs based on rank...")

    gpu_id = rank % 8  # Assuming max 8 GPUs per node
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # For NVIDIA
    os.environ['ROCR_VISIBLE_DEVICES'] = str(gpu_id)  # For AMD
    os.environ['HIP_VISIBLE_DEVICES'] = str(gpu_id)   # Alternative AMD env var

    if rank == 0:
        print(f"Rank 0: Assigned to GPU {gpu_id}")

import jax
import jax.numpy as jnp

# JAX configuration
jax.config.update("jax_enable_x64", False)

from config import Config
from grid.decomposition import create_uniform_decomposition
from grid.domain import Domain
from grid.neighbors import setup_neighbor_connectivity, print_connectivity
from ic_generation.ic import initial_conditions
from time_integration.cfl import compute_dt_and_diagnostics
from time_integration.integrators_mpi import (
    mpi_rk2_step,
    mpi_rk4_step,
    mpi_muscl_hancock_step,
    mpi_rk2_step_substep,
    mpi_rk4_step_substep,
    mpi_muscl_hancock_step_substep,
    _compute_k_stage
)
from hydro.physics_checks import validate_from_diagnostics
from hydro.divergence import flux_divergence_from_2ghost
from parallel.halo_exchange_mpi import (
    update_ghosts_mpi,
    compute_global_dt_mpi,
    gather_field_for_output_mpi
)
from parallel.mpi_flux import create_mpi_flux_fn, get_ghost_slices
from file_io.write_snapshot import write_snapshot
from file_io.gather_mpi import reconstruct_global_array


def main():
    # === MPI INITIALIZATION ===
    # Note: comm and rank are already initialized at module level for GPU setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    # === JAX DEVICE SETUP ===
    # Verify JAX can see the GPU assigned to this rank
    devices = jax.devices()
    if len(devices) == 0:
        if rank == 0:
            print("ERROR: No JAX devices found!")
        sys.exit(1)

    device = devices[0]  # This rank uses device 0 (which is actually GPU rank%8)

    if rank == 0:
        print("="*60)
        print("MULTI-GPU SIMULATION (MPI)")
        print("="*60)
        print(f"MPI ranks: {n_ranks}")
        print(f"JAX backend: {jax.default_backend()}")
        print()

    # === LOAD CONFIGURATION ===
    cfg = Config()

    # === IMPORTANT: Generate IC on rank 0 to update config dimensions ===
    # Some test problems (like Sod-Shock-3D) modify cfg.Ny, cfg.Nz to maintain
    # uniform cell spacing. We need these updated values BEFORE decomposition.
    if rank == 0:
        print("="*60)
        print("INITIAL CONFIGURATION (pre-IC)")
        print("="*60)
        print(f"Global grid: {cfg.Nx} × {cfg.Ny} × {cfg.Nz if cfg.SPATIAL_DIMENSION == 3 else 'N/A'}")
        print(f"Domain: {cfg.Lx} × {cfg.Ly} × {cfg.Lz if cfg.SPATIAL_DIMENSION == 3 else 'N/A'}")
        print()

        # Generate IC to let it update cfg if needed
        print("Generating initial conditions (to determine actual grid size)...")
        U_global = initial_conditions(cfg)
        print(f"Actual grid after IC: {cfg.Nx} × {cfg.Ny} × {cfg.Nz if cfg.SPATIAL_DIMENSION == 3 else 'N/A'}")
        print()
    else:
        U_global = None

    # Broadcast updated config dimensions to all ranks
    cfg_dims = None
    if rank == 0:
        cfg_dims = [cfg.Nx, cfg.Ny, cfg.Nz, cfg.Lx, cfg.Ly, cfg.Lz]
    cfg_dims = comm.bcast(cfg_dims, root=0)

    if rank != 0:
        cfg.Nx, cfg.Ny, cfg.Nz, cfg.Lx, cfg.Ly, cfg.Lz = cfg_dims

    if rank == 0:
        print()
        print("="*60)
        print("CONFIGURATION (after IC adjustment)")
        print("="*60)
        print(f"Global grid: {cfg.Nx} × {cfg.Ny} × {cfg.Nz if cfg.SPATIAL_DIMENSION == 3 else 'N/A'}")
        print(f"Domain: {cfg.Lx} × {cfg.Ly} × {cfg.Lz if cfg.SPATIAL_DIMENSION == 3 else 'N/A'}")
        print(f"Time range: {cfg.t_start} → {cfg.t_end}")
        print(f"CFL: {cfg.CFL}")
        print()

    # === DOMAIN DECOMPOSITION ===
    if rank == 0:
        print("="*60)
        print("DOMAIN DECOMPOSITION")
        print("="*60)

    # Create patches (all ranks do this - it's cheap)
    if cfg.SPATIAL_DIMENSION == 1:
        global_shape = (cfg.Nx,)
        global_extent = (cfg.Lx,)
    elif cfg.SPATIAL_DIMENSION == 2:
        global_shape = (cfg.Nx, cfg.Ny)
        global_extent = (cfg.Lx, cfg.Ly)
    else:
        global_shape = (cfg.Nx, cfg.Ny, cfg.Nz)
        global_extent = (cfg.Lx, cfg.Ly, cfg.Lz)

    patches = create_uniform_decomposition(
        global_shape=global_shape,
        global_extent=global_extent,
        n_gpus=n_ranks,
        boundary_type=cfg.BOUNDARY
    )

    # Setup neighbor connectivity
    setup_neighbor_connectivity(patches, periodic=(cfg.BOUNDARY == 0))

    # Print connectivity (only rank 0)
    if rank == 0:
        print_connectivity(patches)
        print()

    # === EACH RANK TAKES ONE PATCH ===
    my_patch = patches[rank]

    # Allocate data for this patch
    n_fields = 3 if cfg.SPATIAL_DIMENSION == 1 else (4 if cfg.SPATIAL_DIMENSION == 2 else 5)
    my_patch.allocate_data(n_fields, dtype=jnp.float32)

    if rank == 0:
        print("="*60)
        print("PATCH ASSIGNMENT")
        print("="*60)

    # All ranks print their patch info (in order)
    for r in range(n_ranks):
        if rank == r:
            print(f"Rank {rank}: {my_patch}")
            print(f"  Neighbors: {sum(len(n) for n in my_patch.neighbors.values())}")
            boundaries = [d for d in my_patch.boundary_types if my_patch.is_boundary(d)]
            if boundaries:
                print(f"  Domain boundaries: {boundaries}")
        comm.Barrier()  # Print in order

    if rank == 0:
        print()

    # === INITIAL CONDITIONS ===
    if rank == 0:
        print("="*60)
        print("EXTRACTING INITIAL CONDITIONS FOR PATCHES")
        print("="*60)

    # Rank 0 already has U_global from earlier (to determine grid size)
    # Other ranks need to generate it now
    # TODO: In production, each rank should only generate its local region
    if U_global is None:
        U_global = initial_conditions(cfg)

    # Extract this rank's portion
    interior_slice = my_patch.get_interior_slice()
    offset = my_patch.global_offset

    if cfg.SPATIAL_DIMENSION == 1:
        ix_start = offset[0]
        ix_end = offset[0] + my_patch.shape[0]
        U_local = U_global[:, ix_start:ix_end]
    elif cfg.SPATIAL_DIMENSION == 2:
        ix_start, iy_start = offset
        ix_end = ix_start + my_patch.shape[0]
        iy_end = iy_start + my_patch.shape[1]
        U_local = U_global[:, ix_start:ix_end, iy_start:iy_end]
    else:
        ix_start, iy_start, iz_start = offset
        ix_end = ix_start + my_patch.shape[0]
        iy_end = iy_start + my_patch.shape[1]
        iz_end = iz_start + my_patch.shape[2]
        U_local = U_global[:, ix_start:ix_end, iy_start:iy_end, iz_start:iz_end]

    # Place into patch data (interior cells)
    my_patch.data = my_patch.data.at[interior_slice].set(U_local)

    # Initial halo exchange
    update_ghosts_mpi(my_patch, cfg.BOUNDARY, comm)

    # === DEBUG: Check ghost cell filling ===
    if cfg.SPATIAL_DIMENSION == 3:
        ng = my_patch.n_ghost
        # Check if y ghost cells match first interior y (should be equal for zero-gradient BC)
        y_ghost_0 = my_patch.data[:, ng:-ng, 0, ng:-ng]  # y=0 ghost
        y_ghost_1 = my_patch.data[:, ng:-ng, 1, ng:-ng]  # y=1 ghost
        y_interior = my_patch.data[:, ng:-ng, ng, ng:-ng]  # y=ng (first interior)
        y_diff_0 = float(jnp.max(jnp.abs(y_ghost_0 - y_interior)))
        y_diff_1 = float(jnp.max(jnp.abs(y_ghost_1 - y_interior)))

        # Check z ghost cells similarly
        z_ghost_0 = my_patch.data[:, ng:-ng, ng:-ng, 0]  # z=0 ghost
        z_ghost_1 = my_patch.data[:, ng:-ng, ng:-ng, 1]  # z=1 ghost
        z_interior = my_patch.data[:, ng:-ng, ng:-ng, ng]  # z=ng (first interior)
        z_diff_0 = float(jnp.max(jnp.abs(z_ghost_0 - z_interior)))
        z_diff_1 = float(jnp.max(jnp.abs(z_ghost_1 - z_interior)))

        # Check for any y/z variation in interior (should be zero for 1D shock)
        interior_data = my_patch.data[interior_slice]
        rho = interior_data[0]
        mom_y = interior_data[2]  # rho*vy
        mom_z = interior_data[3]  # rho*vz

        # Variation along y (should be 0)
        rho_y_var = float(jnp.max(jnp.std(rho, axis=1)))
        # Variation along z (should be 0)
        rho_z_var = float(jnp.max(jnp.std(rho, axis=2)))

        print(f"Rank {rank} ghost cell check:")
        print(f"  y ghost: max|y0-interior|={y_diff_0:.2e}, max|y1-interior|={y_diff_1:.2e}")
        print(f"  z ghost: max|z0-interior|={z_diff_0:.2e}, max|z1-interior|={z_diff_1:.2e}")
        print(f"  Interior variation: rho_y_std={rho_y_var:.2e}, rho_z_std={rho_z_var:.2e}")
        print(f"  max|rho*vy|={float(jnp.max(jnp.abs(mom_y))):.2e}, max|rho*vz|={float(jnp.max(jnp.abs(mom_z))):.2e}")
    comm.Barrier()
    # === END DEBUG ===

    if rank == 0:
        print("✓ Initial conditions set")
        print()

    # === SETUP PHYSICS FUNCTIONS ===
    dx = my_patch.dx
    dy = my_patch.dy
    dz = my_patch.dz
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy if dy is not None else None
    inv_dz = 1.0 / dz if dz is not None else None

    # In MPI mode, ghost cells are always pre-filled:
    #   - Inter-patch boundaries: filled by halo exchange from neighbors
    #   - Domain edges: filled by apply_boundary_conditions_mpi using cfg.BOUNDARY
    #
    # Use flux_divergence_from_2ghost which properly handles 2 ghost cells per side.
    # This is CRITICAL for correct PLM reconstruction at MPI boundaries.
    # With only 1 ghost cell, the flux at subdomain boundaries uses asymmetric
    # reconstruction (cell-center from ghost, reconstructed from interior), causing
    # each rank to compute different fluxes at their shared interface.
    # With 2 ghost cells, we can compute slopes OF the ghost cells, enabling
    # symmetric reconstruction on both sides of the interface.

    flux_divergence_base = partial(
        flux_divergence_from_2ghost,
        gamma=cfg.gamma,
        order=cfg.ORDER,
        limiter=cfg.LIMITER,
        dx=dx, dy=dy, dz=dz,
        inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz
    )

    # Create MPI flux function that takes BOTH U_interior and U_with_ghost
    # This avoids closure capture of patch.data (which causes stale data in JIT)
    flux_fn = create_mpi_flux_fn(flux_divergence_base, my_patch, cfg)

    # Get slice for extracting data with ALL 2 ghost cells per side
    slice_2g = get_ghost_slices(my_patch, cfg)

    # Use MPI-specific time integrators that pass ghost data as argument
    # Choose between standard (fixed ghost cells) or sub-step (updated ghost cells) versions
    use_substep = getattr(cfg, 'SUBSTEP_HALO_EXCHANGE', False)

    if use_substep:
        # Sub-step versions: ghost cells updated between RK stages (more accurate at boundaries)
        mpi_time_solver_substep_map = {
            0: mpi_rk2_step_substep,
            1: mpi_muscl_hancock_step_substep,
            2: mpi_rk4_step_substep
        }
        time_step_substep_fn = mpi_time_solver_substep_map[cfg.TIME_SOLVER]
        time_step_fn = None  # Will use substep version in main loop
        if rank == 0:
            print("Using SUB-STEP halo exchange for RK stages (more accurate at boundaries)")
    else:
        # Standard versions: ghost cells fixed during RK stages (faster)
        mpi_time_solver_map = {0: mpi_rk2_step, 1: mpi_muscl_hancock_step, 2: mpi_rk4_step}
        time_step_fn = partial(mpi_time_solver_map[cfg.TIME_SOLVER], flux_fn=flux_fn)
        time_step_substep_fn = None
        if rank == 0:
            print("Using STANDARD time integration (ghost cells fixed during RK stages)")

    # === WARMUP (JIT COMPILATION) ===
    if rank == 0:
        print("="*60)
        print("WARMUP (JIT COMPILATION)")
        print("="*60)
        print("Compiling kernels...")

    # Extract interior and ghost data for warmup
    U_interior = my_patch.data[interior_slice]
    U_with_ghost = my_patch.data[slice_2g]  # All 2 ghost cells per side

    # Warmup time step - only for standard mode (substep mode has MPI calls, can't fully JIT)
    dt = 0.001
    if use_substep:
        # For substep mode, just warmup the single-stage compute function
        for _ in range(3):
            _ = _compute_k_stage(U_interior, U_with_ghost, flux_fn)
        jax.block_until_ready(_)
    else:
        # Standard mode: warmup full time step
        for _ in range(3):
            U_interior = time_step_fn(U_interior, U_with_ghost, dt)
        jax.block_until_ready(U_interior)

    if rank == 0:
        print("✓ Compilation complete")
        print()

    # === RESTORE INITIAL CONDITIONS AFTER WARMUP ===
    # The warmup was just for JIT compilation; now restore the original initial conditions
    if cfg.SPATIAL_DIMENSION == 1:
        ix_start = offset[0]
        ix_end = offset[0] + my_patch.shape[0]
        U_local = U_global[:, ix_start:ix_end]
    elif cfg.SPATIAL_DIMENSION == 2:
        ix_start, iy_start = offset
        ix_end = ix_start + my_patch.shape[0]
        iy_end = iy_start + my_patch.shape[1]
        U_local = U_global[:, ix_start:ix_end, iy_start:iy_end]
    else:
        ix_start, iy_start, iz_start = offset
        ix_end = ix_start + my_patch.shape[0]
        iy_end = iy_start + my_patch.shape[1]
        iz_end = iz_start + my_patch.shape[2]
        U_local = U_global[:, ix_start:ix_end, iy_start:iy_end, iz_start:iz_end]

    # Restore initial conditions to patch data
    my_patch.data = my_patch.data.at[interior_slice].set(U_local)

    # Fresh halo exchange with restored initial conditions
    update_ghosts_mpi(my_patch, cfg.BOUNDARY, comm)

    if rank == 0:
        print("✓ Initial conditions restored after warmup")
        print()

    # --- Set output frequency (same as single GPU version) ---
    # Compute number of snapshots based on time between snapshots and total simulation time
    n_snapshots = int(cfg.t_end / cfg.time_bet_snaps)
    snapshot_times = jnp.linspace(cfg.t_start, cfg.t_end, n_snapshots)

    snapshot_idx = 0

    # --- Output directory ---
    if rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)

    # --- Time loop variables ---
    t = cfg.t_start
    step = 0

    # Timing
    step_times = []
    comm_times = []

    if rank == 0:
        print("="*60)
        print("SIMULATION START")
        print("="*60)
        print(f"Time range: {cfg.t_start:.3f} → {cfg.t_end:.3f}")
        print(f"Snapshot times: {[float(st) for st in snapshot_times[:5]]}{'...' if len(snapshot_times) > 5 else ''}")
        print()

    # Start wallclock timer for total runtime
    wall_start = time.time()

    while t < cfg.t_end:
        # === HALO EXCHANGE (FIRST!) ===
        # CRITICAL: Update ghost cells BEFORE time integration!
        # This ensures fluxes are computed with fresh boundary data.
        # For shocks at domain boundaries, using stale ghost cells causes
        # the shock to see incorrect boundary states.
        t_comm_start = time.time()
        update_ghosts_mpi(my_patch, cfg.BOUNDARY, comm)
        t_comm = (time.time() - t_comm_start) * 1000

        # === COMPUTE TIMESTEP ===
        # Get the next snapshot time from the pre-computed array
        t_snapshot = snapshot_times[snapshot_idx] if snapshot_idx < len(snapshot_times) else cfg.t_end

        # Pass t_snapshot to ensure dt is adjusted to hit snapshot times exactly
        # Now uses fresh ghost data for CFL computation
        dt = compute_global_dt_mpi(my_patch, cfg.gamma, cfg.CFL, t, t_snapshot, comm)

        # Don't overshoot end time
        if t + dt > cfg.t_end:
            dt = cfg.t_end - t
            # If dt is now effectively zero, we've reached the end
            if dt < 1e-12:
                break

        # === TIME INTEGRATION ===
        t_compute_start = time.time()

        # Extract interior AND ghost data
        U_interior = my_patch.data[interior_slice]
        U_with_ghost = my_patch.data[slice_2g]

        if use_substep:
            # Sub-step mode: integrator handles its own halo exchanges between stages
            # This is more accurate for shocks at boundaries but slower due to extra MPI calls
            U_interior = time_step_substep_fn(
                my_patch, dt, flux_fn, cfg.BOUNDARY, comm, interior_slice, slice_2g
            )
            # Update interior in patch data (substep fn modified patch.data internally for stages)
            my_patch.data = my_patch.data.at[interior_slice].set(U_interior)
        else:
            # Standard JIT mode: ghost cells fixed during RK stages
            # Time step - pass BOTH interior and ghost data
            # Ghost data is a JAX array argument, not captured from closure,
            # so it will be the CURRENT data, not stale data from JIT compile time
            U_interior = time_step_fn(U_interior, U_with_ghost, dt)

            # Update interior in patch data
            my_patch.data = my_patch.data.at[interior_slice].set(U_interior)

        # Note: block_until_ready is needed for accurate timing AND because
        # the next iteration's halo exchange uses np.asarray() which requires
        # the data to be ready. Without GPU-aware MPI, this sync is unavoidable.
        jax.block_until_ready(my_patch.data)
        t_compute = (time.time() - t_compute_start) * 1000

        # Update time
        t += dt
        step += 1

        # Store timings (after warmup)
        if step > 5:
            step_times.append(t_compute)
            comm_times.append(t_comm)

        # === DIAGNOSTICS ===
        if step % cfg.output_cadence == 0 or t >= cfg.t_end:
            # Compute local diagnostics
            dt_diag, min_rho, min_p, has_nan, has_inf = compute_dt_and_diagnostics(
                U_interior, cfg.gamma, cfg.CFL, t, t_snapshot=t_snapshot, dx=dx, dy=dy, dz=dz
            )

            # Global reduction for diagnostics
            min_rho_global = comm.allreduce(float(min_rho), op=MPI.MIN)
            min_p_global = comm.allreduce(float(min_p), op=MPI.MIN)
            has_nan_global = comm.allreduce(int(has_nan), op=MPI.LOR)
            has_inf_global = comm.allreduce(int(has_inf), op=MPI.LOR)

            # Validate (only rank 0 prints)
            if rank == 0:
                validate_from_diagnostics(
                    min_rho_global, min_p_global,
                    bool(has_nan_global), bool(has_inf_global),
                    step=step, time=t
                )

                # Print progress
                avg_compute = sum(step_times[-100:]) / len(step_times[-100:]) if step_times else 0
                avg_comm = sum(comm_times[-100:]) / len(comm_times[-100:]) if comm_times else 0
                comm_pct = 100 * avg_comm / (avg_compute + avg_comm) if avg_compute + avg_comm > 0 else 0

                print(f"Step {step:5d}, t={t:.5f}, dt={dt:.5e}, "
                      f"compute={avg_compute:.2f}ms, comm={avg_comm:.2f}ms ({comm_pct:.1f}%)")

            # Check for NaN/Inf (inside diagnostics block since these vars are only set here)
            if has_nan_global or has_inf_global:
                if rank == 0:
                    print("ERROR: Simulation became unstable!")
                break

        # === WRITE SNAPSHOT IF TIME MATCHES ===
        # Save snapshot if the current time matches the snapshot time
        if t >= t_snapshot:
            if rank == 0:
                print(f"\n{'─'*60}")
                print(f"💾 Saving snapshot #{snapshot_idx} at t = {t:.5f}")
                print(f"{'─'*60}")

            # Gather all fields from all patches to rank 0
            n_fields = my_patch.data.shape[0]
            U_interior = my_patch.data[interior_slice]

            gathered_fields = []
            for field_idx in range(n_fields):
                local_field = jnp.array(U_interior[field_idx])
                all_patches = comm.gather(jnp.asarray(local_field), root=0)
                if rank == 0:
                    gathered_fields.append(all_patches)

            if rank == 0:
                U_global_output = reconstruct_global_array(
                    gathered_fields, patches, global_shape, cfg.SPATIAL_DIMENSION
                )
                # Debug: print data range to verify evolution
                rho_min, rho_max = float(jnp.min(U_global_output[0])), float(jnp.max(U_global_output[0]))
                print(f"   Density range: [{rho_min:.6f}, {rho_max:.6f}]")

                write_snapshot(U_global_output, cfg, t, snapshot_idx)
                print(f"✓ Snapshot saved successfully\n")

            # Synchronize all ranks after snapshot
            comm.Barrier()

            # Move to next snapshot
            snapshot_idx += 1

            # Ensure we don't overshoot the snapshot time
            if snapshot_idx < len(snapshot_times):
                t_snapshot = snapshot_times[snapshot_idx]
            else:
                t_snapshot = cfg.t_end  # Once all snapshots are saved, set to end time

    # === FINALIZE ===
    wall_end = time.time()
    wall_elapsed = wall_end - wall_start

    if rank == 0:
        print("\n" + "="*60)
        print("SIMULATION COMPLETED")
        print("="*60)

        if step_times:
            avg_compute = sum(step_times) / len(step_times)
            avg_comm = sum(comm_times) / len(comm_times)
            total = avg_compute + avg_comm
            print(f"✓ Total steps: {step}")
            print(f"✓ Final time: {t:.5f}")
            print(f"✓ Wallclock time: {wall_elapsed:.2f} s ({wall_elapsed/60:.2f} min)")
            print()
            print(f"Average timings:")
            print(f"  Computation: {avg_compute:.2f} ms ({100*avg_compute/total:.1f}%)")
            print(f"  Communication: {avg_comm:.2f} ms ({100*avg_comm/total:.1f}%)")
            print(f"  Total per step: {total:.2f} ms")
        print()

    # Synchronize all ranks before plotting
    if rank == 0:
        print("Synchronizing ranks before plotting...")
    comm.Barrier()
    if rank == 0:
        print("All ranks synchronized.")

    # === PLOTTING ===
    if rank == 0 and cfg.PLOT_AFTER_SIM:
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

    # Final synchronization before exit
    comm.Barrier()

    if rank == 0:
        print("="*60)
        print("ALL DONE!")
        print("="*60)

    # MPI cleanup (mpi4py usually handles this automatically, but explicit is safer)
    MPI.Finalize()


if __name__ == "__main__":
    main()
