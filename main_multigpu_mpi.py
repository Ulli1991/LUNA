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

import jax
import jax.numpy as jnp

# JAX configuration
jax.config.update("jax_enable_x64", False)

from config import Config
from grid.decomposition import create_uniform_decomposition
from grid.domain import Domain
from grid.neighbors import setup_neighbor_connectivity, print_connectivity
from ic_generation.ic import initial_conditions
from time_integration.integrators import rk2_step, rk4_step, muscl_hancock_step
from time_integration.cfl import compute_dt_and_diagnostics
from hydro.physics_checks import validate_from_diagnostics
from hydro.divergence import flux_divergence
from parallel.halo_exchange_mpi import (
    update_ghosts_mpi,
    compute_global_dt_mpi,
    gather_field_for_output_mpi
)


def main():
    # === MPI INITIALIZATION ===
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    # === JAX DEVICE SETUP ===
    # Each MPI rank uses one GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank % 8)  # For NVIDIA
    os.environ['ROCR_VISIBLE_DEVICES'] = str(rank % 8)  # For AMD

    # Force JAX to see only this GPU
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

    if rank == 0:
        print("="*60)
        print("CONFIGURATION")
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
        print("INITIAL CONDITIONS")
        print("="*60)

    # Generate initial conditions for the GLOBAL domain
    # TODO: In production, each rank should only generate its local region
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

    # Create specialized flux function for this patch
    # NOTE: For uniform grid, all patches have same dx/dy/dz
    # For AMR, each patch would have different flux_fn
    flux_fn = partial(
        flux_divergence,
        gamma=cfg.gamma,
        order=cfg.ORDER,
        limiter=cfg.LIMITER,
        boundary=1,  # Use ghost cells (already filled by halo exchange)
        dx=dx, dy=dy, dz=dz,
        inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz
    )

    # Time integration function
    time_solver_map = {0: rk2_step, 1: muscl_hancock_step, 2: rk4_step}
    time_step_fn = partial(time_solver_map[cfg.TIME_SOLVER], flux_fn=flux_fn)

    # === WARMUP (JIT COMPILATION) ===
    if rank == 0:
        print("="*60)
        print("WARMUP (JIT COMPILATION)")
        print("="*60)
        print("Compiling kernels...")

    # Extract interior for warmup
    U_interior = my_patch.data[interior_slice]

    # Warmup time step
    dt = 0.001
    for _ in range(3):
        U_interior = time_step_fn(U_interior, dt)

    jax.block_until_ready(U_interior)

    if rank == 0:
        print("✓ Compilation complete")
        print()

    # === MAIN SIMULATION LOOP ===
    if rank == 0:
        print("="*60)
        print("MAIN SIMULATION")
        print("="*60)

    t = cfg.t_start
    step = 0
    output_step = 0

    # Timing
    step_times = []
    comm_times = []

    while t < cfg.t_end:
        # === COMPUTE TIMESTEP ===
        dt = compute_global_dt_mpi(my_patch, cfg.gamma, cfg.CFL, comm)

        # Don't overshoot end time
        if t + dt > cfg.t_end:
            dt = cfg.t_end - t

        # === TIME INTEGRATION ===
        t_compute_start = time.time()

        # Extract interior
        U_interior = my_patch.data[interior_slice]

        # Time step on interior only
        U_interior_new = time_step_fn(U_interior, dt)

        # Update interior in patch data
        my_patch.data = my_patch.data.at[interior_slice].set(U_interior_new)

        jax.block_until_ready(my_patch.data)
        t_compute = (time.time() - t_compute_start) * 1000

        # === HALO EXCHANGE ===
        t_comm_start = time.time()

        update_ghosts_mpi(my_patch, cfg.BOUNDARY, comm)

        t_comm = (time.time() - t_comm_start) * 1000

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
            _, min_rho, min_p, has_nan, has_inf = compute_dt_and_diagnostics(
                U_interior, cfg.gamma, cfg.CFL, t, cfg.t_end, dx, dy, dz
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

                print(f"Step {step:5d}, t={t:.4f}, dt={dt:.2e}, "
                      f"compute={avg_compute:.2f}ms, comm={avg_comm:.2f}ms ({comm_pct:.1f}%)")

            # TODO: Output to file (gather to rank 0 and write)

        # Check for NaN/Inf
        if has_nan_global or has_inf_global:
            if rank == 0:
                print("ERROR: Simulation became unstable!")
            break

    # === FINALIZE ===
    if rank == 0:
        print()
        print("="*60)
        print("SIMULATION COMPLETE")
        print("="*60)

        if step_times:
            avg_compute = sum(step_times) / len(step_times)
            avg_comm = sum(comm_times) / len(comm_times)
            total = avg_compute + avg_comm
            print(f"Average timings:")
            print(f"  Computation: {avg_compute:.2f} ms ({100*avg_compute/total:.1f}%)")
            print(f"  Communication: {avg_comm:.2f} ms ({100*avg_comm/total:.1f}%)")
            print(f"  Total per step: {total:.2f} ms")
            print()
            print(f"Total steps: {step}")
            print(f"Final time: {t:.4f}")


if __name__ == "__main__":
    main()
