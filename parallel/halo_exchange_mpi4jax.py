"""
GPU-native MPI halo exchange using mpi4jax.

mpi4jax provides JAX-compatible MPI operations that work directly on GPU arrays,
eliminating the D2H/H2D transfers that are the main bottleneck in standard MPI.

Key benefits:
- No GPU->CPU->GPU data movement
- Operations can be JIT-compiled with JAX
- Automatic differentiation support (if needed)

NOTE: Using separate send/recv instead of sendrecv for ROCm compatibility.
"""

import jax
import jax.numpy as jnp
from functools import partial
from mpi4py import MPI
import mpi4jax


def create_halo_exchange_fn(patch, cfg, comm):
    """
    Create a JIT-compiled halo exchange function for this patch.

    For X-decomposition, this exchanges left/right ghost cells with neighbors.
    Y and Z boundaries are always domain boundaries (handled by BCs in flux fn).

    Uses separate send/recv operations for ROCm compatibility (sendrecv may not
    have MLIR lowering for ROCm platform).

    Returns
    -------
    exchange_fn : callable
        Function(U_interior) -> (left_ghost, right_ghost)
        Takes interior data, returns ghost cells from neighbors (or dummy for domain BCs)
    """
    ndim = cfg.SPATIAL_DIMENSION
    ng = patch.n_ghost
    rank = comm.Get_rank()

    # Get neighbor ranks for X direction
    left_neighbors = patch.neighbors.get('left', [])
    right_neighbors = patch.neighbors.get('right', [])

    has_left_neighbor = len(left_neighbors) > 0
    has_right_neighbor = len(right_neighbors) > 0

    left_neighbor_rank = left_neighbors[0].rank if has_left_neighbor else -1
    right_neighbor_rank = right_neighbors[0].rank if has_right_neighbor else -1

    # Determine if we're at domain boundaries
    is_left_domain = patch.is_boundary('left')
    is_right_domain = patch.is_boundary('right')

    if ndim == 3:
        def exchange_fn(U_interior):
            """
            Exchange X ghost cells with neighbors using separate send/recv.

            U_interior: (nvar, nx, ny, nz)
            Returns: (left_ghost, right_ghost) each of shape (nvar, ng, ny, nz)

            Pattern for 2 ranks (rank 0 left, rank 1 right):
            - Rank 0: sends right boundary to rank 1, receives from rank 1
            - Rank 1: sends left boundary to rank 0, receives from rank 0

            To avoid deadlock, we use the pattern:
            - Even ranks: send first, then recv
            - Odd ranks: recv first, then send

            Note: mpi4jax>=0.8.0 manages tokens automatically, no explicit tokens needed.
            """
            nvar, nx, ny, nz = U_interior.shape

            # Data to send: boundary slices of interior
            send_to_left = U_interior[:, :ng, :, :]   # Send to left neighbor
            send_to_right = U_interior[:, -ng:, :, :]  # Send to right neighbor

            # Handle left ghost cells
            if is_left_domain:
                left_ghost = jnp.zeros((nvar, ng, ny, nz), dtype=U_interior.dtype)
            else:
                # Exchange with left neighbor
                recv_buf = jnp.zeros((nvar, ng, ny, nz), dtype=U_interior.dtype)

                # Use rank parity to determine send/recv order (avoid deadlock)
                if rank % 2 == 0:
                    # Even rank: send first, then recv
                    mpi4jax.send(send_to_left, dest=left_neighbor_rank, tag=0, comm=comm)
                    left_ghost = mpi4jax.recv(recv_buf, source=left_neighbor_rank, tag=0, comm=comm)
                else:
                    # Odd rank: recv first, then send
                    left_ghost = mpi4jax.recv(recv_buf, source=left_neighbor_rank, tag=0, comm=comm)
                    mpi4jax.send(send_to_left, dest=left_neighbor_rank, tag=0, comm=comm)

            # Handle right ghost cells
            if is_right_domain:
                right_ghost = jnp.zeros((nvar, ng, ny, nz), dtype=U_interior.dtype)
            else:
                # Exchange with right neighbor
                recv_buf = jnp.zeros((nvar, ng, ny, nz), dtype=U_interior.dtype)

                if rank % 2 == 0:
                    # Even rank: send first, then recv
                    mpi4jax.send(send_to_right, dest=right_neighbor_rank, tag=1, comm=comm)
                    right_ghost = mpi4jax.recv(recv_buf, source=right_neighbor_rank, tag=1, comm=comm)
                else:
                    # Odd rank: recv first, then send
                    right_ghost = mpi4jax.recv(recv_buf, source=right_neighbor_rank, tag=1, comm=comm)
                    mpi4jax.send(send_to_right, dest=right_neighbor_rank, tag=1, comm=comm)

            return left_ghost, right_ghost

        return exchange_fn

    elif ndim == 2:
        def exchange_fn(U_interior):
            nvar, nx, ny = U_interior.shape
            send_to_left = U_interior[:, :ng, :]
            send_to_right = U_interior[:, -ng:, :]

            if is_left_domain:
                left_ghost = jnp.zeros((nvar, ng, ny), dtype=U_interior.dtype)
            else:
                recv_buf = jnp.zeros((nvar, ng, ny), dtype=U_interior.dtype)
                if rank % 2 == 0:
                    mpi4jax.send(send_to_left, dest=left_neighbor_rank, tag=0, comm=comm)
                    left_ghost = mpi4jax.recv(recv_buf, source=left_neighbor_rank, tag=0, comm=comm)
                else:
                    left_ghost = mpi4jax.recv(recv_buf, source=left_neighbor_rank, tag=0, comm=comm)
                    mpi4jax.send(send_to_left, dest=left_neighbor_rank, tag=0, comm=comm)

            if is_right_domain:
                right_ghost = jnp.zeros((nvar, ng, ny), dtype=U_interior.dtype)
            else:
                recv_buf = jnp.zeros((nvar, ng, ny), dtype=U_interior.dtype)
                if rank % 2 == 0:
                    mpi4jax.send(send_to_right, dest=right_neighbor_rank, tag=1, comm=comm)
                    right_ghost = mpi4jax.recv(recv_buf, source=right_neighbor_rank, tag=1, comm=comm)
                else:
                    right_ghost = mpi4jax.recv(recv_buf, source=right_neighbor_rank, tag=1, comm=comm)
                    mpi4jax.send(send_to_right, dest=right_neighbor_rank, tag=1, comm=comm)

            return left_ghost, right_ghost

        return exchange_fn

    else:  # 1D
        def exchange_fn(U_interior):
            nvar, nx = U_interior.shape
            send_to_left = U_interior[:, :ng]
            send_to_right = U_interior[:, -ng:]

            if is_left_domain:
                left_ghost = jnp.zeros((nvar, ng), dtype=U_interior.dtype)
            else:
                recv_buf = jnp.zeros((nvar, ng), dtype=U_interior.dtype)
                if rank % 2 == 0:
                    mpi4jax.send(send_to_left, dest=left_neighbor_rank, tag=0, comm=comm)
                    left_ghost = mpi4jax.recv(recv_buf, source=left_neighbor_rank, tag=0, comm=comm)
                else:
                    left_ghost = mpi4jax.recv(recv_buf, source=left_neighbor_rank, tag=0, comm=comm)
                    mpi4jax.send(send_to_left, dest=left_neighbor_rank, tag=0, comm=comm)

            if is_right_domain:
                right_ghost = jnp.zeros((nvar, ng), dtype=U_interior.dtype)
            else:
                recv_buf = jnp.zeros((nvar, ng), dtype=U_interior.dtype)
                if rank % 2 == 0:
                    mpi4jax.send(send_to_right, dest=right_neighbor_rank, tag=1, comm=comm)
                    right_ghost = mpi4jax.recv(recv_buf, source=right_neighbor_rank, tag=1, comm=comm)
                else:
                    right_ghost = mpi4jax.recv(recv_buf, source=right_neighbor_rank, tag=1, comm=comm)
                    mpi4jax.send(send_to_right, dest=right_neighbor_rank, tag=1, comm=comm)

            return left_ghost, right_ghost

        return exchange_fn


def allreduce_min(value, comm):
    """
    Global minimum reduction using mpi4jax.

    Works directly on JAX arrays, JIT-compatible.
    """
    result, _ = mpi4jax.allreduce(value, op=MPI.MIN, comm=comm)
    return result


def compute_global_dt_mpi4jax(U_interior, gamma, cfl, dx, dy, dz, comm):
    """
    Compute global CFL timestep using mpi4jax allreduce.

    This is JIT-compatible and works directly on GPU arrays.
    """
    from time_integration.cfl import compute_dt_local

    # Compute local dt
    dt_local = compute_dt_local(U_interior, gamma, cfl, dx, dy, dz)

    # Global reduction
    dt_global, _ = mpi4jax.allreduce(dt_local, op=MPI.MIN, comm=comm)

    return dt_global
