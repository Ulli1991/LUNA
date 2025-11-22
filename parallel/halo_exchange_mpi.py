"""
MPI-based halo exchange for multi-GPU execution.

This module provides explicit MPI-based ghost cell communication.
Each MPI rank owns one patch and explicitly sends/receives with neighbors.

Design:
- Simple and explicit (easy to debug)
- Standard MPI (mpi4py)
- Works for single-node and multi-node
- Can be optimized to hybrid MPI+pmap later
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, Optional
from mpi4py import MPI
from grid.patch import GridPatch
from grid.domain import Domain


def exchange_halos_mpi(
    patch: GridPatch,
    comm: MPI.Comm,
    use_non_blocking: bool = True
) -> None:
    """
    Exchange ghost cells with all neighboring patches using MPI.

    Each rank owns one patch. This function sends boundary data to neighbors
    and receives their boundary data into ghost cells.

    Parameters
    ----------
    patch : GridPatch
        The local patch owned by this rank
    comm : MPI.Comm
        MPI communicator (usually MPI.COMM_WORLD)
    use_non_blocking : bool
        If True, use non-blocking MPI (Isend/Irecv)
        If False, use blocking MPI (Send/Recv) - simpler but can deadlock

    Notes
    -----
    Communication pattern:
    1. Post all receives first (standard MPI pattern to avoid deadlock)
    2. Post all sends
    3. Wait for all to complete

    Each direction (left, right, top, bottom, back, front) is handled
    independently. Tag scheme ensures messages don't collide.
    """
    if patch.data is None:
        raise ValueError("Patch data not allocated")

    if use_non_blocking:
        _exchange_halos_nonblocking(patch, comm)
    else:
        _exchange_halos_blocking(patch, comm)


def _exchange_halos_blocking(patch: GridPatch, comm: MPI.Comm) -> None:
    """
    Blocking MPI halo exchange (simpler but can deadlock).

    Use this for initial testing/debugging only.
    For production, use non-blocking version.
    """
    rank = comm.Get_rank()

    for direction, neighbors in patch.neighbors.items():
        if not neighbors:
            continue

        neighbor = neighbors[0]  # Single neighbor for fixed grid

        # Get data to send
        send_slice = patch.get_ghost_slice(direction, send=True)
        send_data = np.asarray(patch.data[send_slice])

        # Receive buffer
        recv_data = np.empty_like(send_data)

        # Get opposite direction for receiving
        opposite = _get_opposite_direction(direction)

        # Create unique tags
        send_tag = _create_tag(rank, neighbor.rank, direction)
        recv_tag = _create_tag(neighbor.rank, rank, opposite)

        # WARNING: This can deadlock if all ranks send first!
        # For production, use non-blocking version below.

        # Send to neighbor
        comm.Send(send_data, dest=neighbor.rank, tag=send_tag)

        # Receive from neighbor
        comm.Recv(recv_data, source=neighbor.rank, tag=recv_tag)

        # Copy into ghost cells
        # If I have a neighbor to my RIGHT, I receive INTO my RIGHT ghost cells
        recv_slice = patch.get_ghost_slice(direction, send=False)
        patch.data = patch.data.at[recv_slice].set(jnp.array(recv_data))


def _exchange_halos_nonblocking(patch: GridPatch, comm: MPI.Comm) -> None:
    """
    Non-blocking MPI halo exchange (safe, won't deadlock).

    Standard pattern:
    1. Post all Irecv first
    2. Post all Isend
    3. Wait for all to complete
    """
    rank = comm.Get_rank()

    send_requests = []
    recv_requests = []
    recv_buffers = {}

    # Step 1: Post all receives first (standard MPI pattern)
    for direction, neighbors in patch.neighbors.items():
        if not neighbors:
            continue

        neighbor = neighbors[0]
        opposite = _get_opposite_direction(direction)

        # Get receive slice and allocate buffer
        # If I have a neighbor to my RIGHT, I receive INTO my RIGHT ghost cells
        # (opposite is only used for tag matching, not for the slice)
        recv_slice = patch.get_ghost_slice(direction, send=False)
        recv_shape = patch.data[recv_slice].shape
        recv_buffer = np.empty(recv_shape, dtype=np.float32)

        # Create tag
        recv_tag = _create_tag(neighbor.rank, rank, opposite)

        # Post receive
        req = comm.Irecv(recv_buffer, source=neighbor.rank, tag=recv_tag)
        recv_requests.append(req)

        # Store buffer and slice for later copying
        recv_buffers[direction] = (recv_buffer, recv_slice)

    # Step 2: Post all sends
    for direction, neighbors in patch.neighbors.items():
        if not neighbors:
            continue

        neighbor = neighbors[0]

        # Get data to send
        send_slice = patch.get_ghost_slice(direction, send=True)
        send_data = np.asarray(patch.data[send_slice])

        # Create tag
        send_tag = _create_tag(rank, neighbor.rank, direction)

        # Post send
        req = comm.Isend(send_data, dest=neighbor.rank, tag=send_tag)
        send_requests.append(req)

    # Step 3: Wait for all receives to complete
    MPI.Request.Waitall(recv_requests)

    # Step 4: Copy received data into ghost cells
    for direction, (recv_buffer, recv_slice) in recv_buffers.items():
        patch.data = patch.data.at[recv_slice].set(jnp.array(recv_buffer))

    # Step 5: Wait for all sends to complete (cleanup)
    MPI.Request.Waitall(send_requests)


def apply_boundary_conditions_mpi(
    patch: GridPatch,
    boundary_type: int,
    rank: int
) -> None:
    """
    Apply boundary conditions at domain edges.

    Only applies boundary conditions for faces that are actual domain boundaries
    (not inter-patch boundaries handled by halo exchange).

    Parameters
    ----------
    patch : GridPatch
        The local patch
    boundary_type : int
        Boundary condition type:
        0 = Periodic (handled by neighbor connectivity)
        1 = Zero-gradient
        2 = Reflective
    rank : int
        MPI rank (for debugging messages)

    Notes
    -----
    This should be called AFTER halo exchange.
    Only patches at domain edges will actually do anything here.
    """
    from parallel.halo_exchange import apply_boundary_conditions

    # Reuse the existing boundary condition logic
    apply_boundary_conditions(patch, boundary_type)


def update_ghosts_mpi(
    patch: GridPatch,
    boundary_type: int,
    comm: MPI.Comm
) -> None:
    """
    Convenience function: halo exchange + boundary conditions.

    IMPORTANT: Order matters for correctness at domain boundaries!

    1. First exchange halos (fills inter-patch ghost cells from neighbors)
    2. Then apply boundary conditions (fills domain edge ghost cells)

    This order ensures:
    - Interior patches get fresh neighbor data first
    - Domain boundary ghost cells are filled AFTER exchange (won't be overwritten)
    - Corner cells in 2D/3D are properly handled (face exchanges fill corners,
      then domain BC overwrites only the actual boundary corners)

    Parameters
    ----------
    patch : GridPatch
        Local patch owned by this rank
    boundary_type : int
        Boundary condition type for domain edges
    comm : MPI.Comm
        MPI communicator

    Examples
    --------
    >>> from mpi4py import MPI
    >>> comm = MPI.COMM_WORLD
    >>> rank = comm.Get_rank()
    >>> my_patch = patches[rank]
    >>>
    >>> # After each timestep:
    >>> update_ghosts_mpi(my_patch, boundary_type=1, comm=comm)
    """
    # Step 1: Exchange halos between patches FIRST
    # This fills ghost cells from neighboring patches
    exchange_halos_mpi(patch, comm, use_non_blocking=True)

    # Step 2: Apply boundary conditions at domain edges AFTER
    # This fills domain edge ghost cells (and won't be overwritten by halo exchange)
    # Only patches at domain boundaries will actually do anything here
    rank = comm.Get_rank()
    apply_boundary_conditions_mpi(patch, boundary_type, rank)


def compute_global_dt_mpi(
    patch: GridPatch,
    gamma: float,
    cfl: float,
    t: float,
    t_snapshot: float,
    comm: MPI.Comm
) -> float:
    """
    Compute global CFL timestep across all patches.

    Each rank computes local dt for its patch, then global min via MPI.

    Parameters
    ----------
    patch : GridPatch
        Local patch owned by this rank
    gamma : float
        Ratio of specific heats
    cfl : float
        CFL number
    t : float
        Current simulation time
    t_snapshot : float
        Next snapshot time (used to adjust dt to hit snapshots exactly)
    comm : MPI.Comm
        MPI communicator

    Returns
    -------
    float
        Global timestep (same on all ranks)

    Examples
    --------
    >>> dt_global = compute_global_dt_mpi(my_patch, gamma=1.4, cfl=0.5, t=0.1, t_snapshot=0.2, comm=comm)
    >>> # All ranks get the same dt_global
    """
    from time_integration.cfl import compute_dt_and_diagnostics

    # Extract interior data (without ghost cells)
    interior_slice = patch.get_interior_slice()
    U_interior = patch.data[interior_slice]

    # Compute local dt with proper t_snapshot to hit output times exactly
    dt_local, _, _, _, _ = compute_dt_and_diagnostics(
        U_interior, gamma, cfl,
        t=t, t_snapshot=t_snapshot,
        dx=patch.dx, dy=patch.dy, dz=patch.dz
    )

    # Global reduction
    dt_global = comm.allreduce(float(dt_local), op=MPI.MIN)

    return dt_global


def gather_field_for_output_mpi(
    patch: GridPatch,
    field_idx: int,
    comm: MPI.Comm,
    root: int = 0
) -> Optional[np.ndarray]:
    """
    Gather a field from all patches to root rank for I/O.

    Parameters
    ----------
    patch : GridPatch
        Local patch
    field_idx : int
        Which field to gather (0=density, 1=x-momentum, etc.)
    comm : MPI.Comm
        MPI communicator
    root : int
        Rank to gather to

    Returns
    -------
    np.ndarray or None
        Full field on root rank, None on other ranks

    Examples
    --------
    >>> # Gather density field to rank 0 for output
    >>> density_global = gather_field_for_output_mpi(my_patch, field_idx=0, comm=comm)
    >>> if comm.Get_rank() == 0:
    >>>     # density_global is the full global array
    >>>     save_to_hdf5(density_global, filename)
    """
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    # Extract interior (no ghost cells)
    interior_slice = patch.get_interior_slice()
    local_field = np.asarray(patch.data[interior_slice][field_idx])

    if rank == root:
        # Allocate global array
        # Need to know global shape - for now assume uniform patches
        global_field = np.empty((n_ranks,) + local_field.shape, dtype=np.float32)
    else:
        global_field = None

    # Gather to root
    comm.Gather(local_field, global_field, root=root)

    if rank == root:
        # Reshape from (n_ranks, nx, ny, nz) to proper global shape
        # TODO: Handle non-uniform decompositions properly
        return global_field
    else:
        return None


def _get_opposite_direction(direction: str) -> str:
    """Get opposite direction for halo exchange."""
    opposites = {
        'left': 'right', 'right': 'left',
        'bottom': 'top', 'top': 'bottom',
        'back': 'front', 'front': 'back',
    }
    return opposites[direction]


def _create_tag(src_rank: int, dest_rank: int, direction: str) -> int:
    """
    Create unique MPI tag for this communication.

    Uses hash to ensure different direction/rank combinations get unique tags.

    Parameters
    ----------
    src_rank : int
        Source rank
    dest_rank : int
        Destination rank
    direction : str
        Direction of communication

    Returns
    -------
    int
        Unique tag in range [0, 2^31)
    """
    # Simple hash that's deterministic
    direction_map = {
        'left': 0, 'right': 1,
        'bottom': 2, 'top': 3,
        'back': 4, 'front': 5
    }
    dir_id = direction_map.get(direction, 0)

    # Create tag: ensure it's positive and within MPI tag limits
    tag = (src_rank * 1000 + dest_rank * 10 + dir_id) % (2**31)
    return tag
