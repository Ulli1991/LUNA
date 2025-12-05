"""
Halo exchange for multi-process JAX sharding.

This module handles ghost cell communication between neighboring processes
in a multi-GPU/multi-node setup.
"""
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P


def pad_with_halo_exchange(U, boundary, mesh, sharding):
    """
    Pad array with ghost cells, using halo exchange for inter-shard boundaries
    and applying physical boundary conditions at domain edges.

    Parameters
    ----------
    U : array
        Shape (Nvar, Nx) for 1D, (Nvar, Nx, Ny) for 2D, (Nvar, Nx, Ny, Nz) for 3D.
    boundary : int
        0 = periodic, 1 = zero-gradient, 2 = reflective (for physical boundaries)
    mesh : jax.sharding.Mesh
        The device mesh
    sharding : jax.sharding.NamedSharding
        The sharding specification

    Returns
    -------
    U_pad : array
        Padded array with 1 ghost cell per side in each spatial dimension.
    """

    # Check if we're in multi-process mode
    if jax.process_count() == 1:
        # Single process - use original padding
        from hydro.padding import pad
        return pad(U, boundary)

    ndim = U.ndim - 1  # number of spatial dimensions

    # Get mesh dimensions
    mesh_shape = mesh.devices.shape

    if ndim == 1:
        return pad_1d_multiprocess(U, boundary, mesh, mesh_shape[0])
    elif ndim == 2:
        return pad_2d_multiprocess(U, boundary, mesh, mesh_shape)
    elif ndim == 3:
        return pad_3d_multiprocess(U, boundary, mesh, mesh_shape)
    else:
        raise ValueError(f"Unsupported ndim: {ndim}")


def pad_2d_multiprocess(U, boundary, mesh, mesh_shape):
    """
    2D halo exchange for a 2D process grid using lax.all_gather.

    IMPORTANT: This function is called inside JIT, so we use lax.all_gather
    which is JIT-compatible, instead of multihost_utils.process_allgather.

    For a 2×2 grid:
    Process layout:
      [0, 1]
      [2, 3]

    Each process needs to exchange boundaries with neighbors.
    """
    Nvar, Nx_local, Ny_local = U.shape
    nx_proc, ny_proc = mesh_shape  # e.g., (2, 2) for 4 processes

    # Get this process's position in the mesh - use lax.axis_index
    # We need the mesh to have a named axis for this
    # For now, compute it from the global process index at runtime
    process_idx = lax.axis_index('processes')
    ix = process_idx // ny_proc  # x position in process grid
    iy = process_idx % ny_proc   # y position in process grid

    # ========================================================================
    # STEP 1: X-direction halo exchange
    # ========================================================================

    # Determine left and right neighbors in x-direction
    has_left_neighbor = (ix > 0)
    has_right_neighbor = (ix < nx_proc - 1)

    left_neighbor = (ix - 1) * ny_proc + iy if has_left_neighbor else -1
    right_neighbor = (ix + 1) * ny_proc + iy if has_right_neighbor else -1

    # Extract boundary slices for sending
    left_boundary = U[:, 0:1, :]   # Leftmost column
    right_boundary = U[:, -1:, :]  # Rightmost column

    # Create axis_name for ppermute (we'll use process index directly)
    # ppermute needs pairs of (source, dest)

    # Receive left ghost (from left neighbor's right boundary)
    if has_left_neighbor:
        # We need to send OUR data and receive THEIR data
        # Send right_boundary of left neighbor TO us
        # This is a collective operation across all processes
        from jax.experimental import multihost_utils

        # Create a full array of boundaries from all processes
        all_right_boundaries = multihost_utils.process_allgather(right_boundary, tiled=False)
        # Extract the one from our left neighbor
        left_ghost_received = all_right_boundaries[left_neighbor]
    else:
        # Physical left boundary - apply boundary condition
        left_ghost_received = apply_boundary_x_left(U, boundary)

    # Receive right ghost (from right neighbor's left boundary)
    if has_right_neighbor:
        all_left_boundaries = multihost_utils.process_allgather(left_boundary, tiled=False)
        right_ghost_received = all_left_boundaries[right_neighbor]
    else:
        # Physical right boundary - apply boundary condition
        right_ghost_received = apply_boundary_x_right(U, boundary)

    # Concatenate in x-direction: [left_ghost, U, right_ghost]
    U_x = jnp.concatenate([left_ghost_received, U, right_ghost_received], axis=1)

    # ========================================================================
    # STEP 2: Y-direction halo exchange (on the x-padded array)
    # ========================================================================

    # Determine top and bottom neighbors in y-direction
    has_bottom_neighbor = (iy > 0)
    has_top_neighbor = (iy < ny_proc - 1)

    bottom_neighbor = ix * ny_proc + (iy - 1) if has_bottom_neighbor else -1
    top_neighbor = ix * ny_proc + (iy + 1) if has_top_neighbor else -1

    # Extract boundary slices for sending (from x-padded array)
    bottom_boundary = U_x[:, :, 0:1]   # Bottom row
    top_boundary = U_x[:, :, -1:]      # Top row

    # Receive bottom ghost (from bottom neighbor's top boundary)
    if has_bottom_neighbor:
        all_top_boundaries = multihost_utils.process_allgather(top_boundary, tiled=False)
        bottom_ghost_received = all_top_boundaries[bottom_neighbor]
    else:
        # Physical bottom boundary - apply boundary condition
        bottom_ghost_received = apply_boundary_y_bottom(U_x, boundary)

    # Receive top ghost (from top neighbor's bottom boundary)
    if has_top_neighbor:
        all_bottom_boundaries = multihost_utils.process_allgather(bottom_boundary, tiled=False)
        top_ghost_received = all_bottom_boundaries[top_neighbor]
    else:
        # Physical top boundary - apply boundary condition
        top_ghost_received = apply_boundary_y_top(U_x, boundary)

    # Concatenate in y-direction: [bottom_ghost, U_x, top_ghost]
    U_pad = jnp.concatenate([bottom_ghost_received, U_x, top_ghost_received], axis=2)

    return U_pad


def apply_boundary_x_left(U, boundary):
    """Apply physical boundary condition on left edge (x-direction)."""
    if boundary == 0:  # Periodic
        return U[:, -1:, :]
    elif boundary == 1:  # Zero-gradient
        return U[:, 0:1, :]
    elif boundary == 2:  # Reflective
        ghost = U[:, 0:1, :]
        ghost = ghost.at[1].set(-ghost[1])  # Negate rho*u
        return ghost


def apply_boundary_x_right(U, boundary):
    """Apply physical boundary condition on right edge (x-direction)."""
    if boundary == 0:  # Periodic
        return U[:, 0:1, :]
    elif boundary == 1:  # Zero-gradient
        return U[:, -1:, :]
    elif boundary == 2:  # Reflective
        ghost = U[:, -1:, :]
        ghost = ghost.at[1].set(-ghost[1])  # Negate rho*u
        return ghost


def apply_boundary_y_bottom(U_x, boundary):
    """Apply physical boundary condition on bottom edge (y-direction)."""
    if boundary == 0:  # Periodic
        return U_x[:, :, -2:-1]  # Note: -2 because U_x is already x-padded
    elif boundary == 1:  # Zero-gradient
        return U_x[:, :, 1:2]    # First interior row
    elif boundary == 2:  # Reflective
        ghost = U_x[:, :, 1:2]
        ghost = ghost.at[2].set(-ghost[2])  # Negate rho*v
        return ghost


def apply_boundary_y_top(U_x, boundary):
    """Apply physical boundary condition on top edge (y-direction)."""
    if boundary == 0:  # Periodic
        return U_x[:, :, 1:2]    # Second row (first interior after bottom ghost)
    elif boundary == 1:  # Zero-gradient
        return U_x[:, :, -2:-1]  # Last interior row
    elif boundary == 2:  # Reflective
        ghost = U_x[:, :, -2:-1]
        ghost = ghost.at[2].set(-ghost[2])  # Negate rho*v
        return ghost


def pad_1d_multiprocess(U, boundary, mesh, n_proc):
    """1D halo exchange (TODO: implement if needed)."""
    raise NotImplementedError("1D multi-process halo exchange not yet implemented")


def pad_3d_multiprocess(U, boundary, mesh, mesh_shape):
    """3D halo exchange (TODO: implement if needed)."""
    raise NotImplementedError("3D multi-process halo exchange not yet implemented")
