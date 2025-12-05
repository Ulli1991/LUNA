"""
Simple halo exchange for pmap that works with 2D decomposition.

This version doesn't require passing domain_info through pmap.
Instead, it uses lax.axis_index to determine which GPU and what neighbors.
"""
import jax.numpy as jnp
from jax import lax


def pad_with_halo_2d(U_local, boundary, grid_shape):
    """
    Pad 2D local array with halo exchange for pmap.

    Must be called inside @pmap with axis_name='batch'.

    Parameters
    ----------
    U_local : array (Nvar, nx_local, ny_local)
        Local subdomain WITHOUT ghost cells
    boundary : int
        0=periodic, 1=zero-gradient, 2=reflective
    grid_shape : tuple (nx_proc, ny_proc)
        Process grid shape (e.g., (2, 2) for 4 GPUs)

    Returns
    -------
    U_padded : array (Nvar, nx_local+2, ny_local+2)
        Local subdomain WITH ghost cells
    """
    nx_proc, ny_proc = grid_shape
    n_devices = nx_proc * ny_proc

    # Get this device's rank and coordinates
    rank = lax.axis_index('batch')
    px = rank // ny_proc
    py = rank % ny_proc

    # Determine neighbors (None if at boundary)
    has_left = px > 0
    has_right = px < nx_proc - 1
    has_bottom = py > 0
    has_top = py < ny_proc - 1

    # Extract boundary slices to send
    left_send = U_local[:, 0:1, :]    # (Nvar, 1, ny_local)
    right_send = U_local[:, -1:, :]   # (Nvar, 1, ny_local)
    bottom_send = U_local[:, :, 0:1]  # (Nvar, nx_local, 1)
    top_send = U_local[:, :, -1:]     # (Nvar, nx_local, 1)

    # Gather boundaries from all devices
    all_rights = lax.all_gather(right_send, 'batch', axis=0, tiled=False)
    all_lefts = lax.all_gather(left_send, 'batch', axis=0, tiled=False)
    all_tops = lax.all_gather(top_send, 'batch', axis=0, tiled=False)
    all_bottoms = lax.all_gather(bottom_send, 'batch', axis=0, tiled=False)

    # Extract ghost cells from neighbors OR apply BC
    def get_ghost_or_bc(has_neighbor, neighbor_rank, neighbor_data, bc_data, bc_type, momentum_idx):
        """Select neighbor data if internal, or apply BC if physical boundary."""
        if has_neighbor:
            ghost = neighbor_data[neighbor_rank]
        else:
            # Apply boundary condition
            if bc_type == 0:  # Periodic - wrap around
                # For periodic, we need the opposite side
                # This is a simplification - proper periodic needs opposite edge
                ghost = bc_data
            elif bc_type == 1:  # Zero-gradient
                ghost = bc_data
            elif bc_type == 2:  # Reflective - negate momentum
                ghost = bc_data.at[momentum_idx].set(-bc_data[momentum_idx])
        return ghost

    # Get ghosts for each direction
    # Left/right exchange in x-direction
    left_neighbor = lax.select(has_left, rank - ny_proc, -1)
    right_neighbor = lax.select(has_right, rank + ny_proc, -1)

    # Left ghost
    def get_left_ghost_internal():
        return all_rights[left_neighbor]

    def get_left_ghost_boundary():
        if boundary == 0:  # Periodic
            rightmost_in_row = lax.select(px == 0, (px * ny_proc) + (ny_proc - 1), rank)
            return lax.select(px == 0, all_rights[rightmost_in_row], right_send)
        elif boundary == 1:  # Zero-gradient
            return left_send
        else:  # Reflective
            return left_send.at[1].set(-left_send[1])

    left_ghost = lax.cond(has_left, get_left_ghost_internal, get_left_ghost_boundary)

    # Right ghost
    def get_right_ghost_internal():
        return all_lefts[right_neighbor]

    def get_right_ghost_boundary():
        if boundary == 0:  # Periodic
            leftmost_in_row = lax.select(px == nx_proc - 1, px * ny_proc, rank)
            return lax.select(px == nx_proc - 1, all_lefts[leftmost_in_row], left_send)
        elif boundary == 1:  # Zero-gradient
            return right_send
        else:  # Reflective
            return right_send.at[1].set(-right_send[1])

    right_ghost = lax.cond(has_right, get_right_ghost_internal, get_right_ghost_boundary)

    # Bottom/top exchange in y-direction
    bottom_neighbor = lax.select(has_bottom, rank - 1, -1)
    top_neighbor = lax.select(has_top, rank + 1, -1)

    # Bottom ghost
    def get_bottom_ghost_internal():
        return all_tops[bottom_neighbor]

    def get_bottom_ghost_boundary():
        if boundary == 0:  # Periodic
            topmost_in_col = lax.select(py == 0, rank + (ny_proc - 1), rank)
            return lax.select(py == 0, all_tops[topmost_in_col], top_send)
        elif boundary == 1:  # Zero-gradient
            return bottom_send
        else:  # Reflective
            return bottom_send.at[2].set(-bottom_send[2])

    bottom_ghost = lax.cond(has_bottom, get_bottom_ghost_internal, get_bottom_ghost_boundary)

    # Top ghost
    def get_top_ghost_internal():
        return all_bottoms[top_neighbor]

    def get_top_ghost_boundary():
        if boundary == 0:  # Periodic
            bottommost_in_col = lax.select(py == ny_proc - 1, rank - py, rank)
            return lax.select(py == ny_proc - 1, all_bottoms[bottommost_in_col], bottom_send)
        elif boundary == 1:  # Zero-gradient
            return top_send
        else:  # Reflective
            return top_send.at[2].set(-top_send[2])

    top_ghost = lax.cond(has_top, get_top_ghost_internal, get_top_ghost_boundary)

    # Pad in x-direction first
    U_x = jnp.concatenate([left_ghost, U_local, right_ghost], axis=1)

    # Extend bottom/top ghosts to include corners
    # Corners: use left/right ghost values at bottom/top positions
    bottom_extended = jnp.concatenate([
        left_ghost[:, :, 0:1],
        bottom_ghost,
        right_ghost[:, :, 0:1]
    ], axis=1)

    top_extended = jnp.concatenate([
        left_ghost[:, :, -1:],
        top_ghost,
        right_ghost[:, :, -1:]
    ], axis=1)

    # Pad in y-direction
    U_padded = jnp.concatenate([bottom_extended, U_x, top_extended], axis=2)

    return U_padded


def pad_with_halo_1d(U_local, boundary, grid_shape):
    """
    Pad 1D local array with halo exchange for pmap.

    Parameters
    ----------
    U_local : array (Nvar, nx_local)
    boundary : int
    grid_shape : tuple (nx_proc,)

    Returns
    -------
    U_padded : array (Nvar, nx_local+2)
    """
    nx_proc, = grid_shape
    rank = lax.axis_index('batch')

    has_left = rank > 0
    has_right = rank < nx_proc - 1

    left_send = U_local[:, 0:1]
    right_send = U_local[:, -1:]

    all_rights = lax.all_gather(right_send, 'batch', axis=0, tiled=False)
    all_lefts = lax.all_gather(left_send, 'batch', axis=0, tiled=False)

    # Left ghost
    def get_left_ghost_internal():
        return all_rights[rank - 1]

    def get_left_ghost_boundary():
        if boundary == 0:  # Periodic
            return all_rights[nx_proc - 1]
        elif boundary == 1:  # Zero-gradient
            return left_send
        else:  # Reflective
            return left_send.at[1].set(-left_send[1])

    left_ghost = lax.cond(has_left, get_left_ghost_internal, get_left_ghost_boundary)

    # Right ghost
    def get_right_ghost_internal():
        return all_lefts[rank + 1]

    def get_right_ghost_boundary():
        if boundary == 0:  # Periodic
            return all_lefts[0]
        elif boundary == 1:  # Zero-gradient
            return right_send
        else:  # Reflective
            return right_send.at[1].set(-right_send[1])

    right_ghost = lax.cond(has_right, get_right_ghost_internal, get_right_ghost_boundary)

    return jnp.concatenate([left_ghost, U_local, right_ghost], axis=1)


def create_pad_halo(spatial_dimension, boundary, grid_shape):
    """
    Create a halo padding function for the given spatial dimension.

    Parameters
    ----------
    spatial_dimension : int
        1, 2, or 3
    boundary : int
        Boundary condition type (0=periodic, 1=zero-gradient, 2=reflective)
    grid_shape : tuple
        Process grid shape

    Returns
    -------
    pad_halo : callable
        Function to pad local subdomain with halo exchange
    """
    if spatial_dimension == 1:
        def pad_halo(U):
            return pad_with_halo_1d(U, boundary, grid_shape)
    elif spatial_dimension == 2:
        def pad_halo(U):
            return pad_with_halo_2d(U, boundary, grid_shape)
    else:
        raise NotImplementedError("3D halo exchange not implemented yet")

    return pad_halo
