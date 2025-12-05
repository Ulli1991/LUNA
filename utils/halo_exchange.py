"""
Halo exchange using JAX collectives for multi-GPU communication.

This module handles boundary data exchange between neighboring processes
using JAX's pmap and collective operations.
"""
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial


def pad_with_halo_exchange(U_local, boundary, domain_info, spatial_dim):
    """
    Pad local array with ghost cells, using halo exchange for internal boundaries
    and applying boundary conditions for physical boundaries.

    This function is designed to work with pmap, where each device has its own U_local.

    Parameters
    ----------
    U_local : array
        Local state WITHOUT ghost cells: (Nvar, nx_local, ...)
    boundary : int
        Boundary condition type: 0=periodic, 1=zero-gradient, 2=reflective
    domain_info : dict
        Domain decomposition info from domain_decomp.get_local_domain_info
    spatial_dim : int
        1, 2, or 3

    Returns
    -------
    U_padded : array
        Local state WITH 1 ghost cell per side: (Nvar, nx_local+2, ...)

    NOTE: This function should be called inside pmap with axis_name='devices'.
    The collective operations (ppermute) only work within pmap context.
    """
    neighbors = domain_info['neighbors']

    if spatial_dim == 1:
        return _pad_1d(U_local, boundary, neighbors)
    elif spatial_dim == 2:
        return _pad_2d(U_local, boundary, neighbors)
    elif spatial_dim == 3:
        return _pad_3d(U_local, boundary, neighbors)


def _exchange_boundaries_1d(U_local, neighbors):
    """
    Exchange boundary data in 1D using JAX collectives.

    U_local: (Nvar, nx_local)
    Returns: left_ghost, right_ghost, each (Nvar, 1)
    """
    # Get boundary slices to send
    left_boundary = U_local[:, 0:1]   # Send to left neighbor
    right_boundary = U_local[:, -1:]  # Send to right neighbor

    # Use lax.ppermute for point-to-point communication
    # ppermute requires axis_name from pmap
    # We'll use allgather as a simpler alternative that doesn't require axis_name
    # Each device gathers all boundaries, then selects the ones it needs

    axis_size = lax.psum(1, 'devices')  # Number of devices
    rank = lax.axis_index('devices')    # This device's rank

    # Gather all left boundaries (what each device wants to send to its left neighbor)
    all_right_boundaries = lax.all_gather(right_boundary, 'devices', axis=0, tiled=False)
    # Gather all right boundaries (what each device wants to send to its right neighbor)
    all_left_boundaries = lax.all_gather(left_boundary, 'devices', axis=0, tiled=False)

    # Extract what I need from my neighbors
    # My left ghost comes from my left neighbor's right boundary
    # My right ghost comes from my right neighbor's left boundary

    if neighbors['left'] is not None:
        left_ghost = all_right_boundaries[neighbors['left']]
    else:
        left_ghost = None  # Will be filled by boundary condition

    if neighbors['right'] is not None:
        right_ghost = all_left_boundaries[neighbors['right']]
    else:
        right_ghost = None  # Will be filled by boundary condition

    return left_ghost, right_ghost


def _apply_bc_1d(U_local, boundary, side):
    """
    Apply boundary condition to get ghost cell values.

    Parameters
    ----------
    U_local : array (Nvar, nx_local)
    boundary : int (0=periodic, 1=zero-grad, 2=reflective)
    side : str ('left' or 'right')

    Returns
    -------
    ghost_cell : array (Nvar, 1)
    """
    if side == 'left':
        interior_cell = U_local[:, 0:1]
    else:  # right
        interior_cell = U_local[:, -1:]

    if boundary == 0:  # Periodic - wrap around
        if side == 'left':
            ghost = U_local[:, -1:]
        else:
            ghost = U_local[:, 0:1]
    elif boundary == 1:  # Zero-gradient
        ghost = interior_cell
    elif boundary == 2:  # Reflective
        ghost = interior_cell.at[1].set(-interior_cell[1])  # Negate momentum

    return ghost


def _pad_1d(U_local, boundary, neighbors):
    """Pad 1D array with ghost cells."""
    # Exchange boundaries
    left_ghost, right_ghost = _exchange_boundaries_1d(U_local, neighbors)

    # Apply boundary conditions where needed
    if left_ghost is None:
        left_ghost = _apply_bc_1d(U_local, boundary, 'left')
    if right_ghost is None:
        right_ghost = _apply_bc_1d(U_local, boundary, 'right')

    # Concatenate
    return jnp.concatenate([left_ghost, U_local, right_ghost], axis=1)


def _exchange_boundaries_2d(U_local, neighbors):
    """
    Exchange boundary data in 2D.

    U_local: (Nvar, nx_local, ny_local)
    Returns: dict with keys 'left', 'right', 'bottom', 'top'
            values are ghost strips or None
    """
    # Extract boundaries
    left_boundary = U_local[:, 0:1, :]    # (Nvar, 1, ny_local)
    right_boundary = U_local[:, -1:, :]   # (Nvar, 1, ny_local)
    bottom_boundary = U_local[:, :, 0:1]  # (Nvar, nx_local, 1)
    top_boundary = U_local[:, :, -1:]     # (Nvar, nx_local, 1)

    # Gather all boundaries from all devices
    all_rights = lax.all_gather(right_boundary, 'devices', axis=0, tiled=False)
    all_lefts = lax.all_gather(left_boundary, 'devices', axis=0, tiled=False)
    all_tops = lax.all_gather(top_boundary, 'devices', axis=0, tiled=False)
    all_bottoms = lax.all_gather(bottom_boundary, 'devices', axis=0, tiled=False)

    ghosts = {}

    # Extract from neighbors
    ghosts['left'] = all_rights[neighbors['left']] if neighbors['left'] is not None else None
    ghosts['right'] = all_lefts[neighbors['right']] if neighbors['right'] is not None else None
    ghosts['bottom'] = all_tops[neighbors['bottom']] if neighbors['bottom'] is not None else None
    ghosts['top'] = all_bottoms[neighbors['top']] if neighbors['top'] is not None else None

    return ghosts


def _apply_bc_2d(U_local, boundary, side):
    """
    Apply 2D boundary condition.

    side : str ('left', 'right', 'bottom', 'top')
    Returns: ghost strip matching the boundary shape
    """
    if side == 'left':
        interior = U_local[:, 0:1, :]
        if boundary == 0:  # Periodic
            ghost = U_local[:, -1:, :]
        elif boundary == 1:  # Zero-gradient
            ghost = interior
        elif boundary == 2:  # Reflective - negate x-momentum (index 1)
            ghost = interior.at[1].set(-interior[1])
    elif side == 'right':
        interior = U_local[:, -1:, :]
        if boundary == 0:
            ghost = U_local[:, 0:1, :]
        elif boundary == 1:
            ghost = interior
        elif boundary == 2:
            ghost = interior.at[1].set(-interior[1])
    elif side == 'bottom':
        interior = U_local[:, :, 0:1]
        if boundary == 0:
            ghost = U_local[:, :, -1:]
        elif boundary == 1:
            ghost = interior
        elif boundary == 2:  # Reflective - negate y-momentum (index 2)
            ghost = interior.at[2].set(-interior[2])
    else:  # top
        interior = U_local[:, :, -1:]
        if boundary == 0:
            ghost = U_local[:, :, 0:1]
        elif boundary == 1:
            ghost = interior
        elif boundary == 2:
            ghost = interior.at[2].set(-interior[2])

    return ghost


def _pad_2d(U_local, boundary, neighbors):
    """Pad 2D array with ghost cells."""
    # Exchange boundaries
    ghosts = _exchange_boundaries_2d(U_local, neighbors)

    # Apply BCs where needed
    for side in ['left', 'right', 'bottom', 'top']:
        if ghosts[side] is None:
            ghosts[side] = _apply_bc_2d(U_local, boundary, side)

    # Pad in x-direction first
    U_x = jnp.concatenate([ghosts['left'], U_local, ghosts['right']], axis=1)

    # Then pad in y-direction
    # Need to extend the top/bottom ghosts to include the corners
    bottom_extended = jnp.concatenate([
        ghosts['left'][:, :, 0:1],
        ghosts['bottom'],
        ghosts['right'][:, :, 0:1]
    ], axis=1)

    top_extended = jnp.concatenate([
        ghosts['left'][:, :, -1:],
        ghosts['top'],
        ghosts['right'][:, :, -1:]
    ], axis=1)

    return jnp.concatenate([bottom_extended, U_x, top_extended], axis=2)


def _exchange_boundaries_3d(U_local, neighbors):
    """Exchange boundary data in 3D."""
    # Extract boundaries
    left = U_local[:, 0:1, :, :]
    right = U_local[:, -1:, :, :]
    bottom = U_local[:, :, 0:1, :]
    top = U_local[:, :, -1:, :]
    back = U_local[:, :, :, 0:1]
    front = U_local[:, :, :, -1:]

    # Gather
    all_rights = lax.all_gather(right, 'devices', axis=0, tiled=False)
    all_lefts = lax.all_gather(left, 'devices', axis=0, tiled=False)
    all_tops = lax.all_gather(top, 'devices', axis=0, tiled=False)
    all_bottoms = lax.all_gather(bottom, 'devices', axis=0, tiled=False)
    all_fronts = lax.all_gather(front, 'devices', axis=0, tiled=False)
    all_backs = lax.all_gather(back, 'devices', axis=0, tiled=False)

    ghosts = {
        'left': all_rights[neighbors['left']] if neighbors['left'] is not None else None,
        'right': all_lefts[neighbors['right']] if neighbors['right'] is not None else None,
        'bottom': all_tops[neighbors['bottom']] if neighbors['bottom'] is not None else None,
        'top': all_bottoms[neighbors['top']] if neighbors['top'] is not None else None,
        'back': all_fronts[neighbors['back']] if neighbors['back'] is not None else None,
        'front': all_backs[neighbors['front']] if neighbors['front'] is not None else None,
    }

    return ghosts


def _apply_bc_3d(U_local, boundary, side):
    """Apply 3D boundary condition."""
    sides_config = {
        'left': (1, U_local[:, 0:1, :, :], U_local[:, -1:, :, :]),
        'right': (1, U_local[:, -1:, :, :], U_local[:, 0:1, :, :]),
        'bottom': (2, U_local[:, :, 0:1, :], U_local[:, :, -1:, :]),
        'top': (2, U_local[:, :, -1:, :], U_local[:, :, 0:1, :]),
        'back': (3, U_local[:, :, :, 0:1], U_local[:, :, :, -1:]),
        'front': (3, U_local[:, :, :, -1:], U_local[:, :, :, 0:1]),
    }

    momentum_idx, interior, periodic_source = sides_config[side]

    if boundary == 0:  # Periodic
        ghost = periodic_source
    elif boundary == 1:  # Zero-gradient
        ghost = interior
    elif boundary == 2:  # Reflective
        ghost = interior.at[momentum_idx].set(-interior[momentum_idx])

    return ghost


def _pad_3d(U_local, boundary, neighbors):
    """Pad 3D array with ghost cells."""
    ghosts = _exchange_boundaries_3d(U_local, neighbors)

    # Apply BCs
    for side in ['left', 'right', 'bottom', 'top', 'back', 'front']:
        if ghosts[side] is not None:
            ghosts[side] = _apply_bc_3d(U_local, boundary, side)

    # Pad x
    U_x = jnp.concatenate([ghosts['left'], U_local, ghosts['right']], axis=1)

    # Pad y (extend with corners)
    ny_padded = U_x.shape[2]
    bottom_ext = jnp.concatenate([
        ghosts['left'][:, :, 0:1, :],
        ghosts['bottom'],
        ghosts['right'][:, :, 0:1, :]
    ], axis=1)
    top_ext = jnp.concatenate([
        ghosts['left'][:, :, -1:, :],
        ghosts['top'],
        ghosts['right'][:, :, -1:, :]
    ], axis=1)
    U_xy = jnp.concatenate([bottom_ext, U_x, top_ext], axis=2)

    # Pad z (extend with all corners/edges)
    nx_padded, ny_padded = U_xy.shape[1], U_xy.shape[2]
    back_ext = jnp.concatenate([
        U_xy[:, :, :, 0:1] * 0 + ghosts['back'][0, 0, 0, 0],  # Simplified corner handling
        ghosts['back'],
        U_xy[:, :, :, 0:1] * 0 + ghosts['back'][0, 0, 0, -1]
    ], axis=1)
    # This is a simplified version - full corner handling would be more complex
    # For now, we'll use a simpler approach
    U_xyz = jnp.concatenate([ghosts['back'], U_xy, ghosts['front']], axis=3)

    return U_xyz
