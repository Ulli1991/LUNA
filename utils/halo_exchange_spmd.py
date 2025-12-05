"""
Halo exchange for JAX SPMD multi-process execution.

This module implements ghost cell communication between processes using
JAX's multihost utilities for distributed execution.
"""
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils


def pad_with_halo_2d(U_local, boundary, domain):
    """
    Pad 2D local subdomain with ghost cells using halo exchange.

    For each boundary:
    - If internal (has neighbor): exchange data with neighbor process
    - If physical (domain edge): apply physical boundary condition

    Parameters
    ----------
    U_local : array
        Local subdomain, shape (Nvar, Nx_local, Ny_local)
    boundary : int
        Physical boundary condition: 0=periodic, 1=zero-gradient, 2=reflective
    domain : LocalDomain
        Domain decomposition information

    Returns
    -------
    U_pad : array
        Padded array with 1 ghost cell per side, shape (Nvar, Nx_local+2, Ny_local+2)
    """
    Nvar, Nx_local, Ny_local = U_local.shape

    # ========================================================================
    # X-DIRECTION HALO EXCHANGE
    # ========================================================================

    # Extract boundaries to send
    left_send = U_local[:, 0:1, :]    # Leftmost column to send to left neighbor
    right_send = U_local[:, -1:, :]   # Rightmost column to send to right neighbor

    # Gather all left and right boundaries from all processes
    # This is a collective operation - all processes must participate
    all_left_boundaries = multihost_utils.process_allgather(left_send, tiled=False)
    all_right_boundaries = multihost_utils.process_allgather(right_send, tiled=False)

    # Receive left ghost (from left neighbor's right boundary)
    if not domain.is_physical_boundary('left'):
        # Internal boundary: get from left neighbor
        left_neighbor_id = domain.neighbors['left']
        left_ghost = all_right_boundaries[left_neighbor_id]
    else:
        # Physical boundary: apply boundary condition
        left_ghost = apply_bc_2d_left(U_local, boundary)

    # Receive right ghost (from right neighbor's left boundary)
    if not domain.is_physical_boundary('right'):
        right_neighbor_id = domain.neighbors['right']
        right_ghost = all_left_boundaries[right_neighbor_id]
    else:
        right_ghost = apply_bc_2d_right(U_local, boundary)

    # Concatenate in x-direction: [left_ghost, U_local, right_ghost]
    U_x = jnp.concatenate([left_ghost, U_local, right_ghost], axis=1)
    # Shape is now (Nvar, Nx_local+2, Ny_local)

    # ========================================================================
    # Y-DIRECTION HALO EXCHANGE (on x-padded array)
    # ========================================================================

    # Extract boundaries to send (from x-padded array)
    bottom_send = U_x[:, :, 0:1]   # Bottom row
    top_send = U_x[:, :, -1:]      # Top row

    # Gather all bottom and top boundaries
    all_bottom_boundaries = multihost_utils.process_allgather(bottom_send, tiled=False)
    all_top_boundaries = multihost_utils.process_allgather(top_send, tiled=False)

    # Receive bottom ghost (from bottom neighbor's top boundary)
    if not domain.is_physical_boundary('bottom'):
        bottom_neighbor_id = domain.neighbors['bottom']
        bottom_ghost = all_top_boundaries[bottom_neighbor_id]
    else:
        bottom_ghost = apply_bc_2d_bottom(U_x, boundary)

    # Receive top ghost (from top neighbor's bottom boundary)
    if not domain.is_physical_boundary('top'):
        top_neighbor_id = domain.neighbors['top']
        top_ghost = all_bottom_boundaries[top_neighbor_id]
    else:
        top_ghost = apply_bc_2d_top(U_x, boundary)

    # Concatenate in y-direction: [bottom_ghost, U_x, top_ghost]
    U_pad = jnp.concatenate([bottom_ghost, U_x, top_ghost], axis=2)
    # Final shape: (Nvar, Nx_local+2, Ny_local+2)

    return U_pad


def pad_with_halo_1d(U_local, boundary, domain):
    """
    Pad 1D local subdomain with ghost cells using halo exchange.

    Parameters
    ----------
    U_local : array
        Local subdomain, shape (Nvar, Nx_local)
    boundary : int
        Physical boundary condition: 0=periodic, 1=zero-gradient, 2=reflective
    domain : LocalDomain
        Domain decomposition information

    Returns
    -------
    U_pad : array
        Padded array, shape (Nvar, Nx_local+2)
    """
    Nvar, Nx_local = U_local.shape

    # Extract boundaries
    left_send = U_local[:, 0:1]
    right_send = U_local[:, -1:]

    # Gather
    all_left = multihost_utils.process_allgather(left_send, tiled=False)
    all_right = multihost_utils.process_allgather(right_send, tiled=False)

    # Get ghosts
    if not domain.is_physical_boundary('left'):
        left_ghost = all_right[domain.neighbors['left']]
    else:
        left_ghost = apply_bc_1d_left(U_local, boundary)

    if not domain.is_physical_boundary('right'):
        right_ghost = all_left[domain.neighbors['right']]
    else:
        right_ghost = apply_bc_1d_right(U_local, boundary)

    U_pad = jnp.concatenate([left_ghost, U_local, right_ghost], axis=1)
    return U_pad


def pad_with_halo_3d(U_local, boundary, domain):
    """
    Pad 3D local subdomain with ghost cells using halo exchange.

    Parameters
    ----------
    U_local : array
        Local subdomain, shape (Nvar, Nx_local, Ny_local, Nz_local)
    boundary : int
        Physical boundary condition
    domain : LocalDomain
        Domain decomposition information

    Returns
    -------
    U_pad : array
        Padded array, shape (Nvar, Nx_local+2, Ny_local+2, Nz_local+2)
    """
    Nvar, Nx_local, Ny_local, Nz_local = U_local.shape

    # X-direction
    left_send = U_local[:, 0:1, :, :]
    right_send = U_local[:, -1:, :, :]

    all_left = multihost_utils.process_allgather(left_send, tiled=False)
    all_right = multihost_utils.process_allgather(right_send, tiled=False)

    if not domain.is_physical_boundary('left'):
        left_ghost = all_right[domain.neighbors['left']]
    else:
        left_ghost = apply_bc_3d_left(U_local, boundary)

    if not domain.is_physical_boundary('right'):
        right_ghost = all_left[domain.neighbors['right']]
    else:
        right_ghost = apply_bc_3d_right(U_local, boundary)

    U_x = jnp.concatenate([left_ghost, U_local, right_ghost], axis=1)

    # Y-direction
    bottom_send = U_x[:, :, 0:1, :]
    top_send = U_x[:, :, -1:, :]

    all_bottom = multihost_utils.process_allgather(bottom_send, tiled=False)
    all_top = multihost_utils.process_allgather(top_send, tiled=False)

    if not domain.is_physical_boundary('bottom'):
        bottom_ghost = all_top[domain.neighbors['bottom']]
    else:
        bottom_ghost = apply_bc_3d_bottom(U_x, boundary)

    if not domain.is_physical_boundary('top'):
        top_ghost = all_bottom[domain.neighbors['top']]
    else:
        top_ghost = apply_bc_3d_top(U_x, boundary)

    U_y = jnp.concatenate([bottom_ghost, U_x, top_ghost], axis=2)

    # Z-direction
    back_send = U_y[:, :, :, 0:1]
    front_send = U_y[:, :, :, -1:]

    all_back = multihost_utils.process_allgather(back_send, tiled=False)
    all_front = multihost_utils.process_allgather(front_send, tiled=False)

    if not domain.is_physical_boundary('back'):
        back_ghost = all_front[domain.neighbors['back']]
    else:
        back_ghost = apply_bc_3d_back(U_y, boundary)

    if not domain.is_physical_boundary('front'):
        front_ghost = all_back[domain.neighbors['front']]
    else:
        front_ghost = apply_bc_3d_front(U_y, boundary)

    U_pad = jnp.concatenate([back_ghost, U_y, front_ghost], axis=3)

    return U_pad


# ============================================================================
# PHYSICAL BOUNDARY CONDITIONS
# ============================================================================
# These are applied ONLY at the edges of the global domain

def apply_bc_1d_left(U, boundary):
    """Apply physical BC on left edge (1D)."""
    if boundary == 0:  # Periodic
        return U[:, -1:]
    elif boundary == 1:  # Zero-gradient
        return U[:, 0:1]
    elif boundary == 2:  # Reflective
        ghost = U[:, 0:1]
        ghost = ghost.at[1].set(-ghost[1])  # Negate rho*u
        return ghost

def apply_bc_1d_right(U, boundary):
    """Apply physical BC on right edge (1D)."""
    if boundary == 0:  # Periodic
        return U[:, 0:1]
    elif boundary == 1:  # Zero-gradient
        return U[:, -1:]
    elif boundary == 2:  # Reflective
        ghost = U[:, -1:]
        ghost = ghost.at[1].set(-ghost[1])
        return ghost


def apply_bc_2d_left(U, boundary):
    """Apply physical BC on left edge (2D)."""
    if boundary == 0:
        return U[:, -1:, :]
    elif boundary == 1:
        return U[:, 0:1, :]
    elif boundary == 2:
        ghost = U[:, 0:1, :]
        ghost = ghost.at[1].set(-ghost[1])  # Negate rho*u
        return ghost

def apply_bc_2d_right(U, boundary):
    """Apply physical BC on right edge (2D)."""
    if boundary == 0:
        return U[:, 0:1, :]
    elif boundary == 1:
        return U[:, -1:, :]
    elif boundary == 2:
        ghost = U[:, -1:, :]
        ghost = ghost.at[1].set(-ghost[1])
        return ghost

def apply_bc_2d_bottom(U_x, boundary):
    """Apply physical BC on bottom edge (2D). U_x is already x-padded."""
    if boundary == 0:
        return U_x[:, :, -2:-1]  # -2 because already x-padded
    elif boundary == 1:
        return U_x[:, :, 1:2]
    elif boundary == 2:
        ghost = U_x[:, :, 1:2]
        ghost = ghost.at[2].set(-ghost[2])  # Negate rho*v
        return ghost

def apply_bc_2d_top(U_x, boundary):
    """Apply physical BC on top edge (2D)."""
    if boundary == 0:
        return U_x[:, :, 1:2]
    elif boundary == 1:
        return U_x[:, :, -2:-1]
    elif boundary == 2:
        ghost = U_x[:, :, -2:-1]
        ghost = ghost.at[2].set(-ghost[2])
        return ghost


def apply_bc_3d_left(U, boundary):
    """Apply physical BC on left edge (3D)."""
    if boundary == 0:
        return U[:, -1:, :, :]
    elif boundary == 1:
        return U[:, 0:1, :, :]
    elif boundary == 2:
        ghost = U[:, 0:1, :, :]
        ghost = ghost.at[1].set(-ghost[1])
        return ghost

def apply_bc_3d_right(U, boundary):
    """Apply physical BC on right edge (3D)."""
    if boundary == 0:
        return U[:, 0:1, :, :]
    elif boundary == 1:
        return U[:, -1:, :, :]
    elif boundary == 2:
        ghost = U[:, -1:, :, :]
        ghost = ghost.at[1].set(-ghost[1])
        return ghost

def apply_bc_3d_bottom(U_x, boundary):
    """Apply physical BC on bottom edge (3D)."""
    if boundary == 0:
        return U_x[:, :, -2:-1, :]
    elif boundary == 1:
        return U_x[:, :, 1:2, :]
    elif boundary == 2:
        ghost = U_x[:, :, 1:2, :]
        ghost = ghost.at[2].set(-ghost[2])
        return ghost

def apply_bc_3d_top(U_x, boundary):
    """Apply physical BC on top edge (3D)."""
    if boundary == 0:
        return U_x[:, :, 1:2, :]
    elif boundary == 1:
        return U_x[:, :, -2:-1, :]
    elif boundary == 2:
        ghost = U_x[:, :, -2:-1, :]
        ghost = ghost.at[2].set(-ghost[2])
        return ghost

def apply_bc_3d_back(U_y, boundary):
    """Apply physical BC on back edge (3D)."""
    if boundary == 0:
        return U_y[:, :, :, -2:-1]
    elif boundary == 1:
        return U_y[:, :, :, 1:2]
    elif boundary == 2:
        ghost = U_y[:, :, :, 1:2]
        ghost = ghost.at[3].set(-ghost[3])  # Negate rho*w
        return ghost

def apply_bc_3d_front(U_y, boundary):
    """Apply physical BC on front edge (3D)."""
    if boundary == 0:
        return U_y[:, :, :, 1:2]
    elif boundary == 1:
        return U_y[:, :, :, -2:-1]
    elif boundary == 2:
        ghost = U_y[:, :, :, -2:-1]
        ghost = ghost.at[3].set(-ghost[3])
        return ghost
