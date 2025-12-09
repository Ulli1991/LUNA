"""
Halo exchange for automatic sharding (NamedSharding + with_sharding_constraint).

This version uses operations that JAX can automatically handle across sharded arrays,
unlike the shard_map version which uses explicit ppermute.
"""
import jax.numpy as jnp


def pad_with_halo_auto(U, boundary, spatial_dimension):
    """
    Pad array with halo exchange using automatic sharding.

    JAX automatically handles communication when using operations like
    slicing and concatenation on sharded arrays.

    Parameters
    ----------
    U : array
        Sharded array without ghost cells
        Shape: (Nvar, Nx, Ny) for 2D or (Nvar, Nx) for 1D
    boundary : int
        Boundary condition type (0=periodic, 1=zero-gradient, 2=reflective)
    spatial_dimension : int
        1 or 2

    Returns
    -------
    U_padded : array
        Array with ghost cells
        Shape: (Nvar, Nx+2, Ny+2) for 2D or (Nvar, Nx+2) for 1D
    """
    if spatial_dimension == 1:
        return pad_with_halo_1d_auto(U, boundary)
    elif spatial_dimension == 2:
        return pad_with_halo_2d_auto(U, boundary)
    else:
        raise NotImplementedError("3D not yet implemented")


def pad_with_halo_1d_auto(U, boundary):
    """
    Pad 1D array with halo exchange using automatic sharding.

    Parameters
    ----------
    U : array (Nvar, Nx)
        Sharded array without ghost cells
    boundary : int
        Boundary condition type

    Returns
    -------
    U_padded : array (Nvar, Nx+2)
        Array with ghost cells
    """
    if boundary == 0:  # Periodic
        # Use roll - JAX automatically handles cross-device communication
        left_ghost = U[:, -1:]
        right_ghost = U[:, 0:1]
    else:  # Non-periodic (zero-gradient or reflective)
        # Copy edge values
        left_ghost = U[:, 0:1]
        right_ghost = U[:, -1:]

        if boundary == 2:  # Reflective - negate x-momentum
            left_ghost = left_ghost.at[1].set(-left_ghost[1])
            right_ghost = right_ghost.at[1].set(-right_ghost[1])

    # Concatenate - JAX handles this across sharded arrays
    U_padded = jnp.concatenate([left_ghost, U, right_ghost], axis=1)
    return U_padded


def pad_with_halo_2d_auto(U, boundary):
    """
    Pad 2D array with halo exchange using automatic sharding.

    Parameters
    ----------
    U : array (Nvar, Nx, Ny)
        Sharded array without ghost cells
    boundary : int
        Boundary condition type

    Returns
    -------
    U_padded : array (Nvar, Nx+2, Ny+2)
        Array with ghost cells
    """
    if boundary == 0:  # Periodic
        # Use slicing - JAX automatically handles cross-device communication
        left_ghost = U[:, -1:, :]
        right_ghost = U[:, 0:1, :]
        bottom_ghost = U[:, :, -1:]
        top_ghost = U[:, :, 0:1]
    else:  # Non-periodic
        # Copy edge values
        left_ghost = U[:, 0:1, :]
        right_ghost = U[:, -1:, :]
        bottom_ghost = U[:, :, 0:1]
        top_ghost = U[:, :, -1:]

        if boundary == 2:  # Reflective - negate momentum
            left_ghost = left_ghost.at[1].set(-left_ghost[1])
            right_ghost = right_ghost.at[1].set(-right_ghost[1])
            bottom_ghost = bottom_ghost.at[2].set(-bottom_ghost[2])
            top_ghost = top_ghost.at[2].set(-top_ghost[2])

    # Pad in x-direction first
    U_x = jnp.concatenate([left_ghost, U, right_ghost], axis=1)

    # Get corners from the x-padded array
    corner_bl = U_x[:, 0:1, 0:1]
    corner_br = U_x[:, -1:, 0:1]
    corner_tl = U_x[:, 0:1, -1:]
    corner_tr = U_x[:, -1:, -1:]

    # Extend top/bottom with corners
    bottom_extended = jnp.concatenate([corner_bl, bottom_ghost, corner_br], axis=1)
    top_extended = jnp.concatenate([corner_tl, top_ghost, corner_tr], axis=1)

    # Pad in y-direction
    U_padded = jnp.concatenate([bottom_extended, U_x, top_extended], axis=2)

    return U_padded
