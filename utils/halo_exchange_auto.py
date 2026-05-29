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
    elif spatial_dimension == 3:
        return pad_with_halo_3d_auto(U, boundary)
    else:
        raise NotImplementedError(f"Spatial dimension {spatial_dimension} not supported")


def pad_with_halo_1d_auto(U, boundary):
    """
    Pad 1D array with halo exchange using automatic sharding.
    Uses jnp.roll() for periodic boundaries (slightly more efficient).

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
    if boundary == 0:  # Periodic — wrap-around
        # Left ghost = U[Nx-1] (last interior cell), right ghost = U[0].
        # NOTE: the previous implementation used
        #   jnp.roll(U, 1, axis=1)[:, -1:]   (which gives U[Nx-2], off-by-one)
        # and the mirror on the right side. That bug shifted both ghost
        # cells by one and corrupted the slopes of cells 0 and Nx-1.
        U_padded = jnp.pad(U, ((0, 0), (1, 1)), mode="wrap")

    elif boundary == 1:  # Zero-gradient
        U_padded = jnp.pad(U, ((0, 0), (1, 1)), mode="edge")

    else:  # Reflective
        U_padded = jnp.pad(U, ((0, 0), (1, 1)), mode="edge")
        # Negate x-momentum at boundaries
        U_padded = U_padded.at[1, 0].set(-U_padded[1, 0])
        U_padded = U_padded.at[1, -1].set(-U_padded[1, -1])

    return U_padded


def pad_with_halo_2d_auto(U, boundary):
    """
    Pad 2D array with halo exchange using automatic sharding.
    Uses jnp.roll() for periodic boundaries (slightly more efficient).

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
    if boundary == 0:  # Periodic — wrap-around (was off-by-one with jnp.roll)
        U_padded = jnp.pad(U, ((0, 0), (1, 1), (1, 1)), mode="wrap")

    elif boundary == 1:  # Zero-gradient - use edge mode
        # jnp.pad with 'edge' mode - copies edge values
        U_padded = jnp.pad(U, ((0, 0), (1, 1), (1, 1)), mode="edge")

    else:  # Reflective - use edge then negate momentum
        # First pad with edge mode
        U_padded = jnp.pad(U, ((0, 0), (1, 1), (1, 1)), mode="edge")

        # Negate momentum at boundaries
        # Left edge (x-momentum)
        U_padded = U_padded.at[1, 0, :].set(-U_padded[1, 0, :])
        # Right edge (x-momentum)
        U_padded = U_padded.at[1, -1, :].set(-U_padded[1, -1, :])
        # Bottom edge (y-momentum)
        U_padded = U_padded.at[2, :, 0].set(-U_padded[2, :, 0])
        # Top edge (y-momentum)
        U_padded = U_padded.at[2, :, -1].set(-U_padded[2, :, -1])

    return U_padded


def pad_with_halo_3d_auto(U, boundary):
    """
    Pad 3D array with halo exchange using automatic sharding.

    Parameters
    ----------
    U : array (Nvar, Nx, Ny, Nz)
        Sharded array without ghost cells
    boundary : int
        Boundary condition type

    Returns
    -------
    U_padded : array (Nvar, Nx+2, Ny+2, Nz+2)
        Array with ghost cells
    """
    if boundary == 0:  # Periodic — wrap-around (was off-by-one with jnp.roll)
        U_padded = jnp.pad(U, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="wrap")

    elif boundary == 1:  # Zero-gradient
        U_padded = jnp.pad(U, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="edge")

    else:  # Reflective
        U_padded = jnp.pad(U, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="edge")

        # Negate x-momentum at x-boundaries
        U_padded = U_padded.at[1, 0, :, :].set(-U_padded[1, 0, :, :])
        U_padded = U_padded.at[1, -1, :, :].set(-U_padded[1, -1, :, :])
        # Negate y-momentum at y-boundaries
        U_padded = U_padded.at[2, :, 0, :].set(-U_padded[2, :, 0, :])
        U_padded = U_padded.at[2, :, -1, :].set(-U_padded[2, :, -1, :])
        # Negate z-momentum at z-boundaries
        U_padded = U_padded.at[3, :, :, 0].set(-U_padded[3, :, :, 0])
        U_padded = U_padded.at[3, :, :, -1].set(-U_padded[3, :, :, -1])

    return U_padded
