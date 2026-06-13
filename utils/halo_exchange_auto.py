"""
Halo exchange for automatic sharding (NamedSharding + with_sharding_constraint).

This version uses operations that JAX can automatically handle across sharded arrays,
unlike the shard_map version which uses explicit ppermute.
"""

import jax
import jax.numpy as jnp


def _reshard(x, sharding):
    """No-op if no sharding given; otherwise pin x to the requested layout.

    Used in the 3D periodic halo to stop XLA from collapsing the sharded
    axes between concatenations (which caused a ~10 GB-per-shard scratch
    blowup at 1024^3 fp32 / 8 GPUs).
    """
    if sharding is None:
        return x
    return jax.lax.with_sharding_constraint(x, sharding)


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
    # Pin the input's sharding through each axis expansion so XLA does not
    # drop spec mid-pad (was materialising a single-axis-sharded ~10 GB
    # intermediate per shard at 1024^3 fp32 / 8 GPUs).
    s = getattr(U, "sharding", None)
    if boundary == 0:  # Periodic — wrap-around
        U_padded = jnp.concatenate([U[:, -1:, :, :], U, U[:, 0:1, :, :]], axis=1)
        U_padded = _reshard(U_padded, s)
        U_padded = jnp.concatenate([U_padded[:, :, -1:, :], U_padded, U_padded[:, :, 0:1, :]], axis=2)
        U_padded = _reshard(U_padded, s)
        U_padded = jnp.concatenate([U_padded[:, :, :, -1:], U_padded, U_padded[:, :, :, 0:1]], axis=3)
        U_padded = _reshard(U_padded, s)

    elif boundary == 1:  # Zero-gradient
        U_padded = jnp.concatenate([U[:, 0:1, :, :], U, U[:, -1:, :, :]], axis=1)
        U_padded = _reshard(U_padded, s)
        U_padded = jnp.concatenate([U_padded[:, :, 0:1, :], U_padded, U_padded[:, :, -1:, :]], axis=2)
        U_padded = _reshard(U_padded, s)
        U_padded = jnp.concatenate([U_padded[:, :, :, 0:1], U_padded, U_padded[:, :, :, -1:]], axis=3)
        U_padded = _reshard(U_padded, s)

    elif boundary == 4:  # Diode outflow: zero-gradient, but no inflow.
        # Plain zero-gradient ghosts copy the edge cell's velocity, so under
        # an inward pull (e.g. self-gravity of a central cloud) the boundary
        # becomes an infinite mass reservoir — gas streams IN forever
        # (measured: ×5.7 box mass by t=2 in the isolated BB79 run, with
        # axis-aligned inflow jets at the face centres). The diode keeps the
        # zero-gradient ghosts but clamps the ghost NORMAL velocity against
        # inflow: gas may leave, nothing may enter. Operates on PRIMITIVES
        # (index 1/2/3 = vx/vy/vz in 3D).
        U_padded = jnp.concatenate([U[:, 0:1, :, :], U, U[:, -1:, :, :]], axis=1)
        U_padded = _reshard(U_padded, s)
        U_padded = jnp.concatenate([U_padded[:, :, 0:1, :], U_padded, U_padded[:, :, -1:, :]], axis=2)
        U_padded = _reshard(U_padded, s)
        U_padded = jnp.concatenate([U_padded[:, :, :, 0:1], U_padded, U_padded[:, :, :, -1:]], axis=3)
        U_padded = _reshard(U_padded, s)
        # Low face: inflow = positive normal velocity → clamp to ≤ 0.
        # High face: inflow = negative normal velocity → clamp to ≥ 0.
        U_padded = U_padded.at[1, 0, :, :].set(jnp.minimum(U_padded[1, 0, :, :], 0.0))
        U_padded = U_padded.at[1, -1, :, :].set(jnp.maximum(U_padded[1, -1, :, :], 0.0))
        U_padded = U_padded.at[2, :, 0, :].set(jnp.minimum(U_padded[2, :, 0, :], 0.0))
        U_padded = U_padded.at[2, :, -1, :].set(jnp.maximum(U_padded[2, :, -1, :], 0.0))
        U_padded = U_padded.at[3, :, :, 0].set(jnp.minimum(U_padded[3, :, :, 0], 0.0))
        U_padded = U_padded.at[3, :, :, -1].set(jnp.maximum(U_padded[3, :, :, -1], 0.0))

    else:  # Reflective (edge then negate normal momenta)
        U_padded = jnp.concatenate([U[:, 0:1, :, :], U, U[:, -1:, :, :]], axis=1)
        U_padded = _reshard(U_padded, s)
        U_padded = jnp.concatenate([U_padded[:, :, 0:1, :], U_padded, U_padded[:, :, -1:, :]], axis=2)
        U_padded = _reshard(U_padded, s)
        U_padded = jnp.concatenate([U_padded[:, :, :, 0:1], U_padded, U_padded[:, :, :, -1:]], axis=3)
        U_padded = _reshard(U_padded, s)

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
