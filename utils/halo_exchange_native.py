"""
Halo exchange for single-process multi-GPU JAX execution using native sharding.

This module implements ghost cell communication between devices in a single process
using JAX's shard_map and lax.ppermute for collective communication.
"""
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
from functools import partial


def pad_with_halo_2d_native(U_sharded, boundary, mesh, sharding):
    """
    Pad 2D sharded array with ghost cells using native JAX halo exchange.

    Works in single-process multi-GPU mode where data is sharded across devices.
    Uses shard_map with lax.ppermute for device-to-device communication.

    Parameters
    ----------
    U_sharded : sharded array
        Sharded array, global shape (Nvar, Nx_global, Ny_global)
        Each device holds a local shard of shape (Nvar, Nx_local, Ny_local)
    boundary : int
        Physical boundary condition: 0=periodic, 1=zero-gradient, 2=reflective
    mesh : jax.sharding.Mesh
        Device mesh (e.g., 2x2 grid)
    sharding : jax.sharding.NamedSharding
        Sharding specification

    Returns
    -------
    U_pad : sharded array
        Padded array where each local shard has shape (Nvar, Nx_local+2, Ny_local+2)
    """
    mesh_shape = mesh.shape
    n_devices_x = mesh_shape['x']
    n_devices_y = mesh_shape['y']

    @partial(shard_map, mesh=mesh, in_specs=P('x', 'y'), out_specs=P('x', 'y'), check_rep=False)
    def per_device_pad(U_local):
        """
        Pad a single device's shard by exchanging halos with neighbors.
        This function runs on each device independently with its local shard.
        """
        # Get logical device indices in the mesh
        # lax.axis_index returns the index of the current device along a mesh axis
        dev_x = lax.axis_index('x')
        dev_y = lax.axis_index('y')

        # Determine if at physical boundary
        at_left = (dev_x == 0)
        at_right = (dev_x == n_devices_x - 1)
        at_bottom = (dev_y == 0)
        at_top = (dev_y == n_devices_y - 1)

        # ====================================================================
        # X-DIRECTION HALO EXCHANGE
        # ====================================================================

        # Extract boundaries to exchange
        left_boundary = U_local[:, 0:1, :]    # Leftmost column
        right_boundary = U_local[:, -1:, :]   # Rightmost column

        # Exchange with left neighbor (send right, receive left)
        # ppermute with axis_name 'x' communicates along the x-axis of the mesh
        # perm=[(i, (i-1)%N)] means "I'm at position i, send to position (i-1) mod N"
        if boundary == 0:  # Periodic
            # Shift right: each device sends its right boundary to the right neighbor
            # and receives from left neighbor
            left_ghost = lax.ppermute(
                right_boundary, 'x',
                perm=[(i, (i - 1) % n_devices_x) for i in range(n_devices_x)]
            )
            right_ghost = lax.ppermute(
                left_boundary, 'x',
                perm=[(i, (i + 1) % n_devices_x) for i in range(n_devices_x)]
            )
        else:  # Non-periodic: use physical BC at edges
            # Conditional based on position
            left_ghost = lax.cond(
                at_left,
                lambda _: apply_bc_2d_left_native(U_local, boundary),
                lambda _: lax.ppermute(
                    right_boundary, 'x',
                    perm=[(i, (i - 1) % n_devices_x) for i in range(n_devices_x)]
                ),
                None
            )
            right_ghost = lax.cond(
                at_right,
                lambda _: apply_bc_2d_right_native(U_local, boundary),
                lambda _: lax.ppermute(
                    left_boundary, 'x',
                    perm=[(i, (i + 1) % n_devices_x) for i in range(n_devices_x)]
                ),
                None
            )

        # Concatenate in x
        U_x = jnp.concatenate([left_ghost, U_local, right_ghost], axis=1)
        # Shape: (Nvar, Nx_local+2, Ny_local)

        # ====================================================================
        # Y-DIRECTION HALO EXCHANGE (on x-padded array)
        # ====================================================================

        bottom_boundary = U_x[:, :, 0:1]   # Bottom row
        top_boundary = U_x[:, :, -1:]      # Top row

        if boundary == 0:  # Periodic
            bottom_ghost = lax.ppermute(
                top_boundary, 'y',
                perm=[(j, (j - 1) % n_devices_y) for j in range(n_devices_y)]
            )
            top_ghost = lax.ppermute(
                bottom_boundary, 'y',
                perm=[(j, (j + 1) % n_devices_y) for j in range(n_devices_y)]
            )
        else:  # Non-periodic
            bottom_ghost = lax.cond(
                at_bottom,
                lambda _: apply_bc_2d_bottom_native(U_x, boundary),
                lambda _: lax.ppermute(
                    top_boundary, 'y',
                    perm=[(j, (j - 1) % n_devices_y) for j in range(n_devices_y)]
                ),
                None
            )
            top_ghost = lax.cond(
                at_top,
                lambda _: apply_bc_2d_top_native(U_x, boundary),
                lambda _: lax.ppermute(
                    bottom_boundary, 'y',
                    perm=[(j, (j + 1) % n_devices_y) for j in range(n_devices_y)]
                ),
                None
            )

        # Concatenate in y
        U_pad = jnp.concatenate([bottom_ghost, U_x, top_ghost], axis=2)
        # Final shape: (Nvar, Nx_local+2, Ny_local+2)

        return U_pad

    # Apply the per-device padding function to the sharded array
    return per_device_pad(U_sharded)


# ============================================================================
# PHYSICAL BOUNDARY CONDITIONS (same as SPMD version but for local shards)
# ============================================================================

def apply_bc_2d_left_native(U, boundary):
    """Apply physical BC on left edge (2D)."""
    if boundary == 0:  # Periodic (shouldn't reach here, handled by ppermute)
        return U[:, -1:, :]
    elif boundary == 1:  # Zero-gradient
        return U[:, 0:1, :]
    elif boundary == 2:  # Reflective
        ghost = U[:, 0:1, :]
        ghost = ghost.at[1].set(-ghost[1])  # Negate rho*u
        return ghost

def apply_bc_2d_right_native(U, boundary):
    """Apply physical BC on right edge (2D)."""
    if boundary == 0:
        return U[:, 0:1, :]
    elif boundary == 1:
        return U[:, -1:, :]
    elif boundary == 2:
        ghost = U[:, -1:, :]
        ghost = ghost.at[1].set(-ghost[1])
        return ghost

def apply_bc_2d_bottom_native(U_x, boundary):
    """Apply physical BC on bottom edge (2D). U_x is already x-padded."""
    if boundary == 0:
        return U_x[:, :, -1:]  # Periodic
    elif boundary == 1:
        return U_x[:, :, 0:1]  # Zero-gradient
    elif boundary == 2:
        ghost = U_x[:, :, 0:1]
        ghost = ghost.at[2].set(-ghost[2])  # Negate rho*v
        return ghost

def apply_bc_2d_top_native(U_x, boundary):
    """Apply physical BC on top edge (2D)."""
    if boundary == 0:
        return U_x[:, :, 0:1]
    elif boundary == 1:
        return U_x[:, :, -1:]
    elif boundary == 2:
        ghost = U_x[:, :, -1:]
        ghost = ghost.at[2].set(-ghost[2])
        return ghost
