"""
Halo exchange for sharded arrays using JAX collectives.

This version works with jax.sharding for multi-node parallelism.
Uses explicit mesh topology and collective operations.
"""
import jax
import jax.numpy as jnp
from jax.experimental import shard_map
from jax.sharding import PartitionSpec as P


def create_halo_exchange_2d(mesh, boundary):
    """
    Create a halo exchange function for 2D sharded arrays.

    Parameters
    ----------
    mesh : jax.sharding.Mesh
        Device mesh with axes ('x', 'y')
    boundary : int
        Boundary condition type (0=periodic, 1=zero-gradient, 2=reflective)

    Returns
    -------
    exchange_halos : callable
        Function that takes sharded array and returns padded version
    """

    def exchange_halos_impl(U_local):
        """
        Exchange halos for a local subdomain.

        Must be called inside shard_map.

        Parameters
        ----------
        U_local : array (Nvar, nx_local, ny_local)
            Local subdomain WITHOUT ghost cells

        Returns
        -------
        U_padded : array (Nvar, nx_local+2, ny_local+2)
            Local subdomain WITH ghost cells
        """
        # Get process coordinates from axis_index
        px = jax.lax.axis_index('x')
        py = jax.lax.axis_index('y')

        nx_proc = jax.lax.psum(1, 'x')
        ny_proc = jax.lax.psum(1, 'y')

        # Determine neighbors
        has_left = px > 0
        has_right = px < nx_proc - 1
        has_bottom = py > 0
        has_top = py < ny_proc - 1

        # Extract boundary slices to send
        left_send = U_local[:, 0:1, :]    # (Nvar, 1, ny_local)
        right_send = U_local[:, -1:, :]   # (Nvar, 1, ny_local)
        bottom_send = U_local[:, :, 0:1]  # (Nvar, nx_local, 1)
        top_send = U_local[:, :, -1:]     # (Nvar, nx_local, 1)

        # X-direction exchange (use 'x' axis)
        # ppermute shifts data along the mesh axis
        # For left neighbor: shift right (+1 along x axis)
        # For right neighbor: shift left (-1 along x axis)

        # Left ghost (receive from left, which sends its right)
        def get_left_ghost_internal():
            # Shift data from left neighbor (px-1 sends to px)
            perm = [(i, (i + 1) % nx_proc) for i in range(nx_proc)]
            return jax.lax.ppermute(right_send, 'x', perm=perm)

        def get_left_ghost_boundary():
            if boundary == 0:  # Periodic - use ppermute with wraparound
                perm = [(i, (i + 1) % nx_proc) for i in range(nx_proc)]
                return jax.lax.ppermute(right_send, 'x', perm=perm)
            elif boundary == 1:  # Zero-gradient
                return left_send
            else:  # Reflective
                return left_send.at[1].set(-left_send[1])

        left_ghost = jax.lax.cond(
            jax.lax.select(boundary == 0, True, has_left),
            get_left_ghost_internal,
            get_left_ghost_boundary
        )

        # Right ghost (receive from right, which sends its left)
        def get_right_ghost_internal():
            # Shift data from right neighbor (px+1 sends to px)
            perm = [((i + 1) % nx_proc, i) for i in range(nx_proc)]
            return jax.lax.ppermute(left_send, 'x', perm=perm)

        def get_right_ghost_boundary():
            if boundary == 0:  # Periodic
                perm = [((i + 1) % nx_proc, i) for i in range(nx_proc)]
                return jax.lax.ppermute(left_send, 'x', perm=perm)
            elif boundary == 1:  # Zero-gradient
                return right_send
            else:  # Reflective
                return right_send.at[1].set(-right_send[1])

        right_ghost = jax.lax.cond(
            jax.lax.select(boundary == 0, True, has_right),
            get_right_ghost_internal,
            get_right_ghost_boundary
        )

        # Pad in x-direction first
        U_x = jnp.concatenate([left_ghost, U_local, right_ghost], axis=1)

        # Y-direction exchange (use 'y' axis)
        # Bottom ghost (receive from bottom, which sends its top)
        def get_bottom_ghost_internal():
            perm = [(i, (i + 1) % ny_proc) for i in range(ny_proc)]
            return jax.lax.ppermute(top_send, 'y', perm=perm)

        def get_bottom_ghost_boundary():
            if boundary == 0:  # Periodic
                perm = [(i, (i + 1) % ny_proc) for i in range(ny_proc)]
                return jax.lax.ppermute(top_send, 'y', perm=perm)
            elif boundary == 1:  # Zero-gradient
                return bottom_send
            else:  # Reflective
                return bottom_send.at[2].set(-bottom_send[2])

        bottom_ghost = jax.lax.cond(
            jax.lax.select(boundary == 0, True, has_bottom),
            get_bottom_ghost_internal,
            get_bottom_ghost_boundary
        )

        # Top ghost (receive from top, which sends its bottom)
        def get_top_ghost_internal():
            perm = [((i + 1) % ny_proc, i) for i in range(ny_proc)]
            return jax.lax.ppermute(bottom_send, 'y', perm=perm)

        def get_top_ghost_boundary():
            if boundary == 0:  # Periodic
                perm = [((i + 1) % ny_proc, i) for i in range(ny_proc)]
                return jax.lax.ppermute(bottom_send, 'y', perm=perm)
            elif boundary == 1:  # Zero-gradient
                return top_send
            else:  # Reflective
                return top_send.at[2].set(-top_send[2])

        top_ghost = jax.lax.cond(
            jax.lax.select(boundary == 0, True, has_top),
            get_top_ghost_internal,
            get_top_ghost_boundary
        )

        # Handle corners (similar to pmap version)
        at_bottom_left = jnp.logical_and(jnp.logical_not(has_left), jnp.logical_not(has_bottom))
        at_bottom_right = jnp.logical_and(jnp.logical_not(has_right), jnp.logical_not(has_bottom))
        at_top_left = jnp.logical_and(jnp.logical_not(has_left), jnp.logical_not(has_top))
        at_top_right = jnp.logical_and(jnp.logical_not(has_right), jnp.logical_not(has_top))

        use_local_corners = (boundary == 1)

        corner_bl = jnp.where(
            jnp.logical_and(at_bottom_left, use_local_corners),
            U_local[:, 0:1, 0:1],
            left_ghost[:, :, 0:1]
        )

        corner_br = jnp.where(
            jnp.logical_and(at_bottom_right, use_local_corners),
            U_local[:, -1:, 0:1],
            right_ghost[:, :, 0:1]
        )

        corner_tl = jnp.where(
            jnp.logical_and(at_top_left, use_local_corners),
            U_local[:, 0:1, -1:],
            left_ghost[:, :, -1:]
        )

        corner_tr = jnp.where(
            jnp.logical_and(at_top_right, use_local_corners),
            U_local[:, -1:, -1:],
            right_ghost[:, :, -1:]
        )

        bottom_extended = jnp.concatenate([corner_bl, bottom_ghost, corner_br], axis=1)
        top_extended = jnp.concatenate([corner_tl, top_ghost, corner_tr], axis=1)

        # Pad in y-direction
        U_padded = jnp.concatenate([bottom_extended, U_x, top_extended], axis=2)

        return U_padded

    # Return the exchange function
    # This will be used inside shard_map
    return exchange_halos_impl


def create_halo_exchange_1d(mesh, boundary):
    """
    Create a halo exchange function for 1D sharded arrays.

    Parameters
    ----------
    mesh : jax.sharding.Mesh
        Device mesh with axis ('x',)
    boundary : int
        Boundary condition type

    Returns
    -------
    exchange_halos : callable
        Function that takes sharded array and returns padded version
    """

    def exchange_halos_impl(U_local):
        """
        Exchange halos for 1D local subdomain.

        Parameters
        ----------
        U_local : array (Nvar, nx_local)

        Returns
        -------
        U_padded : array (Nvar, nx_local+2)
        """
        px = jax.lax.axis_index('x')
        nx_proc = jax.lax.psum(1, 'x')

        has_left = px > 0
        has_right = px < nx_proc - 1

        left_send = U_local[:, 0:1]
        right_send = U_local[:, -1:]

        # Left ghost
        def get_left_ghost_internal():
            perm = [(i, (i + 1) % nx_proc) for i in range(nx_proc)]
            return jax.lax.ppermute(right_send, 'x', perm=perm)

        def get_left_ghost_boundary():
            if boundary == 0:  # Periodic
                perm = [(i, (i + 1) % nx_proc) for i in range(nx_proc)]
                return jax.lax.ppermute(right_send, 'x', perm=perm)
            elif boundary == 1:  # Zero-gradient
                return left_send
            else:  # Reflective
                return left_send.at[1].set(-left_send[1])

        left_ghost = jax.lax.cond(
            jax.lax.select(boundary == 0, True, has_left),
            get_left_ghost_internal,
            get_left_ghost_boundary
        )

        # Right ghost
        def get_right_ghost_internal():
            perm = [((i + 1) % nx_proc, i) for i in range(nx_proc)]
            return jax.lax.ppermute(left_send, 'x', perm=perm)

        def get_right_ghost_boundary():
            if boundary == 0:  # Periodic
                perm = [((i + 1) % nx_proc, i) for i in range(nx_proc)]
                return jax.lax.ppermute(left_send, 'x', perm=perm)
            elif boundary == 1:  # Zero-gradient
                return right_send
            else:  # Reflective
                return right_send.at[1].set(-right_send[1])

        right_ghost = jax.lax.cond(
            jax.lax.select(boundary == 0, True, has_right),
            get_right_ghost_internal,
            get_right_ghost_boundary
        )

        return jnp.concatenate([left_ghost, U_local, right_ghost], axis=1)

    return exchange_halos_impl
