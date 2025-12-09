"""
Simplified halo exchange for sharded arrays - works inside shard_map.

This version uses a simpler approach that's easier to debug.
"""
import jax
import jax.numpy as jnp


def pad_with_halo_2d_sharded(U_local, boundary):
    """
    Pad 2D local array with halo exchange for sharded arrays.

    Must be called inside shard_map with mesh axes ('x', 'y').

    Parameters
    ----------
    U_local : array (Nvar, nx_local, ny_local)
        Local subdomain WITHOUT ghost cells
    boundary : int
        0=periodic, 1=zero-gradient, 2=reflective

    Returns
    -------
    U_padded : array (Nvar, nx_local+2, ny_local+2)
        Local subdomain WITH ghost cells
    """
    # Get mesh coordinates
    px = jax.lax.axis_index('x')
    py = jax.lax.axis_index('y')

    nx_proc = jax.lax.psum(1, 'x')
    ny_proc = jax.lax.psum(1, 'y')

    # Extract boundary slices
    left_send = U_local[:, 0:1, :]
    right_send = U_local[:, -1:, :]
    bottom_send = U_local[:, :, 0:1]
    top_send = U_local[:, :, -1:]

    # X-direction exchange using ppermute (efficient neighbor-to-neighbor)
    if boundary == 0:  # Periodic
        # Wraparound communication
        left_perm = [(i, (i + 1) % nx_proc) for i in range(nx_proc)]
        right_perm = [((i + 1) % nx_proc, i) for i in range(nx_proc)]
        left_ghost = jax.lax.ppermute(right_send, 'x', perm=left_perm)
        right_ghost = jax.lax.ppermute(left_send, 'x', perm=right_perm)
    else:
        # Non-periodic: use ppermute for interior, apply BC at boundaries
        # Build permutations using list comprehensions (more efficient)

        # Left ghost: interior processes receive from left neighbor (i-1),
        # leftmost process (i=0) receives from itself
        left_perm = [((i-1) if i > 0 else i, i) for i in range(nx_proc)]
        left_ghost = jax.lax.ppermute(right_send, 'x', perm=left_perm)

        # Right ghost: interior processes receive from right neighbor (i+1),
        # rightmost process receives from itself
        right_perm = [((i+1) if i < nx_proc-1 else i, i) for i in range(nx_proc)]
        right_ghost = jax.lax.ppermute(left_send, 'x', perm=right_perm)

        # Apply boundary conditions at physical boundaries
        at_left = px == 0
        at_right = px == nx_proc - 1

        if boundary == 1:  # Zero-gradient
            # Already correct - ghost = boundary
            pass
        elif boundary == 2:  # Reflective
            # Negate x-momentum at boundaries
            left_ghost = jnp.where(at_left, left_ghost.at[1].set(-left_ghost[1]), left_ghost)
            right_ghost = jnp.where(at_right, right_ghost.at[1].set(-right_ghost[1]), right_ghost)

    # Pad in x
    U_x = jnp.concatenate([left_ghost, U_local, right_ghost], axis=1)

    # Y-direction exchange using ppermute
    if boundary == 0:  # Periodic
        bottom_perm = [(i, (i + 1) % ny_proc) for i in range(ny_proc)]
        top_perm = [((i + 1) % ny_proc, i) for i in range(ny_proc)]
        bottom_ghost = jax.lax.ppermute(top_send, 'y', perm=bottom_perm)
        top_ghost = jax.lax.ppermute(bottom_send, 'y', perm=top_perm)
    else:
        # Non-periodic: use ppermute for interior, apply BC at boundaries

        # Bottom ghost: interior processes receive from bottom neighbor (i-1),
        # bottommost process receives from itself
        bottom_perm = [((i-1) if i > 0 else i, i) for i in range(ny_proc)]
        bottom_ghost = jax.lax.ppermute(top_send, 'y', perm=bottom_perm)

        # Top ghost: interior processes receive from top neighbor (i+1),
        # topmost process receives from itself
        top_perm = [((i+1) if i < ny_proc-1 else i, i) for i in range(ny_proc)]
        top_ghost = jax.lax.ppermute(bottom_send, 'y', perm=top_perm)

        # Apply boundary conditions at physical boundaries
        at_bottom = py == 0
        at_top = py == ny_proc - 1

        if boundary == 1:  # Zero-gradient
            # Already correct
            pass
        elif boundary == 2:  # Reflective
            # Negate y-momentum at boundaries
            bottom_ghost = jnp.where(at_bottom, bottom_ghost.at[2].set(-bottom_ghost[2]), bottom_ghost)
            top_ghost = jnp.where(at_top, top_ghost.at[2].set(-top_ghost[2]), top_ghost)

    # Handle corners (simple version - just use edge values)
    # Get the x-padded array's edges for corners
    left_col = U_x[:, 0:1, :]
    right_col = U_x[:, -1:, :]

    corner_bl = left_col[:, :, 0:1]
    corner_br = right_col[:, :, 0:1]
    corner_tl = left_col[:, :, -1:]
    corner_tr = right_col[:, :, -1:]

    # Extend bottom/top with corners
    bottom_extended = jnp.concatenate([corner_bl, bottom_ghost, corner_br], axis=1)
    top_extended = jnp.concatenate([corner_tl, top_ghost, corner_tr], axis=1)

    # Final padding in y
    U_padded = jnp.concatenate([bottom_extended, U_x, top_extended], axis=2)

    return U_padded


def pad_with_halo_1d_sharded(U_local, boundary):
    """
    Pad 1D local array with halo exchange for sharded arrays.

    Must be called inside shard_map with mesh axis ('x',).

    Parameters
    ----------
    U_local : array (Nvar, nx_local)
        Local subdomain WITHOUT ghost cells
    boundary : int
        Boundary condition type

    Returns
    -------
    U_padded : array (Nvar, nx_local+2)
        Local subdomain WITH ghost cells
    """
    px = jax.lax.axis_index('x')
    nx_proc = jax.lax.psum(1, 'x')

    left_send = U_local[:, 0:1]
    right_send = U_local[:, -1:]

    if boundary == 0:  # Periodic
        left_perm = [(i, (i + 1) % nx_proc) for i in range(nx_proc)]
        right_perm = [((i + 1) % nx_proc, i) for i in range(nx_proc)]
        left_ghost = jax.lax.ppermute(right_send, 'x', perm=left_perm)
        right_ghost = jax.lax.ppermute(left_send, 'x', perm=right_perm)
    else:
        # Non-periodic: use ppermute for neighbor exchange

        # Use list comprehensions for efficient permutation construction
        left_perm = [((i-1) if i > 0 else i, i) for i in range(nx_proc)]
        left_ghost = jax.lax.ppermute(right_send, 'x', perm=left_perm)

        right_perm = [((i+1) if i < nx_proc-1 else i, i) for i in range(nx_proc)]
        right_ghost = jax.lax.ppermute(left_send, 'x', perm=right_perm)

        # Apply boundary conditions at physical boundaries
        at_left = px == 0
        at_right = px == nx_proc - 1

        if boundary == 1:  # Zero-gradient
            # Already correct
            pass
        elif boundary == 2:  # Reflective
            # Negate x-momentum at boundaries
            left_ghost = jnp.where(at_left, left_ghost.at[1].set(-left_ghost[1]), left_ghost)
            right_ghost = jnp.where(at_right, right_ghost.at[1].set(-right_ghost[1]), right_ghost)

    return jnp.concatenate([left_ghost, U_local, right_ghost], axis=1)
