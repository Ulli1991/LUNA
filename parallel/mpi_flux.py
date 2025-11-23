"""
MPI flux function utilities for multi-GPU execution.

This module provides functions to create flux functions that work correctly
with MPI halo exchange, handling the case where ghost cells come from
neighboring ranks rather than local padding.
"""

import jax
import jax.numpy as jnp
from functools import partial


def create_mpi_flux_fn_fast(flux_divergence_fn, patch, cfg):
    """
    OPTIMIZED flux function using concatenation instead of .at[].set().

    For X-decomposition only: uses concatenation for X boundaries (fast)
    and jnp.pad for Y/Z boundaries (optimized JAX op).

    This avoids the expensive .at[interior].set() that copies the entire array.

    Returns:
        flux_fn: function(U_interior, left_mpi_ghost, right_mpi_ghost) -> dU
        is_x_left_domain: bool - True if left X boundary is domain (not MPI)
        is_x_right_domain: bool - True if right X boundary is domain (not MPI)
    """
    ndim = cfg.SPATIAL_DIMENSION
    ng = patch.n_ghost  # Should be 2

    # Determine boundary types for X (can be MPI or domain)
    is_x_left_domain = patch.is_boundary('left')
    is_x_right_domain = patch.is_boundary('right')

    # For X-only decomposition, Y and Z are always domain boundaries
    # We'll use jnp.pad for those

    if ndim == 3:
        @partial(jax.jit)
        def mpi_flux_fn(U_interior, left_mpi_ghost, right_mpi_ghost):
            """
            Fast flux function using concatenation for X, jnp.pad for Y/Z.

            U_interior: (nvar, nx, ny, nz) - interior cells
            left_mpi_ghost: (nvar, ng, ny, nz) - left X ghost from MPI (or dummy if domain BC)
            right_mpi_ghost: (nvar, ng, ny, nz) - right X ghost from MPI (or dummy if domain BC)

            For domain boundaries, ghost cells are computed from U_interior.
            For MPI boundaries, ghost cells come from left_mpi_ghost/right_mpi_ghost.
            """
            # Compute left X ghost cells
            if is_x_left_domain:
                # Domain BC: replicate first interior slice
                left_ghost = jnp.broadcast_to(
                    U_interior[:, :1, :, :],
                    (U_interior.shape[0], ng, U_interior.shape[2], U_interior.shape[3])
                )
            else:
                # MPI boundary: use provided ghost cells
                left_ghost = left_mpi_ghost

            # Compute right X ghost cells
            if is_x_right_domain:
                # Domain BC: replicate last interior slice
                right_ghost = jnp.broadcast_to(
                    U_interior[:, -1:, :, :],
                    (U_interior.shape[0], ng, U_interior.shape[2], U_interior.shape[3])
                )
            else:
                # MPI boundary: use provided ghost cells
                right_ghost = right_mpi_ghost

            # Build X-padded array with concatenation (fast!)
            U_x_padded = jnp.concatenate([left_ghost, U_interior, right_ghost], axis=1)

            # Pad Y and Z with edge mode (domain BCs, optimized JAX op)
            U_padded = jnp.pad(U_x_padded,
                              ((0, 0), (0, 0), (ng, ng), (ng, ng)),
                              mode='edge')

            return flux_divergence_fn(U_padded)

        return mpi_flux_fn, is_x_left_domain, is_x_right_domain

    elif ndim == 2:
        @partial(jax.jit)
        def mpi_flux_fn(U_interior, left_mpi_ghost, right_mpi_ghost):
            if is_x_left_domain:
                left_ghost = jnp.broadcast_to(
                    U_interior[:, :1, :],
                    (U_interior.shape[0], ng, U_interior.shape[2])
                )
            else:
                left_ghost = left_mpi_ghost

            if is_x_right_domain:
                right_ghost = jnp.broadcast_to(
                    U_interior[:, -1:, :],
                    (U_interior.shape[0], ng, U_interior.shape[2])
                )
            else:
                right_ghost = right_mpi_ghost

            U_x_padded = jnp.concatenate([left_ghost, U_interior, right_ghost], axis=1)
            U_padded = jnp.pad(U_x_padded, ((0, 0), (0, 0), (ng, ng)), mode='edge')
            return flux_divergence_fn(U_padded)

        return mpi_flux_fn, is_x_left_domain, is_x_right_domain

    else:  # 1D
        @partial(jax.jit)
        def mpi_flux_fn(U_interior, left_mpi_ghost, right_mpi_ghost):
            if is_x_left_domain:
                left_ghost = jnp.broadcast_to(
                    U_interior[:, :1],
                    (U_interior.shape[0], ng)
                )
            else:
                left_ghost = left_mpi_ghost

            if is_x_right_domain:
                right_ghost = jnp.broadcast_to(
                    U_interior[:, -1:],
                    (U_interior.shape[0], ng)
                )
            else:
                right_ghost = right_mpi_ghost

            U_padded = jnp.concatenate([left_ghost, U_interior, right_ghost], axis=1)
            return flux_divergence_fn(U_padded)

        return mpi_flux_fn, is_x_left_domain, is_x_right_domain


def create_mpi_flux_fn(flux_divergence_fn, patch, cfg):
    """
    Create a flux function for MPI that uses ghost cells from halo exchange.

    IMPORTANT: This returns a function that takes BOTH U_interior AND U_with_ghost
    as arguments. This avoids closure capture of patch.data, which would cause
    stale ghost cells when JIT compiled (JAX captures closure values at trace time).

    The returned function signature is: mpi_flux_fn(U_interior, U_with_ghost)

    NOTE: U_with_ghost should have 2 ghost cells per side for proper PLM reconstruction
    at MPI boundaries. With 2 ghost cells, we can compute slopes OF the ghost cells,
    enabling symmetric flux computation at inter-patch interfaces.

    CRITICAL FIX: After setting U_interior, we re-apply domain boundary conditions
    to ghost cells at PHYSICAL boundaries (not MPI boundaries). During RK sub-steps,
    interior is updated but ghost cells are stale, creating discontinuities that
    cause spurious transverse fluxes.
    """
    ndim = cfg.SPATIAL_DIMENSION
    ng = patch.n_ghost  # Should be 2
    boundary_type = cfg.BOUNDARY

    # Determine which faces are PHYSICAL domain boundaries (not MPI boundaries)
    is_x_left_boundary = patch.is_boundary('left')
    is_x_right_boundary = patch.is_boundary('right')
    is_y_bottom_boundary = patch.is_boundary('bottom') if ndim >= 2 else False
    is_y_top_boundary = patch.is_boundary('top') if ndim >= 2 else False
    is_z_back_boundary = patch.is_boundary('back') if ndim == 3 else False
    is_z_front_boundary = patch.is_boundary('front') if ndim == 3 else False

    # Slice to extract interior from 2-ghost array
    if ndim == 1:
        interior_in_2g = (slice(None), slice(ng, -ng))
    elif ndim == 2:
        interior_in_2g = (slice(None), slice(ng, -ng), slice(ng, -ng))
    else:  # 3D
        interior_in_2g = (slice(None), slice(ng, -ng), slice(ng, -ng), slice(ng, -ng))

    def mpi_flux_fn(U_interior, U_with_ghost):
        """
        Flux function that combines U_interior with ghost cells.

        U_interior: interior data (no ghost cells) - may be intermediate RK state
        U_with_ghost: array with 2 ghost cells on each side (from patch.data)
        Returns: flux divergence with same shape as U_interior

        OPTIMIZATION: Uses slice-based updates to set all ghost cells at once,
        reducing the number of .at[].set() operations from 12 to 6 for 3D.
        """
        # Replace interior with U_interior (handles intermediate RK states)
        U_combined = U_with_ghost.at[interior_in_2g].set(U_interior)

        # Re-apply domain BCs at PHYSICAL boundaries (not MPI boundaries)
        # Use slice-based updates to set both ghost layers at once (faster than individual sets)
        if boundary_type == 1:  # Zero-gradient (Neumann)
            if ndim == 3:
                # Y boundaries - broadcast interior slice to fill all ghost cells
                if is_y_bottom_boundary:
                    interior_val = U_combined[:, :, ng:ng+1, :]  # Keep dim for broadcast
                    U_combined = U_combined.at[:, :, :ng, :].set(
                        jnp.broadcast_to(interior_val, U_combined[:, :, :ng, :].shape))
                if is_y_top_boundary:
                    interior_val = U_combined[:, :, -ng-1:-ng, :]
                    U_combined = U_combined.at[:, :, -ng:, :].set(
                        jnp.broadcast_to(interior_val, U_combined[:, :, -ng:, :].shape))
                # Z boundaries
                if is_z_back_boundary:
                    interior_val = U_combined[:, :, :, ng:ng+1]
                    U_combined = U_combined.at[:, :, :, :ng].set(
                        jnp.broadcast_to(interior_val, U_combined[:, :, :, :ng].shape))
                if is_z_front_boundary:
                    interior_val = U_combined[:, :, :, -ng-1:-ng]
                    U_combined = U_combined.at[:, :, :, -ng:].set(
                        jnp.broadcast_to(interior_val, U_combined[:, :, :, -ng:].shape))
                # X boundaries
                if is_x_left_boundary:
                    interior_val = U_combined[:, ng:ng+1, :, :]
                    U_combined = U_combined.at[:, :ng, :, :].set(
                        jnp.broadcast_to(interior_val, U_combined[:, :ng, :, :].shape))
                if is_x_right_boundary:
                    interior_val = U_combined[:, -ng-1:-ng, :, :]
                    U_combined = U_combined.at[:, -ng:, :, :].set(
                        jnp.broadcast_to(interior_val, U_combined[:, -ng:, :, :].shape))
            elif ndim == 2:
                if is_y_bottom_boundary:
                    interior_val = U_combined[:, :, ng:ng+1]
                    U_combined = U_combined.at[:, :, :ng].set(
                        jnp.broadcast_to(interior_val, U_combined[:, :, :ng].shape))
                if is_y_top_boundary:
                    interior_val = U_combined[:, :, -ng-1:-ng]
                    U_combined = U_combined.at[:, :, -ng:].set(
                        jnp.broadcast_to(interior_val, U_combined[:, :, -ng:].shape))
                if is_x_left_boundary:
                    interior_val = U_combined[:, ng:ng+1, :]
                    U_combined = U_combined.at[:, :ng, :].set(
                        jnp.broadcast_to(interior_val, U_combined[:, :ng, :].shape))
                if is_x_right_boundary:
                    interior_val = U_combined[:, -ng-1:-ng, :]
                    U_combined = U_combined.at[:, -ng:, :].set(
                        jnp.broadcast_to(interior_val, U_combined[:, -ng:, :].shape))
            elif ndim == 1:
                if is_x_left_boundary:
                    interior_val = U_combined[:, ng:ng+1]
                    U_combined = U_combined.at[:, :ng].set(
                        jnp.broadcast_to(interior_val, U_combined[:, :ng].shape))
                if is_x_right_boundary:
                    interior_val = U_combined[:, -ng-1:-ng]
                    U_combined = U_combined.at[:, -ng:].set(
                        jnp.broadcast_to(interior_val, U_combined[:, -ng:].shape))

        return flux_divergence_fn(U_combined)

    return mpi_flux_fn


def get_ghost_slices(patch, cfg):
    """
    Get slices for extracting data with ALL ghost cells (2 per side).

    For proper PLM reconstruction at MPI boundaries, we need 2 ghost cells
    so we can compute slopes OF the ghost cells, enabling proper reconstruction
    at inter-patch interfaces.

    With only 1 ghost cell, the flux at subdomain boundaries uses cell-center
    values for one side (from ghost) and reconstructed values for the other
    (from interior), causing asymmetric flux computation between ranks.
    """
    # Return slice that keeps all data (all ghost cells)
    ndim = cfg.SPATIAL_DIMENSION

    if ndim == 1:
        return (slice(None), slice(None))
    elif ndim == 2:
        return (slice(None), slice(None), slice(None))
    else:  # 3D
        return (slice(None), slice(None), slice(None), slice(None))
