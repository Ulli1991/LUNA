"""
MPI flux function utilities for multi-GPU execution.

This module provides functions to create flux functions that work correctly
with MPI halo exchange, handling the case where ghost cells come from
neighboring ranks rather than local padding.
"""

import jax.numpy as jnp


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
