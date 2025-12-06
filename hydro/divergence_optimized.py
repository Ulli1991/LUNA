"""
Optimized flux divergence functions with fused periodic padding.

OPTIMIZATION: For periodic boundaries, eliminates explicit padding by using
wraparound indexing with jnp.roll. This saves significant memory traffic.

Expected speedup: 10-15% for periodic boundary conditions.
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

from hydro.slopes import slope_limiter, slope_limiter_smooth
from hydro.fluxes import hllc_flux


@partial(jit, static_argnums=(2, 3, 9, 10))
def flux_divergence_periodic_fused(U, gamma, order, limiter, dx, dy, dz,
                                    inv_dx, inv_dy, inv_dz,
                                    use_smooth_limiters=False, sharpness=10.0):
    """
    Optimized flux divergence for PERIODIC boundaries ONLY.

    Eliminates explicit padding by using wraparound indexing (jnp.roll).
    This removes one full array allocation and associated memory traffic.

    PERFORMANCE:
    - Saves 1 allocation of size (Nvar, Nx+2, Ny+2[, Nz+2])
    - Reduces memory traffic by ~30-50 MB per call (1024² grid)
    - Expected speedup: 10-15%

    LIMITATION: Only works for periodic boundaries (boundary=0)

    Parameters
    ----------
    U : array
        State array (Nvar, Nx) or (Nvar, Nx, Ny) or (Nvar, Nx, Ny, Nz)
    gamma : float
        Adiabatic index
    order : int (static)
        0 = first-order, 1 = second-order PLM
    limiter : int (static)
        Slope limiter type (0=minmod, 1=moncen, 2=vanleer)
    dx, dy, dz : float
        Grid spacing
    inv_dx, inv_dy, inv_dz : float
        Inverse grid spacing (precomputed for speed)
    use_smooth_limiters : bool (static)
        Use differentiable smooth limiters
    sharpness : float (static)
        Sharpness parameter for smooth limiters

    Returns
    -------
    dU/dt : array
        Time derivative (same shape as U)
    """
    ndim = U.ndim - 1

    # ==================================================================
    # 1D CASE
    # ==================================================================
    if ndim == 1:
        if order == 0:
            # First-order: direct flux computation with periodic wraparound
            U_L = U
            U_R = jnp.roll(U, shift=-1, axis=1)  # Right neighbor (wraps at boundary)
            Fx = hllc_flux(U_L, U_R, axis=0, gamma=gamma)
            dFdx = (Fx - jnp.roll(Fx, shift=1, axis=1)) * inv_dx
            return -dFdx
        else:
            # Second-order: compute slopes with periodic wraparound
            U_c = U  # Current cells
            U_L = jnp.roll(U, shift=1, axis=1)   # Left neighbors (periodic)
            U_R = jnp.roll(U, shift=-1, axis=1)  # Right neighbors (periodic)

            # Compute slopes
            dx_L = U_c - U_L
            dx_R = U_R - U_c

            if use_smooth_limiters:
                dUx = slope_limiter_smooth(dx_L, dx_R, limiter, sharpness)
            else:
                dUx = slope_limiter(dx_L, dx_R, limiter)

            # Reconstruct at interfaces (half-slopes computed inline)
            half_dUx = 0.5 * dUx
            UL_x = U_c + half_dUx
            UR_x = jnp.roll(U_c - half_dUx, shift=-1, axis=1)

            # Compute fluxes and divergence
            Fx = hllc_flux(UL_x, UR_x, axis=0, gamma=gamma)
            dFdx = (Fx - jnp.roll(Fx, shift=1, axis=1)) * inv_dx
            return -dFdx

    # ==================================================================
    # 2D CASE
    # ==================================================================
    elif ndim == 2:
        if order == 0:
            # First-order: direct flux computation
            # X-direction
            U_R_x = jnp.roll(U, shift=-1, axis=1)
            Fx = hllc_flux(U, U_R_x, axis=0, gamma=gamma)
            dFdx = (Fx - jnp.roll(Fx, shift=1, axis=1)) * inv_dx

            # Y-direction
            U_R_y = jnp.roll(U, shift=-1, axis=2)
            Fy = hllc_flux(U, U_R_y, axis=1, gamma=gamma)
            dFdy = (Fy - jnp.roll(Fy, shift=1, axis=2)) * inv_dy

            return -(dFdx + dFdy)
        else:
            # Second-order: compute slopes with periodic wraparound
            U_c = U

            # X-direction slopes
            U_L_x = jnp.roll(U, shift=1, axis=1)
            U_R_x = jnp.roll(U, shift=-1, axis=1)
            dx_L = U_c - U_L_x
            dx_R = U_R_x - U_c

            if use_smooth_limiters:
                dUx = slope_limiter_smooth(dx_L, dx_R, limiter, sharpness)
            else:
                dUx = slope_limiter(dx_L, dx_R, limiter)

            # Y-direction slopes
            U_L_y = jnp.roll(U, shift=1, axis=2)
            U_R_y = jnp.roll(U, shift=-1, axis=2)
            dy_L = U_c - U_L_y
            dy_R = U_R_y - U_c

            if use_smooth_limiters:
                dUy = slope_limiter_smooth(dy_L, dy_R, limiter, sharpness)
            else:
                dUy = slope_limiter(dy_L, dy_R, limiter)

            # Cache half-slopes (inline computation saves one allocation)
            half_dUx = 0.5 * dUx
            half_dUy = 0.5 * dUy

            # X-direction fluxes
            UL_x = U_c + half_dUx
            UR_x = jnp.roll(U_c - half_dUx, shift=-1, axis=1)
            Fx = hllc_flux(UL_x, UR_x, axis=0, gamma=gamma)
            dFdx = (Fx - jnp.roll(Fx, shift=1, axis=1)) * inv_dx

            # Y-direction fluxes
            UL_y = U_c + half_dUy
            UR_y = jnp.roll(U_c - half_dUy, shift=-1, axis=2)
            Fy = hllc_flux(UL_y, UR_y, axis=1, gamma=gamma)
            dFdy = (Fy - jnp.roll(Fy, shift=1, axis=2)) * inv_dy

            return -(dFdx + dFdy)

    # ==================================================================
    # 3D CASE
    # ==================================================================
    else:  # ndim == 3
        if order == 0:
            # First-order
            # X-direction
            U_R_x = jnp.roll(U, shift=-1, axis=1)
            Fx = hllc_flux(U, U_R_x, axis=0, gamma=gamma)
            dFdx = (Fx - jnp.roll(Fx, shift=1, axis=1)) * inv_dx

            # Y-direction
            U_R_y = jnp.roll(U, shift=-1, axis=2)
            Fy = hllc_flux(U, U_R_y, axis=1, gamma=gamma)
            dFdy = (Fy - jnp.roll(Fy, shift=1, axis=2)) * inv_dy

            # Z-direction
            U_R_z = jnp.roll(U, shift=-1, axis=3)
            Fz = hllc_flux(U, U_R_z, axis=2, gamma=gamma)
            dFdz = (Fz - jnp.roll(Fz, shift=1, axis=3)) * inv_dz

            return -(dFdx + dFdy + dFdz)
        else:
            # Second-order
            U_c = U

            # X-direction slopes
            U_L_x = jnp.roll(U, shift=1, axis=1)
            U_R_x = jnp.roll(U, shift=-1, axis=1)
            dx_L = U_c - U_L_x
            dx_R = U_R_x - U_c

            if use_smooth_limiters:
                dUx = slope_limiter_smooth(dx_L, dx_R, limiter, sharpness)
            else:
                dUx = slope_limiter(dx_L, dx_R, limiter)

            # Y-direction slopes
            U_L_y = jnp.roll(U, shift=1, axis=2)
            U_R_y = jnp.roll(U, shift=-1, axis=2)
            dy_L = U_c - U_L_y
            dy_R = U_R_y - U_c

            if use_smooth_limiters:
                dUy = slope_limiter_smooth(dy_L, dy_R, limiter, sharpness)
            else:
                dUy = slope_limiter(dy_L, dy_R, limiter)

            # Z-direction slopes
            U_L_z = jnp.roll(U, shift=1, axis=3)
            U_R_z = jnp.roll(U, shift=-1, axis=3)
            dz_L = U_c - U_L_z
            dz_R = U_R_z - U_c

            if use_smooth_limiters:
                dUz = slope_limiter_smooth(dz_L, dz_R, limiter, sharpness)
            else:
                dUz = slope_limiter(dz_L, dz_R, limiter)

            # Cache half-slopes
            half_dUx = 0.5 * dUx
            half_dUy = 0.5 * dUy
            half_dUz = 0.5 * dUz

            # X-direction fluxes
            UL_x = U_c + half_dUx
            UR_x = jnp.roll(U_c - half_dUx, shift=-1, axis=1)
            Fx = hllc_flux(UL_x, UR_x, axis=0, gamma=gamma)
            dFdx = (Fx - jnp.roll(Fx, shift=1, axis=1)) * inv_dx

            # Y-direction fluxes
            UL_y = U_c + half_dUy
            UR_y = jnp.roll(U_c - half_dUy, shift=-1, axis=2)
            Fy = hllc_flux(UL_y, UR_y, axis=1, gamma=gamma)
            dFdy = (Fy - jnp.roll(Fy, shift=1, axis=2)) * inv_dy

            # Z-direction fluxes
            UL_z = U_c + half_dUz
            UR_z = jnp.roll(U_c - half_dUz, shift=-1, axis=3)
            Fz = hllc_flux(UL_z, UR_z, axis=2, gamma=gamma)
            dFdz = (Fz - jnp.roll(Fz, shift=1, axis=3)) * inv_dz

            return -(dFdx + dFdy + dFdz)
