"""
Flux divergence with PRIMITIVE variable reconstruction.

This is the Athena++ / modern production code approach:
1. Convert conservative U to primitive W
2. Pad primitive variables
3. Reconstruct primitives at interfaces (more physical!)
4. Convert reconstructed primitives back to conservatives for Riemann solver
5. Compute fluxes and divergence

Benefits:
- More robust for strong shocks
- Avoids unphysical states (negative pressure)
- Better physical behavior (velocity/pressure are natural variables)
"""

import jax.numpy as jnp
from jax import jit, lax
from functools import partial

from hydro.padding import pad
from hydro.slopes import compute_slopes_from_padded
from hydro.fluxes import hllc_flux_primitive
from hydro.conversions import cons_to_prim
from hydro.jacobian import (apply_primitive_jacobian_1d,
                             apply_primitive_jacobian_2d,
                             apply_primitive_jacobian_3d)


def flux_divergence_primitive(U, gamma, order, limiter, boundary, dx, dy, dz,
                               inv_dx, inv_dy, inv_dz, use_smooth_limiters=False, dt=None):
    """
    Compute flux divergence using PRIMITIVE variable reconstruction.

    Flow:
    U (conservative) → W (primitive) → pad → slopes → reconstruct W →
    convert to U_L, U_R → HLLC flux → divergence

    Parameters
    ----------
    U : array
        Conservative state (Nvar, Nx) or (Nvar, Nx, Ny) or (Nvar, Nx, Ny, Nz)
    gamma : float
        Adiabatic index
    order : int (static)
        0 = first-order, 1 = second-order PLM
    limiter : int (static)
        Slope limiter type
    boundary : int (static)
        Boundary condition type
    dx, dy, dz : float
        Grid spacing
    inv_dx, inv_dy, inv_dz : float
        Inverse grid spacing
    use_smooth_limiters : bool (static)
        Use smooth limiters for autodiff
    dt : float or None
        Timestep (required for MUSCL-Hancock predictor step; if None, no predictor)

    Returns
    -------
    dU/dt : array
        Conservative update (same shape as U)
    """
    ndim = U.ndim - 1

    # STEP 1: Convert conservative to primitive
    W = cons_to_prim(U, gamma)

    # STEP 2: Pad primitive variables
    W_pad = pad(W, boundary)

    # STEP 3: First-order vs second-order paths
    def first_order_divergence(W_pad):
        """First-order Godunov: no reconstruction, just use cell-centered values."""
        if ndim == 1:
            # Extract primitives at interfaces (no reconstruction)
            W_L = W_pad[:, :-1]
            W_R = W_pad[:, 1:]
            # Compute flux directly from primitives (no conversion needed!)
            Fx = hllc_flux_primitive(W_L, W_R, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:] - Fx[:, :-1]) * inv_dx
            return dFdx, None, None

        elif ndim == 2:
            # X-direction
            W_L_x = W_pad[:, :-1, 1:-1]
            W_R_x = W_pad[:, 1:, 1:-1]
            Fx = hllc_flux_primitive(W_L_x, W_R_x, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:, :] - Fx[:, :-1, :]) * inv_dx

            # Y-direction
            W_L_y = W_pad[:, 1:-1, :-1]
            W_R_y = W_pad[:, 1:-1, 1:]
            Fy = hllc_flux_primitive(W_L_y, W_R_y, axis=1, gamma=gamma)
            dFdy = (Fy[:, :, 1:] - Fy[:, :, :-1]) * inv_dy

            return dFdx, dFdy, None

        else:  # 3D
            # X-direction
            W_L_x = W_pad[:, :-1, 1:-1, 1:-1]
            W_R_x = W_pad[:, 1:, 1:-1, 1:-1]
            Fx = hllc_flux_primitive(W_L_x, W_R_x, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:, :, :] - Fx[:, :-1, :, :]) * inv_dx

            # Y-direction
            W_L_y = W_pad[:, 1:-1, :-1, 1:-1]
            W_R_y = W_pad[:, 1:-1, 1:, 1:-1]
            Fy = hllc_flux_primitive(W_L_y, W_R_y, axis=1, gamma=gamma)
            dFdy = (Fy[:, :, 1:, :] - Fy[:, :, :-1, :]) * inv_dy

            # Z-direction
            W_L_z = W_pad[:, 1:-1, 1:-1, :-1]
            W_R_z = W_pad[:, 1:-1, 1:-1, 1:]
            Fz = hllc_flux_primitive(W_L_z, W_R_z, axis=2, gamma=gamma)
            dFdz = (Fz[:, :, :, 1:] - Fz[:, :, :, :-1]) * inv_dz

            return dFdx, dFdy, dFdz

    def second_order_divergence(W_pad):
        """Second-order PLM: reconstruct PRIMITIVES at interfaces."""
        # Compute slopes of PRIMITIVE variables at t^n
        dWx, dWy, dWz = compute_slopes_from_padded(W_pad, limiter)

        if ndim == 1:
            Wc = W_pad[:, 1:-1]

            # MUSCL-Hancock predictor step (if dt provided)
            if dt is not None:
                # W^(n+1/2) = W^n - (dt/2) * A_x * ∂W/∂x
                # where ∂W/∂x ≈ dWx * inv_dx
                dWx_scaled = dWx * inv_dx
                A_dW = apply_primitive_jacobian_1d(Wc, dWx_scaled, gamma)
                W_half = Wc - 0.5 * dt * A_dW
            else:
                # No predictor (standard PLM)
                W_half = Wc

            half_dWx = 0.5 * dWx

            # Reconstruct primitives at interfaces using predicted state
            WL = jnp.concatenate([W_pad[:, 0:1], W_half + half_dWx], axis=1)
            WR = jnp.concatenate([W_half - half_dWx, W_pad[:, -1:]], axis=1)

            # Compute fluxes directly from primitives (no conversion needed!)
            Fx = hllc_flux_primitive(WL, WR, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:] - Fx[:, :-1]) * inv_dx
            return dFdx, None, None

        elif ndim == 2:
            Wc = W_pad[:, 1:-1, 1:-1]

            # MUSCL-Hancock predictor step (if dt provided)
            if dt is not None:
                # W^(n+1/2) = W^n - (dt/2) * (A_x * ∂W/∂x + A_y * ∂W/∂y)
                # where ∂W/∂x ≈ dWx * inv_dx and ∂W/∂y ≈ dWy * inv_dy
                # The Jacobian function needs slopes with proper scaling
                dWx_scaled = dWx * inv_dx
                dWy_scaled = dWy * inv_dy
                A_dW = apply_primitive_jacobian_2d(Wc, dWx_scaled, dWy_scaled, gamma)
                W_half = Wc - 0.5 * dt * A_dW
            else:
                # No predictor (standard PLM)
                W_half = Wc

            half_dWx = 0.5 * dWx
            half_dWy = 0.5 * dWy

            # X-direction reconstruction of primitives using predicted state
            WL_x = jnp.concatenate([W_pad[:, 0:1, 1:-1], W_half + half_dWx], axis=1)
            WR_x = jnp.concatenate([W_half - half_dWx, W_pad[:, -1:, 1:-1]], axis=1)

            # Compute X fluxes directly from primitives
            Fx = hllc_flux_primitive(WL_x, WR_x, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:, :] - Fx[:, :-1, :]) * inv_dx

            # Y-direction reconstruction of primitives using predicted state
            WL_y = jnp.concatenate([W_pad[:, 1:-1, 0:1], W_half + half_dWy], axis=2)
            WR_y = jnp.concatenate([W_half - half_dWy, W_pad[:, 1:-1, -1:]], axis=2)

            # Compute Y fluxes directly from primitives
            Fy = hllc_flux_primitive(WL_y, WR_y, axis=1, gamma=gamma)
            dFdy = (Fy[:, :, 1:] - Fy[:, :, :-1]) * inv_dy

            return dFdx, dFdy, None

        else:  # 3D
            Wc = W_pad[:, 1:-1, 1:-1, 1:-1]

            # MUSCL-Hancock predictor step (if dt provided)
            if dt is not None:
                # W^(n+1/2) = W^n - (dt/2) * (A_x * ∂W/∂x + A_y * ∂W/∂y + A_z * ∂W/∂z)
                dWx_scaled = dWx * inv_dx
                dWy_scaled = dWy * inv_dy
                dWz_scaled = dWz * inv_dz
                A_dW = apply_primitive_jacobian_3d(Wc, dWx_scaled, dWy_scaled, dWz_scaled, gamma)
                W_half = Wc - 0.5 * dt * A_dW
            else:
                # No predictor (standard PLM)
                W_half = Wc

            half_dWx = 0.5 * dWx
            half_dWy = 0.5 * dWy
            half_dWz = 0.5 * dWz

            # X-direction reconstruction using predicted state
            WL_x = jnp.concatenate([W_pad[:, 0:1, 1:-1, 1:-1], W_half + half_dWx], axis=1)
            WR_x = jnp.concatenate([W_half - half_dWx, W_pad[:, -1:, 1:-1, 1:-1]], axis=1)
            Fx = hllc_flux_primitive(WL_x, WR_x, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:, :, :] - Fx[:, :-1, :, :]) * inv_dx

            # Y-direction reconstruction using predicted state
            WL_y = jnp.concatenate([W_pad[:, 1:-1, 0:1, 1:-1], W_half + half_dWy], axis=2)
            WR_y = jnp.concatenate([W_half - half_dWy, W_pad[:, 1:-1, -1:, 1:-1]], axis=2)
            Fy = hllc_flux_primitive(WL_y, WR_y, axis=1, gamma=gamma)
            dFdy = (Fy[:, :, 1:, :] - Fy[:, :, :-1, :]) * inv_dy

            # Z-direction reconstruction using predicted state
            WL_z = jnp.concatenate([W_pad[:, 1:-1, 1:-1, 0:1], W_half + half_dWz], axis=3)
            WR_z = jnp.concatenate([W_half - half_dWz, W_pad[:, 1:-1, 1:-1, -1:]], axis=3)
            Fz = hllc_flux_primitive(WL_z, WR_z, axis=2, gamma=gamma)
            dFdz = (Fz[:, :, :, 1:] - Fz[:, :, :, :-1]) * inv_dz

            return dFdx, dFdy, dFdz

    # Select order
    is_first_order = jnp.array(order == 0)
    dFdx, dFdy, dFdz = lax.cond(
        is_first_order,
        first_order_divergence,
        second_order_divergence,
        W_pad
    )

    # Return divergence
    if ndim == 1:
        return -dFdx
    elif ndim == 2:
        return -(dFdx + dFdy)
    else:
        return -(dFdx + dFdy + dFdz)


@partial(jit, static_argnums=(1, 2, 3))
def flux_divergence_from_padded(U_pad, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz, dt=None):
    """
    Compute flux divergence from ALREADY PADDED conservative data using PRIMITIVE reconstruction.

    For multi-GPU pmap mode: ghost cells are pre-filled by halo exchange.
    This skips the pad() call and works directly with the padded conservative state.

    Flow:
    1. Convert padded conservative U_pad to primitive W_pad
    2. Compute slopes of primitives
    3. (Optional) Apply MUSCL-Hancock predictor if dt provided
    4. Reconstruct primitives at interfaces
    5. Compute HLLC fluxes from primitives
    6. Return divergence

    Parameters
    ----------
    U_pad : array
        ALREADY PADDED conservative state with 1 ghost cell per side
        Shape: (Nvar, Nx+2) or (Nvar, Nx+2, Ny+2) or (Nvar, Nx+2, Ny+2, Nz+2)
    gamma : float
        Adiabatic index
    order : int (static)
        0 = first-order, 1 = second-order PLM
    limiter : int (static)
        Slope limiter type
    dx, dy, dz : float
        Grid spacing
    inv_dx, inv_dy, inv_dz : float
        Inverse grid spacing
    dt : float or None
        Timestep for MUSCL-Hancock predictor. If None, no predictor (standard PLM)

    Returns
    -------
    dU/dt : array
        Conservative update with shape (Nvar, Nx) or (Nvar, Nx, Ny) or (Nvar, Nx, Ny, Nz)

    Notes
    -----
    - Pass dt=None for RK2/RK4 (they handle time externally)
    - Pass dt=<value> for MUSCL-Hancock predictor step
    """
    ndim = U_pad.ndim - 1

    # Convert PADDED conservatives to primitives
    W_pad = cons_to_prim(U_pad, gamma)

    # First-order path
    def first_order_divergence(W_pad):
        if ndim == 1:
            W_L = W_pad[:, :-1]
            W_R = W_pad[:, 1:]
            Fx = hllc_flux_primitive(W_L, W_R, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:] - Fx[:, :-1]) * inv_dx
            return dFdx, None, None

        elif ndim == 2:
            W_L_x = W_pad[:, :-1, 1:-1]
            W_R_x = W_pad[:, 1:, 1:-1]
            Fx = hllc_flux_primitive(W_L_x, W_R_x, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:, :] - Fx[:, :-1, :]) * inv_dx

            W_L_y = W_pad[:, 1:-1, :-1]
            W_R_y = W_pad[:, 1:-1, 1:]
            Fy = hllc_flux_primitive(W_L_y, W_R_y, axis=1, gamma=gamma)
            dFdy = (Fy[:, :, 1:] - Fy[:, :, :-1]) * inv_dy

            return dFdx, dFdy, None

        else:  # 3D
            W_L_x = W_pad[:, :-1, 1:-1, 1:-1]
            W_R_x = W_pad[:, 1:, 1:-1, 1:-1]
            Fx = hllc_flux_primitive(W_L_x, W_R_x, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:, :, :] - Fx[:, :-1, :, :]) * inv_dx

            W_L_y = W_pad[:, 1:-1, :-1, 1:-1]
            W_R_y = W_pad[:, 1:-1, 1:, 1:-1]
            Fy = hllc_flux_primitive(W_L_y, W_R_y, axis=1, gamma=gamma)
            dFdy = (Fy[:, :, 1:, :] - Fy[:, :, :-1, :]) * inv_dy

            W_L_z = W_pad[:, 1:-1, 1:-1, :-1]
            W_R_z = W_pad[:, 1:-1, 1:-1, 1:]
            Fz = hllc_flux_primitive(W_L_z, W_R_z, axis=2, gamma=gamma)
            dFdz = (Fz[:, :, :, 1:] - Fz[:, :, :, :-1]) * inv_dz

            return dFdx, dFdy, dFdz

    # Second-order path (optional predictor if dt provided)
    def second_order_divergence(W_pad):
        dWx, dWy, dWz = compute_slopes_from_padded(W_pad, limiter)

        if ndim == 1:
            Wc = W_pad[:, 1:-1]

            # MUSCL-Hancock predictor step (if dt provided)
            if dt is not None:
                dWx_scaled = dWx * inv_dx
                A_dW = apply_primitive_jacobian_1d(Wc, dWx_scaled, gamma)
                W_half = Wc - 0.5 * dt * A_dW
            else:
                W_half = Wc

            half_dWx = 0.5 * dWx
            WL = jnp.concatenate([W_pad[:, 0:1], W_half + half_dWx], axis=1)
            WR = jnp.concatenate([W_half - half_dWx, W_pad[:, -1:]], axis=1)
            Fx = hllc_flux_primitive(WL, WR, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:] - Fx[:, :-1]) * inv_dx
            return dFdx, None, None

        elif ndim == 2:
            Wc = W_pad[:, 1:-1, 1:-1]

            # MUSCL-Hancock predictor step (if dt provided)
            if dt is not None:
                dWx_scaled = dWx * inv_dx
                dWy_scaled = dWy * inv_dy
                A_dW = apply_primitive_jacobian_2d(Wc, dWx_scaled, dWy_scaled, gamma)
                W_half = Wc - 0.5 * dt * A_dW
            else:
                W_half = Wc

            half_dWx = 0.5 * dWx
            half_dWy = 0.5 * dWy

            WL_x = jnp.concatenate([W_pad[:, 0:1, 1:-1], W_half + half_dWx], axis=1)
            WR_x = jnp.concatenate([W_half - half_dWx, W_pad[:, -1:, 1:-1]], axis=1)
            Fx = hllc_flux_primitive(WL_x, WR_x, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:, :] - Fx[:, :-1, :]) * inv_dx

            WL_y = jnp.concatenate([W_pad[:, 1:-1, 0:1], W_half + half_dWy], axis=2)
            WR_y = jnp.concatenate([W_half - half_dWy, W_pad[:, 1:-1, -1:]], axis=2)
            Fy = hllc_flux_primitive(WL_y, WR_y, axis=1, gamma=gamma)
            dFdy = (Fy[:, :, 1:] - Fy[:, :, :-1]) * inv_dy

            return dFdx, dFdy, None

        else:  # 3D
            Wc = W_pad[:, 1:-1, 1:-1, 1:-1]

            # MUSCL-Hancock predictor step (if dt provided)
            if dt is not None:
                dWx_scaled = dWx * inv_dx
                dWy_scaled = dWy * inv_dy
                dWz_scaled = dWz * inv_dz
                A_dW = apply_primitive_jacobian_3d(Wc, dWx_scaled, dWy_scaled, dWz_scaled, gamma)
                W_half = Wc - 0.5 * dt * A_dW
            else:
                W_half = Wc

            half_dWx = 0.5 * dWx
            half_dWy = 0.5 * dWy
            half_dWz = 0.5 * dWz

            WL_x = jnp.concatenate([W_pad[:, 0:1, 1:-1, 1:-1], W_half + half_dWx], axis=1)
            WR_x = jnp.concatenate([W_half - half_dWx, W_pad[:, -1:, 1:-1, 1:-1]], axis=1)
            Fx = hllc_flux_primitive(WL_x, WR_x, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:, :, :] - Fx[:, :-1, :, :]) * inv_dx

            WL_y = jnp.concatenate([W_pad[:, 1:-1, 0:1, 1:-1], W_half + half_dWy], axis=2)
            WR_y = jnp.concatenate([W_half - half_dWy, W_pad[:, 1:-1, -1:, 1:-1]], axis=2)
            Fy = hllc_flux_primitive(WL_y, WR_y, axis=1, gamma=gamma)
            dFdy = (Fy[:, :, 1:, :] - Fy[:, :, :-1, :]) * inv_dy

            WL_z = jnp.concatenate([W_pad[:, 1:-1, 1:-1, 0:1], W_half + half_dWz], axis=3)
            WR_z = jnp.concatenate([W_half - half_dWz, W_pad[:, 1:-1, 1:-1, -1:]], axis=3)
            Fz = hllc_flux_primitive(WL_z, WR_z, axis=2, gamma=gamma)
            dFdz = (Fz[:, :, :, 1:] - Fz[:, :, :, :-1]) * inv_dz

            return dFdx, dFdy, dFdz

    # Select order
    is_first_order = jnp.array(order == 0)
    dFdx, dFdy, dFdz = lax.cond(
        is_first_order,
        first_order_divergence,
        second_order_divergence,
        W_pad
    )

    # Return divergence
    if ndim == 1:
        return -dFdx
    elif ndim == 2:
        return -(dFdx + dFdy)
    else:
        return -(dFdx + dFdy + dFdz)
