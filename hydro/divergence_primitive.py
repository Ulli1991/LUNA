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
from hydro.fluxes_primitive import hllc_flux_primitive
from hydro.conversions import cons_to_prim


@partial(jit, static_argnums=(1, 2, 3, 4, 11))
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
        Timestep (ignored for now; kept for interface compatibility with MUSCL-Hancock)

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
        # Compute slopes of PRIMITIVE variables
        dWx, dWy, dWz = compute_slopes_from_padded(W_pad, limiter)

        if ndim == 1:
            Wc = W_pad[:, 1:-1]
            half_dWx = 0.5 * dWx

            # Reconstruct primitives at interfaces
            WL = jnp.concatenate([W_pad[:, 0:1], Wc + half_dWx], axis=1)
            WR = jnp.concatenate([Wc - half_dWx, W_pad[:, -1:]], axis=1)

            # Compute fluxes directly from primitives (no conversion needed!)
            Fx = hllc_flux_primitive(WL, WR, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:] - Fx[:, :-1]) * inv_dx
            return dFdx, None, None

        elif ndim == 2:
            Wc = W_pad[:, 1:-1, 1:-1]
            half_dWx = 0.5 * dWx
            half_dWy = 0.5 * dWy

            # X-direction reconstruction of primitives
            WL_x = jnp.concatenate([W_pad[:, 0:1, 1:-1], Wc + half_dWx], axis=1)
            WR_x = jnp.concatenate([Wc - half_dWx, W_pad[:, -1:, 1:-1]], axis=1)

            # Compute X fluxes directly from primitives
            Fx = hllc_flux_primitive(WL_x, WR_x, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:, :] - Fx[:, :-1, :]) * inv_dx

            # Y-direction reconstruction of primitives
            WL_y = jnp.concatenate([W_pad[:, 1:-1, 0:1], Wc + half_dWy], axis=2)
            WR_y = jnp.concatenate([Wc - half_dWy, W_pad[:, 1:-1, -1:]], axis=2)

            # Compute Y fluxes directly from primitives
            Fy = hllc_flux_primitive(WL_y, WR_y, axis=1, gamma=gamma)
            dFdy = (Fy[:, :, 1:] - Fy[:, :, :-1]) * inv_dy

            return dFdx, dFdy, None

        else:  # 3D
            Wc = W_pad[:, 1:-1, 1:-1, 1:-1]
            half_dWx = 0.5 * dWx
            half_dWy = 0.5 * dWy
            half_dWz = 0.5 * dWz

            # X-direction
            WL_x = jnp.concatenate([W_pad[:, 0:1, 1:-1, 1:-1], Wc + half_dWx], axis=1)
            WR_x = jnp.concatenate([Wc - half_dWx, W_pad[:, -1:, 1:-1, 1:-1]], axis=1)
            Fx = hllc_flux_primitive(WL_x, WR_x, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:, :, :] - Fx[:, :-1, :, :]) * inv_dx

            # Y-direction
            WL_y = jnp.concatenate([W_pad[:, 1:-1, 0:1, 1:-1], Wc + half_dWy], axis=2)
            WR_y = jnp.concatenate([Wc - half_dWy, W_pad[:, 1:-1, -1:, 1:-1]], axis=2)
            Fy = hllc_flux_primitive(WL_y, WR_y, axis=1, gamma=gamma)
            dFdy = (Fy[:, :, 1:, :] - Fy[:, :, :-1, :]) * inv_dy

            # Z-direction
            WL_z = jnp.concatenate([W_pad[:, 1:-1, 1:-1, 0:1], Wc + half_dWz], axis=3)
            WR_z = jnp.concatenate([Wc - half_dWz, W_pad[:, 1:-1, 1:-1, -1:]], axis=3)
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
