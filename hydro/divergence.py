import jax.numpy as jnp
from jax import jit, lax
from functools import partial

from hydro.padding import pad
from hydro.slopes import compute_slopes_from_padded
from hydro.fluxes import hllc_flux

@partial(jit, static_argnums=(2, 3, 4))
def flux_divergence(U, gamma, order, limiter, boundary, dx, dy=None, dz=None):
    """
    Compute the flux divergence in 1D, 2D, or 3D using first-order or second-order PLM.

    Optimized to avoid redundant flux computations:
    - First-order mode: Direct flux computation from padded states
    - Second-order mode: Reconstruction at ALL interfaces (including boundaries)

    U: (Nvar, Nx) or (Nvar, Nx, Ny) or (Nvar, Nx, Ny, Nz)
    boundary: int, 0=periodic, 1=zero-gradient, 2=reflective
    order: int, 0=first-order Godunov, 1=second-order PLM
    Returns: -(dFdx + dFdy + dFdz) with shapes matching U
    """
    U_pad = pad(U, boundary)
    ndim = U.ndim - 1  # number of spatial dimensions

    # -------------------
    # FIRST-ORDER PATH
    # -------------------
    def first_order_divergence(U_pad):
        """Compute first-order Godunov fluxes directly from padded states."""
        if ndim == 1:
            U_Lx = U_pad[:, :-1]
            U_Rx = U_pad[:, 1:]
            Fx = hllc_flux(U_Lx, U_Rx, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:] - Fx[:, :-1]) / dx
            return dFdx, None, None
        elif ndim == 2:
            U_Lx = U_pad[:, :-1, 1:-1]
            U_Rx = U_pad[:, 1: , 1:-1]
            Fx = hllc_flux(U_Lx, U_Rx, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:, :] - Fx[:, :-1, :]) / dx

            U_By = U_pad[:, 1:-1, :-1]
            U_Ty = U_pad[:, 1:-1, 1: ]
            Fy = hllc_flux(U_By, U_Ty, axis=1, gamma=gamma)
            dFdy = (Fy[:, :, 1:] - Fy[:, :, :-1]) / dy
            return dFdx, dFdy, None
        else:  # 3D
            # X-direction flux
            U_Lx = U_pad[:, :-1, 1:-1, 1:-1]
            U_Rx = U_pad[:, 1: , 1:-1, 1:-1]
            Fx = hllc_flux(U_Lx, U_Rx, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:, :, :] - Fx[:, :-1, :, :]) / dx

            # Y-direction flux
            U_By = U_pad[:, 1:-1, :-1, 1:-1]
            U_Ty = U_pad[:, 1:-1, 1: , 1:-1]
            Fy = hllc_flux(U_By, U_Ty, axis=1, gamma=gamma)
            dFdy = (Fy[:, :, 1:, :] - Fy[:, :, :-1, :]) / dy

            # Z-direction flux
            U_Bz = U_pad[:, 1:-1, 1:-1, :-1]
            U_Fz = U_pad[:, 1:-1, 1:-1, 1: ]
            Fz = hllc_flux(U_Bz, U_Fz, axis=2, gamma=gamma)
            dFdz = (Fz[:, :, :, 1:] - Fz[:, :, :, :-1]) / dz
            return dFdx, dFdy, dFdz

    # -------------------
    # SECOND-ORDER PATH
    # -------------------
    def second_order_divergence(U_pad):
        """
        Compute second-order PLM fluxes with reconstruction at ALL interfaces.
        This eliminates the need for first-order flux computation.
        Slope limiting ensures stability at boundaries.
        """
        # Compute slopes for all real cells (using ghost cells in stencil)
        dUx, dUy, dUz = compute_slopes_from_padded(U_pad, limiter)

        if ndim == 1:
            # Extract real cells (excluding ghost cells)
            Qc = U_pad[:, 1:-1]  # Shape: (Nvar, Nx)

            # Cache half-slopes to avoid redundant computation
            half_dUx = 0.5 * dUx

            # Reconstruct at ALL Nx+1 interfaces (including boundaries)
            # Left states: extrapolate from left cell
            UL = jnp.concatenate([
                U_pad[:, 0:1],                    # Left boundary: use ghost cell value
                Qc + half_dUx                     # Interior + right boundary: cell center + slope
            ], axis=1)

            # Right states: extrapolate from right cell
            UR = jnp.concatenate([
                Qc - half_dUx,                    # Left boundary + interior: cell center - slope
                U_pad[:, -1:]                     # Right boundary: use ghost cell value
            ], axis=1)

            # Compute fluxes at all interfaces
            Fx = hllc_flux(UL, UR, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:] - Fx[:, :-1]) / dx
            return dFdx, None, None

        elif ndim == 2:
            Qc = U_pad[:, 1:-1, 1:-1]  # Shape: (Nvar, Nx, Ny)

            # Cache half-slopes to avoid redundant computation
            half_dUx = 0.5 * dUx
            half_dUy = 0.5 * dUy

            # X-direction reconstruction at ALL Nx+1 interfaces
            UL_x = jnp.concatenate([
                U_pad[:, 0:1, 1:-1],              # Left boundary
                Qc + half_dUx                     # Interior + right boundary
            ], axis=1)
            UR_x = jnp.concatenate([
                Qc - half_dUx,                    # Left boundary + interior
                U_pad[:, -1:, 1:-1]               # Right boundary
            ], axis=1)
            Fx = hllc_flux(UL_x, UR_x, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:, :] - Fx[:, :-1, :]) / dx

            # Y-direction reconstruction at ALL Ny+1 interfaces
            UL_y = jnp.concatenate([
                U_pad[:, 1:-1, 0:1],              # Bottom boundary
                Qc + half_dUy                     # Interior + top boundary
            ], axis=2)
            UR_y = jnp.concatenate([
                Qc - half_dUy,                    # Bottom boundary + interior
                U_pad[:, 1:-1, -1:]               # Top boundary
            ], axis=2)
            Fy = hllc_flux(UL_y, UR_y, axis=1, gamma=gamma)
            dFdy = (Fy[:, :, 1:] - Fy[:, :, :-1]) / dy

            return dFdx, dFdy, None

        else:  # 3D
            Qc = U_pad[:, 1:-1, 1:-1, 1:-1]  # Shape: (Nvar, Nx, Ny, Nz)

            # Cache half-slopes to avoid redundant computation
            half_dUx = 0.5 * dUx
            half_dUy = 0.5 * dUy
            half_dUz = 0.5 * dUz

            # X-direction reconstruction
            UL_x = jnp.concatenate([
                U_pad[:, 0:1, 1:-1, 1:-1],
                Qc + half_dUx
            ], axis=1)
            UR_x = jnp.concatenate([
                Qc - half_dUx,
                U_pad[:, -1:, 1:-1, 1:-1]
            ], axis=1)
            Fx = hllc_flux(UL_x, UR_x, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:, :, :] - Fx[:, :-1, :, :]) / dx

            # Y-direction reconstruction
            UL_y = jnp.concatenate([
                U_pad[:, 1:-1, 0:1, 1:-1],
                Qc + half_dUy
            ], axis=2)
            UR_y = jnp.concatenate([
                Qc - half_dUy,
                U_pad[:, 1:-1, -1:, 1:-1]
            ], axis=2)
            Fy = hllc_flux(UL_y, UR_y, axis=1, gamma=gamma)
            dFdy = (Fy[:, :, 1:, :] - Fy[:, :, :-1, :]) / dy

            # Z-direction reconstruction
            UL_z = jnp.concatenate([
                U_pad[:, 1:-1, 1:-1, 0:1],
                Qc + half_dUz
            ], axis=3)
            UR_z = jnp.concatenate([
                Qc - half_dUz,
                U_pad[:, 1:-1, 1:-1, -1:]
            ], axis=3)
            Fz = hllc_flux(UL_z, UR_z, axis=2, gamma=gamma)
            dFdz = (Fz[:, :, :, 1:] - Fz[:, :, :, :-1]) / dz

            return dFdx, dFdy, dFdz

    # -------------------
    # ORDER SELECTION (JIT-safe with lax.cond)
    # -------------------
    is_first_order = jnp.array(order == 0)  # 0 = first-order, 1 = PLM
    dFdx, dFdy, dFdz = lax.cond(
        is_first_order,
        first_order_divergence,
        second_order_divergence,
        U_pad
    )

    # Return total divergence
    if ndim == 1:
        return -dFdx
    elif ndim == 2:
        return -(dFdx + dFdy)
    else:  # 3D
        return -(dFdx + dFdy + dFdz)

