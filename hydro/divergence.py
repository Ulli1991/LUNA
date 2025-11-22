import jax.numpy as jnp
from jax import jit, lax
from functools import partial

from hydro.padding import pad
from hydro.slopes import compute_slopes_from_padded, slope_limiter
from hydro.fluxes import hllc_flux

@partial(jit, static_argnums=(2, 3, 4))
def flux_divergence(U, gamma, order, limiter, boundary, dx, dy, dz, inv_dx, inv_dy, inv_dz):
    """
    Compute the flux divergence in 1D, 2D, or 3D using first-order or second-order PLM.

    Optimized to avoid redundant flux computations:
    - First-order mode: Direct flux computation from padded states
    - Second-order mode: Reconstruction at ALL interfaces (including boundaries)

    U: (Nvar, Nx) or (Nvar, Nx, Ny) or (Nvar, Nx, Ny, Nz)
    boundary: int, 0=periodic, 1=zero-gradient, 2=reflective
    order: int, 0=first-order Godunov, 1=second-order PLM
    inv_dx, inv_dy, inv_dz: Precomputed 1/dx, 1/dy, 1/dz (faster than division)
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
            dFdx = (Fx[:, 1:] - Fx[:, :-1]) * inv_dx  # Multiplication faster than division
            return dFdx, None, None
        elif ndim == 2:
            U_Lx = U_pad[:, :-1, 1:-1]
            U_Rx = U_pad[:, 1: , 1:-1]
            Fx = hllc_flux(U_Lx, U_Rx, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:, :] - Fx[:, :-1, :]) * inv_dx  # Multiplication faster than division

            U_By = U_pad[:, 1:-1, :-1]
            U_Ty = U_pad[:, 1:-1, 1: ]
            Fy = hllc_flux(U_By, U_Ty, axis=1, gamma=gamma)
            dFdy = (Fy[:, :, 1:] - Fy[:, :, :-1]) * inv_dy  # Multiplication faster than division
            return dFdx, dFdy, None
        else:  # 3D
            # X-direction flux
            U_Lx = U_pad[:, :-1, 1:-1, 1:-1]
            U_Rx = U_pad[:, 1: , 1:-1, 1:-1]
            Fx = hllc_flux(U_Lx, U_Rx, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:, :, :] - Fx[:, :-1, :, :]) * inv_dx  # Multiplication faster than division

            # Y-direction flux
            U_By = U_pad[:, 1:-1, :-1, 1:-1]
            U_Ty = U_pad[:, 1:-1, 1: , 1:-1]
            Fy = hllc_flux(U_By, U_Ty, axis=1, gamma=gamma)
            dFdy = (Fy[:, :, 1:, :] - Fy[:, :, :-1, :]) * inv_dy  # Multiplication faster than division

            # Z-direction flux
            U_Bz = U_pad[:, 1:-1, 1:-1, :-1]
            U_Fz = U_pad[:, 1:-1, 1:-1, 1: ]
            Fz = hllc_flux(U_Bz, U_Fz, axis=2, gamma=gamma)
            dFdz = (Fz[:, :, :, 1:] - Fz[:, :, :, :-1]) * inv_dz  # Multiplication faster than division
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
            dFdx = (Fx[:, 1:] - Fx[:, :-1]) * inv_dx  # Multiplication faster than division
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
            dFdx = (Fx[:, 1:, :] - Fx[:, :-1, :]) * inv_dx  # Multiplication faster than division

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
            dFdy = (Fy[:, :, 1:] - Fy[:, :, :-1]) * inv_dy  # Multiplication faster than division

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
            dFdx = (Fx[:, 1:, :, :] - Fx[:, :-1, :, :]) * inv_dx  # Multiplication faster than division

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
            dFdy = (Fy[:, :, 1:, :] - Fy[:, :, :-1, :]) * inv_dy  # Multiplication faster than division

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
            dFdz = (Fz[:, :, :, 1:] - Fz[:, :, :, :-1]) * inv_dz  # Multiplication faster than division

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


@partial(jit, static_argnums=(2, 3))
def flux_divergence_from_2ghost(U_pad, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz):
    """
    Compute flux divergence from data with 2 GHOST CELLS per side.

    For MPI mode: This version properly handles 2 ghost cells, enabling correct
    PLM reconstruction at inter-patch boundaries. With 2 ghost cells, we can
    compute slopes OF the first ghost cell layer, allowing proper reconstruction
    at the interface between subdomains.

    U_pad: (Nvar, Nx+4) or (Nvar, Nx+4, Ny+4) or (Nvar, Nx+4, Ny+4, Nz+4)
           Data with 2 ghost cells per side already filled.
    Returns: -(dFdx + dFdy + dFdz) with shape (Nvar, Nx) or (Nvar, Nx, Ny) or (Nvar, Nx, Ny, Nz)
    """
    ndim = U_pad.ndim - 1  # number of spatial dimensions

    # FIRST-ORDER PATH (same logic, just adjust slicing for 2 ghost cells)
    def first_order_divergence(U_pad):
        if ndim == 1:
            # With 2 ghost cells: indices 0,1 are left ghost, -2,-1 are right ghost
            # Interior is 2:-2, so Nx+4 - 4 = Nx cells
            U_Lx = U_pad[:, 1:-2]  # cells 1 to -3 (Nx+1 values)
            U_Rx = U_pad[:, 2:-1]  # cells 2 to -2 (Nx+1 values)
            Fx = hllc_flux(U_Lx, U_Rx, axis=0, gamma=gamma)
            # dFdx = F[i+1/2] - F[i-1/2] for interior cells
            dFdx = (Fx[:, 1:] - Fx[:, :-1]) * inv_dx
            return dFdx, None, None
        elif ndim == 2:
            U_Lx = U_pad[:, 1:-2, 2:-2]
            U_Rx = U_pad[:, 2:-1, 2:-2]
            Fx = hllc_flux(U_Lx, U_Rx, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:, :] - Fx[:, :-1, :]) * inv_dx

            U_By = U_pad[:, 2:-2, 1:-2]
            U_Ty = U_pad[:, 2:-2, 2:-1]
            Fy = hllc_flux(U_By, U_Ty, axis=1, gamma=gamma)
            dFdy = (Fy[:, :, 1:] - Fy[:, :, :-1]) * inv_dy
            return dFdx, dFdy, None
        else:  # 3D
            U_Lx = U_pad[:, 1:-2, 2:-2, 2:-2]
            U_Rx = U_pad[:, 2:-1, 2:-2, 2:-2]
            Fx = hllc_flux(U_Lx, U_Rx, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:, :, :] - Fx[:, :-1, :, :]) * inv_dx

            U_By = U_pad[:, 2:-2, 1:-2, 2:-2]
            U_Ty = U_pad[:, 2:-2, 2:-1, 2:-2]
            Fy = hllc_flux(U_By, U_Ty, axis=1, gamma=gamma)
            dFdy = (Fy[:, :, 1:, :] - Fy[:, :, :-1, :]) * inv_dy

            U_Bz = U_pad[:, 2:-2, 2:-2, 1:-2]
            U_Fz = U_pad[:, 2:-2, 2:-2, 2:-1]
            Fz = hllc_flux(U_Bz, U_Fz, axis=2, gamma=gamma)
            dFdz = (Fz[:, :, :, 1:] - Fz[:, :, :, :-1]) * inv_dz
            return dFdx, dFdy, dFdz

    # SECOND-ORDER PATH
    def second_order_divergence(U_pad):
        # With 2 ghost cells, we can compute slopes for ghost cell layer 1
        # This enables proper reconstruction at the MPI boundary interface

        if ndim == 1:
            # Compute slopes for cells 1 to -2 (interior + 1 ghost layer each side)
            # Shape: Nx+2 cells (Nx interior + 1 ghost on each side)
            Qc_ext = U_pad[:, 1:-1]  # Extended: includes 1 ghost on each side
            dx_L = Qc_ext - U_pad[:, :-2]
            dx_R = U_pad[:, 2:] - Qc_ext
            dUx_ext = slope_limiter(dx_L, dx_R, limiter)

            half_dUx = 0.5 * dUx_ext

            # Reconstruct at Nx+1 interfaces (for Nx interior cells)
            # UL at interface i+1/2 = cell i + half_slope
            # UR at interface i+1/2 = cell i+1 - half_slope
            # For interior cells (indices 2:-2 in U_pad, or 1:-1 in Qc_ext):
            #   Need fluxes at interfaces 0.5 to Nx+0.5 (Nx+1 interfaces)
            UL = Qc_ext[:, :-1] + half_dUx[:, :-1]  # Nx+1 left states
            UR = Qc_ext[:, 1:] - half_dUx[:, 1:]    # Nx+1 right states
            Fx = hllc_flux(UL, UR, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:] - Fx[:, :-1]) * inv_dx  # Nx values for interior
            return dFdx, None, None

        elif ndim == 2:
            # Extended cells: interior + 1 ghost layer on each side
            Qc_ext = U_pad[:, 1:-1, 1:-1]  # Shape: (Nvar, Nx+2, Ny+2)

            # X-direction slopes
            dx_L = Qc_ext - U_pad[:, :-2, 1:-1]
            dx_R = U_pad[:, 2:, 1:-1] - Qc_ext
            dUx_ext = slope_limiter(dx_L, dx_R, limiter)

            # Y-direction slopes
            dy_D = Qc_ext - U_pad[:, 1:-1, :-2]
            dy_U = U_pad[:, 1:-1, 2:] - Qc_ext
            dUy_ext = slope_limiter(dy_D, dy_U, limiter)

            half_dUx = 0.5 * dUx_ext
            half_dUy = 0.5 * dUy_ext

            # X-fluxes: need y to be in interior range (1:-1 of Qc_ext = 2:-2 of U_pad)
            UL_x = Qc_ext[:, :-1, 1:-1] + half_dUx[:, :-1, 1:-1]
            UR_x = Qc_ext[:, 1:, 1:-1] - half_dUx[:, 1:, 1:-1]
            Fx = hllc_flux(UL_x, UR_x, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:, :] - Fx[:, :-1, :]) * inv_dx

            # Y-fluxes
            UL_y = Qc_ext[:, 1:-1, :-1] + half_dUy[:, 1:-1, :-1]
            UR_y = Qc_ext[:, 1:-1, 1:] - half_dUy[:, 1:-1, 1:]
            Fy = hllc_flux(UL_y, UR_y, axis=1, gamma=gamma)
            dFdy = (Fy[:, :, 1:] - Fy[:, :, :-1]) * inv_dy

            return dFdx, dFdy, None

        else:  # 3D
            # Extended cells: interior + 1 ghost layer on each side
            Qc_ext = U_pad[:, 1:-1, 1:-1, 1:-1]  # Shape: (Nvar, Nx+2, Ny+2, Nz+2)

            # X-direction slopes
            dx_L = Qc_ext - U_pad[:, :-2, 1:-1, 1:-1]
            dx_R = U_pad[:, 2:, 1:-1, 1:-1] - Qc_ext
            dUx_ext = slope_limiter(dx_L, dx_R, limiter)

            # Y-direction slopes
            dy_D = Qc_ext - U_pad[:, 1:-1, :-2, 1:-1]
            dy_U = U_pad[:, 1:-1, 2:, 1:-1] - Qc_ext
            dUy_ext = slope_limiter(dy_D, dy_U, limiter)

            # Z-direction slopes
            dz_B = Qc_ext - U_pad[:, 1:-1, 1:-1, :-2]
            dz_F = U_pad[:, 1:-1, 1:-1, 2:] - Qc_ext
            dUz_ext = slope_limiter(dz_B, dz_F, limiter)

            half_dUx = 0.5 * dUx_ext
            half_dUy = 0.5 * dUy_ext
            half_dUz = 0.5 * dUz_ext

            # X-fluxes: need y,z to be in interior range
            UL_x = Qc_ext[:, :-1, 1:-1, 1:-1] + half_dUx[:, :-1, 1:-1, 1:-1]
            UR_x = Qc_ext[:, 1:, 1:-1, 1:-1] - half_dUx[:, 1:, 1:-1, 1:-1]
            Fx = hllc_flux(UL_x, UR_x, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:, :, :] - Fx[:, :-1, :, :]) * inv_dx

            # Y-fluxes
            UL_y = Qc_ext[:, 1:-1, :-1, 1:-1] + half_dUy[:, 1:-1, :-1, 1:-1]
            UR_y = Qc_ext[:, 1:-1, 1:, 1:-1] - half_dUy[:, 1:-1, 1:, 1:-1]
            Fy = hllc_flux(UL_y, UR_y, axis=1, gamma=gamma)
            dFdy = (Fy[:, :, 1:, :] - Fy[:, :, :-1, :]) * inv_dy

            # Z-fluxes
            UL_z = Qc_ext[:, 1:-1, 1:-1, :-1] + half_dUz[:, 1:-1, 1:-1, :-1]
            UR_z = Qc_ext[:, 1:-1, 1:-1, 1:] - half_dUz[:, 1:-1, 1:-1, 1:]
            Fz = hllc_flux(UL_z, UR_z, axis=2, gamma=gamma)
            dFdz = (Fz[:, :, :, 1:] - Fz[:, :, :, :-1]) * inv_dz

            return dFdx, dFdy, dFdz

    # ORDER SELECTION
    is_first_order = jnp.array(order == 0)
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


@partial(jit, static_argnums=(2, 3))
def flux_divergence_from_padded(U_pad, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz):
    """
    Compute flux divergence from ALREADY PADDED data.

    For MPI mode: ghost cells are pre-filled by halo exchange and boundary conditions.
    This skips the pad() call entirely.

    U_pad: (Nvar, Nx+2) or (Nvar, Nx+2, Ny+2) or (Nvar, Nx+2, Ny+2, Nz+2)
           Data with 1 ghost cell per side already filled.
    Returns: -(dFdx + dFdy + dFdz) with shape (Nvar, Nx) or (Nvar, Nx, Ny) or (Nvar, Nx, Ny, Nz)
    """
    ndim = U_pad.ndim - 1  # number of spatial dimensions

    # FIRST-ORDER PATH
    def first_order_divergence(U_pad):
        if ndim == 1:
            U_Lx = U_pad[:, :-1]
            U_Rx = U_pad[:, 1:]
            Fx = hllc_flux(U_Lx, U_Rx, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:] - Fx[:, :-1]) * inv_dx
            return dFdx, None, None
        elif ndim == 2:
            U_Lx = U_pad[:, :-1, 1:-1]
            U_Rx = U_pad[:, 1: , 1:-1]
            Fx = hllc_flux(U_Lx, U_Rx, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:, :] - Fx[:, :-1, :]) * inv_dx

            U_By = U_pad[:, 1:-1, :-1]
            U_Ty = U_pad[:, 1:-1, 1: ]
            Fy = hllc_flux(U_By, U_Ty, axis=1, gamma=gamma)
            dFdy = (Fy[:, :, 1:] - Fy[:, :, :-1]) * inv_dy
            return dFdx, dFdy, None
        else:  # 3D
            U_Lx = U_pad[:, :-1, 1:-1, 1:-1]
            U_Rx = U_pad[:, 1: , 1:-1, 1:-1]
            Fx = hllc_flux(U_Lx, U_Rx, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:, :, :] - Fx[:, :-1, :, :]) * inv_dx

            U_By = U_pad[:, 1:-1, :-1, 1:-1]
            U_Ty = U_pad[:, 1:-1, 1: , 1:-1]
            Fy = hllc_flux(U_By, U_Ty, axis=1, gamma=gamma)
            dFdy = (Fy[:, :, 1:, :] - Fy[:, :, :-1, :]) * inv_dy

            U_Bz = U_pad[:, 1:-1, 1:-1, :-1]
            U_Fz = U_pad[:, 1:-1, 1:-1, 1: ]
            Fz = hllc_flux(U_Bz, U_Fz, axis=2, gamma=gamma)
            dFdz = (Fz[:, :, :, 1:] - Fz[:, :, :, :-1]) * inv_dz
            return dFdx, dFdy, dFdz

    # SECOND-ORDER PATH
    def second_order_divergence(U_pad):
        dUx, dUy, dUz = compute_slopes_from_padded(U_pad, limiter)

        if ndim == 1:
            Qc = U_pad[:, 1:-1]
            half_dUx = 0.5 * dUx
            UL = jnp.concatenate([U_pad[:, 0:1], Qc + half_dUx], axis=1)
            UR = jnp.concatenate([Qc - half_dUx, U_pad[:, -1:]], axis=1)
            Fx = hllc_flux(UL, UR, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:] - Fx[:, :-1]) * inv_dx
            return dFdx, None, None

        elif ndim == 2:
            Qc = U_pad[:, 1:-1, 1:-1]
            half_dUx = 0.5 * dUx
            half_dUy = 0.5 * dUy

            UL_x = jnp.concatenate([U_pad[:, 0:1, 1:-1], Qc + half_dUx], axis=1)
            UR_x = jnp.concatenate([Qc - half_dUx, U_pad[:, -1:, 1:-1]], axis=1)
            Fx = hllc_flux(UL_x, UR_x, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:, :] - Fx[:, :-1, :]) * inv_dx

            UL_y = jnp.concatenate([U_pad[:, 1:-1, 0:1], Qc + half_dUy], axis=2)
            UR_y = jnp.concatenate([Qc - half_dUy, U_pad[:, 1:-1, -1:]], axis=2)
            Fy = hllc_flux(UL_y, UR_y, axis=1, gamma=gamma)
            dFdy = (Fy[:, :, 1:] - Fy[:, :, :-1]) * inv_dy

            return dFdx, dFdy, None

        else:  # 3D
            Qc = U_pad[:, 1:-1, 1:-1, 1:-1]
            half_dUx = 0.5 * dUx
            half_dUy = 0.5 * dUy
            half_dUz = 0.5 * dUz

            UL_x = jnp.concatenate([U_pad[:, 0:1, 1:-1, 1:-1], Qc + half_dUx], axis=1)
            UR_x = jnp.concatenate([Qc - half_dUx, U_pad[:, -1:, 1:-1, 1:-1]], axis=1)
            Fx = hllc_flux(UL_x, UR_x, axis=0, gamma=gamma)
            dFdx = (Fx[:, 1:, :, :] - Fx[:, :-1, :, :]) * inv_dx

            UL_y = jnp.concatenate([U_pad[:, 1:-1, 0:1, 1:-1], Qc + half_dUy], axis=2)
            UR_y = jnp.concatenate([Qc - half_dUy, U_pad[:, 1:-1, -1:, 1:-1]], axis=2)
            Fy = hllc_flux(UL_y, UR_y, axis=1, gamma=gamma)
            dFdy = (Fy[:, :, 1:, :] - Fy[:, :, :-1, :]) * inv_dy

            UL_z = jnp.concatenate([U_pad[:, 1:-1, 1:-1, 0:1], Qc + half_dUz], axis=3)
            UR_z = jnp.concatenate([Qc - half_dUz, U_pad[:, 1:-1, 1:-1, -1:]], axis=3)
            Fz = hllc_flux(UL_z, UR_z, axis=2, gamma=gamma)
            dFdz = (Fz[:, :, :, 1:] - Fz[:, :, :, :-1]) * inv_dz

            return dFdx, dFdy, dFdz

    # ORDER SELECTION
    is_first_order = jnp.array(order == 0)
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

