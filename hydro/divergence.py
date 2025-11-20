import jax.numpy as jnp
from jax import jit, lax

from hydro.padding import pad
from hydro.slopes import compute_slopes
from hydro.fluxes import hllc_flux

@jit
def flux_divergence(U, gamma, order, limiter, boundary, dx, dy=None, dz=None):
    """
    Compute the flux divergence in 1D or 2D using first-order or second-order PLM.
    
    U: (Nvar, Nx) or (Nvar, Nx, Ny)
    boundary: int, 0=periodic, 1=zero-gradient, 2=reflective
    Returns: -(dFdx + dFdy) with shapes matching U
    """
    U_pad = pad(U, boundary)  # <-- use new general padding
    ndim = U.ndim - 1  # number of spatial dimensions
    Nx = U.shape[1]
    if ndim == 2:
        Ny = U.shape[2]

    # -------------------
    # FIRST-ORDER FLUXES
    # -------------------
    if ndim == 1:
        U_Lx = U_pad[:, :-1]
        U_Rx = U_pad[:, 1:]
        Fx = hllc_flux(U_Lx, U_Rx, axis=0, gamma=gamma)
        dFdx = (Fx[:, 1:] - Fx[:, :-1]) / dx
        dFdy = None
        dFdz = None
    elif ndim == 2:
        U_Lx = U_pad[:, :-1, 1:-1]
        U_Rx = U_pad[:, 1: , 1:-1]
        Fx = hllc_flux(U_Lx, U_Rx, axis=0, gamma=gamma)
        dFdx = (Fx[:, 1:, :] - Fx[:, :-1, :]) / dx

        U_By = U_pad[:, 1:-1, :-1]
        U_Ty = U_pad[:, 1:-1, 1: ]
        Fy = hllc_flux(U_By, U_Ty, axis=1, gamma=gamma)
        dFdy = (Fy[:, :, 1:] - Fy[:, :, :-1]) / dy
        dFdz = None
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

    # -------------------
    # SECOND-ORDER PLM
    # -------------------
    def second_order(U_pad, dx, dy, dz, gamma, limiter, boundary, dFdx, dFdy, dFdz):
        dUx, dUy, dUz = compute_slopes(U, limiter, boundary)  # <-- now returns 3 values
        if ndim == 1:
            Qc = U_pad[:, 1:-1]
            UL = Qc[:, :-1] + 0.5 * dUx[:, :-1]
            UR = Qc[:, 1:]  - 0.5 * dUx[:, 1:]
            Fx2 = hllc_flux(UL, UR, axis=0, gamma=gamma)
            dFdx2 = (Fx2[:, 1:] - Fx2[:, :-1]) / dx
            dFdx = dFdx.at[:, 1:-1].set(dFdx2)
            return dFdx, None, None
        elif ndim == 2:
            Qc = U_pad[:, 1:-1, 1:-1]

            # X-direction reconstruction
            UL = Qc[:, :-1, :] + 0.5 * dUx[:, :-1, :]
            UR = Qc[:, 1:,  :] - 0.5 * dUx[:, 1:, :]
            Fx2 = hllc_flux(UL, UR, axis=0, gamma=gamma)
            dFdx2 = (Fx2[:, 1:, :] - Fx2[:, :-1, :]) / dx
            dFdx = dFdx.at[:, 1:-1, :].set(dFdx2)

            # Y-direction reconstruction
            UB = Qc[:, :, :-1] + 0.5 * dUy[:, :, :-1]
            UT = Qc[:, :, 1:]  - 0.5 * dUy[:, :, 1:]
            Fy2 = hllc_flux(UB, UT, axis=1, gamma=gamma)
            dFdy2 = (Fy2[:, :, 1:] - Fy2[:, :, :-1]) / dy
            dFdy = dFdy.at[:, :, 1:-1].set(dFdy2)

            return dFdx, dFdy, None
        else:  # 3D
            Qc = U_pad[:, 1:-1, 1:-1, 1:-1]

            # X-direction reconstruction
            UL = Qc[:, :-1, :, :] + 0.5 * dUx[:, :-1, :, :]
            UR = Qc[:, 1:,  :, :] - 0.5 * dUx[:, 1:, :, :]
            Fx2 = hllc_flux(UL, UR, axis=0, gamma=gamma)
            dFdx2 = (Fx2[:, 1:, :, :] - Fx2[:, :-1, :, :]) / dx
            dFdx = dFdx.at[:, 1:-1, :, :].set(dFdx2)

            # Y-direction reconstruction
            UB = Qc[:, :, :-1, :] + 0.5 * dUy[:, :, :-1, :]
            UT = Qc[:, :, 1:,  :] - 0.5 * dUy[:, :, 1:, :]
            Fy2 = hllc_flux(UB, UT, axis=1, gamma=gamma)
            dFdy2 = (Fy2[:, :, 1:, :] - Fy2[:, :, :-1, :]) / dy
            dFdy = dFdy.at[:, :, 1:-1, :].set(dFdy2)

            # Z-direction reconstruction
            UBz = Qc[:, :, :, :-1] + 0.5 * dUz[:, :, :, :-1]
            UFz = Qc[:, :, :, 1:]  - 0.5 * dUz[:, :, :, 1:]
            Fz2 = hllc_flux(UBz, UFz, axis=2, gamma=gamma)
            dFdz2 = (Fz2[:, :, :, 1:] - Fz2[:, :, :, :-1]) / dz
            dFdz = dFdz.at[:, :, :, 1:-1].set(dFdz2)

            return dFdx, dFdy, dFdz

    # -------------------
    # ORDER SELECTION (JIT-safe)
    # -------------------
    is_first_order = jnp.array(order == 0)  # 0 = first-order, 1 = PLM
    operand = (U_pad, dx, dy, dz, gamma, limiter, boundary, dFdx, dFdy, dFdz)

    def first_order_fun(ops):
        _, _, _, _, _, _, _, dFdx, dFdy, dFdz = ops
        return dFdx, dFdy, dFdz

    def second_order_fun(ops):
        U_pad, dx, dy, dz, gamma, limiter, boundary, dFdx, dFdy, dFdz = ops
        return second_order(U_pad, dx, dy, dz, gamma, limiter, boundary, dFdx, dFdy, dFdz)

    dFdx, dFdy, dFdz = lax.cond(is_first_order, first_order_fun, second_order_fun, operand)

    if ndim == 1:
        return -dFdx
    elif ndim == 2:
        return -(dFdx + dFdy)
    else:  # 3D
        return -(dFdx + dFdy + dFdz)

