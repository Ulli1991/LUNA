from utils.padding_full import pad
from hydro.slopes_new2 import compute_slopes
from hydro.fluxes_new import hllc_flux
import jax.numpy as jnp
from jax import jit, lax

@jit
def flux_divergence(U, gamma, order, limiter, dx, dy=None, dz=None, boundary=0):
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
    else:  # 2D
        U_Lx = U_pad[:, :-1, 1:-1]
        U_Rx = U_pad[:, 1: , 1:-1]
        Fx = hllc_flux(U_Lx, U_Rx, axis=0, gamma=gamma)
        dFdx = (Fx[:, 1:, :] - Fx[:, :-1, :]) / dx

        U_By = U_pad[:, 1:-1, :-1]
        U_Ty = U_pad[:, 1:-1, 1: ]
        Fy = hllc_flux(U_By, U_Ty, axis=1, gamma=gamma)
        dFdy = (Fy[:, :, 1:] - Fy[:, :, :-1]) / dy

    # -------------------
    # SECOND-ORDER PLM
    # -------------------
    def second_order(U_pad, dx, dy, gamma, limiter, dFdx, dFdy):
        dUx, dUy = compute_slopes(U, limiter, boundary)  # <-- pass boundary
        if ndim == 1:
            Qc = U_pad[:, 1:-1]
            UL = Qc[:, :-1] + 0.5 * dUx[:, :-1]
            UR = Qc[:, 1:]  - 0.5 * dUx[:, 1:]
            Fx2 = hllc_flux(UL, UR, axis=0, gamma=gamma)
            dFdx2 = (Fx2[:, 1:] - Fx2[:, :-1]) / dx
            dFdx = dFdx.at[:, 1:-1].set(dFdx2)
            return dFdx, None
        else:  # 2D
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

            return dFdx, dFdy

    # -------------------
    # ORDER SELECTION (JIT-safe)
    # -------------------
    is_first_order = jnp.array(order == 0)  # 0 = first-order, 1 = PLM
    operand = (U_pad, dx, dy, gamma, limiter, dFdx, dFdy)

    def first_order_fun(ops):
        _, _, _, _, _, dFdx, dFdy = ops
        return dFdx, dFdy

    def second_order_fun(ops):
        U_pad, dx, dy, gamma, limiter, dFdx, dFdy = ops
        return second_order(U_pad, dx, dy, gamma, limiter, dFdx, dFdy)

    dFdx, dFdy = lax.cond(is_first_order, first_order_fun, second_order_fun, operand)

    if ndim == 1:
        return -dFdx
    else:
        return -(dFdx + dFdy)

