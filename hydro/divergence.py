from utils.padding import periodic_pad
from hydro.slopes import compute_slopes
from hydro.reconstruction import plm_reconstruct
from hydro.fluxes import hllc_flux
import jax.numpy as jnp
from jax import jit, lax

@jit
def flux_divergence(U, dx, dy, gamma, order, limiter):
    """
    Compute divergence of fluxes for 2D Euler equations.
    
    Parameters
    ----------
    U : array
        Conservative variables (4, Nx, Ny)
    dx, dy : float
        Grid spacing
    gamma : float
        Adiabatic index
    order : int
        Reconstruction order: 0 → first-order, 1 → second-order PLM
    limiter : float
        Slope limiter parameter for PLM

    Returns
    -------
    divF : array
        Negative divergence of fluxes (4, Nx, Ny)
    """

    # Pad for periodic boundaries
    U_pad = periodic_pad(U)

    # --- First-order fluxes ---
    Fx = hllc_flux(U_pad[:, :-1, 1:-1], U_pad[:, 1:, 1:-1], axis=0, gamma=gamma)
    Fy = hllc_flux(U_pad[:, 1:-1, :-1], U_pad[:, 1:-1, 1:], axis=1, gamma=gamma)

    dFdx = (Fx[:, 1:, :] - Fx[:, :-1, :]) / dx
    dFdy = (Fy[:, :, 1:] - Fy[:, :, :-1]) / dy

    # --- Higher-order fluxes ---
    def second_order_fluxes(U_pad, dx, dy, gamma, limiter, dFdx, dFdy):
        # Compute slopes using the limiter
        dUx, dUy = compute_slopes(U_pad, limiter)
        Qc = U_pad[:, 1:-1, 1:-1]  # interior (4, Nx, Ny)

        # ----- X-direction reconstruction -----
        UL = Qc[:, :-1, :] + 0.5 * dUx[:, :-1, :]  # left state at i+1/2
        UR = Qc[:, 1:, :] - 0.5 * dUx[:, 1:, :]    # right state at i+1/2

        Fx2 = hllc_flux(UL, UR, axis=0, gamma=gamma)  # shape (4, Nx-1, Ny)
        dFdx2 = (Fx2[:, 1:, :] - Fx2[:, :-1, :]) / dx
        dFdx = dFdx.at[:, 1:-1, :].set(dFdx2)  # update interior only

        # ----- Y-direction reconstruction -----
        UB = Qc[:, :, :-1] + 0.5 * dUy[:, :, :-1]  # bottom state at j+1/2
        UT = Qc[:, :, 1:] - 0.5 * dUy[:, :, 1:]    # top state at j+1/2

        Fy2 = hllc_flux(UB, UT, axis=1, gamma=gamma)  # shape (4, Nx, Ny-1)
        dFdy2 = (Fy2[:, :, 1:] - Fy2[:, :, :-1]) / dy
        dFdy = dFdy.at[:, :, 1:-1].set(dFdy2)  # update interior only

        return dFdx, dFdy

    # Conditionally apply second-order fluxes (PLM) if order == 1
    dFdx, dFdy = lax.cond(
        order == 1,  # If order is 1, apply second-order fluxes (PLM)
        second_order_fluxes,  # Function to apply second-order fluxes
        lambda *args: (dFdx, dFdy),  # Otherwise, keep the first-order fluxes
        U_pad, dx, dy, gamma, limiter, dFdx, dFdy
    )

    # Return the negative divergence of fluxes
    return -(dFdx + dFdy)

