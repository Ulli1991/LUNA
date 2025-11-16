# fluxes.py
"""
Unified HLLC Riemann solver for 2D Euler equations.
Used for both x- and y-direction flux calculations.
"""

import jax
import jax.numpy as jnp

# -------------------------------------------------------------
# HLLC FLUX FUNCTION
# -------------------------------------------------------------
def hllc_flux(U_L, U_R, axis, gamma):
    """
    Direction-agnostic HLLC Riemann solver.

    Parameters
    ----------
    U_L, U_R : arrays with shape (4, Nx, Ny or interface count)
        Left and right conservative states.

    axis : int
        0 → x-direction flux
        1 → y-direction flux

    gamma : float
        Adiabatic index (ratio of specific heats)

    Returns
    -------
    Flux : array (4, ...)
        HLLC flux in the selected direction.
    """

    rho_L, rhou_L, rhov_L, E_L = U_L
    rho_R, rhou_R, rhov_R, E_R = U_R

    # Select normal & transverse velocities depending on axis
    if axis == 0:  # x-direction
        uL_n, uL_t = rhou_L / rho_L, rhov_L / rho_L
        uR_n, uR_t = rhou_R / rho_R, rhov_R / rho_R
        momL_n, momL_t = rhou_L, rhov_L
        momR_n, momR_t = rhou_R, rhov_R
    else:          # y-direction
        uL_n, uL_t = rhov_L / rho_L, rhou_L / rho_L
        uR_n, uR_t = rhov_R / rho_R, rhou_R / rho_R
        momL_n, momL_t = rhov_L, rhou_L
        momR_n, momR_t = rhov_R, rhou_R

    # Thermodynamic quantities
    p_L = (gamma - 1.0) * (E_L - 0.5 * rho_L * (uL_n**2 + uL_t**2))
    p_R = (gamma - 1.0) * (E_R - 0.5 * rho_R * (uR_n**2 + uR_t**2))

    c_L = jnp.sqrt(gamma * p_L / rho_L)
    c_R = jnp.sqrt(gamma * p_R / rho_R)

    # Wave speeds
    S_L = jnp.minimum(uL_n - c_L, uR_n - c_R)
    S_R = jnp.maximum(uL_n + c_L, uR_n + c_R)

    S_M = (p_R - p_L + rho_L * uL_n * (S_L - uL_n) - rho_R * uR_n * (S_R - uR_n)) / \
          (rho_L * (S_L - uL_n) - rho_R * (S_R - uR_n))

    # Star region densities
    rho_star_L = rho_L * (S_L - uL_n) / (S_L - S_M)
    rho_star_R = rho_R * (S_R - uR_n) / (S_R - S_M)

    # Star region energies per mass
    E_star_L = E_L / rho_L + (S_M - uL_n) * (S_M + p_L / (rho_L * (S_L - uL_n)))
    E_star_R = E_R / rho_R + (S_M - uR_n) * (S_M + p_R / (rho_R * (S_R - uR_n)))

    # Construct star states and fluxes
    if axis == 0:
        U_star_L = jnp.stack([rho_star_L, rho_star_L * S_M, rho_star_L * uL_t, rho_star_L * E_star_L], axis=0)
        U_star_R = jnp.stack([rho_star_R, rho_star_R * S_M, rho_star_R * uR_t, rho_star_R * E_star_R], axis=0)

        F_L = jnp.stack([momL_n, momL_n * uL_n + p_L, momL_n * uL_t, (E_L + p_L) * uL_n], axis=0)
        F_R = jnp.stack([momR_n, momR_n * uR_n + p_R, momR_n * uR_t, (E_R + p_R) * uR_n], axis=0)
    else:  # axis == 1 (y-direction)
        U_star_L = jnp.stack([rho_star_L, rho_star_L * uL_t, rho_star_L * S_M, rho_star_L * E_star_L], axis=0)
        U_star_R = jnp.stack([rho_star_R, rho_star_R * uR_t, rho_star_R * S_M, rho_star_R * E_star_R], axis=0)

        F_L = jnp.stack([momL_n, momL_n * uL_t, momL_n * uL_n + p_L, (E_L + p_L) * uL_n], axis=0)
        F_R = jnp.stack([momR_n, momR_n * uR_t, momR_n * uR_n + p_R, (E_R + p_R) * uR_n], axis=0)

    # Final HLLC flux decision
    flux = jnp.where(
        S_L >= 0, F_L,
        jnp.where(
            S_M >= 0,
            F_L + S_L * (U_star_L - U_L),
            jnp.where(
                S_R > 0,
                F_R + S_R * (U_star_R - U_R),
                F_R
            )
        )
    )

    return flux

