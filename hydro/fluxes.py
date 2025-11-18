# fluxes.py
"""
Unified HLLC Riemann solver for 1D and 2D Euler equations.
Follows the old 2D code line by line.
"""

import jax
import jax.numpy as jnp

def hllc_flux(U_L, U_R, axis, gamma):
    """
    HLLC Riemann solver for 1D and 2D.

    Parameters
    ----------
    U_L, U_R : array
        Left and right conservative states.
        1D: (3, Nx)
        2D: (4, Nx, Ny)
        
    axis : int
        0 → x-direction flux
        1 → y-direction flux

    gamma : float
        Ratio of specific heats
    """

    Nvar = U_L.shape[0]

    # --- 1D case ---
    if Nvar == 3:
        rho_L, rhou_L, E_L = U_L
        rho_R, rhou_R, E_R = U_R

        # velocities
        uL = rhou_L / rho_L
        uR = rhou_R / rho_R

        # pressure
        p_L = (gamma - 1.0) * (E_L - 0.5 * rho_L * uL**2)
        p_R = (gamma - 1.0) * (E_R - 0.5 * rho_R * uR**2)

        # sound speed
        c_L = jnp.sqrt(gamma * p_L / rho_L)
        c_R = jnp.sqrt(gamma * p_R / rho_R)

        # wave speeds
        S_L = jnp.minimum(uL - c_L, uR - c_R)
        S_R = jnp.maximum(uL + c_L, uR + c_R)

        # middle speed
        S_M = (p_R - p_L + rho_L*uL*(S_L - uL) - rho_R*uR*(S_R - uR)) / \
              (rho_L*(S_L - uL) - rho_R*(S_R - uR))

        # star states
        rho_star_L = rho_L * (S_L - uL) / (S_L - S_M)
        rho_star_R = rho_R * (S_R - uR) / (S_R - S_M)
        E_star_L = E_L / rho_L + (S_M - uL)*(S_M + p_L / (rho_L*(S_L - uL)))
        E_star_R = E_R / rho_R + (S_M - uR)*(S_M + p_R / (rho_R*(S_R - uR)))

        # star states vectors
        U_star_L = jnp.stack([rho_star_L, rho_star_L*S_M, rho_star_L*E_star_L])
        U_star_R = jnp.stack([rho_star_R, rho_star_R*S_M, rho_star_R*E_star_R])

        # fluxes
        F_L = jnp.stack([rho_L*uL, rho_L*uL**2 + p_L, (E_L + p_L)*uL])
        F_R = jnp.stack([rho_R*uR, rho_R*uR**2 + p_R, (E_R + p_R)*uR])

        # final flux
        flux = jnp.where(
            S_L >= 0, F_L,
            jnp.where(
                S_M >= 0,
                F_L + S_L*(U_star_L - U_L),
                jnp.where(
                    S_R > 0,
                    F_R + S_R*(U_star_R - U_R),
                    F_R
                )
            )
        )

    # --- 2D case ---
    elif Nvar == 4:
        rho_L, rhou_L, rhov_L, E_L = U_L
        rho_R, rhou_R, rhov_R, E_R = U_R

        if axis == 0:  # x-direction flux
            uL_n, uL_t = rhou_L / rho_L, rhov_L / rho_L
            uR_n, uR_t = rhou_R / rho_R, rhov_R / rho_R
            momL_n, momL_t = rhou_L, rhov_L
            momR_n, momR_t = rhou_R, rhov_R

            # pressure
            p_L = (gamma - 1.0) * (E_L - 0.5*rho_L*(uL_n**2 + uL_t**2))
            p_R = (gamma - 1.0) * (E_R - 0.5*rho_R*(uR_n**2 + uR_t**2))

            # sound speed
            c_L = jnp.sqrt(gamma * p_L / rho_L)
            c_R = jnp.sqrt(gamma * p_R / rho_R)

            # wave speeds
            S_L = jnp.minimum(uL_n - c_L, uR_n - c_R)
            S_R = jnp.maximum(uL_n + c_L, uR_n + c_R)

            # middle speed
            S_M = (p_R - p_L + rho_L*uL_n*(S_L - uL_n) - rho_R*uR_n*(S_R - uR_n)) / \
                  (rho_L*(S_L - uL_n) - rho_R*(S_R - uR_n))

            # star states
            rho_star_L = rho_L*(S_L - uL_n)/(S_L - S_M)
            rho_star_R = rho_R*(S_R - uR_n)/(S_R - S_M)
            E_star_L = E_L/rho_L + (S_M - uL_n)*(S_M + p_L/(rho_L*(S_L - uL_n)))
            E_star_R = E_R/rho_R + (S_M - uR_n)*(S_M + p_R/(rho_R*(S_R - uR_n)))

            U_star_L = jnp.stack([rho_star_L, rho_star_L*S_M, rho_star_L*uL_t, rho_star_L*E_star_L])
            U_star_R = jnp.stack([rho_star_R, rho_star_R*S_M, rho_star_R*uR_t, rho_star_R*E_star_R])

            F_L = jnp.stack([momL_n, momL_n*uL_n + p_L, momL_n*uL_t, (E_L + p_L)*uL_n])
            F_R = jnp.stack([momR_n, momR_n*uR_n + p_R, momR_n*uR_t, (E_R + p_R)*uR_n])

        else:  # y-direction flux
            uL_n, uL_t = rhov_L / rho_L, rhou_L / rho_L
            uR_n, uR_t = rhov_R / rho_R, rhou_R / rho_R
            momL_n, momL_t = rhov_L, rhou_L
            momR_n, momR_t = rhov_R, rhou_R

            # pressure
            p_L = (gamma - 1.0) * (E_L - 0.5*rho_L*(uL_n**2 + uL_t**2))
            p_R = (gamma - 1.0) * (E_R - 0.5*rho_R*(uR_n**2 + uR_t**2))

            # sound speed
            c_L = jnp.sqrt(gamma * p_L / rho_L)
            c_R = jnp.sqrt(gamma * p_R / rho_R)

            # wave speeds
            S_L = jnp.minimum(uL_n - c_L, uR_n - c_R)
            S_R = jnp.maximum(uL_n + c_L, uR_n + c_R)

            # middle speed
            S_M = (p_R - p_L + rho_L*uL_n*(S_L - uL_n) - rho_R*uR_n*(S_R - uR_n)) / \
                  (rho_L*(S_L - uL_n) - rho_R*(S_R - uR_n))

            # star states
            rho_star_L = rho_L*(S_L - uL_n)/(S_L - S_M)
            rho_star_R = rho_R*(S_R - uR_n)/(S_R - S_M)
            E_star_L = E_L/rho_L + (S_M - uL_n)*(S_M + p_L/(rho_L*(S_L - uL_n)))
            E_star_R = E_R/rho_R + (S_M - uR_n)*(S_M + p_R/(rho_R*(S_R - uR_n)))

            U_star_L = jnp.stack([rho_star_L, rho_star_L*uL_t, rho_star_L*S_M, rho_star_L*E_star_L])
            U_star_R = jnp.stack([rho_star_R, rho_star_R*uR_t, rho_star_R*S_M, rho_star_R*E_star_R])

            F_L = jnp.stack([momL_n, momL_n*uL_t, momL_n*uL_n + p_L, (E_L + p_L)*uL_n])
            F_R = jnp.stack([momR_n, momR_n*uR_t, momR_n*uR_n + p_R, (E_R + p_R)*uR_n])

        # final flux
        flux = jnp.where(
            S_L >= 0, F_L,
            jnp.where(
                S_M >= 0,
                F_L + S_L*(U_star_L - U_L),
                jnp.where(
                    S_R > 0,
                    F_R + S_R*(U_star_R - U_R),
                    F_R
                )
            )
        )

    else:
        raise ValueError("HLLC solver only supports 1D (3 vars) or 2D (4 vars)")

    return flux

