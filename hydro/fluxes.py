# fluxes.py
"""
Unified HLLC Riemann solver for 1D and 2D Euler equations.
Follows the old 2D code line by line.
"""

import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(2,))
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

    # --- 3D case ---
    elif Nvar == 5:
        rho_L, rhou_L, rhov_L, rhow_L, E_L = U_L
        rho_R, rhou_R, rhov_R, rhow_R, E_R = U_R

        if axis == 0:  # x-direction flux
            # Normal velocity: vx, Tangent velocities: vy, vz
            uL_n, uL_t1, uL_t2 = rhou_L / rho_L, rhov_L / rho_L, rhow_L / rho_L
            uR_n, uR_t1, uR_t2 = rhou_R / rho_R, rhov_R / rho_R, rhow_R / rho_R
            momL_n, momL_t1, momL_t2 = rhou_L, rhov_L, rhow_L
            momR_n, momR_t1, momR_t2 = rhou_R, rhov_R, rhow_R

            # Pressure (includes all 3 velocity components)
            p_L = (gamma - 1.0) * (E_L - 0.5*rho_L*(uL_n**2 + uL_t1**2 + uL_t2**2))
            p_R = (gamma - 1.0) * (E_R - 0.5*rho_R*(uR_n**2 + uR_t1**2 + uR_t2**2))

            # Sound speed
            c_L = jnp.sqrt(gamma * p_L / rho_L)
            c_R = jnp.sqrt(gamma * p_R / rho_R)

            # Wave speeds (based on normal velocity)
            S_L = jnp.minimum(uL_n - c_L, uR_n - c_R)
            S_R = jnp.maximum(uL_n + c_L, uR_n + c_R)

            # Middle speed
            S_M = (p_R - p_L + rho_L*uL_n*(S_L - uL_n) - rho_R*uR_n*(S_R - uR_n)) / \
                  (rho_L*(S_L - uL_n) - rho_R*(S_R - uR_n))

            # Star states
            rho_star_L = rho_L*(S_L - uL_n)/(S_L - S_M)
            rho_star_R = rho_R*(S_R - uR_n)/(S_R - S_M)
            E_star_L = E_L/rho_L + (S_M - uL_n)*(S_M + p_L/(rho_L*(S_L - uL_n)))
            E_star_R = E_R/rho_R + (S_M - uR_n)*(S_M + p_R/(rho_R*(S_R - uR_n)))

            # Star state vectors: [rho, rho*vx_new, rho*vy, rho*vz, rho*E]
            U_star_L = jnp.stack([rho_star_L, rho_star_L*S_M, rho_star_L*uL_t1, rho_star_L*uL_t2, rho_star_L*E_star_L])
            U_star_R = jnp.stack([rho_star_R, rho_star_R*S_M, rho_star_R*uR_t1, rho_star_R*uR_t2, rho_star_R*E_star_R])

            # Fluxes in x-direction: [rho*vx, rho*vx*vx + p, rho*vx*vy, rho*vx*vz, (E+p)*vx]
            F_L = jnp.stack([momL_n, momL_n*uL_n + p_L, momL_n*uL_t1, momL_n*uL_t2, (E_L + p_L)*uL_n])
            F_R = jnp.stack([momR_n, momR_n*uR_n + p_R, momR_n*uR_t1, momR_n*uR_t2, (E_R + p_R)*uR_n])

        elif axis == 1:  # y-direction flux
            # Normal velocity: vy, Tangent velocities: vx, vz
            uL_n, uL_t1, uL_t2 = rhov_L / rho_L, rhou_L / rho_L, rhow_L / rho_L
            uR_n, uR_t1, uR_t2 = rhov_R / rho_R, rhou_R / rho_R, rhow_R / rho_R
            momL_n, momL_t1, momL_t2 = rhov_L, rhou_L, rhow_L
            momR_n, momR_t1, momR_t2 = rhov_R, rhou_R, rhow_R

            # Pressure
            p_L = (gamma - 1.0) * (E_L - 0.5*rho_L*(uL_n**2 + uL_t1**2 + uL_t2**2))
            p_R = (gamma - 1.0) * (E_R - 0.5*rho_R*(uR_n**2 + uR_t1**2 + uR_t2**2))

            # Sound speed
            c_L = jnp.sqrt(gamma * p_L / rho_L)
            c_R = jnp.sqrt(gamma * p_R / rho_R)

            # Wave speeds
            S_L = jnp.minimum(uL_n - c_L, uR_n - c_R)
            S_R = jnp.maximum(uL_n + c_L, uR_n + c_R)

            # Middle speed
            S_M = (p_R - p_L + rho_L*uL_n*(S_L - uL_n) - rho_R*uR_n*(S_R - uR_n)) / \
                  (rho_L*(S_L - uL_n) - rho_R*(S_R - uR_n))

            # Star states
            rho_star_L = rho_L*(S_L - uL_n)/(S_L - S_M)
            rho_star_R = rho_R*(S_R - uR_n)/(S_R - S_M)
            E_star_L = E_L/rho_L + (S_M - uL_n)*(S_M + p_L/(rho_L*(S_L - uL_n)))
            E_star_R = E_R/rho_R + (S_M - uR_n)*(S_M + p_R/(rho_R*(S_R - uR_n)))

            # Star state vectors: [rho, rho*vx, rho*vy_new, rho*vz, rho*E]
            U_star_L = jnp.stack([rho_star_L, rho_star_L*uL_t1, rho_star_L*S_M, rho_star_L*uL_t2, rho_star_L*E_star_L])
            U_star_R = jnp.stack([rho_star_R, rho_star_R*uR_t1, rho_star_R*S_M, rho_star_R*uR_t2, rho_star_R*E_star_R])

            # Fluxes in y-direction: [rho*vy, rho*vy*vx, rho*vy*vy + p, rho*vy*vz, (E+p)*vy]
            F_L = jnp.stack([momL_n, momL_n*uL_t1, momL_n*uL_n + p_L, momL_n*uL_t2, (E_L + p_L)*uL_n])
            F_R = jnp.stack([momR_n, momR_n*uR_t1, momR_n*uR_n + p_R, momR_n*uR_t2, (E_R + p_R)*uR_n])

        else:  # axis == 2, z-direction flux
            # Normal velocity: vz, Tangent velocities: vx, vy
            uL_n, uL_t1, uL_t2 = rhow_L / rho_L, rhou_L / rho_L, rhov_L / rho_L
            uR_n, uR_t1, uR_t2 = rhow_R / rho_R, rhou_R / rho_R, rhov_R / rho_R
            momL_n, momL_t1, momL_t2 = rhow_L, rhou_L, rhov_L
            momR_n, momR_t1, momR_t2 = rhow_R, rhou_R, rhov_R

            # Pressure
            p_L = (gamma - 1.0) * (E_L - 0.5*rho_L*(uL_n**2 + uL_t1**2 + uL_t2**2))
            p_R = (gamma - 1.0) * (E_R - 0.5*rho_R*(uR_n**2 + uR_t1**2 + uR_t2**2))

            # Sound speed
            c_L = jnp.sqrt(gamma * p_L / rho_L)
            c_R = jnp.sqrt(gamma * p_R / rho_R)

            # Wave speeds
            S_L = jnp.minimum(uL_n - c_L, uR_n - c_R)
            S_R = jnp.maximum(uL_n + c_L, uR_n + c_R)

            # Middle speed
            S_M = (p_R - p_L + rho_L*uL_n*(S_L - uL_n) - rho_R*uR_n*(S_R - uR_n)) / \
                  (rho_L*(S_L - uL_n) - rho_R*(S_R - uR_n))

            # Star states
            rho_star_L = rho_L*(S_L - uL_n)/(S_L - S_M)
            rho_star_R = rho_R*(S_R - uR_n)/(S_R - S_M)
            E_star_L = E_L/rho_L + (S_M - uL_n)*(S_M + p_L/(rho_L*(S_L - uL_n)))
            E_star_R = E_R/rho_R + (S_M - uR_n)*(S_M + p_R/(rho_R*(S_R - uR_n)))

            # Star state vectors: [rho, rho*vx, rho*vy, rho*vz_new, rho*E]
            U_star_L = jnp.stack([rho_star_L, rho_star_L*uL_t1, rho_star_L*uL_t2, rho_star_L*S_M, rho_star_L*E_star_L])
            U_star_R = jnp.stack([rho_star_R, rho_star_R*uR_t1, rho_star_R*uR_t2, rho_star_R*S_M, rho_star_R*E_star_R])

            # Fluxes in z-direction: [rho*vz, rho*vz*vx, rho*vz*vy, rho*vz*vz + p, (E+p)*vz]
            F_L = jnp.stack([momL_n, momL_n*uL_t1, momL_n*uL_t2, momL_n*uL_n + p_L, (E_L + p_L)*uL_n])
            F_R = jnp.stack([momR_n, momR_n*uR_t1, momR_n*uR_t2, momR_n*uR_n + p_R, (E_R + p_R)*uR_n])

        # Final flux (same HLLC logic for all directions)
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
        raise ValueError("HLLC solver only supports 1D (3 vars), 2D (4 vars), or 3D (5 vars)")

    return flux

