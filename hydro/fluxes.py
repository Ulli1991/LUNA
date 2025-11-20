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

        # Compute primitives (fused to reduce intermediates)
        uL = rhou_L / rho_L
        uR = rhou_R / rho_R

        # Kinetic energy (reused in pressure calculation)
        ke_L = 0.5 * rho_L * uL * uL
        ke_R = 0.5 * rho_R * uR * uR

        # Pressure
        gm1 = gamma - 1.0
        p_L = gm1 * (E_L - ke_L)
        p_R = gm1 * (E_R - ke_R)

        # Sound speed (fused computation)
        c_L = jnp.sqrt(gamma * p_L / rho_L)
        c_R = jnp.sqrt(gamma * p_R / rho_R)

        # Wave speeds
        S_L = jnp.minimum(uL - c_L, uR - c_R)
        S_R = jnp.maximum(uL + c_L, uR + c_R)

        # Cached quantities for star state computation
        denom_L = rho_L * (S_L - uL)
        denom_R = rho_R * (S_R - uR)

        # Middle speed (uses cached denominators to avoid recomputation)
        S_M = (p_R - p_L + denom_L*uL - denom_R*uR) / (denom_L - denom_R)

        # Star states (cache rho_star for reuse)
        rho_star_L = denom_L / (S_L - S_M)
        rho_star_R = denom_R / (S_R - S_M)
        E_star_L = E_L / rho_L + (S_M - uL)*(S_M + p_L / denom_L)
        E_star_R = E_R / rho_R + (S_M - uR)*(S_M + p_R / denom_R)

        # Star state vectors (needed for flux computation)
        U_star_L = jnp.stack([rho_star_L, rho_star_L*S_M, rho_star_L*E_star_L])
        U_star_R = jnp.stack([rho_star_R, rho_star_R*S_M, rho_star_R*E_star_R])

        # Flux vectors (cache rho*u to avoid recomputation)
        F_L = jnp.stack([rhou_L, rhou_L*uL + p_L, (E_L + p_L)*uL])
        F_R = jnp.stack([rhou_R, rhou_R*uR + p_R, (E_R + p_R)*uR])

        # Final flux selection using HLLC logic
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

        # Compute all velocities upfront (outside axis branches)
        uL_x = rhou_L / rho_L
        uR_x = rhou_R / rho_R
        uL_y = rhov_L / rho_L
        uR_y = rhov_R / rho_R

        if axis == 0:  # x-direction flux
            # Assign normal (x) and tangent (y) velocities
            uL_n, uL_t = uL_x, uL_y
            uR_n, uR_t = uR_x, uR_y
            momL_n, momL_t = rhou_L, rhov_L
            momR_n, momR_t = rhou_R, rhov_R

            # Kinetic energy (fused computation)
            ke_L = 0.5 * rho_L * (uL_n*uL_n + uL_t*uL_t)
            ke_R = 0.5 * rho_R * (uR_n*uR_n + uR_t*uR_t)

            # Pressure
            gm1 = gamma - 1.0
            p_L = gm1 * (E_L - ke_L)
            p_R = gm1 * (E_R - ke_R)

            # Sound speed
            c_L = jnp.sqrt(gamma * p_L / rho_L)
            c_R = jnp.sqrt(gamma * p_R / rho_R)

            # Wave speeds
            S_L = jnp.minimum(uL_n - c_L, uR_n - c_R)
            S_R = jnp.maximum(uL_n + c_L, uR_n + c_R)

            # Cached denominators
            denom_L = rho_L * (S_L - uL_n)
            denom_R = rho_R * (S_R - uR_n)

            # Middle speed (uses cached denominators to avoid recomputation)
            S_M = (p_R - p_L + denom_L*uL_n - denom_R*uR_n) / (denom_L - denom_R)

            # Star states
            rho_star_L = denom_L / (S_L - S_M)
            rho_star_R = denom_R / (S_R - S_M)
            E_star_L = E_L/rho_L + (S_M - uL_n)*(S_M + p_L/denom_L)
            E_star_R = E_R/rho_R + (S_M - uR_n)*(S_M + p_R/denom_R)

            U_star_L = jnp.stack([rho_star_L, rho_star_L*S_M, rho_star_L*uL_t, rho_star_L*E_star_L])
            U_star_R = jnp.stack([rho_star_R, rho_star_R*S_M, rho_star_R*uR_t, rho_star_R*E_star_R])

            F_L = jnp.stack([momL_n, momL_n*uL_n + p_L, momL_n*uL_t, (E_L + p_L)*uL_n])
            F_R = jnp.stack([momR_n, momR_n*uR_n + p_R, momR_n*uR_t, (E_R + p_R)*uR_n])

        else:  # y-direction flux
            # Assign normal (y) and tangent (x) velocities
            uL_n, uL_t = uL_y, uL_x
            uR_n, uR_t = uR_y, uR_x
            momL_n, momL_t = rhov_L, rhou_L
            momR_n, momR_t = rhov_R, rhou_R

            # Kinetic energy (fused computation)
            ke_L = 0.5 * rho_L * (uL_n*uL_n + uL_t*uL_t)
            ke_R = 0.5 * rho_R * (uR_n*uR_n + uR_t*uR_t)

            # Pressure
            gm1 = gamma - 1.0
            p_L = gm1 * (E_L - ke_L)
            p_R = gm1 * (E_R - ke_R)

            # Sound speed
            c_L = jnp.sqrt(gamma * p_L / rho_L)
            c_R = jnp.sqrt(gamma * p_R / rho_R)

            # Wave speeds
            S_L = jnp.minimum(uL_n - c_L, uR_n - c_R)
            S_R = jnp.maximum(uL_n + c_L, uR_n + c_R)

            # Cached denominators
            denom_L = rho_L * (S_L - uL_n)
            denom_R = rho_R * (S_R - uR_n)

            # Middle speed (uses cached denominators to avoid recomputation)
            S_M = (p_R - p_L + denom_L*uL_n - denom_R*uR_n) / (denom_L - denom_R)

            # Star states
            rho_star_L = denom_L / (S_L - S_M)
            rho_star_R = denom_R / (S_R - S_M)
            E_star_L = E_L/rho_L + (S_M - uL_n)*(S_M + p_L/denom_L)
            E_star_R = E_R/rho_R + (S_M - uR_n)*(S_M + p_R/denom_R)

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

        # Compute all velocities upfront (outside axis branches)
        uL_x = rhou_L / rho_L
        uR_x = rhou_R / rho_R
        uL_y = rhov_L / rho_L
        uR_y = rhov_R / rho_R
        uL_z = rhow_L / rho_L
        uR_z = rhow_R / rho_R

        if axis == 0:  # x-direction flux
            # Assign normal (vx) and tangent (vy, vz) velocities
            uL_n, uL_t1, uL_t2 = uL_x, uL_y, uL_z
            uR_n, uR_t1, uR_t2 = uR_x, uR_y, uR_z
            momL_n, momL_t1, momL_t2 = rhou_L, rhov_L, rhow_L
            momR_n, momR_t1, momR_t2 = rhou_R, rhov_R, rhow_R

            # Kinetic energy (fused computation, reused in pressure)
            ke_L = 0.5 * rho_L * (uL_n*uL_n + uL_t1*uL_t1 + uL_t2*uL_t2)
            ke_R = 0.5 * rho_R * (uR_n*uR_n + uR_t1*uR_t1 + uR_t2*uR_t2)

            # Pressure
            gm1 = gamma - 1.0
            p_L = gm1 * (E_L - ke_L)
            p_R = gm1 * (E_R - ke_R)

            # Sound speed
            c_L = jnp.sqrt(gamma * p_L / rho_L)
            c_R = jnp.sqrt(gamma * p_R / rho_R)

            # Wave speeds (based on normal velocity)
            S_L = jnp.minimum(uL_n - c_L, uR_n - c_R)
            S_R = jnp.maximum(uL_n + c_L, uR_n + c_R)

            # Cached denominators
            denom_L = rho_L * (S_L - uL_n)
            denom_R = rho_R * (S_R - uR_n)

            # Middle speed (uses cached denominators to avoid recomputation)
            S_M = (p_R - p_L + denom_L*uL_n - denom_R*uR_n) / (denom_L - denom_R)

            # Star states
            rho_star_L = denom_L / (S_L - S_M)
            rho_star_R = denom_R / (S_R - S_M)
            E_star_L = E_L/rho_L + (S_M - uL_n)*(S_M + p_L/denom_L)
            E_star_R = E_R/rho_R + (S_M - uR_n)*(S_M + p_R/denom_R)

            # Star state vectors: [rho, rho*vx_new, rho*vy, rho*vz, rho*E]
            U_star_L = jnp.stack([rho_star_L, rho_star_L*S_M, rho_star_L*uL_t1, rho_star_L*uL_t2, rho_star_L*E_star_L])
            U_star_R = jnp.stack([rho_star_R, rho_star_R*S_M, rho_star_R*uR_t1, rho_star_R*uR_t2, rho_star_R*E_star_R])

            # Fluxes in x-direction: [rho*vx, rho*vx*vx + p, rho*vx*vy, rho*vx*vz, (E+p)*vx]
            F_L = jnp.stack([momL_n, momL_n*uL_n + p_L, momL_n*uL_t1, momL_n*uL_t2, (E_L + p_L)*uL_n])
            F_R = jnp.stack([momR_n, momR_n*uR_n + p_R, momR_n*uR_t1, momR_n*uR_t2, (E_R + p_R)*uR_n])

        elif axis == 1:  # y-direction flux
            # Assign normal (vy) and tangent (vx, vz) velocities
            uL_n, uL_t1, uL_t2 = uL_y, uL_x, uL_z
            uR_n, uR_t1, uR_t2 = uR_y, uR_x, uR_z
            momL_n, momL_t1, momL_t2 = rhov_L, rhou_L, rhow_L
            momR_n, momR_t1, momR_t2 = rhov_R, rhou_R, rhow_R

            # Kinetic energy (fused computation, reused in pressure)
            ke_L = 0.5 * rho_L * (uL_n*uL_n + uL_t1*uL_t1 + uL_t2*uL_t2)
            ke_R = 0.5 * rho_R * (uR_n*uR_n + uR_t1*uR_t1 + uR_t2*uR_t2)

            # Pressure
            gm1 = gamma - 1.0
            p_L = gm1 * (E_L - ke_L)
            p_R = gm1 * (E_R - ke_R)

            # Sound speed
            c_L = jnp.sqrt(gamma * p_L / rho_L)
            c_R = jnp.sqrt(gamma * p_R / rho_R)

            # Wave speeds
            S_L = jnp.minimum(uL_n - c_L, uR_n - c_R)
            S_R = jnp.maximum(uL_n + c_L, uR_n + c_R)

            # Cached denominators
            denom_L = rho_L * (S_L - uL_n)
            denom_R = rho_R * (S_R - uR_n)

            # Middle speed (uses cached denominators to avoid recomputation)
            S_M = (p_R - p_L + denom_L*uL_n - denom_R*uR_n) / (denom_L - denom_R)

            # Star states
            rho_star_L = denom_L / (S_L - S_M)
            rho_star_R = denom_R / (S_R - S_M)
            E_star_L = E_L/rho_L + (S_M - uL_n)*(S_M + p_L/denom_L)
            E_star_R = E_R/rho_R + (S_M - uR_n)*(S_M + p_R/denom_R)

            # Star state vectors: [rho, rho*vx, rho*vy_new, rho*vz, rho*E]
            U_star_L = jnp.stack([rho_star_L, rho_star_L*uL_t1, rho_star_L*S_M, rho_star_L*uL_t2, rho_star_L*E_star_L])
            U_star_R = jnp.stack([rho_star_R, rho_star_R*uR_t1, rho_star_R*S_M, rho_star_R*uR_t2, rho_star_R*E_star_R])

            # Fluxes in y-direction: [rho*vy, rho*vy*vx, rho*vy*vy + p, rho*vy*vz, (E+p)*vy]
            F_L = jnp.stack([momL_n, momL_n*uL_t1, momL_n*uL_n + p_L, momL_n*uL_t2, (E_L + p_L)*uL_n])
            F_R = jnp.stack([momR_n, momR_n*uR_t1, momR_n*uR_n + p_R, momR_n*uR_t2, (E_R + p_R)*uR_n])

        else:  # axis == 2, z-direction flux
            # Assign normal (vz) and tangent (vx, vy) velocities
            uL_n, uL_t1, uL_t2 = uL_z, uL_x, uL_y
            uR_n, uR_t1, uR_t2 = uR_z, uR_x, uR_y
            momL_n, momL_t1, momL_t2 = rhow_L, rhou_L, rhov_L
            momR_n, momR_t1, momR_t2 = rhow_R, rhou_R, rhov_R

            # Kinetic energy (fused computation, reused in pressure)
            ke_L = 0.5 * rho_L * (uL_n*uL_n + uL_t1*uL_t1 + uL_t2*uL_t2)
            ke_R = 0.5 * rho_R * (uR_n*uR_n + uR_t1*uR_t1 + uR_t2*uR_t2)

            # Pressure
            gm1 = gamma - 1.0
            p_L = gm1 * (E_L - ke_L)
            p_R = gm1 * (E_R - ke_R)

            # Sound speed
            c_L = jnp.sqrt(gamma * p_L / rho_L)
            c_R = jnp.sqrt(gamma * p_R / rho_R)

            # Wave speeds
            S_L = jnp.minimum(uL_n - c_L, uR_n - c_R)
            S_R = jnp.maximum(uL_n + c_L, uR_n + c_R)

            # Cached denominators
            denom_L = rho_L * (S_L - uL_n)
            denom_R = rho_R * (S_R - uR_n)

            # Middle speed (uses cached denominators to avoid recomputation)
            S_M = (p_R - p_L + denom_L*uL_n - denom_R*uR_n) / (denom_L - denom_R)

            # Star states
            rho_star_L = denom_L / (S_L - S_M)
            rho_star_R = denom_R / (S_R - S_M)
            E_star_L = E_L/rho_L + (S_M - uL_n)*(S_M + p_L/denom_L)
            E_star_R = E_R/rho_R + (S_M - uR_n)*(S_M + p_R/denom_R)

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

