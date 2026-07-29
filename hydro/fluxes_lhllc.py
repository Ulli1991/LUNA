"""Low-Mach HLLC (L-HLLC) Riemann solver (Velasco-Romero & Teyssier 2025,
appendix A): the contact-pressure dissipation term is scaled by
phi = chi(2 - chi), chi = min(1, max|v_n|/max c), so it vanishes in the
low-Mach limit and recovers standard HLLC at sonic speeds. Same signature as
hydro.fluxes.hllc_flux_primitive (swap via cfg.riemann_solver).
"""

import jax.numpy as jnp

from hydro.fluxes import solve_in_normal_frame


def _lhllc_core(rho_L, rho_R, vn_L, vn_R, P_L, P_R, vt1_L, vt1_R, vt2_L, vt2_R, gamma):
    """L-HLLC flux for one face direction: returns the conservative stack
    (rho v_n, rho v_n^2 + P, rho v_n v_t1, rho v_n v_t2, (E+P) v_n);
    the caller zero-pads missing tangential velocities."""
    gm1 = gamma - 1.0

    c_L = jnp.sqrt(gamma * P_L / rho_L)
    c_R = jnp.sqrt(gamma * P_R / rho_R)
    c_max = jnp.maximum(c_L, c_R)

    # Low-Mach correction factor
    chi = jnp.minimum(1.0, jnp.maximum(jnp.abs(vn_L), jnp.abs(vn_R)) / c_max)
    phi = chi * (2.0 - chi)

    # Wave speeds (Einfeldt-style)
    S_L = jnp.minimum(vn_L, vn_R) - c_max
    S_R = jnp.maximum(vn_L, vn_R) + c_max

    # Auxiliary positive quantities (Toro notation, both >= 0)
    A_L = rho_L * (vn_L - S_L)
    A_R = rho_R * (S_R - vn_R)
    A_sum = A_L + A_R

    # Contact wave speed (same as standard HLLC)
    S_M = (P_L - P_R + A_L * vn_L + A_R * vn_R) / A_sum

    # L-HLLC contact pressure: pressure-average + phi * velocity-difference term
    P_star = (A_R * P_L + A_L * P_R + phi * A_L * A_R * (vn_L - vn_R)) / A_sum

    # Kinetic + total energy on the two sides
    ke_L = 0.5 * (vn_L * vn_L + vt1_L * vt1_L + vt2_L * vt2_L)
    ke_R = 0.5 * (vn_R * vn_R + vt1_R * vt1_R + vt2_R * vt2_R)
    E_L = P_L / gm1 + rho_L * ke_L
    E_R = P_R / gm1 + rho_R * ke_R

    # Star densities and energies (Toro 10.73)
    rho_star_L = rho_L * (S_L - vn_L) / (S_L - S_M)
    rho_star_R = rho_R * (S_R - vn_R) / (S_R - S_M)
    E_star_L = ((S_L - vn_L) * E_L - P_L * vn_L + P_star * S_M) / (S_L - S_M)
    E_star_R = ((S_R - vn_R) * E_R - P_R * vn_R + P_star * S_M) / (S_R - S_M)

    # Star-state conservative variables (tangential vt preserved across contact)
    U_L = jnp.stack([rho_L, rho_L * vn_L, rho_L * vt1_L, rho_L * vt2_L, E_L])
    U_R = jnp.stack([rho_R, rho_R * vn_R, rho_R * vt1_R, rho_R * vt2_R, E_R])
    U_star_L = jnp.stack(
        [rho_star_L, rho_star_L * S_M, rho_star_L * vt1_L, rho_star_L * vt2_L, E_star_L]
    )
    U_star_R = jnp.stack(
        [rho_star_R, rho_star_R * S_M, rho_star_R * vt1_R, rho_star_R * vt2_R, E_star_R]
    )

    # Physical fluxes on the two sides
    F_L = jnp.stack(
        [
            rho_L * vn_L,
            rho_L * vn_L * vn_L + P_L,
            rho_L * vn_L * vt1_L,
            rho_L * vn_L * vt2_L,
            (E_L + P_L) * vn_L,
        ]
    )
    F_R = jnp.stack(
        [
            rho_R * vn_R,
            rho_R * vn_R * vn_R + P_R,
            rho_R * vn_R * vt1_R,
            rho_R * vn_R * vt2_R,
            (E_R + P_R) * vn_R,
        ]
    )

    # HLLC flux selection on wave-speed signs
    F_star_L = F_L + S_L * (U_star_L - U_L)
    F_star_R = F_R + S_R * (U_star_R - U_R)
    F = jnp.where(
        S_L >= 0.0, F_L, jnp.where(S_M >= 0.0, F_star_L, jnp.where(S_R >= 0.0, F_star_R, F_R))
    )
    return F  # shape (5, *face_shape)


def lhllc_flux_primitive_1d(W_L, W_R, axis, gamma):
    """1D L-HLLC. axis ignored (signature compat). W = (rho, vx, P)."""
    return solve_in_normal_frame(_lhllc_core, W_L, W_R, axis, gamma)


def lhllc_flux_primitive_2d(W_L, W_R, axis, gamma):
    """2D L-HLLC. W = (rho, vx, vy, P). axis 0 -> x flux, 1 -> y flux."""
    return solve_in_normal_frame(_lhllc_core, W_L, W_R, axis, gamma)


def lhllc_flux_primitive_3d(W_L, W_R, axis, gamma):
    """3D L-HLLC. W = (rho, vx, vy, vz, P). axis 0/1/2 -> x/y/z flux."""
    return solve_in_normal_frame(_lhllc_core, W_L, W_R, axis, gamma)


def lhllc_flux_primitive(W_L, W_R, axis, gamma):
    """Dispatcher for L-HLLC, matching hllc_flux_primitive's signature."""
    Nvar = W_L.shape[0]
    if Nvar not in (3, 4, 5):
        raise ValueError(f"lhllc_flux_primitive supports 1D/2D/3D, got Nvar={Nvar}")
    return solve_in_normal_frame(_lhllc_core, W_L, W_R, axis, gamma)
