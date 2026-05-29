"""Low-Mach HLLC (L-HLLC) Riemann solver.

Based on Velasco-Romero & Teyssier 2025 (MNRAS 537, 2387), appendix A.

The standard HLLC contact pressure (Toro 10.71) contains a velocity-difference
dissipation term proportional to (v_L - v_R). At low Mach numbers this term is
small in magnitude but still dominates the truncation error in the acoustic
modes, washing out subsonic flow features. L-HLLC multiplies just that term
by a Mach-dependent factor

    chi = min(1, max(|v_n_L|, |v_n_R|) / max(c_L, c_R))
    phi = chi * (2 - chi)

so that phi -> 0 in the low-Mach limit (no spurious dissipation) and phi -> 1
in the sonic / supersonic limit (standard HLLC, full upwinding).

Same input/output signature as hydro.fluxes.hllc_flux_primitive so the
modular integrator can swap solvers via cfg.riemann_solver.
"""

import jax.numpy as jnp


def _lhllc_core(rho_L, rho_R, vn_L, vn_R, P_L, P_R, vt1_L, vt1_R, vt2_L, vt2_R, gamma):
    """Compute L-HLLC numerical flux for one face direction.

    All inputs are JAX arrays of identical shape (the face grid).
    Returns a stack of conservative-form flux components:
      (rho v_n, rho v_n^2 + P, rho v_n v_t1, rho v_n v_t2, (E+P) v_n)

    The caller pads vt1, vt2 with zeros for missing tangential directions
    (1D: both zero; 2D: vt2 zero).
    """
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
    rho_L, vx_L, P_L = W_L
    rho_R, vx_R, P_R = W_R
    zeros = jnp.zeros_like(rho_L)
    F = _lhllc_core(rho_L, rho_R, vx_L, vx_R, P_L, P_R, zeros, zeros, zeros, zeros, gamma)
    return jnp.stack([F[0], F[1], F[4]])


def lhllc_flux_primitive_2d(W_L, W_R, axis, gamma):
    """2D L-HLLC. W = (rho, vx, vy, P). axis 0 -> x flux, 1 -> y flux."""
    rho_L, vx_L, vy_L, P_L = W_L
    rho_R, vx_R, vy_R, P_R = W_R
    zeros = jnp.zeros_like(rho_L)
    if axis == 0:
        vn_L, vt_L = vx_L, vy_L
        vn_R, vt_R = vx_R, vy_R
    else:
        vn_L, vt_L = vy_L, vx_L
        vn_R, vt_R = vy_R, vx_R
    F = _lhllc_core(rho_L, rho_R, vn_L, vn_R, P_L, P_R, vt_L, vt_R, zeros, zeros, gamma)
    # Unpack: F[0]=mass, F[1]=normal momentum, F[2]=vt mom, F[4]=energy
    if axis == 0:
        return jnp.stack([F[0], F[1], F[2], F[4]])
    else:
        return jnp.stack([F[0], F[2], F[1], F[4]])


def lhllc_flux_primitive_3d(W_L, W_R, axis, gamma):
    """3D L-HLLC. W = (rho, vx, vy, vz, P). axis 0/1/2 -> x/y/z flux."""
    rho_L, vx_L, vy_L, vz_L, P_L = W_L
    rho_R, vx_R, vy_R, vz_R, P_R = W_R
    if axis == 0:
        vn_L, vt1_L, vt2_L = vx_L, vy_L, vz_L
        vn_R, vt1_R, vt2_R = vx_R, vy_R, vz_R
    elif axis == 1:
        vn_L, vt1_L, vt2_L = vy_L, vx_L, vz_L
        vn_R, vt1_R, vt2_R = vy_R, vx_R, vz_R
    else:
        vn_L, vt1_L, vt2_L = vz_L, vx_L, vy_L
        vn_R, vt1_R, vt2_R = vz_R, vx_R, vy_R
    F = _lhllc_core(rho_L, rho_R, vn_L, vn_R, P_L, P_R, vt1_L, vt1_R, vt2_L, vt2_R, gamma)
    # Reorder back to (rho, mvx, mvy, mvz, E)
    if axis == 0:
        return jnp.stack([F[0], F[1], F[2], F[3], F[4]])
    elif axis == 1:
        return jnp.stack([F[0], F[2], F[1], F[3], F[4]])
    else:
        return jnp.stack([F[0], F[2], F[3], F[1], F[4]])


def lhllc_flux_primitive(W_L, W_R, axis, gamma):
    """Dispatcher for L-HLLC, matching hllc_flux_primitive's signature."""
    Nvar = W_L.shape[0]
    if Nvar == 3:
        return lhllc_flux_primitive_1d(W_L, W_R, axis, gamma)
    elif Nvar == 4:
        return lhllc_flux_primitive_2d(W_L, W_R, axis, gamma)
    elif Nvar == 5:
        return lhllc_flux_primitive_3d(W_L, W_R, axis, gamma)
    else:
        raise ValueError(f"lhllc_flux_primitive supports 1D/2D/3D, got Nvar={Nvar}")
