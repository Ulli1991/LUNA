"""HLLC and HLL Riemann solvers: primitive face states in, conservative flux out."""

from functools import partial

import jax
import jax.numpy as jnp


def rotate_primitive_pair(W_L, W_R, axis):
    """Rotate L/R primitive states into the (normal, transverse) frame of ``_hllc_core``.

    Returns (rho_L, rho_R, vn_L, vn_R, P_L, P_R, vt1_L, vt1_R, vt2_L, vt2_R); missing
    transverse velocities in 1D/2D are zero.
    """
    ndim = W_L.shape[0] - 2
    rho_L, rho_R = W_L[0], W_R[0]
    P_L, P_R = W_L[ndim + 1], W_R[ndim + 1]
    z = jnp.zeros_like(rho_L)
    order = [axis] + [k for k in range(ndim) if k != axis]
    v_L = [W_L[1 + k] for k in order] + [z] * (3 - ndim)
    v_R = [W_R[1 + k] for k in order] + [z] * (3 - ndim)
    return (rho_L, rho_R, v_L[0], v_R[0], P_L, P_R,
            v_L[1], v_R[1], v_L[2], v_R[2])


def unrotate_flux(F_rho, F_mn, F_mt1, F_mt2, F_E, *, axis, ndim):
    """Stack normal-frame flux components back into (rho, momenta..., E) order."""
    if ndim == 1:
        return jnp.stack([F_rho, F_mn, F_E])
    F_t = (F_mt1, F_mt2)
    mom = []
    t = 0
    for k in range(ndim):
        if k == axis:
            mom.append(F_mn)
        else:
            mom.append(F_t[t])
            t += 1
    return jnp.stack([F_rho, *mom, F_E])


def solve_in_normal_frame(core, W_L, W_R, axis, gamma):
    """Rotate face states into the normal frame, run ``core``, rotate the flux back."""
    ndim = W_L.shape[0] - 2
    if ndim == 1:
        axis = 0
    F = core(*rotate_primitive_pair(W_L, W_R, axis), gamma)
    return unrotate_flux(*F, axis=axis, ndim=ndim)


def _hllc_core(rho_L, rho_R, vn_L, vn_R, P_L, P_R,
               vt1_L, vt1_R, vt2_L, vt2_R, gamma):
    """HLLC flux (F_rho, F_mn, F_mt1, F_mt2, F_E) in the normal frame (Toro 2009, ch. 10)."""
    # Reciprocals reused: 7 divisions per face instead of 11.
    inv_gm1 = 1.0 / (gamma - 1.0)
    inv_rho_L = 1.0 / rho_L
    inv_rho_R = 1.0 / rho_R
    # A face state with c^2 < 0 (negative rho or P) made c NaN: every branch test failed, F_R was taken, and
    # the NaN sqrt partial poisoned the VJP. Same selection, safe sqrt.
    c2_L = gamma * P_L * inv_rho_L
    c2_R = gamma * P_R * inv_rho_R
    ok = (c2_L >= 0) & (c2_R >= 0)
    c_L = jnp.where(c2_L > 0, jnp.sqrt(jnp.where(c2_L > 0, c2_L, 1.0)), 0.0)
    c_R = jnp.where(c2_R > 0, jnp.sqrt(jnp.where(c2_R > 0, c2_R, 1.0)), 0.0)
    S_L = jnp.minimum(vn_L - c_L, vn_R - c_R)
    S_R = jnp.maximum(vn_L + c_L, vn_R + c_R)
    denom_L = rho_L * (S_L - vn_L)
    denom_R = rho_R * (S_R - vn_R)

    # Divisions are kept off exact zero so the VJP of unselected branches cannot NaN.
    def _nz(d, eps=1e-20):
        return jnp.where(jnp.abs(d) < eps, jnp.where(d < 0, -eps, eps), d)

    S_M = (P_R - P_L + denom_L * vn_L - denom_R * vn_R) / _nz(denom_L - denom_R)
    rho_star_L = denom_L / _nz(S_L - S_M)
    rho_star_R = denom_R / _nz(S_R - S_M)

    ke_L = 0.5 * (vn_L * vn_L + vt1_L * vt1_L + vt2_L * vt2_L)
    ke_R = 0.5 * (vn_R * vn_R + vt1_R * vt1_R + vt2_R * vt2_R)
    E_L = P_L * inv_gm1 + rho_L * ke_L
    E_R = P_R * inv_gm1 + rho_R * ke_R
    E_star_L = E_L * inv_rho_L + (S_M - vn_L) * (S_M + P_L * (1.0 / _nz(denom_L)))
    E_star_R = E_R * inv_rho_R + (S_M - vn_R) * (S_M + P_R * (1.0 / _nz(denom_R)))

    rho_vn_L = rho_L * vn_L
    rho_vn_R = rho_R * vn_R

    def comp(F_L_i, F_R_i, U_L_i, U_R_i, U_star_L_i, U_star_R_i):
        f_starL = F_L_i + S_L * (U_star_L_i - U_L_i)
        f_starR = F_R_i + S_R * (U_star_R_i - U_R_i)
        return jnp.where(
            ok & (S_L >= 0), F_L_i,
            jnp.where(ok & (S_M >= 0), f_starL, jnp.where(ok & (S_R > 0), f_starR, F_R_i)))

    F_rho = comp(rho_vn_L, rho_vn_R, rho_L, rho_R, rho_star_L, rho_star_R)
    F_mn = comp(rho_vn_L * vn_L + P_L, rho_vn_R * vn_R + P_R,
                rho_L * vn_L, rho_R * vn_R, rho_star_L * S_M, rho_star_R * S_M)
    F_mt1 = comp(rho_vn_L * vt1_L, rho_vn_R * vt1_R,
                 rho_L * vt1_L, rho_R * vt1_R,
                 rho_star_L * vt1_L, rho_star_R * vt1_R)
    F_mt2 = comp(rho_vn_L * vt2_L, rho_vn_R * vt2_R,
                 rho_L * vt2_L, rho_R * vt2_R,
                 rho_star_L * vt2_L, rho_star_R * vt2_R)
    F_E = comp((E_L + P_L) * vn_L, (E_R + P_R) * vn_R, E_L, E_R,
               rho_star_L * E_star_L, rho_star_R * E_star_R)
    return F_rho, F_mn, F_mt1, F_mt2, F_E


def hllc_flux_primitive_1d(W_L, W_R, axis, gamma):
    """HLLC flux for 1D Euler, W = [rho, vx, P]."""
    return solve_in_normal_frame(_hllc_core, W_L, W_R, axis, gamma)


def hllc_flux_primitive_2d(W_L, W_R, axis, gamma):
    """HLLC flux for 2D Euler, W = [rho, vx, vy, P]."""
    return solve_in_normal_frame(_hllc_core, W_L, W_R, axis, gamma)


def hllc_flux_primitive_3d(W_L, W_R, axis, gamma):
    """HLLC flux for 3D Euler, W = [rho, vx, vy, vz, P]."""
    return solve_in_normal_frame(_hllc_core, W_L, W_R, axis, gamma)


@partial(jax.jit, static_argnums=(2, 3))
def hllc_flux_primitive(W_L, W_R, axis, gamma):
    """HLLC conservative flux from primitive face states (1D/2D/3D by variable count)."""
    Nvar = W_L.shape[0]
    if Nvar not in (3, 4, 5):
        raise ValueError(
            f"hllc_flux_primitive only supports 1D (3), 2D (4), or 3D (5) vars, got {Nvar}"
        )
    return solve_in_normal_frame(_hllc_core, W_L, W_R, axis, gamma)


def _hll_core(rho_L, rho_R, vn_L, vn_R, P_L, P_R,
              vt1_L, vt1_R, vt2_L, vt2_R, gamma):
    """Two-wave HLL flux (F_rho, F_mn, F_mt1, F_mt2, F_E) in the normal frame; no contact wave."""
    # Reciprocals reused: 3 divisions per face instead of 5.
    inv_gm1 = 1.0 / (gamma - 1.0)
    inv_rho_L = 1.0 / rho_L
    inv_rho_R = 1.0 / rho_R
    c2_L = gamma * P_L * inv_rho_L     # c^2 < 0: c = 0 by a safe sqrt (finite VJP)
    c2_R = gamma * P_R * inv_rho_R
    c_L = jnp.where(c2_L > 0, jnp.sqrt(jnp.where(c2_L > 0, c2_L, 1.0)), 0.0)
    c_R = jnp.where(c2_R > 0, jnp.sqrt(jnp.where(c2_R > 0, c2_R, 1.0)), 0.0)
    S_L = jnp.minimum(vn_L - c_L, vn_R - c_R)
    S_R = jnp.maximum(vn_L + c_L, vn_R + c_R)

    ke_L = 0.5 * (vn_L * vn_L + vt1_L * vt1_L + vt2_L * vt2_L)
    ke_R = 0.5 * (vn_R * vn_R + vt1_R * vt1_R + vt2_R * vt2_R)
    E_L = P_L * inv_gm1 + rho_L * ke_L
    E_R = P_R * inv_gm1 + rho_R * ke_R

    inv_dS = 1.0 / (S_R - S_L)

    def comp(F_L_i, F_R_i, U_L_i, U_R_i):
        f_mid = (S_R * F_L_i - S_L * F_R_i + S_L * S_R * (U_R_i - U_L_i)) * inv_dS
        return jnp.where(S_L >= 0.0, F_L_i, jnp.where(S_R <= 0.0, F_R_i, f_mid))

    rho_vn_L = rho_L * vn_L
    rho_vn_R = rho_R * vn_R
    F_rho = comp(rho_vn_L, rho_vn_R, rho_L, rho_R)
    F_mn = comp(rho_vn_L * vn_L + P_L, rho_vn_R * vn_R + P_R, rho_L * vn_L, rho_R * vn_R)
    F_mt1 = comp(rho_vn_L * vt1_L, rho_vn_R * vt1_R, rho_L * vt1_L, rho_R * vt1_R)
    F_mt2 = comp(rho_vn_L * vt2_L, rho_vn_R * vt2_R, rho_L * vt2_L, rho_R * vt2_R)
    F_E = comp((E_L + P_L) * vn_L, (E_R + P_R) * vn_R, E_L, E_R)
    return F_rho, F_mn, F_mt1, F_mt2, F_E


@partial(jax.jit, static_argnums=(2, 3))
def hll_flux_primitive(W_L, W_R, axis, gamma):
    """HLL conservative flux from primitive face states (1D/2D/3D by variable count)."""
    Nvar = W_L.shape[0]
    if Nvar not in (3, 4, 5):
        raise ValueError(
            f"hll_flux_primitive only supports 1D (3), 2D (4), or 3D (5) vars, got {Nvar}"
        )
    return solve_in_normal_frame(_hll_core, W_L, W_R, axis, gamma)
