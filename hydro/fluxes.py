"""
HLLC Riemann solver that takes PRIMITIVE variables as input.

This eliminates the redundant cons→prim conversion that happens when
reconstructing primitives but then converting back to conservatives
just to convert to primitives again inside the Riemann solver.

OPTIMIZATION: Saves one prim↔cons conversion per flux evaluation.
Expected speedup: ~5-8%
"""

from functools import partial

import jax
import jax.numpy as jnp


def _hllc_core(rho_L, rho_R, vn_L, vn_R, P_L, P_R,
               vt1_L, vt1_R, vt2_L, vt2_R, gamma):
    """Core HLLC flux in (normal, transverse) coordinates.

    Returns the five conservative flux components
    ``(F_rho, F_mn, F_mt1, F_mt2, F_E)`` where ``F_mn`` is the NORMAL-momentum
    flux and ``F_mt1/F_mt2`` the transverse ones. The 1D/2D/3D wrappers below
    map array axes onto (normal, transverse) and pad the unused transverse
    velocities with zeros, exactly as the L-HLLC solver does.

    Memory-friendly (per-component ``jnp.where``, no stacked intermediates) and
    autodiff-safe (every division is guarded away from exact zero, so the VJP
    of the unselected branch cannot pull out a NaN; guarded faces are never
    selected, so forward results are unchanged).
    """
    gm1 = gamma - 1.0
    c_L = jnp.sqrt(gamma * P_L / rho_L)
    c_R = jnp.sqrt(gamma * P_R / rho_R)
    S_L = jnp.minimum(vn_L - c_L, vn_R - c_R)
    S_R = jnp.maximum(vn_L + c_L, vn_R + c_R)
    denom_L = rho_L * (S_L - vn_L)
    denom_R = rho_R * (S_R - vn_R)

    def _nz(d, eps=1e-20):
        return jnp.where(jnp.abs(d) < eps, jnp.where(d < 0, -eps, eps), d)

    S_M = (P_R - P_L + denom_L * vn_L - denom_R * vn_R) / _nz(denom_L - denom_R)
    rho_star_L = denom_L / _nz(S_L - S_M)
    rho_star_R = denom_R / _nz(S_R - S_M)

    ke_L = 0.5 * (vn_L * vn_L + vt1_L * vt1_L + vt2_L * vt2_L)
    ke_R = 0.5 * (vn_R * vn_R + vt1_R * vt1_R + vt2_R * vt2_R)
    E_L = P_L / gm1 + rho_L * ke_L
    E_R = P_R / gm1 + rho_R * ke_R
    E_star_L = E_L / rho_L + (S_M - vn_L) * (S_M + P_L / _nz(denom_L))
    E_star_R = E_R / rho_R + (S_M - vn_R) * (S_M + P_R / _nz(denom_R))

    rho_vn_L = rho_L * vn_L
    rho_vn_R = rho_R * vn_R

    def comp(F_L_i, F_R_i, U_L_i, U_R_i, U_star_L_i, U_star_R_i):
        f_starL = F_L_i + S_L * (U_star_L_i - U_L_i)
        f_starR = F_R_i + S_R * (U_star_R_i - U_R_i)
        return jnp.where(
            S_L >= 0, F_L_i,
            jnp.where(S_M >= 0, f_starL, jnp.where(S_R > 0, f_starR, F_R_i)))

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
    """HLLC flux, 1D Euler. W = [rho, vx, P]; returns conservative flux."""
    rho_L, vx_L, P_L = W_L
    rho_R, vx_R, P_R = W_R
    z = jnp.zeros_like(rho_L)
    F_rho, F_mn, _, _, F_E = _hllc_core(
        rho_L, rho_R, vx_L, vx_R, P_L, P_R, z, z, z, z, gamma)
    return jnp.stack([F_rho, F_mn, F_E])


def hllc_flux_primitive_2d(W_L, W_R, axis, gamma):
    """HLLC flux, 2D Euler. W = [rho, vx, vy, P]; axis 0->x, 1->y."""
    rho_L, vx_L, vy_L, P_L = W_L
    rho_R, vx_R, vy_R, P_R = W_R
    z = jnp.zeros_like(rho_L)
    if axis == 0:                       # normal = x, transverse = y
        F_rho, F_mn, F_mt, _, F_E = _hllc_core(
            rho_L, rho_R, vx_L, vx_R, P_L, P_R, vy_L, vy_R, z, z, gamma)
        return jnp.stack([F_rho, F_mn, F_mt, F_E])
    else:                               # normal = y, transverse = x
        F_rho, F_mn, F_mt, _, F_E = _hllc_core(
            rho_L, rho_R, vy_L, vy_R, P_L, P_R, vx_L, vx_R, z, z, gamma)
        return jnp.stack([F_rho, F_mt, F_mn, F_E])


def hllc_flux_primitive_3d(W_L, W_R, axis, gamma):
    """HLLC flux, 3D Euler. W = [rho, vx, vy, vz, P]; axis 0->x, 1->y, 2->z."""
    rho_L, vx_L, vy_L, vz_L, P_L = W_L
    rho_R, vx_R, vy_R, vz_R, P_R = W_R
    if axis == 0:                       # normal = x, transverse = y, z
        F_rho, F_mn, F_t1, F_t2, F_E = _hllc_core(
            rho_L, rho_R, vx_L, vx_R, P_L, P_R, vy_L, vy_R, vz_L, vz_R, gamma)
        return jnp.stack([F_rho, F_mn, F_t1, F_t2, F_E])
    elif axis == 1:                     # normal = y, transverse = x, z
        F_rho, F_mn, F_t1, F_t2, F_E = _hllc_core(
            rho_L, rho_R, vy_L, vy_R, P_L, P_R, vx_L, vx_R, vz_L, vz_R, gamma)
        return jnp.stack([F_rho, F_t1, F_mn, F_t2, F_E])
    else:                               # normal = z, transverse = x, y
        F_rho, F_mn, F_t1, F_t2, F_E = _hllc_core(
            rho_L, rho_R, vz_L, vz_R, P_L, P_R, vx_L, vx_R, vy_L, vy_R, gamma)
        return jnp.stack([F_rho, F_t1, F_t2, F_mn, F_E])


# Dispatcher
@partial(jax.jit, static_argnums=(2, 3))
def hllc_flux_primitive(W_L, W_R, axis, gamma):
    """
    HLLC Riemann solver dispatcher for PRIMITIVE variables.

    Automatically selects 1D or 2D version based on input shape.

    Parameters
    ----------
    W_L, W_R : array
        Left and right PRIMITIVE states
        1D: (3, Nx) = [rho, vx, P]
        2D: (4, Nx, Ny) = [rho, vx, vy, P]
    axis : int (static)
        Flux direction
    gamma : float
        Ratio of specific heats

    Returns
    -------
    flux : array
        HLLC flux in CONSERVATIVE form (same shape as W_L)
    """
    Nvar = W_L.shape[0]

    if Nvar == 3:
        return hllc_flux_primitive_1d(W_L, W_R, axis, gamma)
    elif Nvar == 4:
        return hllc_flux_primitive_2d(W_L, W_R, axis, gamma)
    elif Nvar == 5:
        return hllc_flux_primitive_3d(W_L, W_R, axis, gamma)
    else:
        raise ValueError(
            f"hllc_flux_primitive only supports 1D (3), 2D (4), or 3D (5) vars, got {Nvar}"
        )


def hll_flux_primitive_3d(W_L, W_R, axis, gamma):
    """HLL Riemann solver for 3D Euler — PRIMITIVE inputs, conservative flux.

    Two-wave solver: no contact wave, so the contact/shear field is smeared
    relative to HLLC. That extra dissipation is exactly what suppresses the
    grid-aligned (carbuncle-type) instability in strong, grid-aligned
    collapse flows. Use as a diagnostic alternative or hybrid fallback.

        F_HLL = F_L                              if S_L ≥ 0
              = F_R                              if S_R ≤ 0
              = (S_R F_L − S_L F_R + S_L S_R (U_R − U_L)) / (S_R − S_L)
    """
    rho_L, vx_L, vy_L, vz_L, P_L = W_L
    rho_R, vx_R, vy_R, vz_R, P_R = W_R
    gm1 = gamma - 1.0

    if axis == 0:
        vn_L, vn_R = vx_L, vx_R
    elif axis == 1:
        vn_L, vn_R = vy_L, vy_R
    else:
        vn_L, vn_R = vz_L, vz_R

    c_L = jnp.sqrt(gamma * P_L / rho_L)
    c_R = jnp.sqrt(gamma * P_R / rho_R)
    S_L = jnp.minimum(vn_L - c_L, vn_R - c_R)
    S_R = jnp.maximum(vn_L + c_L, vn_R + c_R)

    ke_L = 0.5 * (vx_L * vx_L + vy_L * vy_L + vz_L * vz_L)
    ke_R = 0.5 * (vx_R * vx_R + vy_R * vy_R + vz_R * vz_R)
    E_L = P_L / gm1 + rho_L * ke_L
    E_R = P_R / gm1 + rho_R * ke_R

    inv_dS = 1.0 / (S_R - S_L)

    def hll_component(F_L_i, F_R_i, U_L_i, U_R_i):
        f_mid = (S_R * F_L_i - S_L * F_R_i + S_L * S_R * (U_R_i - U_L_i)) * inv_dS
        return jnp.where(S_L >= 0.0, F_L_i, jnp.where(S_R <= 0.0, F_R_i, f_mid))

    rho_vn_L = rho_L * vn_L
    rho_vn_R = rho_R * vn_R

    flux_0 = hll_component(rho_vn_L, rho_vn_R, rho_L, rho_R)
    flux_E = hll_component((E_L + P_L) * vn_L, (E_R + P_R) * vn_R, E_L, E_R)

    flux_mx = hll_component(
        rho_vn_L * vx_L + (P_L if axis == 0 else 0.0),
        rho_vn_R * vx_R + (P_R if axis == 0 else 0.0),
        rho_L * vx_L, rho_R * vx_R,
    )
    flux_my = hll_component(
        rho_vn_L * vy_L + (P_L if axis == 1 else 0.0),
        rho_vn_R * vy_R + (P_R if axis == 1 else 0.0),
        rho_L * vy_L, rho_R * vy_R,
    )
    flux_mz = hll_component(
        rho_vn_L * vz_L + (P_L if axis == 2 else 0.0),
        rho_vn_R * vz_R + (P_R if axis == 2 else 0.0),
        rho_L * vz_L, rho_R * vz_R,
    )
    return jnp.stack([flux_0, flux_mx, flux_my, flux_mz, flux_E])


@partial(jax.jit, static_argnums=(2, 3))
def hll_flux_primitive(W_L, W_R, axis, gamma):
    """HLL dispatcher (3D only for now — diagnostic use)."""
    Nvar = W_L.shape[0]
    if Nvar == 5:
        return hll_flux_primitive_3d(W_L, W_R, axis, gamma)
    raise ValueError(f"hll_flux_primitive currently supports 3D (5 vars), got {Nvar}")
