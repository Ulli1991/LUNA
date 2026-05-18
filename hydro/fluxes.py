"""HLLC Riemann solver with the AthenaK algorithm structure.

Ported from /raven/u/uli/athenak/src/hydro/rsolvers/hllc_hyd.hpp.

Differences vs the previous LUNA implementation:
- Wave speeds use the PVRS (primitive-variable Riemann solver) middle-pressure
  estimate plus the Batten et al. (1997) pressure-dependent correction q_L/q_R,
  instead of the simple Davis (1988) min/max estimate.
- Fluxes are expressed in SHIFTED form fl = F_L - bm·U_L, fr = F_R - bp·U_R
  rather than building star states U*_L, U*_R explicitly.
- The final flux is a WEIGHTED COMBINATION of fl, fr, and an explicit
  contact-pressure injection (qe·cp on the normal momentum, qe·cp·am on the
  energy). Selection is a TWO-BRANCH where(am >= 0, …, …) on the contact
  wave speed, replacing the previous 4-case jnp.select on (S_L, S_M, S_R).

For smooth flows the two formulations are mathematically equivalent; in
floating-point they accumulate roundoff differently. References:
- Toro, "Riemann Solvers", 2nd ed., Springer 1999, chpt. 10.
- Batten et al., SIAM J. Sci. Stat. Comput. 18 (1997) 1553.
"""

import jax
import jax.numpy as jnp
from functools import partial


def hllc_flux_primitive_1d(W_L, W_R, axis, gamma):
    """1D HLLC. axis is ignored (signature compatibility).

    W_L, W_R: (3, ...) primitives [rho, vx, P].
    Returns conservative flux (3, ...) = [F_rho, F_mvx, F_E].
    """
    rho_L, vx_L, P_L = W_L
    rho_R, vx_R, P_R = W_R

    gm1 = gamma - 1.0
    igm1 = 1.0 / gm1
    alpha = (gamma + 1.0) / (2.0 * gamma)

    c_L = jnp.sqrt(gamma * P_L / rho_L)
    c_R = jnp.sqrt(gamma * P_R / rho_R)
    e_L = P_L * igm1 + 0.5 * rho_L * vx_L * vx_L
    e_R = P_R * igm1 + 0.5 * rho_R * vx_R * vx_R

    # Step 2: PVRS middle-pressure estimate (Toro 10.5.2)
    rho_c_avg = 0.25 * (rho_L + rho_R) * (c_L + c_R)
    P_mid = 0.5 * (P_L + P_R + (vx_L - vx_R) * rho_c_avg)

    # Step 3: Batten et al. pressure-dependent corrections
    q_L = jnp.where(P_mid <= P_L, 1.0, jnp.sqrt(1.0 + alpha * (P_mid / P_L - 1.0)))
    q_R = jnp.where(P_mid <= P_R, 1.0, jnp.sqrt(1.0 + alpha * (P_mid / P_R - 1.0)))

    # Step 4: max/min wave speeds with TINY clamp (AthenaK ll.78-79)
    al = vx_L - c_L * q_L
    ar = vx_R + c_R * q_R
    bp = jnp.where(ar > 0.0, ar, 1.0e-20)
    bm = jnp.where(al < 0.0, al, -1.0e-20)

    # Step 5: contact wave speed and pressure at the contact
    vxl_shift = vx_L - bm
    vxr_shift = vx_R - bp
    tl = P_L + vxl_shift * rho_L * vx_L
    tr = P_R + vxr_shift * rho_R * vx_R
    ml = rho_L * vxl_shift
    mr = -(rho_R * vxr_shift)
    am = (tl - tr) / (ml + mr)
    cp = jnp.maximum((ml * tr + mr * tl) / (ml + mr), 0.0)

    # Step 6: shifted left/right fluxes (F - S*U)
    qe = rho_L * (vx_L - bm)
    qf = rho_R * (vx_R - bp)

    fl_d  = qe
    fl_mx = qe * vx_L + P_L
    fl_e  = e_L * (vx_L - bm) + P_L * vx_L

    fr_d  = qf
    fr_mx = qf * vx_R + P_R
    fr_e  = e_R * (vx_R - bp) + P_R * vx_R

    # Step 8: weights (two-branch on sign of am)
    inv_am_minus_bm = 1.0 / (am - bm)
    inv_bp_minus_am = 1.0 / (bp - am)
    am_geq_0 = am >= 0.0
    wc = jnp.where(am_geq_0, am * inv_am_minus_bm, 0.0)
    wd = jnp.where(am_geq_0, 0.0, -am * inv_bp_minus_am)
    we = jnp.where(am_geq_0, -bm * inv_am_minus_bm, bp * inv_bp_minus_am)

    # Step 9: combine with contact-pressure injection on normal mom + energy
    flux_d  = wc * fl_d  + wd * fr_d
    flux_mx = wc * fl_mx + wd * fr_mx + we * cp
    flux_e  = wc * fl_e  + wd * fr_e  + we * cp * am

    return jnp.stack([flux_d, flux_mx, flux_e])


def _hllc_core(rho_L, rho_R, vn_L, vn_R, P_L, P_R, vt_L, vt_R, vt2_L, vt2_R,
               e_L, e_R, gamma):
    """Internal core of AthenaK HLLC.

    Computes the shifted-flux + weighted-combination HLLC, returning
    six scalars per face: flux_d, flux_normal_mom, flux_vt_mom, flux_vt2_mom,
    flux_e, plus am (contact speed) and cp (contact pressure). For 2D
    `vt2_L/R` should be zero arrays (their flux component is just 0).
    """
    gm1 = gamma - 1.0
    alpha = (gamma + 1.0) / (2.0 * gamma)

    c_L = jnp.sqrt(gamma * P_L / rho_L)
    c_R = jnp.sqrt(gamma * P_R / rho_R)

    rho_c_avg = 0.25 * (rho_L + rho_R) * (c_L + c_R)
    P_mid = 0.5 * (P_L + P_R + (vn_L - vn_R) * rho_c_avg)

    q_L = jnp.where(P_mid <= P_L, 1.0, jnp.sqrt(1.0 + alpha * (P_mid / P_L - 1.0)))
    q_R = jnp.where(P_mid <= P_R, 1.0, jnp.sqrt(1.0 + alpha * (P_mid / P_R - 1.0)))

    al = vn_L - c_L * q_L
    ar = vn_R + c_R * q_R
    bp = jnp.where(ar > 0.0, ar, 1.0e-20)
    bm = jnp.where(al < 0.0, al, -1.0e-20)

    vnl_shift = vn_L - bm
    vnr_shift = vn_R - bp
    tl = P_L + vnl_shift * rho_L * vn_L
    tr = P_R + vnr_shift * rho_R * vn_R
    ml = rho_L * vnl_shift
    mr = -(rho_R * vnr_shift)
    am = (tl - tr) / (ml + mr)
    cp = jnp.maximum((ml * tr + mr * tl) / (ml + mr), 0.0)

    qe = rho_L * (vn_L - bm)
    qf = rho_R * (vn_R - bp)

    # Shifted fluxes
    fl_d   = qe
    fl_n   = qe * vn_L + P_L
    fl_t   = qe * vt_L
    fl_t2  = qe * vt2_L
    fl_e   = e_L * (vn_L - bm) + P_L * vn_L

    fr_d   = qf
    fr_n   = qf * vn_R + P_R
    fr_t   = qf * vt_R
    fr_t2  = qf * vt2_R
    fr_e   = e_R * (vn_R - bp) + P_R * vn_R

    inv_am_minus_bm = 1.0 / (am - bm)
    inv_bp_minus_am = 1.0 / (bp - am)
    am_geq_0 = am >= 0.0
    wc = jnp.where(am_geq_0, am * inv_am_minus_bm, 0.0)
    wd = jnp.where(am_geq_0, 0.0, -am * inv_bp_minus_am)
    we = jnp.where(am_geq_0, -bm * inv_am_minus_bm, bp * inv_bp_minus_am)

    flux_d  = wc * fl_d  + wd * fr_d
    flux_n  = wc * fl_n  + wd * fr_n  + we * cp
    flux_t  = wc * fl_t  + wd * fr_t
    flux_t2 = wc * fl_t2 + wd * fr_t2
    flux_e  = wc * fl_e  + wd * fr_e  + we * cp * am

    return flux_d, flux_n, flux_t, flux_t2, flux_e


def hllc_flux_primitive_2d(W_L, W_R, axis, gamma):
    """2D HLLC with axis ∈ {0, 1}.

    W_L, W_R: (4, Nx, Ny) primitives [rho, vx, vy, P].
    Returns conservative flux [F_rho, F_mvx, F_mvy, F_E].
    """
    rho_L, vx_L, vy_L, P_L = W_L
    rho_R, vx_R, vy_R, P_R = W_R

    igm1 = 1.0 / (gamma - 1.0)
    e_L = P_L * igm1 + 0.5 * rho_L * (vx_L * vx_L + vy_L * vy_L)
    e_R = P_R * igm1 + 0.5 * rho_R * (vx_R * vx_R + vy_R * vy_R)

    if axis == 0:  # x-direction: normal = vx, tangential = vy
        vn_L, vn_R = vx_L, vx_R
        vt_L, vt_R = vy_L, vy_R
    else:  # y-direction: normal = vy, tangential = vx
        vn_L, vn_R = vy_L, vy_R
        vt_L, vt_R = vx_L, vx_R

    zeros = jnp.zeros_like(rho_L)
    flux_d, flux_n, flux_t, _, flux_e = _hllc_core(
        rho_L, rho_R, vn_L, vn_R, P_L, P_R, vt_L, vt_R, zeros, zeros,
        e_L, e_R, gamma,
    )

    # Assemble in [F_rho, F_mvx, F_mvy, F_E] order
    if axis == 0:
        return jnp.stack([flux_d, flux_n, flux_t, flux_e])
    else:
        return jnp.stack([flux_d, flux_t, flux_n, flux_e])


def hllc_flux_primitive_3d(W_L, W_R, axis, gamma):
    """3D HLLC with axis ∈ {0, 1, 2}.

    W_L, W_R: (5, Nx, Ny, Nz) primitives [rho, vx, vy, vz, P].
    Returns conservative flux [F_rho, F_mvx, F_mvy, F_mvz, F_E].
    """
    rho_L, vx_L, vy_L, vz_L, P_L = W_L
    rho_R, vx_R, vy_R, vz_R, P_R = W_R

    igm1 = 1.0 / (gamma - 1.0)
    e_L = P_L * igm1 + 0.5 * rho_L * (vx_L * vx_L + vy_L * vy_L + vz_L * vz_L)
    e_R = P_R * igm1 + 0.5 * rho_R * (vx_R * vx_R + vy_R * vy_R + vz_R * vz_R)

    if axis == 0:    # x: normal=vx, t1=vy, t2=vz
        vn_L, vn_R = vx_L, vx_R
        vt_L, vt_R = vy_L, vy_R
        vt2_L, vt2_R = vz_L, vz_R
    elif axis == 1:  # y: normal=vy, t1=vx, t2=vz
        vn_L, vn_R = vy_L, vy_R
        vt_L, vt_R = vx_L, vx_R
        vt2_L, vt2_R = vz_L, vz_R
    else:            # z: normal=vz, t1=vx, t2=vy
        vn_L, vn_R = vz_L, vz_R
        vt_L, vt_R = vx_L, vx_R
        vt2_L, vt2_R = vy_L, vy_R

    flux_d, flux_n, flux_t, flux_t2, flux_e = _hllc_core(
        rho_L, rho_R, vn_L, vn_R, P_L, P_R, vt_L, vt_R, vt2_L, vt2_R,
        e_L, e_R, gamma,
    )

    if axis == 0:
        return jnp.stack([flux_d, flux_n,  flux_t,  flux_t2, flux_e])
    elif axis == 1:
        return jnp.stack([flux_d, flux_t,  flux_n,  flux_t2, flux_e])
    else:
        return jnp.stack([flux_d, flux_t,  flux_t2, flux_n,  flux_e])


@partial(jax.jit, static_argnums=(2, 3))
def hllc_flux_primitive(W_L, W_R, axis, gamma):
    """Dispatcher: selects 1D/2D/3D HLLC based on input shape."""
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
