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


def hllc_flux_primitive_2d(W_L, W_R, axis, gamma):
    """
    HLLC Riemann solver for 2D Euler equations using PRIMITIVE variables.

    Parameters
    ----------
    W_L, W_R : array, shape (4, Nx, Ny)
        Left and right PRIMITIVE states: [rho, vx, vy, P]
    axis : int (static)
        0 → x-direction flux
        1 → y-direction flux
    gamma : float
        Ratio of specific heats

    Returns
    -------
    flux : array, shape (4, Nx, Ny)
        HLLC flux in CONSERVATIVE form [rho*v, rho*v*v+P, rho*v*v, (E+P)*v]
    """
    # Extract primitives (already have them!)
    rho_L, vx_L, vy_L, P_L = W_L
    rho_R, vx_R, vy_R, P_R = W_R

    # Precompute gamma - 1
    gm1 = gamma - 1.0

    if axis == 0:  # x-direction flux
        # Normal and tangential velocities
        vn_L, vt_L = vx_L, vy_L
        vn_R, vt_R = vx_R, vy_R

        # Sound speed (primitives already available!)
        c_L = jnp.sqrt(gamma * P_L / rho_L)
        c_R = jnp.sqrt(gamma * P_R / rho_R)

        # Wave speeds
        S_L = jnp.minimum(vn_L - c_L, vn_R - c_R)
        S_R = jnp.maximum(vn_L + c_L, vn_R + c_R)

        # Cached denominators
        denom_L = rho_L * (S_L - vn_L)
        denom_R = rho_R * (S_R - vn_R)

        # Middle speed
        S_M = (P_R - P_L + denom_L * vn_L - denom_R * vn_R) / (denom_L - denom_R)

        # Star states (densities)
        rho_star_L = denom_L / (S_L - S_M)
        rho_star_R = denom_R / (S_R - S_M)

        # Energy (need to compute from primitives)
        ke_L = 0.5 * (vn_L * vn_L + vt_L * vt_L)
        ke_R = 0.5 * (vn_R * vn_R + vt_R * vt_R)
        E_L = P_L / gm1 + rho_L * ke_L
        E_R = P_R / gm1 + rho_R * ke_R

        # Star energies
        E_star_L = E_L / rho_L + (S_M - vn_L) * (S_M + P_L / denom_L)
        E_star_R = E_R / rho_R + (S_M - vn_R) * (S_M + P_R / denom_R)

        # Momentum (from primitives)
        rho_vn_L = rho_L * vn_L
        rho_vn_R = rho_R * vn_R
        rho_vt_L = rho_L * vt_L
        rho_vt_R = rho_R * vt_R

        # Conservative states (for star regions)
        U_L = jnp.stack([rho_L, rho_vn_L, rho_vt_L, E_L])
        U_R = jnp.stack([rho_R, rho_vn_R, rho_vt_R, E_R])

        U_star_L = jnp.stack(
            [rho_star_L, rho_star_L * S_M, rho_star_L * vt_L, rho_star_L * E_star_L]
        )
        U_star_R = jnp.stack(
            [rho_star_R, rho_star_R * S_M, rho_star_R * vt_R, rho_star_R * E_star_R]
        )

        # Fluxes (in conservative form)
        F_L = jnp.stack([rho_vn_L, rho_vn_L * vn_L + P_L, rho_vn_L * vt_L, (E_L + P_L) * vn_L])
        F_R = jnp.stack([rho_vn_R, rho_vn_R * vn_R + P_R, rho_vn_R * vt_R, (E_R + P_R) * vn_R])

    else:  # y-direction flux
        # Normal and tangential velocities
        vn_L, vt_L = vy_L, vx_L
        vn_R, vt_R = vy_R, vx_R

        # Sound speed
        c_L = jnp.sqrt(gamma * P_L / rho_L)
        c_R = jnp.sqrt(gamma * P_R / rho_R)

        # Wave speeds
        S_L = jnp.minimum(vn_L - c_L, vn_R - c_R)
        S_R = jnp.maximum(vn_L + c_L, vn_R + c_R)

        # Cached denominators
        denom_L = rho_L * (S_L - vn_L)
        denom_R = rho_R * (S_R - vn_R)

        # Middle speed
        S_M = (P_R - P_L + denom_L * vn_L - denom_R * vn_R) / (denom_L - denom_R)

        # Star states
        rho_star_L = denom_L / (S_L - S_M)
        rho_star_R = denom_R / (S_R - S_M)

        # Energy
        ke_L = 0.5 * (vn_L * vn_L + vt_L * vt_L)
        ke_R = 0.5 * (vn_R * vn_R + vt_R * vt_R)
        E_L = P_L / gm1 + rho_L * ke_L
        E_R = P_R / gm1 + rho_R * ke_R

        E_star_L = E_L / rho_L + (S_M - vn_L) * (S_M + P_L / denom_L)
        E_star_R = E_R / rho_R + (S_M - vn_R) * (S_M + P_R / denom_R)

        # Momentum
        rho_vn_L = rho_L * vn_L
        rho_vn_R = rho_R * vn_R
        rho_vt_L = rho_L * vt_L
        rho_vt_R = rho_R * vt_R

        # Conservative states (note order: [rho, rho*vx, rho*vy, E])
        U_L = jnp.stack([rho_L, rho_vt_L, rho_vn_L, E_L])
        U_R = jnp.stack([rho_R, rho_vt_R, rho_vn_R, E_R])

        U_star_L = jnp.stack(
            [rho_star_L, rho_star_L * vt_L, rho_star_L * S_M, rho_star_L * E_star_L]
        )
        U_star_R = jnp.stack(
            [rho_star_R, rho_star_R * vt_R, rho_star_R * S_M, rho_star_R * E_star_R]
        )

        # Fluxes (in conservative form, order: [rho*vy, rho*vy*vx, rho*vy*vy+P, (E+P)*vy])
        F_L = jnp.stack([rho_vn_L, rho_vn_L * vt_L, rho_vn_L * vn_L + P_L, (E_L + P_L) * vn_L])
        F_R = jnp.stack([rho_vn_R, rho_vn_R * vt_R, rho_vn_R * vn_R + P_R, (E_R + P_R) * vn_R])

    # HLLC flux selection
    conditions = [
        S_L >= 0,
        (S_L < 0) & (S_M >= 0),
        (S_M < 0) & (S_R > 0),
    ]

    choices = [
        F_L,
        F_L + S_L * (U_star_L - U_L),
        F_R + S_R * (U_star_R - U_R),
    ]

    flux = jnp.select(conditions, choices, default=F_R)

    return flux


def hllc_flux_primitive_3d(W_L, W_R, axis, gamma):
    """
    HLLC Riemann solver for 3D Euler equations using PRIMITIVE variables.

    Parameters
    ----------
    W_L, W_R : array, shape (5, Nx, Ny, Nz)
        Left and right PRIMITIVE states: [rho, vx, vy, vz, P]
    axis : int (static)
        0 → x-direction flux
        1 → y-direction flux
        2 → z-direction flux
    gamma : float
        Ratio of specific heats

    Returns
    -------
    flux : array, shape (5, Nx, Ny, Nz)
        HLLC flux in CONSERVATIVE form [rho*vn, rho*vn*vx, rho*vn*vy, rho*vn*vz, (E+P)*vn]
    """
    rho_L, vx_L, vy_L, vz_L, P_L = W_L
    rho_R, vx_R, vy_R, vz_R, P_R = W_R

    gm1 = gamma - 1.0

    # Select normal and tangential velocities based on axis
    if axis == 0:  # x-direction
        vn_L, vt1_L, vt2_L = vx_L, vy_L, vz_L
        vn_R, vt1_R, vt2_R = vx_R, vy_R, vz_R
    elif axis == 1:  # y-direction
        vn_L, vt1_L, vt2_L = vy_L, vx_L, vz_L
        vn_R, vt1_R, vt2_R = vy_R, vx_R, vz_R
    else:  # z-direction
        vn_L, vt1_L, vt2_L = vz_L, vx_L, vy_L
        vn_R, vt1_R, vt2_R = vz_R, vx_R, vy_R

    # Sound speeds
    c_L = jnp.sqrt(gamma * P_L / rho_L)
    c_R = jnp.sqrt(gamma * P_R / rho_R)

    # Wave speeds
    S_L = jnp.minimum(vn_L - c_L, vn_R - c_R)
    S_R = jnp.maximum(vn_L + c_L, vn_R + c_R)

    # Cached denominators
    denom_L = rho_L * (S_L - vn_L)
    denom_R = rho_R * (S_R - vn_R)

    # Middle speed
    S_M = (P_R - P_L + denom_L * vn_L - denom_R * vn_R) / (denom_L - denom_R)

    # Star densities
    rho_star_L = denom_L / (S_L - S_M)
    rho_star_R = denom_R / (S_R - S_M)

    # Total kinetic energy and total energy
    ke_L = 0.5 * (vn_L * vn_L + vt1_L * vt1_L + vt2_L * vt2_L)
    ke_R = 0.5 * (vn_R * vn_R + vt1_R * vt1_R + vt2_R * vt2_R)
    E_L = P_L / gm1 + rho_L * ke_L
    E_R = P_R / gm1 + rho_R * ke_R

    # Star energies
    E_star_L = E_L / rho_L + (S_M - vn_L) * (S_M + P_L / denom_L)
    E_star_R = E_R / rho_R + (S_M - vn_R) * (S_M + P_R / denom_R)

    # Normal momentum
    rho_vn_L = rho_L * vn_L
    rho_vn_R = rho_R * vn_R

    # Memory-friendly HLLC: compute each of the 5 flux components separately
    # via jnp.where chains, then stack ONLY the final scalar fields. The old
    # code built 8 stacked (5, *shard) intermediates (U_L, U_R, U_star_L/R,
    # F_L, F_R + 3 choices for jnp.select) — that allocation was ~9 GB per
    # direction at 1024^3 fp32 / 32 GPUs and dominated the OOM budget.

    def hllc_component(F_L_i, F_R_i, U_L_i, U_R_i, U_star_L_i, U_star_R_i):
        """Compute one component of the HLLC flux using nested jnp.where."""
        f_starL = F_L_i + S_L * (U_star_L_i - U_L_i)
        f_starR = F_R_i + S_R * (U_star_R_i - U_R_i)
        return jnp.where(
            S_L >= 0, F_L_i, jnp.where(S_M >= 0, f_starL, jnp.where(S_R > 0, f_starR, F_R_i))
        )

    # Component 0: rho
    flux_0 = hllc_component(
        F_L_i=rho_vn_L,
        F_R_i=rho_vn_R,
        U_L_i=rho_L,
        U_R_i=rho_R,
        U_star_L_i=rho_star_L,
        U_star_R_i=rho_star_R,
    )
    # Energy (same for all axes)
    F_L_E = (E_L + P_L) * vn_L
    F_R_E = (E_R + P_R) * vn_R
    U_star_L_E = rho_star_L * E_star_L
    U_star_R_E = rho_star_R * E_star_R
    flux_E = hllc_component(F_L_E, F_R_E, E_L, E_R, U_star_L_E, U_star_R_E)

    # Momentum components depend on which is normal vs tangential.
    if axis == 0:  # normal = x, tangential = y, z
        flux_mx = hllc_component(
            rho_vn_L * vx_L + P_L,
            rho_vn_R * vx_R + P_R,
            rho_L * vx_L,
            rho_R * vx_R,
            rho_star_L * S_M,
            rho_star_R * S_M,
        )
        flux_my = hllc_component(
            rho_vn_L * vy_L,
            rho_vn_R * vy_R,
            rho_L * vy_L,
            rho_R * vy_R,
            rho_star_L * vt1_L,
            rho_star_R * vt1_R,
        )
        flux_mz = hllc_component(
            rho_vn_L * vz_L,
            rho_vn_R * vz_R,
            rho_L * vz_L,
            rho_R * vz_R,
            rho_star_L * vt2_L,
            rho_star_R * vt2_R,
        )
    elif axis == 1:  # normal = y, tangential = x, z
        flux_mx = hllc_component(
            rho_vn_L * vx_L,
            rho_vn_R * vx_R,
            rho_L * vx_L,
            rho_R * vx_R,
            rho_star_L * vt1_L,
            rho_star_R * vt1_R,
        )
        flux_my = hllc_component(
            rho_vn_L * vy_L + P_L,
            rho_vn_R * vy_R + P_R,
            rho_L * vy_L,
            rho_R * vy_R,
            rho_star_L * S_M,
            rho_star_R * S_M,
        )
        flux_mz = hllc_component(
            rho_vn_L * vz_L,
            rho_vn_R * vz_R,
            rho_L * vz_L,
            rho_R * vz_R,
            rho_star_L * vt2_L,
            rho_star_R * vt2_R,
        )
    else:  # axis == 2: normal = z, tangential = x, y
        flux_mx = hllc_component(
            rho_vn_L * vx_L,
            rho_vn_R * vx_R,
            rho_L * vx_L,
            rho_R * vx_R,
            rho_star_L * vt1_L,
            rho_star_R * vt1_R,
        )
        flux_my = hllc_component(
            rho_vn_L * vy_L,
            rho_vn_R * vy_R,
            rho_L * vy_L,
            rho_R * vy_R,
            rho_star_L * vt2_L,
            rho_star_R * vt2_R,
        )
        flux_mz = hllc_component(
            rho_vn_L * vz_L + P_L,
            rho_vn_R * vz_R + P_R,
            rho_L * vz_L,
            rho_R * vz_R,
            rho_star_L * S_M,
            rho_star_R * S_M,
        )

    return jnp.stack([flux_0, flux_mx, flux_my, flux_mz, flux_E])


def hllc_flux_primitive_1d(W_L, W_R, axis, gamma):
    """
    HLLC Riemann solver for 1D Euler using PRIMITIVE variables.

    Parameters
    ----------
    W_L, W_R : array, shape (3, Nx)
        Left and right PRIMITIVE states: [rho, vx, P]
    axis : int (static, ignored for 1D)
        Direction (ignored for 1D)
    gamma : float
        Ratio of specific heats

    Returns
    -------
    flux : array, shape (3, Nx)
        HLLC flux in CONSERVATIVE form
    """
    rho_L, vx_L, P_L = W_L
    rho_R, vx_R, P_R = W_R

    gm1 = gamma - 1.0

    # Sound speed
    c_L = jnp.sqrt(gamma * P_L / rho_L)
    c_R = jnp.sqrt(gamma * P_R / rho_R)

    # Wave speeds
    S_L = jnp.minimum(vx_L - c_L, vx_R - c_R)
    S_R = jnp.maximum(vx_L + c_L, vx_R + c_R)

    # Denominators
    denom_L = rho_L * (S_L - vx_L)
    denom_R = rho_R * (S_R - vx_R)

    # Middle speed
    S_M = (P_R - P_L + denom_L * vx_L - denom_R * vx_R) / (denom_L - denom_R)

    # Star densities
    rho_star_L = denom_L / (S_L - S_M)
    rho_star_R = denom_R / (S_R - S_M)

    # Energy
    E_L = P_L / gm1 + 0.5 * rho_L * vx_L * vx_L
    E_R = P_R / gm1 + 0.5 * rho_R * vx_R * vx_R

    E_star_L = E_L / rho_L + (S_M - vx_L) * (S_M + P_L / denom_L)
    E_star_R = E_R / rho_R + (S_M - vx_R) * (S_M + P_R / denom_R)

    # Momentum
    rho_vx_L = rho_L * vx_L
    rho_vx_R = rho_R * vx_R

    # Conservative states
    U_L = jnp.stack([rho_L, rho_vx_L, E_L])
    U_R = jnp.stack([rho_R, rho_vx_R, E_R])

    U_star_L = jnp.stack([rho_star_L, rho_star_L * S_M, rho_star_L * E_star_L])
    U_star_R = jnp.stack([rho_star_R, rho_star_R * S_M, rho_star_R * E_star_R])

    # Fluxes
    F_L = jnp.stack([rho_vx_L, rho_vx_L * vx_L + P_L, (E_L + P_L) * vx_L])
    F_R = jnp.stack([rho_vx_R, rho_vx_R * vx_R + P_R, (E_R + P_R) * vx_R])

    # HLLC selection
    conditions = [
        S_L >= 0,
        (S_L < 0) & (S_M >= 0),
        (S_M < 0) & (S_R > 0),
    ]

    choices = [
        F_L,
        F_L + S_L * (U_star_L - U_L),
        F_R + S_R * (U_star_R - U_R),
    ]

    flux = jnp.select(conditions, choices, default=F_R)

    return flux


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
