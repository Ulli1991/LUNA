"""External, position-dependent gravitational accelerations from analytic
static potentials. Used for galaxy-scale simulations (CGOLS-style M82) where
the gas self-gravity is negligible compared to the fixed stellar + halo
potential.

All quantities are in the Arepo unit system (kpc, 1e10 Msun, km/s), so:

    G_code  = 43.0071   (km/s)^2 * kpc / (1e10 Msun)
    Lengths in kpc, masses in 1e10 Msun, velocities in km/s.

Supported components:

  * Miyamoto-Nagai disk:
        Phi(R, z) = -G * M / sqrt(R^2 + (a + sqrt(z^2 + b^2))^2)
        g_R       = -G * M * R / D^(3/2)
        g_z       = -G * M * (a + sqrt(z^2+b^2)) * z / (sqrt(z^2+b^2) * D^(3/2))
        D         = R^2 + (a + sqrt(z^2+b^2))^2

  * NFW spherical halo:
        rho(r)  = rho_s / [(r/r_s)(1 + r/r_s)^2]
        M(<r)   = 4*pi*rho_s*r_s^3 * [ln(1+r/r_s) - (r/r_s)/(1+r/r_s)]
        g_r     = -G*M(<r)/r^2

    Parametrised by (M_200, c). We compute r_200 from M_200 and an
    z=0 critical density rho_crit_code = 1.4e-8 (1e10 Msun / kpc^3),
    then r_s = r_200 / c and rho_s from the normalization integral.
"""

from __future__ import annotations

import jax.numpy as jnp

# Gravitational constant in Arepo code units (kpc, 1e10 Msun, km/s).
G_CODE = 43007.1

# Critical density at z=0 in Arepo code units (1e10 Msun / kpc^3).
RHO_CRIT_CODE = 1.4e-8


def _miyamoto_nagai_accel(X, Y, Z, M, a, b):
    """Cartesian acceleration from a Miyamoto-Nagai disk centered at origin.
    Returns (g_x, g_y, g_z) with same shape as X, Y, Z.
    """
    R2 = X * X + Y * Y
    zb = jnp.sqrt(Z * Z + b * b)
    a_zb = a + zb
    D = R2 + a_zb * a_zb
    D32 = D * jnp.sqrt(D)  # D^(3/2)
    inv_D32 = 1.0 / D32
    g_x = -G_CODE * M * X * inv_D32
    g_y = -G_CODE * M * Y * inv_D32
    g_z = -G_CODE * M * a_zb * Z * inv_D32 / zb
    return g_x, g_y, g_z


def _nfw_params_from_M200_c(M_200, c):
    """Return (r_s, rho_s) given M_200 in 1e10 Msun and concentration c."""
    # r_200^3 = M_200 / (4/3 pi * 200 * rho_crit)
    r_200_cubed = M_200 / (4.0 / 3.0 * jnp.pi * 200.0 * RHO_CRIT_CODE)
    r_200 = r_200_cubed ** (1.0 / 3.0)
    r_s = r_200 / c
    # rho_s from M_200 normalization
    f_c = jnp.log(1.0 + c) - c / (1.0 + c)
    rho_s = M_200 / (4.0 * jnp.pi * r_s**3 * f_c)
    return r_s, rho_s


def _nfw_accel(X, Y, Z, M_200, c):
    """Spherical NFW halo acceleration. Returns (g_x, g_y, g_z)."""
    r_s, rho_s = _nfw_params_from_M200_c(M_200, c)
    r2 = X * X + Y * Y + Z * Z
    r = jnp.sqrt(r2 + 1e-10)  # softening to avoid r=0
    x = r / r_s
    # M_enclosed = 4 pi rho_s r_s^3 (ln(1+x) - x/(1+x))
    M_enc = 4.0 * jnp.pi * rho_s * r_s**3 * (jnp.log(1.0 + x) - x / (1.0 + x))
    g_mag = -G_CODE * M_enc / (r2 + 1e-10)
    g_x = g_mag * X / r
    g_y = g_mag * Y / r
    g_z = g_mag * Z / r
    return g_x, g_y, g_z


def combined_acceleration(
    X, Y, Z, MN_M, MN_a, MN_b, NFW_M200, NFW_c, center_x=0.0, center_y=0.0, center_z=0.0
):
    """Combined Miyamoto-Nagai disk + NFW halo, centered on (center_*)."""
    Xc = X - center_x
    Yc = Y - center_y
    Zc = Z - center_z
    gmn = _miyamoto_nagai_accel(Xc, Yc, Zc, MN_M, MN_a, MN_b)
    gnfw = _nfw_accel(Xc, Yc, Zc, NFW_M200, NFW_c)
    return tuple(a + b for a, b in zip(gmn, gnfw))


def vcirc_at_midplane(R, MN_M, MN_a, MN_b, NFW_M200, NFW_c):
    """Circular velocity in the midplane (z=0) from combined MN+NFW.
    v_circ^2 = R * |g_R(R, 0)|.  Used for setting disk rotation.
    """
    # In the midplane (z=0): MN g_R = -G M R / (R^2 + (a+b)^2)^(3/2)
    a_b = MN_a + MN_b
    D = R * R + a_b * a_b
    g_R_mn = G_CODE * MN_M * R / (D * jnp.sqrt(D))
    # NFW spherical:
    r_s, rho_s = _nfw_params_from_M200_c(NFW_M200, NFW_c)
    x = R / r_s
    M_enc = 4.0 * jnp.pi * rho_s * r_s**3 * (jnp.log(1.0 + x) - x / (1.0 + x))
    g_R_nfw = G_CODE * M_enc / (R * R + 1e-10)
    g_R_tot = g_R_mn + g_R_nfw  # both pointing toward origin
    return jnp.sqrt(R * g_R_tot)
