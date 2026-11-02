"""Corrections from Gong, Ostriker & Wolfire 2017 (ApJ 843, 38; "GOW17") to the chemcool networks.

Used by a network's rhs when fixes="gow17"; docs/CHEMISTRY.md lists each fix and its source.
"""
import functools

import jax.numpy as jnp
import numpy as np

from luna.chemistry import tables as tb
from luna.chemistry.photo import AV_PER_NH

K_CO_THIN = 2.4e-10      # CO photodissociation in the Draine field [s^-1] (GOW17 T2 #16, Heays+17)
GAMMA_CO = 3.88          # its dust exponent (GOW17 T2 #16 = Heays+17 Table 18 gamma_exp,ISM)
G0_DRAINE = 1.7          # the Draine field in Habing units
K_HEP_CO = 1.6e-9        # He+ + CO -> C+ + O + He [cm^3 s^-1] (GOW17 T1 #8)
K_HEP_H2_HP = 1.26e-13   # He+ + H2 -> H+ + He + H, times exp(-22.5/T) (GOW17 T1 #7)
K_HEP_H2_H2P = 7.2e-15   # He+ + H2 -> H2+ + He (GOW17 T1 #25)
CR_HE = 1.1              # cr + He -> He+ + e, per xi_H (GOW17 T2 #8)
CR_PHOTO_CO = 90.0       # CR-induced gamma + CO -> C + O, per xi_H times f_mol (GOW17 T2 #12)
SEC_ATOMIC, SEC_MOLECULAR = 1.5, 1.15   # total / primary H ionisation (GOW17 App. A.1 item 3)


@functools.cache
def visser09_table():
    """(log10 N_CO nodes, log10 N_H2 nodes, log10 Theta[N_H2, N_CO]) of the Visser+09 grid."""
    d = np.loadtxt(tb.data_path("co_shield_visser09.txt"))
    return np.log10(np.maximum(d[0, 1:], 1.0)), np.log10(d[1:, 0]), np.log10(d[1:, 1:])


def _cell(lx, ax):
    """Lower node index and fraction of lx on the ascending axis ax (lx inside the axis)."""
    i = jnp.clip(jnp.searchsorted(ax, lx, side="right") - 1, 0, ax.size - 2)
    return i, (lx - ax[i]) / (ax[i + 1] - ax[i])


def co_shielding_visser09(n_co_col, n_h2_col):
    """Theta(N_CO, N_H2): bilinear in log Theta vs (log N_CO, log N_H2), columns clamped to the grid."""
    ax, ay, lt = (jnp.asarray(a) for a in visser09_table())
    lx = jnp.clip(jnp.log10(jnp.maximum(n_co_col, 1.0)), ax[0], ax[-1])
    ly = jnp.clip(jnp.log10(jnp.maximum(n_h2_col, 1.0)), ay[0], ay[-1])
    i, fx = _cell(lx, ax)
    j, fy = _cell(ly, ay)
    v = ((1 - fx) * (1 - fy) * lt[j, i] + fx * (1 - fy) * lt[j, i + 1]
         + (1 - fx) * fy * lt[j + 1, i] + fx * fy * lt[j + 1, i + 1])
    return 10.0 ** v


def co_photo_rate(g0, n_h_col, n_h2_col, n_co_col, dgr, av_local=0.0):
    """CO photodissociation rate [s^-1], GOW17 eq. 8: pixel mean of exp(-3.88 A_V) Theta (G0 in Habing).

    Columns are (..., n_pix); each pixel has weight 1 / n_pix.
    """
    g0 = jnp.asarray(g0, jnp.float64)
    dgr = jnp.asarray(dgr, jnp.float64)
    av = AV_PER_NH * dgr[..., None] * n_h_col + jnp.asarray(av_local)[..., None]
    f = jnp.exp(-GAMMA_CO * av) * co_shielding_visser09(n_co_col, n_h2_col)
    return K_CO_THIN * (g0 / G0_DRAINE) * jnp.mean(f, axis=-1)


def molecular_fraction(x_hi, x_h2):
    """Fraction of neutral H nuclei in H2, 2 x_H2 / (x_HI + 2 x_H2); 0 without neutral H."""
    tot = x_hi + 2.0 * x_h2
    return jnp.where(tot > 0.0, 2.0 * x_h2 / jnp.where(tot > 0.0, tot, 1.0), 0.0)


def xi_from_zeta(zeta, x_hi, x_h2):
    """GOW17 primary xi_H from chemcool's total zeta, equal total ionisation per neutral H nucleus."""
    f = molecular_fraction(x_hi, x_h2)
    return zeta / (SEC_ATOMIC * (1.0 - f) + SEC_MOLECULAR * f)


def he_grain_coefficients(temp):
    """WD01 He+ fit terms (5115 T^3.903e-7, -0.4956 - 5.494e-7 ln T): chemcool chtab(135/136)."""
    temp = jnp.asarray(temp, jnp.float64)
    return 5.115e3 * temp**3.903e-7, -0.4956 - 5.494e-7 * jnp.log(temp)


def he_grain_table():
    """chtab(135/136) on chemcool's temperature grid, as cheminmo.F:690-694 fills them (numpy)."""
    T = tb.temptab()
    return 5.115e3 * T**3.903e-7, -0.4956 - 5.494e-7 * np.log(T)


def he_grain_recombination(temp, phi, dgr, coeffs=None):
    """He+ + e + grain -> He [cm^3 s^-1], WD01 as GOW17 T2 #4 and chemcool rate_eq.F:364-371.

    phi = G_dust sqrt(T) / n_e as the network computes it; coeffs = (ch135, ch136) if tabulated.
    """
    c135, c136 = he_grain_coefficients(temp) if coeffs is None else coeffs
    phi_s = jnp.maximum(phi, 1e-6)
    slow = 1.0 + 3.185e-7 * phi_s**1.512 * (1.0 + c135 * phi_s**c136)
    return jnp.where(phi < 1e-6, 5.572e-14 * dgr, 5.572e-14 * dgr / slow)


def grain_phi(g_dust, temp, n, x_e):
    """WD01 charging parameter G_dust sqrt(T) / n_e as chemcool's rate_eq computes it (1e20 without e-)."""
    has_e = x_e != 0.0
    return jnp.where(has_e, g_dust * jnp.sqrt(temp) / (n * jnp.where(has_e, x_e, 1.0)), 1e20)


def he_grain_rate(g_dust, temp, n, x_e, dgr):
    """He+ grain recombination coefficient [cm^3 s^-1] from the network's state: the network-17 hook."""
    return he_grain_recombination(temp, grain_phi(g_dust, temp, n, x_e), dgr)


def he_case_b(temp):
    """He+ + e -> He, case B (Hummer & Storey 1998 fit as GOW17 T1 #14) [cm^3 s^-1]."""
    lt = jnp.log10(temp)
    return 1e-11 / jnp.sqrt(temp) * (11.19 - 1.676 * lt - 0.2852 * lt**2 + 0.04433 * lt**3)


def he_plus_equilibrium(xi, temp, n, x_h2, x_e, x_co, k_he_gr):
    """Equilibrium x_He+ for a network without He+: CR ionisation over the GOW17 He+ sinks."""
    sinks = (k_he_gr + (K_HEP_H2_HP * jnp.exp(-22.5 / temp) + K_HEP_H2_H2P) * x_h2
             + he_case_b(temp) * x_e + K_HEP_CO * x_co)
    return CR_HE * xi * tb.ABHE / jnp.maximum(n * sinks, 1e-300)


def co_cr_destruction(xi, temp, n, x_hi, x_h2, x_e, x_co, k_he_gr):
    """Cosmic-ray CO destruction per CO [s^-1]: He+ + CO (T1 #8) + CR-induced photons (T2 #12)."""
    x_hep = he_plus_equilibrium(xi, temp, n, x_h2, x_e, x_co, k_he_gr)
    return K_HEP_CO * x_hep * n + CR_PHOTO_CO * xi * molecular_fraction(x_hi, x_h2), x_hep
