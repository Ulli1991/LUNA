"""Photodissociation, photoelectric field and shielding for networks 5 and 17 (calc_photo.F, photoinit_ism.F).
Columns are arrays whose last axis is the TreeCol pixel, each weighted 1 / n_pix; docs/CHEMISTRY.md.
"""
import jax.numpy as jnp
import numpy as np

from luna.chemistry import tables as tb

AV_PER_NH = 5.348e-22            # A_V per H column at dust_to_gas = 1 (AV_conversion_factor)
SIGMA_DUST_LW = 2e-21            # LW dust cross-section per H (calc_photo.F:273)
K_H2_THIN = 3.3e-11              # H2 photodissociation rate at G0 = 1 [s^-1] (photoinit_ism.F:50)

K_CO_THIN = 1.235e-10            # CO photodissociation rate at G0 = 1 [s^-1] (photoinit_ism.F:101)
K_CHX = 2.94e-10                 # CH_x photodestruction per G_dust, NL97 (rate_eq_simple.F:213)


def _pixel_mean(values):
    """sum_i (1/n_pix) v_i over the last axis (calc_photo.F fac_uv_pix = 1/nray), in loop order."""
    w = values * (1.0 / values.shape[-1])
    s = w[..., 0]
    for k in range(1, values.shape[-1]):         # not a reduction: same bits for any batch shape
        s = s + w[..., k]
    return s


def chi_isrf(av):
    """Dust-attenuated ISRF for dust heating, chi(A_V) of calc_photo.F:683 (Mathis+83), linear in A_V."""
    tab = tb.device_table("chi", lambda: jnp.asarray(tb.raw()["chi_tab_mathis"]))
    av = jnp.asarray(av, jnp.float64)
    a = jnp.clip(av, 0.0, 199.999999)
    i = jnp.floor(a).astype(jnp.int32)
    lo = jnp.where(i == 0, 1.0, tab[jnp.maximum(i - 1, 0)])
    hi = tab[jnp.minimum(i, 199)]
    chi = lo + (a - i) * (hi - lo)
    chi = jnp.where(av > 2e2, tab[199], chi)
    return jnp.where(av == 0.0, 1.0, chi)


def h2_self_shielding(n_h2_col, temp):
    """H2 self-shielding, DB96 eq. 37 with b = sqrt(k T / m_H), the mean over pixels (calc_photo.F:745)."""
    b5 = jnp.sqrt(tb.KBOLTZ * jnp.asarray(temp, jnp.float64) / tb.MH) / 1e5
    x = n_h2_col / 5e14
    sq = jnp.sqrt(1.0 + x)
    f = 9.65e-1 / ((1.0 + x / b5[..., None]) * (1.0 + x / b5[..., None])) \
        + 3.5e-2 * jnp.exp(-8.5e-4 * sq) / sq
    return _pixel_mean(f)


def hd_self_shielding(n_hd_local, n_h2_col, x_h2, x_hd, abund_d, temp, weight_both=False):
    """HD self-shielding, calc_HD_shielding_factor (iphoto = 6); 1/n_pix on the first term only unless weight_both."""
    b5 = jnp.sqrt(tb.KBOLTZ * jnp.asarray(temp, jnp.float64) / tb.MH) / 1e5
    either0 = (x_h2 == 0.0) | (x_hd == 0.0)
    ratio = jnp.where(either0, abund_d, x_hd / jnp.where(either0, 1.0, x_h2))
    x = (n_hd_local[..., None] + ratio[..., None] * n_h2_col) / 5e14
    sq = jnp.sqrt(1.0 + x)
    a = jnp.where(x < 1e1, 9.5e-1, jnp.where(x < 1e4, 9.4e-1, 9.85e-1))
    tail = jnp.where(x < 1e1, 5.0e-2 * jnp.exp(-8.5e-4 * sq) / sq,
                     jnp.where(x < 1e4, 6.0e-2 / sq**1.5, 4.0e-3 / sq**0.9))
    d = 1.0 + x / b5[..., None]
    w = 1.0 / n_h2_col.shape[-1]
    terms = w * (a / (d * d) + tail) if weight_both else w * a / (d * d) + tail
    s = terms[..., 0]
    for k in range(1, terms.shape[-1]):          # the Fortran's pixel loop order
        s = s + terms[..., k]
    return s


def _shield_table(lo, n, columns, factors):
    """cheminmo.F:1893: Lee+96 factors resampled to 0.05 dex from log N = lo."""
    logn = lo + 0.05 * np.arange(n)
    j = np.searchsorted(columns, logn, side="right") - 1
    dn = (logn - columns[j]) / (columns[j + 1] - columns[j])
    return dn * factors[j + 1] + (1.0 - dn) * factors[j]


def co_shielding(n_co_col, n_h2_col):
    """CO shielding by CO and H2 lines (Lee+96, CO_SHIELDING only): the product of the two pixel means."""
    r = tb.raw()
    s_co = jnp.asarray(_shield_table(12.0, 161, r["CO_shield_columns"], r["CO_shield_factors"]))
    s_h2 = jnp.asarray(_shield_table(13.0, 181, r["H2_shield_columns"], r["H2_shield_factors"]))

    def lookup(col, tab, nmin, nmax, ioff):
        lg = jnp.log10(jnp.clip(col, nmin, nmax))
        ipos = jnp.floor(lg / 0.05).astype(jnp.int32) - ioff
        ip = jnp.clip(ipos, 1, tab.size - 1)
        dth = tab[ip] - tab[ip - 1]
        dnact = lg - (ip + ioff) * 0.05
        val = tab[ip - 1] + dth * (dnact / 0.05)
        val = jnp.where(ipos == 0, 1.0, val)
        val = jnp.where(col <= nmin, 1.0, val)
        return jnp.where(col >= nmax, tab[-1], val)

    theta_co = _pixel_mean(lookup(n_co_col, s_co, 1e12, 1e20, 239))
    theta_h2 = _pixel_mean(lookup(n_h2_col, s_h2, 1e13, 1e22, 259))
    return theta_h2 * theta_co


def photo_rates(g0, n_h_col, n_h2_col, n_co_col, dgr, av_local=0.0, g0_lw=None,
                co_shield=False):
    """State-independent photo rates (calc_photo, iphoto = 6): dict k_h2_dust, k_co, g_dust, chi."""
    g0 = jnp.asarray(g0, jnp.float64)
    dgr = jnp.asarray(dgr, jnp.float64)
    g0_lw = g0 if g0_lw is None else g0_lw
    f_dust = _pixel_mean(jnp.exp(-(SIGMA_DUST_LW * dgr)[..., None] * n_h_col))
    av = AV_PER_NH * dgr[..., None] * n_h_col + jnp.asarray(av_local)[..., None]
    chi = _pixel_mean(chi_isrf(av))
    f_simp = _pixel_mean(jnp.exp(-2.5 * av))
    k_co = K_CO_THIN * g0 * f_simp
    if co_shield:
        k_co = k_co * co_shielding(n_co_col, n_h2_col)
    on = g0 != 0.0
    return {"k_h2_dust": jnp.where(on, K_H2_THIN * g0_lw * f_dust, 0.0),
            "k_co": jnp.where(on, k_co, 0.0),
            "g_dust": jnp.where(on, 1.0 * g0 * f_simp, 0.0),
            "chi": chi}
