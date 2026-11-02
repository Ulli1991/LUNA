"""Table lookups for the generated network-5 code: each returns values and in-cell slopes.
The values equal the existing lookups' (cooling, thermo, photo); a slope is the derivative jax.jacfwd takes there.
"""
import math

import jax.numpy as jnp

from luna.chemistry import cooling, photo, rates_net5, thermo
from luna.chemistry import tables as tb

LN10 = math.log(10.0)           # a Python constant: a kernel would take log(10.0) in FP32


def gtw(a, b):
    """d max(a, b) / d a as JAX takes it: 1 if a > b, 1/2 on a tie, else 0."""
    return jnp.where(a > b, 1.0, jnp.where(a == b, 0.5, 0.0))


def _dclip(x, lo, hi):
    """d clip(x, lo, hi) / dx with JAX's tie rule (clip = min(hi, max(lo, x)))."""
    m = jnp.maximum(lo, x)
    return gtw(x, lo) * gtw(hi, m)


def photo_terms(p):
    """photo.photo_rates for network 5's arguments (state-independent)."""
    av_local = photo.AV_PER_NH * p.dgr * (p.dl * p.n_h)
    return photo.photo_rates(p.g0, p.n_h_col, p.n_h2_col, p.n_co_col, p.dgr, av_local)


_ROWS = {"chem_t": (cooling.chem_rows, True), "cool_t": (cooling.cool_rows, True),
         "cool_f": (cooling.cool_rows, False)}


def rows(kind, temp, vkeys, skeys):
    """Rows vkeys of a T table at temp and the slopes of rows skeys (as cooling.interp_rows)."""
    table, clamp = _ROWS[kind]
    tab = table()
    i, dtemp = cooling._node_index(temp, clamp)
    on = (temp > tb.TMIN) & (temp < tb.TMAX) if clamp else temp > tb.TMIN
    vals = {k: tab[k][0][i] + dtemp * tab[k][1][i] for k in vkeys}
    slopes = {k: jnp.where(on, tab[k][1][i], 0.0) for k in skeys}
    return vals, slopes


def temp_gamma(ekn, x_h2, slopes):
    """thermo.temperature_gamma_table and, if slopes, (dT/dekn, dT/dx, dg/dekn, dg/dx)."""
    temp_t, gamma_t = thermo._tables()
    atomic = (ekn < 1.0) | (ekn > 5e4) | (x_h2 == 0.0)
    e_safe = jnp.clip(ekn, 1.0, 5e4)
    de = _dclip(ekn, 1.0, 5e4)
    idx_en = jnp.floor(e_safe)
    diff_en = e_safe - idx_en
    idx_en = idx_en.astype(jnp.int32)
    f01 = thermo._F01
    idx_h2 = jnp.clip(jnp.floor(x_h2 / f01).astype(jnp.int32), 0, 4)
    lo = (f01 * idx_h2.astype(jnp.float32)).astype(jnp.float64)
    diff_h2 = (x_h2 - lo) / f01

    def row(block):
        i = 50000 * block + idx_en - 1
        dt, dg = temp_t[i + 1] - temp_t[i], gamma_t[i + 1] - gamma_t[i]
        return temp_t[i] + dt * diff_en, gamma_t[i] + dg * diff_en, dt * de, dg * de

    t_hi, g_hi, dt_hi, dg_hi = row(idx_h2)
    t_lo, g_lo, dt_lo, dg_lo = row(jnp.maximum(idx_h2 - 1, 0))
    first = idx_h2 == 0
    t_lo, dt_lo = jnp.where(first, (2.0 / 3.0) * e_safe, t_lo), jnp.where(first, (2.0 / 3.0) * de, dt_lo)
    g_lo, dg_lo = jnp.where(first, 5.0 / 3.0, g_lo), jnp.where(first, 0.0, dg_lo)
    temp = t_lo + (t_hi - t_lo) * diff_h2
    gamma = g_lo + (g_hi - g_lo) * diff_h2
    t_half, g_half, dt_half, dg_half = row(4)
    half = x_h2 == 0.5
    temp = jnp.where(atomic, (2.0 / 3.0) * ekn, jnp.where(half, t_half, temp))
    gamma = jnp.where(atomic, 5.0 / 3.0, jnp.where(half, g_half, gamma))
    if not slopes:
        return temp, gamma
    t_e = jnp.where(atomic, 2.0 / 3.0, jnp.where(half, dt_half, dt_lo + (dt_hi - dt_lo) * diff_h2))
    g_e = jnp.where(atomic | half, jnp.where(atomic, 0.0, dg_half), dg_lo + (dg_hi - dg_lo) * diff_h2)
    t_x = jnp.where(atomic | half, 0.0, (t_hi - t_lo) / f01)
    g_x = jnp.where(atomic | half, 0.0, (g_hi - g_lo) / f01)
    return temp, gamma, t_e, t_x, g_e, g_x


def co(temp, n_col, slopes):
    """Raw cooling.co_cool tables (log L0, log LTE, alpha, log n05, log vib) at (temp, n_col), with slopes."""
    c = cooling.co_arrays()
    ti = jnp.floor(jnp.clip(temp, 5.0, 2000.0))
    it = jnp.where(temp >= 2000.0, 1995, ti.astype(jnp.int32) - 5)
    t_on = ~((temp >= 2000.0) | (temp == 5.0))
    dt = jnp.where(t_on, temp - ti, 0.0)
    lo, hi = c["n_rot"][0], c["n_rot"][-1]
    iN = jnp.floor(10 * jnp.clip(n_col, lo, hi) - 144).astype(jnp.int32) - 1
    iN = jnp.where(n_col <= lo, 0, jnp.where(n_col >= hi, 45, iN))
    n_on = ~((n_col <= lo) | (n_col >= hi))
    dN = jnp.where(n_on, (n_col - c["n_rot"][iN]) / 0.1, 0.0)
    iN2 = jnp.minimum(iN + 1, 45)
    vals = {"l0": c["L0"][it] + dt * c["dL0"][it]}
    d_t = {"l0": jnp.where(t_on, c["dL0"][it], 0.0)}
    d_n = {}
    for k in ("lte", "alp", "n05"):
        v1 = c[k][iN, it] + dt * c["d" + k][iN, it]
        v2 = c[k][iN2, it] + dt * c["d" + k][iN2, it]
        last = iN == 45
        vals[k] = jnp.where(last, v1, v1 + (v2 - v1) * dN)
        if slopes:
            s1 = jnp.where(t_on, c["d" + k][iN, it], 0.0)
            s2 = jnp.where(t_on, c["d" + k][iN2, it], 0.0)
            d_t[k] = jnp.where(last, s1, s1 + (s2 - s1) * dN)
            d_n[k] = jnp.where(last | ~n_on, 0.0, (v2 - v1) / 0.1)
    tv = jnp.floor(jnp.clip(temp, 100.0, 4000.0))
    itv = jnp.where(temp >= 4000.0, 3900, tv.astype(jnp.int32) - 100)
    itv = jnp.where(temp <= 100.0, 0, itv)
    tv_on = ~((temp <= 100.0) | (temp >= 4000.0))
    dtv = jnp.where(tv_on, temp - tv, 0.0)
    lo, hi = c["n_vib"][0], c["n_vib"][-1]
    jN = jnp.floor(10 * jnp.clip(n_col, lo, hi) - 129).astype(jnp.int32) - 1
    jN = jnp.where(n_col <= lo, 0, jnp.where(n_col >= hi, 60, jN))
    nv_on = ~((n_col <= lo) | (n_col >= hi))
    dNv = jnp.where(nv_on, (n_col - c["n_vib"][jN]) / 0.1, 0.0)
    jN2 = jnp.minimum(jN + 1, 60)
    v1 = c["vib"][jN, itv] + dtv * c["dvib"][jN, itv]
    v2 = c["vib"][jN2, itv] + dtv * c["dvib"][jN2, itv]
    last = jN == 60
    vals["vib"] = jnp.where(last, v1, v1 + (v2 - v1) * dNv)
    if not slopes:
        return vals, {}, {}
    s1 = jnp.where(tv_on, c["dvib"][jN, itv], 0.0)
    s2 = jnp.where(tv_on, c["dvib"][jN2, itv], 0.0)
    d_t["vib"] = jnp.where(last, s1, s1 + (s2 - s1) * dNv)
    d_n["vib"] = jnp.where(last | ~nv_on, 0.0, (v2 - v1) / 0.1)
    return vals, d_t, d_n


def h2_opacity(temp, n_eff, slopes):
    """cooling.h2_opacity and, if slopes, its derivatives in temp and n_eff."""
    r = tb.raw()
    lt, lc, op = tb.device_table("h2_opac", lambda: (
        jnp.asarray(r["h2_opac_temp"]), jnp.asarray(r["h2_opac_column"]), jnp.asarray(r["h2_opac"])))
    cmin, cmax = 10 ** float(r["h2_opac_column"][0]), 10 ** float(r["h2_opac_column"][-1])
    logn = jnp.log10(jnp.clip(n_eff, cmin, cmax))
    jj = jnp.clip((1 + jnp.floor(10 * (logn - 17.0))).astype(jnp.int32), 1, lc.size - 1) - 1
    dn = (logn - lc[jj]) / (lc[jj + 1] - lc[jj])
    logt = jnp.log10(temp)
    ii = jnp.clip((1 + jnp.floor((jnp.clip(logt, lt[0], lt[-1]) - 1.5) / tb.f32(0.03))).astype(
        jnp.int32), 1, lt.size - 1) - 1
    edge = (logt <= lt[0]) | (logt >= lt[-1])
    ii = jnp.where(logt <= lt[0], 0, jnp.where(logt >= lt[-1], lt.size - 1, ii))
    span = jnp.where(edge, 1.0, lt[jnp.minimum(ii + 1, lt.size - 1)] - lt[ii])   # 0 on the edge
    dT = jnp.where(edge, 0.0, (logt - lt[ii]) / span)
    i2 = jnp.minimum(ii + 1, lt.size - 1)
    inT = dT > 0
    o1 = jnp.where(inT, op[ii, jj] + dT * (op[i2, jj] - op[ii, jj]), op[ii, jj])
    o2 = jnp.where(inT, op[ii, jj + 1] + dT * (op[i2, jj + 1] - op[ii, jj + 1]), op[ii, jj + 1])
    opac = 10.0 ** (o1 + dn * (o2 - o1))
    lowc, highc = n_eff <= cmin, n_eff >= cmax
    val = jnp.where(lowc, 1.0, jnp.where(highc, 0.0, opac))
    if not slopes:
        return val
    dlt = lt[i2] - lt[ii]
    do1 = jnp.where(inT, (op[i2, jj] - op[ii, jj]) / jnp.where(dlt == 0, 1.0, dlt), 0.0)
    do2 = jnp.where(inT, (op[i2, jj + 1] - op[ii, jj + 1]) / jnp.where(dlt == 0, 1.0, dlt), 0.0)
    d_t = opac * (do1 + dn * (do2 - do1)) / temp
    dlogn = _dclip(n_eff, cmin, cmax) / (n_eff * LN10)
    d_n = opac * LN10 * (o2 - o1) / (lc[jj + 1] - lc[jj]) * dlogn
    off = lowc | highc
    return val, jnp.where(off, 0.0, d_t), jnp.where(off, 0.0, d_n)


def dust_td(n_h, temp, x_hi, chi, g0_dust, dgr, tdust_in, h2_form_ex, h2_form_kin, rtol,
            iters=None):
    """T_dust from the existing Newton solve (cooling.dust_temperature), iters steps if given."""
    return cooling.dust_temperature(n_h, temp, x_hi, chi, g0_dust, dgr, tdust_in, h2_form_ex,
                                    h2_form_kin, rtol, iters)[2]


def photo_terms_1(q):
    """photo.photo_rates for one pixel whose columns are per-cell values (no pixel axis)."""
    av = photo.AV_PER_NH * q.dgr * q.n_h_col + photo.AV_PER_NH * q.dgr * (q.dl * q.n_h)
    f_simp = jnp.exp(-2.5 * av)
    on = q.g0 != 0.0
    f_dust = jnp.exp(-(photo.SIGMA_DUST_LW * q.dgr) * q.n_h_col)
    return {"k_h2_dust": jnp.where(on, photo.K_H2_THIN * q.g0 * f_dust, 0.0),
            "k_co": jnp.where(on, photo.K_CO_THIN * q.g0 * f_simp, 0.0),
            "g_dust": jnp.where(on, 1.0 * q.g0 * f_simp, 0.0), "chi": photo.chi_isrf(av)}


def invalid(y0, y1, y2, y3):
    """rates_net5's validate_input flag."""
    return rates_net5.invalid_state(y0, y1, y2, y3)
