"""Heating and cooling of network-5 and network-17 gas (cool_func.F, 28 terms) and the dust temperature.

Lambda > 0 is cooling [erg s^-1 cm^-3], de/dt = -Lambda. Port of chemcool; the terms in docs/CHEMISTRY.md."""
import jax
import jax.numpy as jnp

from luna.chemistry import tables as tb

KBOLTZ, EV, MH = tb.KBOLTZ, tb.EV, tb.MH
TMIN, TMAX, NMD, DTLOG = tb.TMIN, tb.TMAX, tb.NMD, tb.DTLOG
N_DUST_ITER = 100                # at most this many Newton iterations (Fortran: up to 20000)
DUST_EMISSION = 4.68e-31         # dust cooling / (n dgr T_d^6) [erg s^-1 K^-6] (cool_util.F:891)
ISRF_DUST_HEATING = 5.8e-24      # ISRF dust heating / (n dgr G0/1.7) [erg s^-1] (cool_util.F:859)


def _arrays(d):
    return {k: (jnp.asarray(v), jnp.asarray(s)) for k, (v, s) in d.items()}


_CACHE = {}


def _get(name, build):
    def cached():
        if name not in _CACHE:
            with jax.ensure_compile_time_eval():  # a concrete array, even if first built under jit
                _CACHE[name] = build()
        return _CACHE[name]
    return tb.device_table(name, cached)


def cool_rows():
    return _get("cl", lambda: _arrays(tb.cool_table()))


def chem_rows():
    return _get("ch", lambda: _arrays(tb.chem_table()))


def cie_rows():
    return _get("cie", lambda: _arrays(tb.cool_table_cie()))


def wss_table():
    return _get("wss", lambda: jnp.asarray(tb.wss_cie_table()))


def he_cie_tables():
    return _get("he", lambda: (jnp.asarray(tb.cie_raw()["hei_tbl"]),
                               jnp.asarray(tb.cie_raw()["heii_tbl"])))


def chem17_rows():
    return _get("ch17", lambda: _arrays(tb.chem_table17()))


def chem17_fixed_rows():
    return _get("ch17fix", lambda: _arrays(tb.chem_table17_fixed()))


def hd_rows():
    return _get("hd", lambda: _arrays(tb.cool_table_hd()))


def _node_index(temp, clamp_top):
    """Table bin and T - T_i: clamp_top (rate_eq) holds the last node from 1e8 K, else (cool_func) extrapolates."""
    temps = _get("temptab", lambda: jnp.asarray(tb.temptab()))
    t = jnp.clip(temp, TMIN, TMAX)
    i = jnp.clip(jnp.floor(jnp.log10(t) / DTLOG).astype(jnp.int32), 0, NMD - 1)
    dtemp = temp - temps[i]
    dtemp = jnp.where(temp <= TMIN, 0.0, dtemp)
    if clamp_top:
        dtemp = jnp.where(temp >= TMAX, 0.0, dtemp)
        i = jnp.where(temp >= TMAX, NMD - 1, i)
    else:
        i = jnp.where(temp > TMAX, NMD - 1, i)
        dtemp = jnp.where(temp > TMAX, temp - temps[NMD - 1], dtemp)
    return i, dtemp


def interp_rows(rows, temp, clamp_top=True):
    """Every table row at temperature `temp`: value_i + (T - T_i) * slope_i."""
    i, dtemp = _node_index(temp, clamp_top)
    return {k: v[i] + dtemp * s[i] for k, (v, s) in rows.items()}


# --------------------------------------------------------------------------
# dust temperature (cool_util.F calc_dust_temp)
# --------------------------------------------------------------------------
def dust_temperature(n_h, temp, x_hi, chi, g0_dust, dgr, tdust_in,
                     h2_form_ex=0.84, h2_form_kin=0.12, rtol=1e-2, iters=None):
    """(R_H2, R_dust, T_dust, converged) by Newton from tdust_in; iters: that many steps, no loop."""
    ch7 = interp_rows({7: chem_rows()[7]}, temp)[7] * dgr   # HM79, per unit dgr in tables
    cl14 = interp_rows({14: cool_rows()[14]}, temp)[14]
    active = (dgr > 1e-8) & (temp < 1e5)
    isrf_heat = ISRF_DUST_HEATING * n_h * (g0_dust * chi / tb.f32(1.7)) * dgr
    temp2 = temp * 1e-2
    h2_form_dust = 1.0 - h2_form_ex - h2_form_kin

    def rates(td):
        tdust2 = td * 1e-2
        fa = 1.0 / (1.0 + 1e4 * jnp.exp(-6e2 / td))
        stick = 1.0 / (1.0 + 0.4 * (temp2 + tdust2) ** 0.5 + 0.2 * temp2 + 0.08 * temp2**2.0)
        return fa, stick, ch7 * fa * stick

    def step(td):
        fa, stick, rdust = rates(td)
        dust_cool = DUST_EMISSION * td**6 * n_h * dgr
        gas_dust = cl14 * (temp - td) * n_h**2 * dgr
        dfa = -fa * (1 - fa) * (6e2 / td**2)
        rdust_rat = dfa / fa - 2e-2 * stick / jnp.sqrt(td + temp)
        dust_h2_heat = rdust * x_hi * n_h**2 * h2_form_dust * 4.48 * EV
        f_nr = isrf_heat - dust_cool + gas_dust + jnp.where(rdust == 0.0, 0.0, dust_h2_heat)
        df_nr = (-(6.0 * dust_cool / td) - cl14 * n_h**2
                 + jnp.where(rdust == 0.0, 0.0, dust_h2_heat * rdust_rat))
        new = td - f_nr / df_nr
        new = jnp.where(new != new, TMIN, new)
        return jnp.clip(new, TMIN, 1.5e3)

    def newton(carry):
        td, done = carry
        new = step(td)
        # the Fortran exits with tdust = the iterate whose Newton step was
        # below 1 %, not the step's result (cool_util.F:884-974)
        stop = done | (jnp.abs((new - td) / td) < rtol)
        return jnp.where(stop, td, new), stop

    # a while loop that ends once every cell has stopped: the same iterates as
    # N_DUST_ITER masked ones, forward-mode differentiable (jacfwd), not reverse
    td0 = jnp.where(active, tdust_in, 10.0)
    done0 = td0 * 0.0 > 1.0           # from td0, so it varies like td0 under shard_map
    if iters is None:
        _, (td, done) = jax.lax.while_loop(
            lambda c: (c[0] < N_DUST_ITER) & ~jnp.all(c[1][1]),
            lambda c: (c[0] + 1, newton(c[1])), (0, (td0, done0)))
    else:                             # straight-line: both AD modes, and inside a kernel
        td, done = td0, done0
        for _ in range(iters):
            td, done = newton((td, done))
    td = _polish(td, rtol, step)
    return _finish(td, done, active, temp, x_hi, n_h, rates)


def _polish(td, rtol, step):
    """Converged mode (rtol < 1 %): one more Newton step, so T_d carries its implicit derivative."""
    return jnp.where(rtol < 1e-2, step(td), td)


def _finish(td, done, active, temp, x_hi, n_h, rates):
    _, _, rdust = rates(td)
    rh2 = rdust * x_hi * n_h**2
    tdust = jnp.where(active, td, jnp.minimum(temp, 1.5e3))
    return (jnp.where(active, rh2, 0.0), jnp.where(active, rdust, 0.0), tdust,
            done | ~active)


# --------------------------------------------------------------------------
# CO cooling tables (cool_util.F co_cool) and H2 opacity
# --------------------------------------------------------------------------
def co_arrays():
    return _get("co", lambda: {k: jnp.asarray(v) for k, v in tb.co_tables().items()})


def co_cool(temp, n_rot, n_vib):
    """CO cooling (L0, L_LTE, alpha, n_1/2, L_vib) of NK93/NLM95 at T and log10 column (cool_util.F)."""
    c = co_arrays()
    # rotational: 1 K bins from 5 K, 0.1 dex from log N = 14.5
    ti = jnp.floor(jnp.clip(temp, 5.0, 2000.0))
    it = jnp.where(temp >= 2000.0, 1995, ti.astype(jnp.int32) - 5)
    dt = jnp.where((temp >= 2000.0) | (temp == 5.0), 0.0, temp - ti)
    lo, hi = c["n_rot"][0], c["n_rot"][-1]
    iN = jnp.floor(10 * jnp.clip(n_rot, lo, hi) - 144).astype(jnp.int32) - 1
    iN = jnp.where(n_rot <= lo, 0, jnp.where(n_rot >= hi, 45, iN))
    dN = jnp.where((n_rot <= lo) | (n_rot >= hi), 0.0, (n_rot - c["n_rot"][iN]) / 0.1)
    iN2 = jnp.minimum(iN + 1, 45)

    def at(tab, dtab, j):
        return tab[j, it] + dt * dtab[j, it]

    l0 = c["L0"][it] + dt * c["dL0"][it]
    out = []
    for k in ("lte", "alp", "n05"):
        v1 = at(c[k], c["d" + k], iN)
        v2 = at(c[k], c["d" + k], iN2)
        out.append(jnp.where(iN == 45, v1, v1 + (v2 - v1) * dN))
    lte, alp, n05 = out
    below = temp < 5.0
    l0 = jnp.where(below, 0.0, 10.0 ** (-l0))
    lte = jnp.where(below, 0.0, 10.0 ** (-lte))
    n05 = jnp.where(below, 1.0, 10.0**n05)
    alp = jnp.where(below, 1.0, alp)
    # vibrational: 1 K bins from 100 K, 0.1 dex from log N = 13.0
    tv = jnp.floor(jnp.clip(temp, 100.0, 4000.0))
    itv = jnp.where(temp >= 4000.0, 3900, tv.astype(jnp.int32) - 100)
    itv = jnp.where(temp <= 100.0, 0, itv)
    dtv = jnp.where((temp <= 100.0) | (temp >= 4000.0), 0.0, temp - tv)
    lo, hi = c["n_vib"][0], c["n_vib"][-1]
    jN = jnp.floor(10 * jnp.clip(n_rot, lo, hi) - 129).astype(jnp.int32) - 1
    jN = jnp.where(n_vib <= lo, 0, jnp.where(n_vib >= hi, 60, jN))
    dNv = jnp.where((n_vib <= lo) | (n_vib >= hi), 0.0, (n_vib - c["n_vib"][jN]) / 0.1)
    v1 = c["vib"][jN, itv] + dtv * c["dvib"][jN, itv]
    v2 = c["vib"][jnp.minimum(jN + 1, 60), itv] + dtv * c["dvib"][jnp.minimum(jN + 1, 60), itv]
    vib = jnp.where(jN == 60, v1, v1 + (v2 - v1) * dNv)
    vib = jnp.where(below, 0.0, 10.0 ** (-vib) * jnp.exp(-3.08e3 / temp))
    return l0, lte, alp, n05, vib


def h2_opacity(temp, n_h2_eff):
    """Optically thick H2 line cooling correction (Sobolev, YOHA06), cool_util.F:506."""
    r = tb.raw()
    lt, lc, op = (jnp.asarray(r["h2_opac_temp"]), jnp.asarray(r["h2_opac_column"]),
                  jnp.asarray(r["h2_opac"]))
    cmin, cmax = 10 ** float(r["h2_opac_column"][0]), 10 ** float(r["h2_opac_column"][-1])
    logn = jnp.log10(jnp.clip(n_h2_eff, cmin, cmax))
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
    o1 = jnp.where(dT > 0, op[ii, jj] + dT * (op[i2, jj] - op[ii, jj]), op[ii, jj])
    o2 = jnp.where(dT > 0, op[ii, jj + 1] + dT * (op[i2, jj + 1] - op[ii, jj + 1]),
                   op[ii, jj + 1])
    opac = 10.0 ** (o1 + dn * (o2 - o1))
    return jnp.where(n_h2_eff <= cmin, 1.0, jnp.where(n_h2_eff >= cmax, 0.0, opac))


# --------------------------------------------------------------------------
# fine-structure helpers (cool_util.F)
# --------------------------------------------------------------------------
def _stim(a10, e10, t_rad):
    """compute_stim: A10 / (exp(E/kT_rad) - 1), zero for E/kT_rad >= 5."""
    x = e10 / (KBOLTZ * t_rad)
    return jnp.where(x < 5.0, a10 / (jnp.exp(jnp.minimum(x, 5.0)) - 1.0), 0.0)


def _three_level(r01, r02, r12, r10, r20, r21):
    """three_level_pops: statistical equilibrium of a 3-level atom."""
    a1, a2, a3 = r01 + r02, -r10, -r20
    b1, b2, b3 = r01, -(r10 + r12), r21
    den = (a1 - a2) * (a1 * b3 - b1 * a3) - (a1 - a3) * (a1 * b2 - b1 * a2)
    zero = (r01 == 0.0) & (r02 == 0.0)
    den = jnp.where(zero, 1.0, den)
    n2 = -a1 * (a1 * b2 - b1 * a2) / den
    a12 = jnp.where(zero, 1.0, a1 - a2)
    n1 = (a1 / a12) - ((a1 - a3) / a12) * n2
    n0 = 1.0 - n1 - n2
    return (jnp.where(zero, 1.0, n0), jnp.where(zero, 0.0, n1), jnp.where(zero, 0.0, n2))


def _safe_div(a, b):
    return a / jnp.where(b == 0.0, 1.0, b)


# --------------------------------------------------------------------------
# WSS_CIE_COOL (wss_z_collis.F, cool_func.F:391): hot-gas metals, He in CIE
# --------------------------------------------------------------------------
def wss_cie_elements(temp):
    """get_coolrates_wssCIE per element (C N O Ne Mg Si S Ca Fe) at solar [erg cm^3 s^-1]."""
    nlt = tb.WSS_NODES - 1
    idx_real = nlt * (jnp.log10(temp) - 2.0) / tb.WSS_LOGT_RANGE + 1.0
    below, above = idx_real < 1.0, idx_real >= nlt
    i = jnp.clip(jnp.floor(idx_real), 1, nlt - 1)
    # clamped at both ends; above the second-to-last node the last interval is unused
    x = jnp.where(below, 0.0, jnp.where(above, 1.0, idx_real - i))[..., None]
    j = i.astype(jnp.int32) - 1
    tab = wss_table()
    return 10.0 ** ((1.0 - x) * tab[j] + x * tab[j + 1])


def wss_cie_cooling(temp, metallicity):
    """WSS metal cooling per n_H^2 [erg cm^3 s^-1] for the solar pattern scaled by metallicity."""
    r = wss_cie_elements(temp) * jnp.asarray(metallicity)[..., None]
    s = r[..., 0]
    for k in range(1, r.shape[-1]):         # C + N + ... + Fe, the Fortran's order
        s = s + r[..., k]
    return s


def he_cie_fractions(temp, abhe=tb.ABHE):
    """He I, II, III per H in CIE (cool_func.F:391): all He I to 1e4 K, all He III from 1e7 K."""
    hei, heii = he_cie_tables()
    idx_real = 50.0 * (jnp.log10(temp) - 4.0) / 3.0 + 1.0
    i = jnp.clip(jnp.floor(idx_real), 1, 50)
    w = idx_real - i
    j = i.astype(jnp.int32) - 1
    x1 = abhe * ((1.0 - w) * hei[j] + w * hei[j + 1])
    x2 = abhe * ((1.0 - w) * heii[j] + w * heii[j + 1])
    x3 = abhe - x1 - x2
    cold, hot = temp <= 1e4, temp >= 1e7
    return (jnp.where(cold, abhe, jnp.where(hot, 0.0, x1)),
            jnp.where(cold | hot, 0.0, x2),
            jnp.where(cold, 0.0, jnp.where(hot, abhe, x3)))


# --------------------------------------------------------------------------
# cool_func.F
# --------------------------------------------------------------------------
def cool_func(temp, n_h, dl, divv, g_dust, ab, tdust, abundc, abundo, abundsi, dgr, zeta,
              redshift=0.0, cie_cool=False, metallicity=1.0, lvg=True):
    """The 28 cool_func.F terms (..., 28) and T_dust; lvg=False: divv is 0, one CO lookup for all isotopologues."""
    rows = cool_rows()
    if cie_cool:                             # He in CIE replaces any He abundances given
        rows = {**rows, **cie_rows()}
        ab = dict(ab, **dict(zip(("hei", "heii", "heiii"), he_cie_fractions(temp))))
    c = interp_rows(rows, temp, clamp_top=False)
    t_cmb = 2.726 * (1.0 + redshift)
    ynh2, ynh = ab["h2"] * n_h, ab["hi"] * n_h
    yne, ynhp = ab["e"] * n_h, ab["hp"] * n_h
    zero = jnp.zeros_like(temp)
    r = {}

    # (1) gas-grain energy transfer (HM89); dense limit: dust emission
    dense = n_h * dgr > 1e10
    tdust_hi = jnp.minimum(temp, 1.5e3)
    r[1] = jnp.where(dense, DUST_EMISSION * tdust_hi**6 * n_h * dgr,
                     c[14] * (temp - tdust) * n_h**2 * dgr)
    tdust = jnp.where(dense, tdust_hi, tdust)

    # (2) H2 rovibrational (LPF99 / GP98 form): n-dependent interpolation
    # between the low-density rate h2var0 and the LTE rate cl1
    if "hei" in ab:                          # He as a collider (cl64)
        ynhe = ab["hei"] * n_h
        h2var0 = ynh * c[4] + ynh2 * c[5] + ynhe * c[64] + ynhp * c[65] + yne * c[66]
    else:
        ynhe = None
        h2var0 = ynh * c[4] + ynh2 * c[5] + ynhp * c[65] + yne * c[66]
    lte_lim = ynh2 * c[1] / (1.0 + c[1] / jnp.where(h2var0 == 0.0, 1.0, h2var0))
    r2 = jnp.where(h2var0 < 1e-4 * c[1], h2var0 * ynh2, lte_lim)
    # optically thick above n = 1e9 cm^-3: H2 column in a Jeans length or LVG
    vth = jnp.sqrt(KBOLTZ * temp / MH)
    n_jeans = ynh2 / (vth / dl)
    n_lvg = jnp.where(divv == 0.0, jnp.inf, ynh2 / jnp.where(divv == 0.0, 1.0, jnp.abs(divv)))
    opac = h2_opacity(temp, jnp.minimum(n_lvg, n_jeans))
    r2 = jnp.where(n_h < 1e9, r2, r2 * opac)
    r[2] = jnp.where(ynh2 == 0.0, 0.0, r2)

    # (25) Lyman alpha; (28) bremsstrahlung is in the zeroed atomic table
    r[25] = c[52] * ynh * yne

    # (7, 21, 22) CO rotational, (8, 9, 24) CO vibrational (NK93, NLM95)
    r.update(_co_terms(c, temp, n_h, ab, divv, abundo, ynh, ynh2, ynhe, lvg))

    # (11) cosmic-ray heating: 20 eV per ionisation (Goldsmith & Langer 1978)
    r[11] = -3.2e-11 * (ynh2 + ynh) * zeta

    # (12) photoelectric heating (Bakes & Tielens 1994, Wolfire+ 2003)
    pe_on = (g_dust != 0.0) & (dgr >= 1e-8) & (yne >= 1e-20 * g_dust * c[16])
    # 1 where off: the 0.73 power has an infinite slope at 0 (a NaN reverse-mode gradient)
    pevar0 = jnp.where(pe_on, _safe_div(g_dust * c[16], jnp.where(pe_on, yne, 1.0)), 1.0)
    eps = 4.9e-2 / (1.0 + 4e-3 * pevar0 ** tb.f32(0.73)) + c[17] / (1.0 + 2e-4 * pevar0)
    r[12] = jnp.where(pe_on, -1.3e-24 * eps * g_dust * n_h * dgr, 0.0)

    # (13) OI fine structure, three-level atom with CMB pumping
    r[13] = _oi(c, temp, t_cmb, ynh, ynh2, yne, ynhp, ab["o"], abundo, n_h)

    # (16) CII 158 micron, two-level atom with LVG / Jeans escape probability
    r[16] = _cii(c, temp, t_cmb, n_h, ynh, ynh2, yne, ab["cp"], abundc, dl, divv)

    # (17) SiII 35 micron, two-level atom
    c10 = c[49] * ynh + c[50] * yne
    b10 = _stim(tb.SIII_A10, tb.SIII_E10, t_cmb)
    var0, var1 = c10 + tb.SIII_A10 + b10, c[51] * c10 + 2.0 * b10
    n0, n1 = var0 / (var0 + var1), var1 / (var0 + var1)
    si_on = (c10 != 0.0) & (ab["si"] > 1e-5 * abundsi)
    r[17] = jnp.where(si_on, ((tb.SIII_A10 + b10) * n1 - 2.0 * b10 * n0)
                      * tb.SIII_E10 * ab["si"] * n_h, 0.0)

    # (18) Compton cooling off the CMB
    r[18] = 1.017e-37 * t_cmb**4 * (temp - t_cmb) * yne

    # (20) H2 collision-induced emission (RA04), CIE opacity cut (M. Turk)
    tau = jnp.maximum((n_h / 1.4e16) ** 2.8, 1e-5)
    r[20] = c[58] * n_h * (2.0 * ynh2) * jnp.minimum((1.0 - jnp.exp(-tau)) / tau, 1.0)

    # (19) HD rotational lines (LNA05 fit in log n_H; collisions with H)
    if "hd" in ab:
        r[19] = _hd_cooling(temp, n_h, ynh, ab["hd"])

    if cie_cool:
        r.update(_cie_terms(c, temp, n_h, ab, yne, ynhe, metallicity))

    rates = jnp.stack([r.get(k, zero) * jnp.ones_like(temp) for k in range(1, 29)], axis=-1)
    return rates, tdust


def _cie_terms(c, temp, n_h, ab, yne, ynhe, metallicity):
    """WSS_CIE_COOL terms: (3) WSS metals, (26) HeI, (27) HeII excitation, (28) free-free."""
    xhe2, xhe3 = ab["heii"], ab["heiii"]
    out = {3: wss_cie_cooling(temp, metallicity) * n_h**2,
           26: c[59] * ynhe * yne + c[60] * xhe2 * n_h * yne**2,
           27: c[61] * xhe2 * n_h * yne}
    ff = c[62] * (ab["hp"] + xhe2) * n_h * yne + c[63] * xhe3 * n_h * yne
    out[28] = jnp.where(temp < 100.0, 0.0, ff)
    return out


def _hd_cooling(temp, n_h, ynh, x_hd):
    """HD line cooling, cool_func.F (R19): LNA05 fit in x = log10 n_HI."""
    c = interp_rows(hd_rows(), temp, clamp_top=False)
    dense, thin = n_h >= 1e8, ynh < 1.0
    lnh = jnp.where(dense, 8.0, jnp.where(thin, 0.0, jnp.log10(jnp.where(thin, 1.0, ynh))))
    scale = jnp.where(~dense & thin, ynh, 1.0)
    rate = 10.0 ** (c[53] + lnh * (c[54] + lnh * (c[55] + lnh * (c[56] + lnh * c[57]))))
    return jnp.where(x_hd > 0.0, rate * (x_hd * n_h) * scale, 0.0)


def _co_terms(c, temp, n_h, ab, divv, abundo, ynh, ynh2, ynhe=None, lvg_on=True):
    co_on = ab["co"] > 1e-5 * abundo
    colmax_rot, colmax_vib = 19.0, 19.0          # co_colntab(46), co_vib_colntab(61)
    dv = 1e-5 * jnp.abs(jnp.where(divv == 0.0, 1.0, divv))
    n_co = jnp.log10(jnp.where(co_on, ab["co"], 1.0) * n_h / dv)
    n13, n18 = n_co - 2.0, n_co - 2.69897
    lvg = divv != 0.0

    def clip3(a, b, m):
        # cool_func.F:475: clip the most abundant isotopes first
        a2 = jnp.where((n18 > m) | (b > m) | (a > m), m, a)
        b2 = jnp.where((n18 > m) | (b > m), m, b)
        c2 = jnp.where(n18 > m, m, n18)
        return a2, b2, c2

    rot = [jnp.where(lvg, v, colmax_rot) for v in clip3(n_co, n13, colmax_rot)]
    vib = [jnp.where(lvg, v, colmax_vib) for v in clip3(n_co, n13, colmax_vib)]
    t_co = jnp.maximum(temp, 5.0)
    thin = rot[0] <= 14.5                         # co_colntab(1)
    p_main = co_cool(t_co, rot[0], vib[0])
    if lvg_on:
        p13 = [jnp.where(thin, a, b) for a, b in zip(p_main, co_cool(t_co, rot[1], vib[1]))]
        p18 = [jnp.where(thin, a, b) for a, b in zip(p_main, co_cool(t_co, rot[2], vib[2]))]
    else:                                         # every column is colmax: the same lookup
        p13 = p18 = p_main

    out = {}
    neff = ynh2 + jnp.sqrt(2.0) * ynh
    if ynhe is not None:                     # He as a collider
        neff = neff + 0.5 * ynhe
    rot_on = co_on & (neff != 0.0)
    low_t = jnp.where(temp <= 5.0, jnp.exp(-1e1 / temp) / jnp.exp(-1e1 / 5.0), 1.0)
    for k, frac, p in ((7, 1.0, p_main), (21, 1e-2, p13), (22, 2e-3, p18)):
        l0, lte, alp, n05, _ = p
        l0s, ltes = jnp.where(rot_on, l0, 1.0), jnp.where(rot_on, lte, 1.0)
        inv = (1.0 / l0s + neff / ltes
               + (1.0 / l0s) * (1.0 - n05 * l0s / ltes) * (neff / n05) ** alp)
        val = frac * ab["co"] * neff * n_h / inv
        out[k] = jnp.where(rot_on, val * low_t, 0.0)
    neff = ynh2 + ynh
    vib_on = co_on & (neff != 0.0) & (temp >= 1e2)
    l0_vib = c[12] * ab["h2"] + c[13] * ab["hi"]
    for k, frac, p in ((8, 1.0, p_main), (9, 1e-2, p13), (24, 2e-3, p18)):
        lte = jnp.where(vib_on, p[4], 1.0)
        inv = 1.0 / jnp.where(vib_on, l0_vib, 1.0) + n_h / lte
        out[k] = jnp.where(vib_on, frac * ab["co"] * n_h * neff * (1.0 / inv), 0.0)
    return out


def _oi(c, temp, t_cmb, ynh, ynh2, yne, ynhp, x_o, abundo, n_h):
    c10 = c[18] * ynh + c[21] * ynh2 + c[24] * yne + c[27] * ynhp
    c20 = c[19] * ynh + c[22] * ynh2 + c[25] * yne + c[28] * ynhp
    c21 = c[20] * ynh + c[23] * ynh2 + c[26] * yne + c[29] * ynhp
    e10, e20, e21 = tb.OX_E10, tb.OX_E20, tb.OX_E21
    a10, a20, a21 = tb.OX_A10, tb.OX_A20, tb.OX_A21
    oa = jnp.exp(-e10 / (KBOLTZ * temp))
    ob = jnp.exp(-e20 / (KBOLTZ * temp))
    oc = jnp.exp(-e21 / (KBOLTZ * temp))
    c01, c02, c12 = 0.6 * c10 * oa, 0.2 * c20 * ob, (1.0 / 3.0) * c21 * oc
    b10, b20, b21 = _stim(a10, e10, t_cmb), _stim(a20, e20, t_cmb), _stim(a21, e21, t_cmb)
    b01, b02, b12 = 0.6 * b10, 0.2 * b20, (1.0 / 3.0) * b21
    r01, r02, r12 = c01 + b01, c02 + b02, c12 + b12
    r10, r20, r21 = c10 + a10 + b10, c20 + a20 + b20, c21 + a21 + b21
    n0, n1, n2 = _three_level(r01, r02, r12, r10, r20, r21)
    tiny = r01 < 1e-40
    n0, n1, n2 = jnp.where(tiny, 1.0, n0), jnp.where(tiny, 0.0, n1), jnp.where(tiny, 0.0, n2)
    lam1 = (a10 + b10) * e10 * n1 + ((a20 + b20) * e20 + (a21 + b21) * e21) * n2
    lam2 = (b01 * e10 + b02 * e20) * n0 + b12 * e21 * n1
    on = (x_o > 1e-5 * abundo) & ~((c10 == 0.0) & (c20 == 0.0) & (c21 == 0.0))
    return jnp.where(on, (lam1 - lam2) * x_o * n_h, 0.0)


def _cii(c, temp, t_cmb, n_h, ynh, ynh2, yne, x_cp, abundc, dl, divv):
    c10 = c[45] * ynh + c[46] * ynh2 + c[47] * yne
    c01 = c[48] * c10
    a10, e10 = tb.CII_A10, tb.CII_E10
    b10 = _stim(a10, e10, t_cmb)
    b01 = 2.0 * b10
    vth = jnp.sqrt(KBOLTZ * temp / (6.0 * MH))
    col_jeans = (n_h * x_cp) / (vth / dl)
    col_lvg = jnp.where(divv == 0.0, jnp.inf,
                        (n_h * x_cp) / jnp.abs(jnp.where(divv == 0.0, 1.0, divv)))
    tau = 7.2e-13 * jnp.minimum(col_jeans, col_lvg)
    a10_eff = a10 * (1.0 / (1.0 + 3.0 * tau))
    var0, var1 = c10 + a10_eff + b10, c01 + b01
    n0, n1 = var0 / (var0 + var1), var1 / (var0 + var1)
    on = (c10 != 0.0) & (x_cp > 1e-5 * abundc)
    return jnp.where(on, ((a10_eff + b10) * e10 * n1 - b01 * e10 * n0) * x_cp * n_h, 0.0)
