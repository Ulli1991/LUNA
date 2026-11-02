"""Network 17: primordial H/He chemistry, NL97 CO and D+/HD, with the gas energy, as one RHS.

Port of rate_eq_ulli.F (S. Glover's chemcool; D+/HD channels added by U. Steinwandel).
State y = (x_H2, x_H+, x_CO, x_He+, x_He++, x_D+, x_HD, e); see docs/CHEMISTRY.md.
"""
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

from luna.chemistry import cooling, photo, rates_net5, thermo
from luna.chemistry import tables as tb

EV, KBOLTZ, PI, GRAVITY = tb.EV, tb.KBOLTZ, tb.PI, tb.GRAVITY
ABHE = tb.ABHE_NET17
MH = 1.6726e-24                    # rate_eq_ulli's own (double) proton mass
CST1, CST75 = 6.4e-10, 7.2e-15     # H2+ + H -> H2 + H+; He+ + H2 -> He + H2+ (const_rates.F)
CR_HE = 1.09                       # He cosmic-ray ionisation / H (cheminmo.F crtab(6))
K0_NL97, K1_NL97 = rates_net5.K0_NL97, rates_net5.K1_NL97
EPS_MAX = rates_net5.EPS_MAX
N_EQB_ITER, EQB_TOL = 25, 1e-3     # rate_eq_ulli's niter, eqtol
PH4_THIN, PH5_THIN, PH35_THIN = 4.2e-7, 6.5e-10, 3.3e-11   # photoinit_ism.F, per G0
MSM_SHIFT = (tb.f32(-1.164408), tb.f32(-1.174242))       # REAL*4 n_crit offsets, CID and DT
UV_PUMPS = 6.94                    # UV pumps per H2 photodissociation
NH_LOCAL = 0.5                     # rate_eq_ulli's NH = 0.5 dl n_H (H2 self-shielding)
NH_LOCAL_FIXED = 1.0               # the fixed Fortran's NH = dl n_H
DUST_EXP_HM, DUST_EXP_H2P = 0.5, 1.9   # df05, df19: dust attenuation exp(-c A_V) of ph4, ph5 (fixed)
NSPEC = 8
FIXES17 = ("chemcool", "fixed")


class Params(NamedTuple):
    """Everything the RHS needs besides the state; columns have shape (..., n_pix)."""
    n_h: float                 # H nuclei [cm^-3]
    n_h_col: jnp.ndarray       # TreeCol total H column per pixel [cm^-2]
    n_h2_col: jnp.ndarray      # TreeCol H2 column per pixel
    n_co_col: jnp.ndarray      # TreeCol CO column per pixel
    g0: float                  # FUV field, Habing units
    g0_dust: float             # field heating the dust (G0_dust)
    zeta: float                # cosmic-ray ionisation rate of H [s^-1]
    dgr: float                 # dust-to-gas ratio / solar
    abundc: float              # total C per H
    abundo: float              # total O per H
    abund_d: float = 2.6e-5    # total D per H (0 switches the D chemistry off)
    dl: float = 0.0            # local shielding length [cm]
    divv: float = 0.0          # velocity divergence [s^-1]
    tdust: float = 10.0        # dust temperature guess (the Fortran carries it)
    h2_form_kin: float = 0.12  # fraction of the 4.48 eV H2 binding energy heating gas
    h2_form_ex: float = 0.84   # fraction into H2 excitation
    dust_rtol: float = 1e-2    # dust Newton stops at the first step below this (Fortran: 1 %)
    photo_temp: Optional[float] = None   # T, x_H2, x_HD at which calc_photo ran
    photo_x_h2: Optional[float] = None   # (the step start in GIZMO); None: the
    photo_x_hd: Optional[float] = None   # current state, as photo_state(y) gives


def make_params(n_h, g0, n_h_col=0.0, n_h2_col=0.0, n_co_col=0.0, n_pix=12, zeta=3e-17,
                z=1.0, **kw):
    """Params at metallicity z (C, O and dust-to-gas ratio scale with it), G0_dust = G0."""
    col = lambda c: jnp.broadcast_to(jnp.asarray(c, jnp.float64)[..., None],  # noqa: E731
                                     jnp.shape(c) + (n_pix,))
    kw.setdefault("abundc", 1.41e-4 * z)
    kw.setdefault("abundo", 3.16e-4 * z)
    kw.setdefault("dgr", z)
    return Params(n_h=n_h, n_h_col=col(n_h_col), n_h2_col=col(n_h2_col),
                  n_co_col=col(n_co_col), g0=g0, g0_dust=kw.pop("g0_dust", g0), zeta=zeta, **kw)


def _eos(fixes17, eos):
    """The EOS of a fixes17 mode: the table for "chemcool", the Newton solve for "fixed" (fix 5)."""
    if fixes17 not in FIXES17:
        raise ValueError(f"fixes17={fixes17!r}")
    return ("table" if fixes17 == "chemcool" else "newton") if eos is None else eos


def photo_state(y, p: Params, fixes17="chemcool", eos=None):
    """(T, x_H2, x_HD) for calc_photo as chemcool.cc computes them (electrons = H+ only)."""
    y = jnp.asarray(y, jnp.float64)
    ekn = y[..., 7] / (KBOLTZ * (1.0 + ABHE - y[..., 0] + y[..., 1]) * p.n_h)
    temp, _ = thermo.temperature_gamma(y[..., 0], ekn, ABHE, _eos(fixes17, eos))
    return temp, y[..., 0], y[..., 6]


def chem_coefficients(temp, fixes17="chemcool"):
    """The rate coefficients network 17 reads, linear in T between table nodes."""
    rows = {**cooling.chem_rows(), **cooling.chem17_rows()}
    if fixes17 == "fixed":
        rows.update(cooling.chem17_fixed_rows())
    return cooling.interp_rows(rows, temp, clamp_top=True)


def photo_rates(p: Params, temp, x_h2, x_hd, fixes17="chemcool"):
    """calc_photo.F for network 17 (iphoto = 6): k_h2_dust, k_hm, k_h2p, k_co, k_hd, g_dust, chi."""
    g0, dgr = p.g0, p.dgr
    local = p.dl * p.n_h
    av = photo.AV_PER_NH * dgr[..., None] * p.n_h_col \
        + (photo.AV_PER_NH * dgr * local)[..., None]
    f_dust = photo._pixel_mean(jnp.exp(-(photo.SIGMA_DUST_LW * dgr)[..., None] * p.n_h_col))
    chi = photo._pixel_mean(photo.chi_isrf(av))
    df25 = photo._pixel_mean(jnp.exp(-2.5 * av))
    # dust-free gas takes the Fortran's other branch: A_V = 0, every factor 1;
    # with dust, chemcool's iphoto = 6 loop never accumulates df05 and df19 (stay 0)
    fixed = fixes17 == "fixed"
    if fixed:
        df05 = photo._pixel_mean(jnp.exp(-DUST_EXP_HM * av))
        df19 = photo._pixel_mean(jnp.exp(-DUST_EXP_H2P * av))
    else:
        df05 = df19 = jnp.zeros_like(df25)
    nodust = dgr == 0.0
    f_dust, chi, df25, df05, df19 = (jnp.where(nodust, 1.0, f)
                                     for f in (f_dust, chi, df25, df05, df19))
    shield_hd = jnp.where(p.abund_d == 0.0, 1.0, photo.hd_self_shielding(
        local * x_hd, p.n_h2_col, x_h2, x_hd, p.abund_d, temp, weight_both=fixed))
    on = g0 != 0.0
    rate = lambda r: jnp.where(on, r, 0.0)  # noqa: E731
    return {"k_h2_dust": rate(photo.K_H2_THIN * g0 * f_dust),
            "k_hm": rate(PH4_THIN * g0 * df05), "k_h2p": rate(PH5_THIN * g0 * df19),
            "k_co": rate(photo.K_CO_THIN * g0 * 1.0 * df25),
            "k_hd": rate(PH35_THIN * g0 * shield_hd * f_dust),
            "g_dust": rate(1.0 * g0 * df25), "chi": chi}


def _pow(b, a):
    """b**a for b >= 0 with a finite derivative at b = 0 (0**a = 0 for a > 0, 1 for a = 0)."""
    pos = b > 0.0
    return jnp.where(pos, jnp.where(pos, b, 1.0) ** a, jnp.where(a > 0.0, 0.0, 1.0))


def _comp_equilib(num, denom, max_ab):
    """comp_equilib (rate_eq_ulli.F) with the caller's clip at 0: num/denom, or 0."""
    q = num / jnp.where(denom == 0.0, 1.0, denom)
    bad = (denom == 0.0) | (q > max_ab)
    return jnp.where(num == 0.0, 0.0, jnp.where(bad, 0.0, jnp.maximum(q, 0.0)))


def equilibrium_species(ch, n, x_h2, x_hp, x_hep, x_e, x_hi, x_hei, ph4, ph5, zeta, h2p_dr,
                        h2p_cdh):
    """H-, H2+ and HeH+ in equilibrium by rate_eq_ulli.F's sweeps (at most 25).

    Returns hm, h2p, hehp, the corrected e, hi, hei, m (H2+ collider sum) and sweeps.
    """
    cr2 = 2.0 * zeta
    shape = jnp.shape(x_e)
    zero = jnp.zeros(shape)

    def sweep(c, i):
        hm0, h2p0, hehp0, e, hi, hei, m, done, nit = c
        num = ch[21] * hi * e + ch[126] * x_h2 * e
        den = (ch[22] * hi + ch[26] * e + ch[27] * hi + ch[28] * x_hp + ch[29] * x_hp + (ph4 / n)
               + ch[8] * x_hep + ch[31] * hei + 2.0 * ch[127] * h2p0)
        hm = _comp_equilib(num, den, 1.0)
        e1 = jnp.maximum(0.0, e - (hm - hm0))
        hi1 = jnp.maximum(0.0, hi - (hm - hm0))
        m1 = hi1 + x_h2 / jnp.sqrt(2.0) + hei / 2.0
        num = (ch[23] * hi1 * x_hp + ch[24] * x_h2 * x_hp + ch[29] * x_hp * hm
               + ch[156] * hehp0 * hi1 + (cr2 * x_h2 / n) + CST75 * x_hep * x_h2)
        den = (CST1 * hi1 + h2p_dr * e1 + (ph5 / n) + 2.0 * ch[127] * hm + ch[154] * hei
               + h2p_cdh * m1)
        h2p = _comp_equilib(num, den, 0.5)
        e1 = jnp.maximum(0.0, e1 + (h2p - h2p0))
        hi1 = jnp.maximum(0.0, hi1 - 2.0 * (h2p - h2p0))
        num = ch[155] * hei * x_hp + ch[154] * hei * h2p
        den = ch[156] * hi1 + ch[153] * e1
        hehp = _comp_equilib(num, den, ABHE)
        e1 = jnp.maximum(0.0, e1 + (hehp - hehp0))
        hi1 = jnp.maximum(0.0, hi1 - (hehp - hehp0))
        hei1 = jnp.maximum(0.0, hei - (hehp - hehp0))
        conv = ((i >= 1) & (jnp.abs(hm0 - hm) < EQB_TOL * hm)
                & (jnp.abs(h2p0 - h2p) < EQB_TOL * h2p) & (jnp.abs(hehp0 - hehp) < EQB_TOL * hehp))
        keep = lambda old, new: jnp.where(done, old, new)  # noqa: E731
        out = (keep(hm0, hm), keep(h2p0, h2p), keep(hehp0, hehp), keep(e, e1), keep(hi, hi1),
               keep(hei, hei1), keep(m, m1), done | conv, jnp.where(done, nit, i + 1))
        return out, None

    c0 = (zero, zero, zero, x_e + zero, x_hi + zero, x_hei + zero, zero,
          jnp.zeros(shape, bool), jnp.zeros(shape, jnp.int32))
    (hm, h2p, hehp, e, hi, hei, m, done, nit), _ = jax.lax.scan(
        sweep, c0, jnp.arange(N_EQB_ITER, dtype=jnp.int32))
    nit = jnp.where(done, nit, N_EQB_ITER + 1)
    return {"hm": hm, "h2p": h2p, "hehp": hehp, "e": e, "hi": hi, "hei": hei, "m": m,
            "sweeps": nit}


def _validate(y):
    """validate_input (validate_rates.F): clip to [0, 1] (H2 to 0.5); flag overshoots and NaN."""
    x = [y[..., k] for k in range(NSPEC)]
    bad = jnp.zeros(jnp.shape(x[0]), bool)
    for k in range(NSPEC):
        bad = bad | jnp.isnan(x[k])
        if k == 0:
            bad, x[k] = bad | (x[k] > 0.5 + EPS_MAX), jnp.clip(x[k], 0.0, 0.5)
        elif k < NSPEC - 1:
            bad, x[k] = bad | (x[k] > 1.0 + EPS_MAX), jnp.clip(x[k], 0.0, 1.0)
        else:
            x[k] = jnp.maximum(x[k], 0.0)
    return x, bad


def rhs(y, p: Params, fixes17="chemcool", eos=None):
    """dy/dt of network 17 (rate_eq_ulli.F) and diagnostics (ydot, aux); fixes17="fixed": GIZMO a58f5faf + Newton EOS."""
    eos = _eos(fixes17, eos)
    fixed = fixes17 == "fixed"
    y = jnp.asarray(y, jnp.float64)
    p = Params(*[None if v is None else jnp.asarray(v, jnp.float64) for v in p])
    (x_h2, x_hp, x_co, x_hep, x_hepp, x_dp, x_hd, energy), bad = _validate(y)
    n = p.n_h
    d_on = p.abund_d > 0.0
    x_dp = jnp.where(d_on, x_dp, 0.0)
    x_hd = jnp.where(d_on, x_hd, 0.0)
    x_di = jnp.where(d_on, jnp.maximum(0.0, p.abund_d - x_dp - x_hd), 0.0)
    x_cp = jnp.maximum(0.0, p.abundc - x_co)
    x_e = x_hp + x_hep + 2.0 * x_hepp + x_cp + x_dp
    x_hi = jnp.maximum(1.0 - 2.0 * x_h2 - x_hp, 0.0)
    x_hei = jnp.maximum(0.0, ABHE - x_hep - x_hepp)

    # temperature (calc_temp.F); T <= 0 is evaluated at 1 K and flagged
    yntot = (1.0 + ABHE + x_e - x_h2) * n
    bad = bad | ~(yntot > 0.0)
    ekn = energy / (KBOLTZ * jnp.where(yntot > 0.0, yntot, 1.0))
    temp, gamma = thermo.temperature_gamma(x_h2, ekn, ABHE, eos)
    rho = (1.0 + 4.0 * ABHE) * MH * n
    cs2 = jnp.maximum(gamma * (gamma - 1.0) * energy / rho, 0.0)
    l_jeans = jnp.sqrt(PI * cs2 / (GRAVITY * rho))
    cold = temp <= 0.0
    temp = jnp.where(cold, tb.TMIN, temp)
    ch = chem_coefficients(temp, fixes17)

    # H2 + H dissociation, collision-induced + dissociative tunnelling (MSM96)
    def msm(hi_n, lo_n, hi_t, lo_t, ncr, a, shift):
        u = n * x_hi / ch[ncr]
        u2 = n * x_hi / (10.0 ** shift * ch[ncr])
        return 10.0 ** ((ch[hi_n] - (ch[hi_n] - ch[lo_n]) / (1.0 + _pow(u, ch[a])))
                        + (ch[hi_t] - (ch[hi_t] - ch[lo_t]) / (1.0 + _pow(u2, ch[a]))))
    k_dh = (msm(161, 163, 162, 164, 165, 166, MSM_SHIFT[0])
            + msm(167, 169, 168, 170, 171, 172, MSM_SHIFT[1]))
    ncrinv = ch[5] + x_hei * ch[139]
    h2var0 = 1.0 / (1.0 + n * ncrinv)
    k_dh2 = ch[2] * ch[4] ** h2var0
    k_dhe = jnp.where(ch[138] < 1e-100, 1e-100,
                      ch[138] * _pow(ch[137] / jnp.where(ch[138] < 1e-100, 1.0, ch[138]), h2var0))
    # H2 + H+ -> H2+ + H, density-interpolated to the LTE rate
    c24 = jnp.where(jnp.abs(ch[24]) < 1e-100, 1e-100, ch[24])
    c173 = jnp.maximum(ch[173], 1e-100)
    k24 = c173 * _pow(c24 / c173, h2var0)
    # H2+ dissociative recombination and collisional dissociation, low n -> LTE
    ncrit_inv = jnp.where((x_hi > 0.0) | (x_e > 0.0), (x_hi / (4e2 * (1e4 / temp))) + (x_e / 5e1),
                          1.0 / 4e2)
    alpha = 1.0 / (1.0 + n * ncrit_inv)
    c157 = jnp.maximum(ch[157], 1e-100)
    h2p_dr = c157 * _pow(ch[30] / c157, alpha)
    c160 = jnp.maximum(ch[160], 1e-100)
    h2p_cdh = c160 * _pow(ch[159] / c160, alpha)

    # photo rates: fixed at the photo state; H2 self-shielding at the current state
    if p.photo_temp is None:
        t_ph, xh2_ph, xhd_ph = photo_state(y, p, fixes17, eos)
    else:
        t_ph, xh2_ph, xhd_ph = p.photo_temp, p.photo_x_h2, p.photo_x_hd
    ph = photo_rates(p, t_ph, xh2_ph, xhd_ph, fixes17)
    nh_local = NH_LOCAL * p.dl * n
    nh_shield = (NH_LOCAL_FIXED if fixed else NH_LOCAL) * p.dl * n
    k_dis = ph["k_h2_dust"] * photo.h2_self_shielding(
        (x_h2 * nh_shield)[..., None] + p.n_h2_col, temp)
    t_cmb = 2.726
    ph4 = (ph["k_hm"] + 0.11 * t_cmb**2.13 * jnp.exp(-8.823e3 / t_cmb)
           + 8.0e-8 * t_cmb**1.3 * jnp.exp(-2.3e3 / t_cmb))
    pd_low = ph["k_h2p"]
    pd_lte = 2.2e-11 * p.g0 * temp**-0.22
    pd_lte = jnp.where((temp < 3.15e3) & (pd_lte > pd_low), pd_low, pd_lte)
    pd_lte = jnp.maximum(pd_lte, 1e-100)
    ph5 = jnp.where(p.g0 == 0.0, 0.0, pd_lte * _pow(pd_low / pd_lte, alpha))
    g_dust = ph["g_dust"]
    gamma_co = ph["k_co"]
    gamma_chx = photo.K_CHX * g_dust

    h_gr = rates_net5.grain_recombination_hp(g_dust, temp, n, x_e, p.dgr, ch)
    rh2, rdust, tdust, dust_ok = cooling.dust_temperature(
        n, temp, x_hi, ph["chi"], p.g0_dust, p.dgr, p.tdust, p.h2_form_ex, p.h2_form_kin,
        p.dust_rtol)
    cr1, cr2, cr6 = p.zeta, 2.0 * p.zeta, CR_HE * p.zeta
    c18 = jnp.where(2.0 * nh_local * x_hi < 1.6e17, 0.0, ch[18])

    ch = {**ch, 24: k24}                     # the Fortran overwrites ch24
    eq = equilibrium_species(ch, n, x_h2, x_hp, x_hep, x_e, x_hi, x_hei, ph4, ph5, p.zeta,
                             h2p_dr, h2p_cdh)
    hm, h2p, e, hi, hei, m = eq["hm"], eq["h2p"], eq["e"], eq["hi"], eq["hei"], eq["m"]

    ynsq = n * n
    dn = lambda terms: sum(jnp.abs(t) for t in terms)  # noqa: E731
    # H+
    hp_in = [ch[11] * e * hi, ch[144] * hi * hi, ch[158] * hei * hi, CST1 * h2p * hi,
             -ch[13] * x_hp * e, -h_gr * x_hp * 1.0, -ch[23] * hi * x_hp, -k24 * x_h2 * x_hp,
             -ch[28] * x_hp * hm, -ch[29] * x_hp * hm, c18 * x_hep * e, ch[32] * x_hep * hi,
             -ch[33] * hei * x_hp, h2p_cdh * h2p * m, ch[9] * x_hep * x_h2]
    hp_d = [ch[99] * x_dp * hi, -ch[98] * x_hp * x_di, ch[100] * x_dp * x_h2,
            -ch[101] * x_hd * x_hp]
    hp_out = [ph5 * h2p, cr1 * hi]
    # H2
    h2_3b = [ch[108] * hi**3, ch[109] * hi**2 * x_h2, ch[110] * hi**2 * hei]
    h2_in = [ch[22] * hi * hm, rdust * hi * 1.0, CST1 * hi * h2p, -k24 * x_hp * x_h2,
             -ch[25] * e * x_h2, -k_dh * hi * x_h2, -k_dh2 * x_h2 * x_h2, -k_dhe * hei * x_h2,
             -ch[9] * x_hep * x_h2, -CST75 * x_hep * x_h2, -ch[126] * x_h2 * e,
             ch[127] * hm * h2p]
    h2_d = [ch[101] * x_hd * x_hp, -ch[100] * x_h2 * x_dp, -ch[102] * x_h2 * x_di,
            ch[103] * x_hd * hi]
    h2_out = [-k_dis * x_h2, -cr2 * x_h2]
    # He+, He++
    hep_in = [ch[14] * hei * e, -ch[15] * x_hep * e, -ch[17] * x_hep * e, ch[20] * x_hepp * e,
              -ch[32] * x_hep * hi, ch[33] * hei * x_hp, -ch[8] * x_hep * hm,
              -ch[9] * x_hep * x_h2, -CST75 * x_hep * x_h2, -ch[92] * x_hep * x_co]
    hep_d = [ch[33] * hei * x_dp, -ch[32] * x_hep * x_di]
    hep_out = [cr6 * hei]
    hepp_in = [ch[15] * x_hep * e, -ch[20] * x_hepp * e]
    # D+, HD
    dp_in = [ch[98] * x_hp * x_di, -ch[99] * x_dp * hi, -ch[100] * x_dp * x_h2,
             ch[101] * x_hd * x_hp, -ch[13] * x_dp * e, ch[11] * x_di * e,
             ch[32] * x_hep * x_di, -ch[33] * hei * x_dp]
    dp_out = [cr1 * x_di]
    hd_3b = (ch[108] * hi + ch[109] * x_h2 + ch[110] * hei) * hi * x_di * ynsq
    hd_in = [ch[100] * x_h2 * x_dp, -ch[101] * x_hd * x_hp, ch[102] * x_h2 * x_di,
             -ch[103] * x_hd * hi]
    hd_out = [-ph["k_hd"] * x_hd]
    # CO (NL97), destroyed also by He+
    x_o = jnp.maximum(0.0, p.abundo - x_co)
    den = K1_NL97 * x_o + gamma_chx / n
    beta = jnp.where(gamma_chx < 1e-30, 1.0, K1_NL97 * x_o / jnp.where(den == 0.0, 1.0, den))
    beta = jnp.where(x_o == 0.0, 0.0, beta)
    co_terms = [K0_NL97 * x_cp * x_h2 * beta * n, -ch[92] * x_hep * x_co * n, -gamma_co * x_co]

    def lin(terms):
        s = terms[0]
        for t in terms[1:]:
            s = s + t
        return s

    dx_hp = lin(hp_in) * n + lin(hp_out)
    dx_hp = dx_hp + jnp.where(d_on, lin(hp_d) * n, 0.0)
    dx_h2 = lin(h2_3b) * n**2 + lin(h2_in) * n + lin(h2_out)
    dx_h2 = dx_h2 + jnp.where(d_on, lin(h2_d) * n, 0.0)
    he_on = ABHE != 0.0
    dx_hep = jnp.where(he_on, lin(hep_in) * n + lin(hep_out), 0.0)
    dx_hep = dx_hep + jnp.where(d_on, lin(hep_d) * n, 0.0)
    dx_hepp = jnp.where(he_on, lin(hepp_in) * n, 0.0)
    dx_dp = jnp.where(d_on, lin(dp_in) * n + lin(dp_out), 0.0)
    dx_hd = jnp.where(d_on, hd_3b + lin(hd_in) * n + lin(hd_out), 0.0)
    dx_co = lin(co_terms)

    chem = _chem_terms(temp, n, x_h2, x_hp, x_hep, x_hepp, hi, hei, e, hm, h2p, ch, k_dh, k_dh2,
                       k_dhe, k24, k_dis, h_gr, h2var0, rh2, p.h2_form_kin)
    ab = {"h2": x_h2, "o": x_o * (1.0 - x_hp), "co": x_co, "cp": x_cp,
          "si": jnp.zeros_like(x_h2), "e": e, "hp": x_hp, "hi": hi, "hei": hei}
    if fixed:
        ab["hd"] = x_hd                      # chemcool never sets abundances(13): no HD cooling
    cool, tdust = cooling.cool_func(temp, n, l_jeans, p.divv, g_dust, ab, tdust, p.abundc,
                                    p.abundo, 0.0, p.dgr, p.zeta)
    de = -(rates_net5._seqsum(cool) + rates_net5._seqsum(chem))

    ydot = jnp.stack([dx_h2, dx_hp, dx_co, dx_hep, dx_hepp, dx_dp, dx_hd, de], axis=-1)
    bad = bad | jnp.any(jnp.isnan(ydot), axis=-1)
    don = lambda t: jnp.where(d_on, dn(t) * n, 0.0)  # noqa: E731
    gross = jnp.stack([
        dn(h2_3b) * n**2 + dn(h2_in) * n + dn(h2_out) + don(h2_d),
        dn(hp_in) * n + dn(hp_out) + don(hp_d),
        dn(co_terms),
        dn(hep_in) * n + dn(hep_out) + don(hep_d),
        dn(hepp_in) * n,
        don(dp_in) + jnp.where(d_on, dn(dp_out), 0.0),
        jnp.where(d_on, jnp.abs(hd_3b) + dn(hd_in) * n + dn(hd_out), 0.0),
        jnp.sum(jnp.abs(cool), -1) + jnp.sum(jnp.abs(chem), -1)], axis=-1)
    ydot = jnp.where(bad[..., None], 0.0, ydot)
    aux = {"T": temp, "gamma": gamma, "tdust": tdust, "cool": cool, "chem": chem, "eqb": eq,
           "invalid": bad | cold, "dust_converged": dust_ok, "k_h2_dis": k_dis,
           "g_dust": g_dust, "photo": ph, "ph4": ph4, "ph5": ph5, "h2var0": h2var0,
           "alpha_h2p": alpha, "gross": gross}
    return ydot, aux


def _chem_terms(temp, n, x_h2, x_hp, x_hep, x_hepp, hi, hei, e, hm, h2p, ch, k_dh, k_dh2, k_dhe,
                k24, k_dis, h_gr, h2var0, rh2, f_kin):
    """rates_chem(1..22) of rate_eq_ulli.F [erg s^-1 cm^-3]; heating is negative."""
    ynsq = n * n
    zero = jnp.zeros_like(temp)
    fpheat = rates_net5.uv_pump_heating_fraction(temp, n, x_h2, hi)
    three_body = ch[108] * hi**3 + ch[109] * hi**2 * x_h2 + ch[110] * hi**2 * hei
    r = [
        4.48 * EV * (k_dh * hi * x_h2 + k_dh2 * x_h2 * x_h2 + k_dhe * hei * x_h2
                     + ch[25] * e * x_h2) * ynsq,                       # H2 coll. dissociation
        1.83 * EV * k24 * x_h2 * x_hp * ynsq,                          # H2 + H+ charge transfer
        -4e-1 * EV * x_h2 * n * k_dis,                                 # H2 photodissociation
        -2.0 * EV * x_h2 * n * k_dis * UV_PUMPS * fpheat,              # UV pumping
        -3.73 * EV * ch[22] * hi * hm * ynsq * (1.0 - h2var0),         # H2 via H-
        -1.83 * EV * CST1 * hi * h2p * ynsq * (1.0 - h2var0),          # H2 via H2+
        -4.48 * EV * three_body * ynsq * n,                            # three-body H2
        13.6 * EV * (ch[11] * e + ch[144] * hi + ch[158] * hei) * hi * ynsq,   # H ionisation
        24.6 * EV * ch[14] * e * hei * ynsq,                           # He ionisation
        54.4 * EV * ch[15] * e * x_hep * ynsq,                         # He+ ionisation
        KBOLTZ * temp * x_hp * (ch[13] * e + h_gr) * ynsq,             # H+ recombination
        40.8 * EV * ch[10] * x_hep * e * ynsq,                         # He+ dielectronic rec.
        KBOLTZ * temp * (ch[17] - ch[10]) * x_hep * e * ynsq,          # He+ radiative rec.
        KBOLTZ * temp * x_hepp * ch[20] * e * ynsq,                    # He++ recombination
        zero, zero, zero, zero, zero, zero,                            # photo-heating (off)
        (0.75 * EV + KBOLTZ * temp) * ch[21] * hi * e * ynsq,          # H- formation
        -4.48 * EV * f_kin * rh2]                                      # H2 formation on dust
    return jnp.stack([t * jnp.ones_like(temp) for t in r], axis=-1)
