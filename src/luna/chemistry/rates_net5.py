"""Network 5 (chemcool rate_eq_simple.F): H2, H+ and CO with the gas energy, y = (x_H2, x_H+, x_CO, e).

Port of S. Glover's chemcool (SGChem); the network and its references are in docs/CHEMISTRY.md."""
from typing import NamedTuple

import jax.numpy as jnp

from luna.chemistry import cooling, gow17, photo, thermo
from luna.chemistry import tables as tb

EV, KBOLTZ, MH, PI, GRAVITY, ABHE = tb.EV, tb.KBOLTZ, tb.MH, tb.PI, tb.GRAVITY, tb.ABHE
ZETA_FACTORS = {"cr1": 1.0, "cr2": 2.0, "cr10": 3.7e-2, "cr11": 2.2e-1}   # cheminmo.F:1857
K0_NL97, K1_NL97 = 5e-16, 5e-10    # C+ + H2 -> CH_x; CH_x + O -> CO  [cm^3 s^-1]
EPS_MAX = 1e-4                     # allowed overshoot before rate_eq rejects a state


class Params(NamedTuple):
    """Everything the RHS needs besides the state. Columns: (..., n_pix)."""
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
    abundsi: float = 0.0       # Si per H (all Si+)

    dl: float = 0.0            # local shielding length: adds n_H dl to the columns
    divv: float = 0.0          # velocity divergence [s^-1] (LVG cooling columns)
    tdust: float = 10.0        # dust temperature guess (the Fortran carries it)

    h2_form_kin: float = 0.12  # fraction of the 4.48 eV H2 binding energy heating gas
    h2_form_ex: float = 0.84   # fraction into H2 excitation (enters the dust balance)

    dust_rtol: float = 1e-2    # dust Newton stops at the first step below this (Fortran: 1 %)
    metallicity: float = 1.0   # Z / Z_sun of the WSS metal pattern (only with cie_cool)


def make_params(n_h, g0, n_h_col=0.0, n_h2_col=0.0, n_co_col=0.0, n_pix=12, zeta=3e-17,
                dgr=1.0, abundc=1.41e-4, abundo=3.16e-4, **kw):
    """Params with scalar columns spread over n_pix identical pixels, G0_dust = G0."""
    col = lambda c: jnp.broadcast_to(jnp.asarray(c, jnp.float64)[..., None],  # noqa: E731
                                     jnp.shape(c) + (n_pix,))
    return Params(n_h=n_h, n_h_col=col(n_h_col), n_h2_col=col(n_h2_col),
                  n_co_col=col(n_co_col), g0=g0, g0_dust=kw.pop("g0_dust", g0), zeta=zeta,
                  dgr=dgr, abundc=abundc, abundo=abundo, **kw)


def chem_coefficients(temp):
    """The chtab rows network 5 uses at T [cm^3 s^-1], linear in T between nodes (rate_eq)."""
    return cooling.interp_rows(cooling.chem_rows(), temp, clamp_top=True)


def invalid_state(x_h2, x_hp, x_co, e):
    """validate_input's flag: a NaN, or a fraction overshooting its bound by more than EPS_MAX."""
    bad = (x_h2 > 0.5 + EPS_MAX) | (x_hp > 1.0 + EPS_MAX) | (x_co > 1.0 + EPS_MAX)
    return bad | jnp.isnan(x_h2) | jnp.isnan(x_hp) | jnp.isnan(x_co) | jnp.isnan(e)


def _validate(y):
    """validate_input: clip to physical bounds, flag states overshooting by more than EPS_MAX."""
    x_h2, x_hp, x_co, e = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
    bad = invalid_state(x_h2, x_hp, x_co, e)
    x_h2 = jnp.clip(x_h2, 0.0, 0.5)
    x_hp = jnp.clip(x_hp, 0.0, 1.0)
    x_co = jnp.clip(x_co, 0.0, 1.0)
    return x_h2, x_hp, x_co, jnp.maximum(e, 0.0), bad


def rhs(y, p: Params, co_shield=False, cie_cool=False, fixes="chemcool", eos="table", lvg=True,
        dust_iters=None):
    """(ydot, aux) of network 5 (rate_eq_simple.F); options and aux keys in docs/CHEMISTRY.md."""
    if fixes not in ("chemcool", "gow17") or (fixes == "gow17" and co_shield):
        raise ValueError(f"fixes={fixes!r}, co_shield={co_shield}")
    y = jnp.asarray(y, jnp.float64)
    p = Params(*[jnp.asarray(v, jnp.float64) for v in p])
    if not lvg:
        p = p._replace(divv=jnp.zeros((), jnp.float64))
    x_h2, x_hp, x_co, energy, bad = _validate(y)
    n = p.n_h
    x_hi = jnp.maximum(1.0 - 2.0 * x_h2 - x_hp, 0.0)
    x_cp = jnp.maximum(0.0, p.abundc - x_co)
    x_e = x_hp + x_cp + p.abundsi

    # temperature from e (calc_temp.F); T <= 0 is evaluated at 1 K and flagged
    ekn = thermo.ekn_from_energy(energy, x_h2, x_e, n)
    temp, gamma = thermo.temperature_gamma(x_h2, ekn, ABHE, eos)
    cold = temp <= 0.0
    temp = jnp.where(cold, tb.TMIN, temp)
    ch = chem_coefficients(temp)

    # collisional dissociation of H2, density-interpolated between the
    # low- and high-n limits (rate_eq_simple.F:193)
    ncrinv = 2.0 * x_h2 * (ch[6] - ch[5]) + ch[5]
    h2var0 = 1.0 / (1.0 + n * ncrinv)
    k_dh = ch[1] * ch[3] ** h2var0           # H2 + H
    k_dh2 = ch[2] * ch[4] ** h2var0          # H2 + H2

    # photo rates for the fixed columns; H2 self-shielding with the local column
    av_local = photo.AV_PER_NH * p.dgr * (p.dl * n)
    ph = photo.photo_rates(p.g0, p.n_h_col, p.n_h2_col, p.n_co_col, p.dgr, av_local,
                           co_shield=co_shield)
    n_h2_col = x_h2[..., None] * (n * p.dl)[..., None] + p.n_h2_col
    k_dis = ph["k_h2_dust"] * photo.h2_self_shielding(n_h2_col, temp)
    g_dust = ph["g_dust"]
    gamma_co = ph["k_co"]
    if fixes == "gow17":
        n_co_col = x_co[..., None] * (n * p.dl)[..., None] + p.n_co_col
        gamma_co = gow17.co_photo_rate(p.g0, p.n_h_col, n_h2_col, n_co_col, p.dgr, av_local)
    gamma_chx = photo.K_CHX * g_dust

    h_gr = grain_recombination_hp(g_dust, temp, n, x_e, p.dgr, ch)

    # dust temperature and H2 formation on dust (cool_util.F calc_dust_temp)
    dust_kw = {} if dust_iters is None else {"iters": dust_iters}
    rh2, rdust, tdust, dust_ok = cooling.dust_temperature(
        n, temp, x_hi, ph["chi"], p.g0_dust, p.dgr, p.tdust, p.h2_form_ex, p.h2_form_kin,
        p.dust_rtol, **dust_kw)

    cr = {k: f * p.zeta for k, f in ZETA_FACTORS.items()}
    # H+: collisional ionisation - case-B recombination - grain recombination
    #     + cosmic-ray ionisation of H and dissociative ionisation of H2
    hp_terms = [ch[11] * x_e * x_hi * n, ch[13] * x_hp * x_e * n, h_gr * x_hp * n,
                cr["cr1"] * x_hi, cr["cr10"] * x_h2]
    dx_hp = ((ch[11] * x_e * x_hi - ch[13] * x_hp * x_e - h_gr * x_hp * 1.0) * n
             + cr["cr1"] * x_hi + cr["cr10"] * x_h2)
    # H2: - collisional dissociation (e-, H, H2) + formation on dust - photodissociation
    #     - cosmic rays (the 0.5 on cr2 is the Fortran's "fudge factor of two")
    h2_terms = [ch[25] * x_e * x_h2 * n, k_dh * x_hi * x_h2 * n, k_dh2 * x_h2 * x_h2 * n,
                rdust * x_hi * n, k_dis * x_h2, 0.5 * cr["cr2"] * x_h2, cr["cr10"] * x_h2,
                cr["cr11"] * x_h2]
    dx_h2 = ((-ch[25] * x_e * x_h2 - k_dh * x_hi * x_h2 - k_dh2 * x_h2 * x_h2
              + rdust * x_hi * 1.0) * n
             - k_dis * x_h2 - 0.5 * cr["cr2"] * x_h2 - cr["cr10"] * x_h2 - cr["cr11"] * x_h2)

    # CO, NL97: C+ + H2 -> CH_x (k0); CH_x + O -> CO (k1) vs photodestruction
    x_o = jnp.maximum(0.0, p.abundo - x_co)
    den = K1_NL97 * x_o + gamma_chx / n
    beta = jnp.where(gamma_chx < 1e-30, 1.0,
                     K1_NL97 * x_o / jnp.where(den == 0.0, 1.0, den))
    beta = jnp.where(x_o == 0.0, 0.0, beta)
    dx_co = K0_NL97 * x_cp * x_h2 * beta * n - gamma_co * x_co
    co_terms = jnp.abs(K0_NL97 * x_cp * x_h2 * beta * n) + jnp.abs(gamma_co * x_co)
    if fixes == "gow17":
        # cosmic rays: He+ (in equilibrium) + CO and CR-induced photons; C goes to C+
        xi = gow17.xi_from_zeta(p.zeta, x_hi, x_h2)
        k_he_gr = gow17.he_grain_recombination(temp, gow17.grain_phi(g_dust, temp, n, x_e), p.dgr)
        k_cr_co, x_hep = gow17.co_cr_destruction(xi, temp, n, x_hi, x_h2, x_e, x_co, k_he_gr)
        dx_co = dx_co - k_cr_co * x_co
        co_terms = co_terms + jnp.abs(k_cr_co * x_co)

    # chemical heating and cooling (rate_eq_simple.F:331-395)
    chem = _chem_terms(temp, n, x_h2, x_hi, x_hp, x_e, k_dh, k_dh2, k_dis, ch, h_gr, rh2,
                       p.h2_form_kin)

    # cool_func with the Jeans length for its escape columns
    rho = (1.0 + 4.0 * ABHE) * MH * n
    cs2 = gamma * (gamma - 1.0) * energy / rho
    l_jeans = jnp.sqrt(PI * cs2 / (GRAVITY * rho))
    ab = {"h2": x_h2, "o": x_o, "co": x_co, "cp": x_cp, "si": p.abundsi * jnp.ones_like(x_h2),
          "e": x_e, "hp": x_hp, "hi": x_hi}
    cool, tdust = cooling.cool_func(temp, n, l_jeans, p.divv, g_dust, ab, tdust, p.abundc,
                                    p.abundo, p.abundsi, p.dgr, p.zeta,
                                    cie_cool=cie_cool, metallicity=p.metallicity, lvg=lvg)
    de = -(_seqsum(cool) + _seqsum(chem))

    ydot = jnp.stack([dx_h2, dx_hp, dx_co, de], axis=-1)
    # sum of |terms| per component: the scale on which ydot is accurate
    gross = jnp.stack([sum(jnp.abs(t) for t in h2_terms), sum(jnp.abs(t) for t in hp_terms),
                       co_terms,
                       jnp.sum(jnp.abs(cool), -1) + jnp.sum(jnp.abs(chem), -1)], axis=-1)
    ydot = jnp.where(bad[..., None], 0.0, ydot)
    aux = {"T": temp, "gamma": gamma, "tdust": tdust, "cool": cool, "chem": chem,
           "invalid": bad | cold, "dust_converged": dust_ok, "k_h2_dis": k_dis,
           "g_dust": g_dust, "gross": gross, "k_co": gamma_co}
    if fixes == "gow17":
        aux["x_hep"] = x_hep
    return ydot, aux


def grain_recombination_hp(g_dust, temp, n, x_e, dgr, ch):
    """H+ recombination on grains [cm^3 s^-1], WD01 fit (rate_eq_simple.F:230)."""
    phi = jnp.where(x_e == 0.0, 1e20,
                    g_dust * jnp.sqrt(temp) / (n * jnp.where(x_e == 0.0, 1.0, x_e)))
    phi_s = jnp.maximum(phi, 1e-6)
    h_gr = jnp.where(phi < 1e-6, 1.225e-13 * dgr,
                     1.225e-13 * dgr / (1.0 + 8.074e-6 * phi_s**1.378
                                        * (1.0 + ch[34] * phi_s ** ch[35])))
    return h_gr * jnp.exp(-temp / 3e4)


def uv_pump_heating_fraction(temp, n, x_h2, x_hi):
    """Fraction of UV-pumped H2 that is collisionally de-excited (heats the gas)."""
    cdex = ((tb.f32(1.4) * jnp.exp(-1.81e4 / (temp + 1200)) * x_h2
             + 1.0 * jnp.exp(-1e3 / temp) * x_hi) * 1e-12 * jnp.sqrt(temp) * n)
    return cdex / (cdex + 2e-7)


def _seqsum(a):
    """a[..., 0] + a[..., 1] + ... in the Fortran's left-to-right order."""
    s = a[..., 0]
    for k in range(1, a.shape[-1]):
        s = s + a[..., k]
    return s


def _chem_terms(temp, n, x_h2, x_hi, x_hp, x_e, k_dh, k_dh2, k_dis, ch, h_gr, rh2, f_kin):
    """lambda_chem(1..6): energy in chemical reactions [erg s^-1 cm^-3]."""
    # 1: H2 collisional dissociation takes 4.48 eV per molecule
    l1 = 4.48 * EV * (k_dh * x_hi * x_h2 + k_dh2 * x_h2 * x_h2 + ch[25] * x_e * x_h2) * n**2
    # 2: H2 photodissociation heats with 0.4 eV per dissociation
    l2 = -4e-1 * EV * x_h2 * n * k_dis
    # 3: UV pumping of H2, 2 eV per pump, 8.5 pumps per dissociation (BHT90,
    #    DB96), times the collisional de-excitation fraction
    l3 = -2.0 * EV * x_h2 * n * k_dis * 8.5 * uv_pump_heating_fraction(temp, n, x_h2, x_hi)
    # 4: H2 formation on dust heats the gas with h2_form_kin of 4.48 eV
    l4 = -4.48 * EV * f_kin * rh2
    # 5: collisional ionisation of H takes 13.6 eV
    l5 = 13.6 * EV * ch[11] * x_e * x_hi * n**2
    # 6: recombination radiates kT (gas phase and on grains), off below 100 K
    l6 = jnp.where(temp < 100.0, 0.0, KBOLTZ * temp * x_hp * (ch[13] * x_e + h_gr) * n**2)
    return jnp.stack([l1, l2, l3, l4, l5, l6], axis=-1)
