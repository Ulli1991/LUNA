"""The JAX port of chemcool network 5 against the Fortran, function by function.

Golden values: $LUNA_DATA/chemistry/chemistry_net5_golden.npz, produced by
the unmodified chemcool Fortran (see docs/CHEMISTRY.md).
Every comparison is to 1e-10 relative in FP64. Values below 1e-290 (the
denormal range, which XLA flushes to zero on CPU) must be below 1e-290 on
both sides. Where a quantity is a difference of terms (a net rate), it is
held to 1e-10 of the sum of the magnitudes of its terms.

The teeth tests break one function at a time (a table row shifted by one
node, a constant or exponent off by 1 %, a single-precision literal read in
double, ...) and require the matching golden comparison to fail.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


from luna.chemistry import cooling, photo, rates_net5, thermo  # noqa: E402
from luna.chemistry import tables as tb  # noqa: E402

GOLDEN = tb.data_path("chemistry_net5_golden.npz")
RTOL = 1e-10
TINY = 1e-290
G = dict(np.load(GOLDEN))


def rel_err(a, b):
    """Max relative error; entries with |b| <= TINY need |a| <= TINY."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    small = np.abs(b) <= TINY
    r = np.where(small, np.where(np.abs(a) <= TINY, 0.0, np.inf),
                 np.abs(a - b) / np.where(small, 1.0, np.abs(b)))
    return float(np.max(r)) if r.size else 0.0


def scaled_err(a, b, scale):
    """Max |a - b| / scale (for net rates: scale = sum of |terms|)."""
    a, b, s = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float),
                                  np.asarray(scale, float))
    ok = (a == b)
    r = np.where(ok, 0.0, np.abs(a - b) / np.where(s > 0, s, 1.0))
    r = np.where(~ok & (s <= TINY), np.where(np.abs(a - b) <= TINY, 0.0, np.inf), r)
    return float(np.max(r))


def pix(x, n=12):
    return jnp.broadcast_to(jnp.asarray(x)[..., None], np.shape(x) + (n,))


@pytest.fixture(autouse=True)
def fresh_tables():
    """Every test starts (and ends) with clean table caches."""
    def clear():
        cooling._CACHE.clear()
        thermo._TEMP = thermo._GAMMA = None
        for f in (tb.chem_table, tb.cool_table, tb.co_tables, tb.energy_table,
                  tb.energy_table_regenerated, tb.raw):
            f.cache_clear()
    clear()
    yield
    clear()


# --------------------------------------------------------------------------
# the comparisons (each returns the max error; tests assert <= RTOL)
# --------------------------------------------------------------------------
def err_tables():
    every = int(G["table_every"])
    T = tb.temptab()
    ch, cl = tb.chem_table(), tb.cool_table()
    errs = [rel_err(tb.temptab()[::every], G["temptab"])]
    rows = list(G["chtab_rows"])
    for i, r in enumerate(rows):
        errs.append(rel_err(ch[r][0][::every], G["chtab"][:, i]))
    rows = list(G["cltab_rows"])
    for i, r in enumerate(rows):
        if r in cl:
            errs.append(rel_err(cl[r][0][::every], G["cltab"][:, i]))
        elif r not in (6, 15):           # tabulated, but multiply zero for network 5
            errs.append(rel_err(0.0, G["cltab"][:, i]))
    # slopes: accurate to 1e-10 of the values they interpolate between
    dT = np.diff(T)[::every][:-1]
    for i, r in enumerate(G["chtab_rows"]):
        v, s = ch[r][0][::every][:-1], ch[r][1][::every][:-1]
        errs.append(scaled_err(s * dT, G["chtab"][:-1, 12 + i] * dT, np.abs(v) + np.abs(s * dT)))
    co = tb.co_tables()
    ev = int(G["co_lte_every"])
    errs.append(rel_err(co["lte"].T[::ev], G["co_lte"]))
    errs.append(rel_err(co["vib"].T[::ev], G["co_vib"]))
    return max(errs)


def err_rate_coefficients():
    T = G["chinterp_in"][:, 0]
    ch = rates_net5.chem_coefficients(jnp.asarray(T))
    rows = list(G["chtab_rows"])
    return max(rel_err(ch[r], G["chinterp_out"][:, i]) for i, r in enumerate(rows))


def err_temperature():
    x = G["temp_gamma_in"]
    t, g = thermo.temperature_gamma(jnp.asarray(x[:, 0]), jnp.asarray(x[:, 1]))
    return max(rel_err(t, G["temp_gamma_out"][:, 0]), rel_err(g, G["temp_gamma_out"][:, 1]))


def err_chi():
    return rel_err(photo.chi_isrf(jnp.asarray(G["chi_in"][:, 0])), G["chi_out"][:, 0])


def err_h2_shielding():
    x = G["h2shield_in"]
    col = pix(x[:, 1] * x[:, 2] + x[:, 3])
    return rel_err(photo.h2_self_shielding(col, jnp.asarray(x[:, 0])), G["h2shield_out"][:, 0])


def _photo(co_shield):
    x = G["photo_in"]
    g0, dgr = jnp.asarray(x[:, 0]), jnp.asarray(x[:, 1])
    av_local = photo.AV_PER_NH * dgr * (x[:, 6] * x[:, 5])
    return photo.photo_rates(g0, pix(x[:, 2]), pix(x[:, 3]), pix(x[:, 4]), dgr, av_local,
                             co_shield=co_shield)


def err_photo():
    ph, out = _photo(False), G["photo_out"]
    e = [rel_err(ph["k_h2_dust"], out[:, 0]), rel_err(ph["k_co"], out[:, 1]),
         rel_err(ph["g_dust"], out[:, 4]), rel_err(0.0, out[:, 2]), rel_err(0.0, out[:, 3])]
    lit = G["photo_in"][:, 0] > 0           # the Fortran leaves chi stale when G0 = 0
    e.append(rel_err(np.asarray(ph["chi"])[lit], out[lit, 5]))
    return max(e)


def err_photo_co_shielding():
    ph = _photo(True)
    return rel_err(ph["k_co"], G["photo_cos_out"][:, 1])


def err_dust():
    x = G["dust_in"]
    a = [jnp.asarray(x[:, i]) for i in range(7)]
    rh2, rdust, td, ok = cooling.dust_temperature(a[0], a[2], a[3], a[1], a[5], a[6], a[4])
    assert bool(np.all(ok))
    out = G["dust_out"]
    return max(rel_err(rh2, out[:, 0]), rel_err(rdust, out[:, 1]), rel_err(td, out[:, 2]))


def err_co_cool():
    x = G["cocool_in"]
    got = cooling.co_cool(jnp.asarray(x[:, 0]), jnp.asarray(x[:, 1]), jnp.asarray(x[:, 2]))
    out = G["cocool_out"]
    # co_cool returns (L0, LTE, alpha, n05, vib), the Fortran argument order
    return max(rel_err(v, out[:, k]) for k, v in enumerate(got))


def err_h2_opacity():
    x = G["h2opac_in"]
    return rel_err(cooling.h2_opacity(jnp.asarray(x[:, 0]), jnp.asarray(x[:, 1])),
                   G["h2opac_out"][:, 0])


def _cool_func():
    x = jnp.asarray(G["coolfunc_in"])
    c = [x[:, i] for i in range(19)]
    ab = {"h2": c[5], "o": c[6], "co": c[7], "cp": c[8], "si": c[9], "e": c[10], "hp": c[11],
          "hi": c[12]}
    return cooling.cool_func(c[0], c[1], c[2], c[3], c[4], ab, c[13], c[14], c[15], c[16],
                             c[17], c[18])


def err_cool_terms():
    rates, tdust = _cool_func()
    rates = np.array(rates)
    out = G["coolfunc_out"].copy()
    # Above ~1.5e7 K the gas-grain table cl14 (~ exp(-T/3e4)) is denormal. XLA
    # flushes denormals to zero on CPU, the Fortran does not and multiplies
    # them by n^2 up to ~1e-288. There the port must give exactly 0.
    T = G["coolfunc_in"][:, 0]
    i, _ = cooling._node_index(jnp.asarray(T), clamp_top=False)
    denorm = np.abs(tb.cool_table()[14][0][np.asarray(i)]) < np.finfo(float).tiny
    flushed = denorm & (rates[:, 0] == 0.0)
    out[flushed, 0] = rates[flushed, 0] = 1.0
    e = [rel_err(rates[:, k], out[:, k]) for k in range(28)]
    e.append(rel_err(tdust, out[:, 28]))
    return max(e)


def _rhs(fn=rates_net5.rhs):
    x = G["rateeq_in"]
    c = [jnp.asarray(x[:, i]) for i in range(17)]
    p = rates_net5.Params(n_h=c[0], n_h_col=pix(c[6]), n_h2_col=pix(c[7]), n_co_col=pix(c[8]),
                          g0=c[5], g0_dust=c[5], zeta=c[11], dgr=c[12], abundc=c[13],
                          abundo=c[14], abundsi=c[15], dl=c[9], divv=c[10], tdust=c[16])
    y = jnp.stack([c[2], c[3], c[4], c[1]], axis=-1)
    return fn(y, p)


def err_rhs(fn=rates_net5.rhs):
    ydot, aux = _rhs(fn)
    out = G["rateeq_out"]
    ok = out[:, 44] == 0                    # ipar(1): rejected states return ydot = 0
    e = [rel_err(np.asarray(ydot)[~ok], 0.0 * out[~ok, :4]),
         scaled_err(ydot, out[:, :4], np.asarray(aux["gross"])),
         rel_err(np.asarray(aux["tdust"])[ok], out[ok, 38])]
    e += [rel_err(np.asarray(aux["cool"])[ok, k], out[ok, 4 + k]) for k in range(28)]
    e += [rel_err(np.asarray(aux["chem"])[ok, k], out[ok, 32 + k]) for k in range(6)]
    assert bool(np.all(np.asarray(aux["invalid"]) == ~ok))
    return max(e)


CHECKS = {
    "tables": err_tables, "rate_coefficients": err_rate_coefficients,
    "temperature": err_temperature, "chi": err_chi, "h2_shielding": err_h2_shielding,
    "photo": err_photo, "photo_co_shielding": err_photo_co_shielding, "dust": err_dust,
    "co_cool": err_co_cool, "h2_opacity": err_h2_opacity, "cool_terms": err_cool_terms,
    "rhs": err_rhs,
}


@pytest.mark.parametrize("name", list(CHECKS))
def test_matches_fortran(name):
    err = CHECKS[name]()
    print(f"{name}: max error {err:.3e}")
    assert err <= RTOL, f"{name}: {err:.3e} > {RTOL}"


# --------------------------------------------------------------------------
# teeth: break one thing, the matching comparison must fail
# --------------------------------------------------------------------------
def _shift_row(table_fn, row):
    """Replace a table builder by one whose `row` is shifted by one T node."""
    orig = table_fn.__wrapped__

    def broken():
        d = dict(orig())
        v, s = d[row]
        d[row] = (np.roll(v, 1), np.roll(s, 1))
        return d
    return broken


def _scaled_row(table_fn, row, f):
    orig = table_fn.__wrapped__

    def broken():
        d = dict(orig())
        v, s = d[row]
        d[row] = (v * f, s * f)
        return d
    return broken


def _patch_cache(monkeypatch, key, builder):
    monkeypatch.setitem(cooling._CACHE, key, cooling._arrays(builder()))


BREAKS = {
    # table row shifted by one node
    "ch13_shifted": ("rate_coefficients",
                     lambda mp: _patch_cache(mp, "ch", _shift_row(tb.chem_table, 13))),
    "cl18_shifted": ("cool_terms",
                     lambda mp: _patch_cache(mp, "cl", _shift_row(tb.cool_table, 18))),
    "cl66_shifted": ("cool_terms",
                     lambda mp: _patch_cache(mp, "cl", _shift_row(tb.cool_table, 66))),
    # a rate 1 % off
    "ch25_1pct": ("rhs", lambda mp: _patch_cache(mp, "ch", _scaled_row(tb.chem_table, 25, 1.01))),
    "cl52_1pct": ("cool_terms",
                  lambda mp: _patch_cache(mp, "cl", _scaled_row(tb.cool_table, 52, 1.01))),
    "k0_nl97_1pct": ("rhs", lambda mp: mp.setattr(rates_net5, "K0_NL97", 5.05e-16)),
    "k_co_1pct": ("photo", lambda mp: mp.setattr(photo, "K_CO_THIN", 1.235e-10 * 1.01)),
    "sigma_lw_1pct": ("photo", lambda mp: mp.setattr(photo, "SIGMA_DUST_LW", 2.02e-21)),
    "kboltz_1pct": ("cool_terms", lambda mp: mp.setattr(cooling, "KBOLTZ", tb.KBOLTZ * 1.01)),
    # exponent off by 1 %: phi**ch35 in the WD01 grain recombination of H+
    "ch35_exponent_1pct": ("rhs", lambda mp: _patch_cache(
        mp, "ch", _scaled_row(tb.chem_table, 35, 1.01))),
    # thermo: the 0.1 of lookup_temp_gamma read in double instead of single
    "thermo_f01_double": ("temperature", lambda mp: mp.setattr(thermo, "_F01", 0.1)),
    # energy table shifted by one e/kn row
    "energy_table_shift": ("temperature", lambda mp: mp.setattr(
        thermo, "_table", lambda: tuple(jnp.roll(a, 1) for a in _ENERGY_TABLE()))),
    # the regenerated stand-in for Glover's energy table (T to 3e-4, gamma to 15 %)
    "energy_table_regenerated": ("rhs", lambda mp: mp.setattr(
        tb, "energy_table", tb.energy_table_regenerated)),
    # chi(A_V) table shifted
    "chi_shift": ("chi", lambda mp: mp.setitem(
        tb.raw(), "chi_tab_mathis", np.roll(tb.raw()["chi_tab_mathis"], 1))),
    # CO cooling table: temperature axis shifted by one bin
    "co_lte_shift": ("co_cool", lambda mp: mp.setitem(
        cooling.co_arrays(), "lte", jnp.roll(cooling.co_arrays()["lte"], 1, axis=1))),
    # dust: emission coefficient 1 % off
    "dust_emission_1pct": ("dust", lambda mp: mp.setattr(cooling, "DUST_EMISSION", 4.7268e-31)),
    # H2 self-shielding: DB96 0.965 -> 0.975
    "h2shield_coef": ("h2_shielding", lambda mp: mp.setattr(
        photo, "h2_self_shielding", _db96_broken)),
    # H2 opacity table shifted by one column bin
    "h2opac_shift": ("h2_opacity", lambda mp: mp.setitem(
        tb.raw(), "h2_opac", np.roll(tb.raw()["h2_opac"], 1, axis=1))),
    # table builder: undecorated Fortran literals (e.g. ch2's T**0.485) read in double
    "f32_literals_in_double": ("tables", lambda mp: mp.setattr(tb, "f32", lambda x: float(x))),
}


def _db96_broken(n_h2_col, temp):
    b5 = jnp.sqrt(tb.KBOLTZ * jnp.asarray(temp) / tb.MH) / 1e5
    x = n_h2_col / 5e14
    sq = jnp.sqrt(1.0 + x)
    f = 9.75e-1 / (1.0 + x / b5[..., None]) ** 2 + 3.5e-2 * jnp.exp(-8.5e-4 * sq) / sq
    return jnp.mean(f, axis=-1)


_ENERGY_TABLE = thermo._table


@pytest.mark.parametrize("brk", list(BREAKS))
def test_teeth(brk, monkeypatch):
    check, apply = BREAKS[brk]
    apply(monkeypatch)
    err = CHECKS[check]()
    print(f"broken {brk}: {check} error {err:.3e}")
    assert err > RTOL, f"breaking {brk} went unnoticed by the {check} comparison"


# --------------------------------------------------------------------------
# differentiability and jit
# --------------------------------------------------------------------------
# T0 sets e = 1.5 k T0 n_tot, so e/(k n_tot) = 1.5 T0; keep that off the integer
# nodes of the energy table, where T(e) has a kink and a central difference
# averages the two one-sided slopes; likewise x_H2 off the 0.1 blocks
STATES = [  # (x_H2, x_H+, x_CO, T0), n, G0, shielding
    ((1e-3, 1e-4, 2e-9, 100.37), 1e2, 1.7, "thin"),
    ((0.31, 2e-6, 5e-5, 30.21), 1e4, 1.7, "av5"),
    ((0.45, 1e-7, 1.3e-4, 15.13), 1e5, 10.0, "av1"),
    ((1e-6, 5e-2, 1e-12, 8000.3), 1.0, 1.7, "thin"),
    ((0.11, 1e-3, 1e-6, 300.29), 1e3, 0.0, "thin"),
]
SHIELD = {"thin": (0.0, 0.0, 0.0), "av1": (1.87e21, 5e20, 1e15), "av5": (9.35e21, 4.5e21, 1e17)}


def _state(s):
    (xh2, xhp, xco, T), n, g0, sh = s
    p = rates_net5.make_params(n, g0, *SHIELD[sh], tdust=12.0)
    ntot = (1 + tb.ABHE - xh2 + xhp + (1.41e-4 - xco)) * n
    e = 1.5 * tb.KBOLTZ * T * ntot
    return jnp.array([xh2, xhp, xco, e]), p


@pytest.mark.parametrize("k", range(len(STATES)))
def test_jacfwd_matches_finite_differences(k):
    y0, p = _state(STATES[k])
    f = lambda y: rates_net5.rhs(y, p)[0]  # noqa: E731
    J = np.asarray(jax.jacfwd(f)(y0))
    assert np.all(np.isfinite(J))
    # central differences, step 1e-7 of each component (T bins are 0.18 % wide)
    fd = np.zeros_like(J)
    for j in range(4):
        h = 1e-7 * max(abs(float(y0[j])), 1e-12 if j < 3 else 1e-20)
        e = np.zeros(4)
        e[j] = h
        fd[:, j] = (np.asarray(f(y0 + e)) - np.asarray(f(y0 - e))) / (2 * h)
    # each row relative to its own scale (rows differ by ~10 orders of magnitude)
    for i in range(4):
        scale = np.max(np.abs(J[i] * np.asarray(y0))) + 1e-300
        err = np.max(np.abs((J[i] - fd[i]) * np.asarray(y0))) / scale
        print(f"state {k} row {i}: |J - FD| / scale = {err:.2e}")
        assert err < 1e-7


def test_jit_matches_fortran():
    """The jitted RHS (XLA may reorder sums: last-bit changes) against rate_eq."""
    err = err_rhs(jax.jit(rates_net5.rhs))
    print(f"jitted rhs: max error {err:.3e}")
    assert err <= RTOL
