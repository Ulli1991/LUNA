"""WSS_CIE_COOL hot-gas cooling (cie_cool=True) against chemcool built with the flag."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


from luna.chemistry import cooling, rates_net5  # noqa: E402
from luna.ode import rodas3 as ig  # noqa: E402
from luna.chemistry import networks  # noqa: E402
from luna.chemistry import tables as tb  # noqa: E402
from tests.test_chemistry_net5 import RTOL, pix, rel_err, scaled_err  # noqa: E402
from tools.chem_integrator import validate_net5 as val  # noqa: E402

G = dict(np.load(tb.data_path("chemistry_cie_golden.npz")))
TRAJ = tb.data_path("chemistry_cie_trajectories.npz")
TARGET, EQ_TARGET = 1e-3, 1e-4         # integrator vs DVODE, as tests/test_chemistry_integrator.py


@pytest.fixture(autouse=True)
def fresh_tables():
    """Every test starts (and ends) with clean table caches."""
    def clear():
        cooling._CACHE.clear()
        for f in (tb.chem_table, tb.cool_table, tb.cool_table_cie, tb.wss_cie_table, tb.cie_raw,
                  tb.co_tables, tb.raw):
            f.cache_clear()
    clear()
    yield
    clear()


# --------------------------------------------------------------------------
# the comparisons (each returns the max error; tests assert <= RTOL)
# --------------------------------------------------------------------------
def err_tables():
    rows = {**tb.cool_table(), **tb.cool_table_cie()}
    every = int(G["table_every"])
    errs = [rel_err(tb.temptab()[::every], G["temptab"])]
    for i, r in enumerate(G["cltab_rows"]):
        if r != 6:                       # SD93 atomic table: rates(3) never uses it (docs)
            errs.append(rel_err(rows[r][0][::every], G["cltab"][:, i]))
    # the WSS tables hold log10 rates: compare the rates
    errs.append(rel_err(10.0 ** tb.wss_cie_table(), 10.0 ** G["wss_tbl"]))
    raw = tb.cie_raw()
    errs.append(rel_err(np.stack([raw["hei_tbl"], raw["heii_tbl"]], 1), G["he_tbl"]))
    return max(errs)


def err_wss():
    t, z = jnp.asarray(G["wss_in"][:, 0]), G["wss_in"][:, 1]
    out = G["wss_out"]
    per = np.asarray(cooling.wss_cie_elements(t)) * z[:, None]
    e = [rel_err(cooling.wss_cie_cooling(t, jnp.asarray(z)), out[:, 0])]
    e += [rel_err(per[:, k], out[:, 1 + k]) for k in range(9)]
    return max(e)


def _cool_func(fn=cooling.cool_func):
    x = jnp.asarray(G["coolfunc_in"])
    c = [x[:, i] for i in range(20)]
    ab = {"h2": c[5], "o": c[6], "co": c[7], "cp": c[8], "si": c[9], "e": c[10], "hp": c[11],
          "hi": c[12]}
    return fn(c[0], c[1], c[2], c[3], c[4], ab, c[13], c[14], c[15], c[16], c[17], c[18],
              cie_cool=True, metallicity=c[19])


def _flush_denormal_gas_grain(rates, out, temp):
    """Above ~1.5e7 K cl14 (or its slope) is denormal: XLA flushes it, the Fortran does not."""
    i, _ = cooling._node_index(jnp.asarray(temp), clamp_top=False)
    v, s = (a[np.asarray(i)] for a in tb.cool_table()[14])
    denorm = (np.abs(v) < np.finfo(float).tiny) | (np.abs(s) < np.finfo(float).tiny)
    flushed = denorm & (np.abs(rates[:, 0]) < 1e-280)
    out[flushed, 0] = rates[flushed, 0] = 1.0


def err_cool_terms():
    rates, tdust = _cool_func()
    rates, out = np.array(rates), G["coolfunc_out"].copy()
    _flush_denormal_gas_grain(rates, out, G["coolfunc_in"][:, 0])
    return max([rel_err(rates[:, k], out[:, k]) for k in range(28)]
               + [rel_err(tdust, out[:, 28])])


def _params(x):
    c = [jnp.asarray(x[:, i]) for i in range(x.shape[1])]
    return rates_net5.Params(n_h=c[0], n_h_col=pix(c[6]), n_h2_col=pix(c[7]),
                             n_co_col=pix(c[8]), g0=c[5], g0_dust=c[5], zeta=c[11], dgr=c[12],
                             abundc=c[13], abundo=c[14], abundsi=c[15], dl=c[9], divv=c[10],
                             tdust=c[16], metallicity=c[17])


def err_rhs(fn=None):
    fn = fn or (lambda y, p: rates_net5.rhs(y, p, cie_cool=True))
    x = G["rateeq_in"]
    y = jnp.asarray(x[:, [2, 3, 4, 1]])
    ydot, aux = fn(y, _params(x))
    out = G["rateeq_out"]
    ok = out[:, 44] == 0
    cool, chem, lam = np.array(aux["cool"]), np.asarray(aux["chem"]), out[:, 4:32].copy()
    _flush_denormal_gas_grain(cool, lam, np.asarray(aux["T"]))
    e = [scaled_err(ydot, out[:, :4], np.asarray(aux["gross"])),
         rel_err(np.asarray(aux["tdust"])[ok], out[ok, 38])]
    e += [rel_err(cool[ok, k], lam[ok, k]) for k in range(28)]
    e += [rel_err(chem[ok, k], out[ok, 32 + k]) for k in range(6)]
    assert bool(np.all(np.asarray(aux["invalid"]) == ~ok))
    return max(e)


def err_trajectory_states():
    """rate_eq at every 5th recorded state of the 75 one-zone runs (T_dust as carried)."""
    meta, s = G["traj_meta"], G["traj_states"]
    p = rates_net5.make_params(jnp.asarray(meta[:, 1]), jnp.asarray(meta[:, 2]),
                               tdust=jnp.asarray(s[:, 4]), metallicity=jnp.asarray(meta[:, 3]))
    ydot, aux = rates_net5.rhs(jnp.asarray(s[:, :4]), p, cie_cool=True)
    cool, out = np.array(aux["cool"]), s[:, 9:37].copy()
    _flush_denormal_gas_grain(cool, out, np.asarray(aux["T"]))
    return max(scaled_err(ydot, s[:, 5:9], np.asarray(aux["gross"])), rel_err(cool, out),
               rel_err(aux["chem"], s[:, 37:43]))


CHECKS = {"tables": err_tables, "wss": err_wss, "cool_terms": err_cool_terms, "rhs": err_rhs,
          "trajectory_states": err_trajectory_states}


@pytest.mark.parametrize("name", list(CHECKS))
def test_matches_fortran(name):
    err = CHECKS[name]()
    print(f"{name}: max error {err:.3e}")
    assert err <= RTOL, f"{name}: {err:.3e} > {RTOL}"


def test_jit_matches_fortran():
    err = err_rhs(jax.jit(lambda y, p: rates_net5.rhs(y, p, cie_cool=True)))
    print(f"jitted rhs: max error {err:.3e}")
    assert err <= RTOL


def test_metal_term_vanishes_at_zero_metallicity():
    z0 = G["coolfunc_in"][:, 19] == 0.0
    assert z0.sum() > 100 and np.all(G["coolfunc_out"][z0, 2] == 0.0)
    rates, _ = _cool_func()
    assert np.all(np.asarray(rates)[z0, 2] == 0.0)
    assert np.all(np.asarray(rates)[~z0, 2] > 0.0)


def test_off_ignores_metallicity_and_adds_nothing():
    """cie_cool=False (the default): metallicity has no effect, terms 3 and 26-28 are zero."""
    x = G["rateeq_in"]
    y = jnp.asarray(x[:, [2, 3, 4, 1]])
    a = rates_net5.rhs(y, _params(x))
    b = rates_net5.rhs(y, _params(x)._replace(metallicity=jnp.full(x.shape[0], 7.0)))
    assert np.array_equal(np.asarray(a[0]), np.asarray(b[0]))
    assert np.all(np.asarray(a[1]["cool"])[:, [2, 25, 26, 27]] == 0.0)


# --------------------------------------------------------------------------
# teeth: break one thing, the matching comparison must fail
# --------------------------------------------------------------------------
def _wss_variant(no_abs=False, no_fac=False, swap=None):
    r = tb.cie_raw()
    rates = r["wss_rates"].copy()
    if swap:
        rates[:, list(swap)] = rates[:, list(swap[::-1])]
    fac = np.exp(-((1e4 / r["wss_temp"]) ** 2))
    fac = np.where(fac < tb.f32(1e-20), tb.f32(1e-20), fac)
    fac = np.ones_like(fac) if no_fac else fac
    with np.errstate(divide="ignore", invalid="ignore"):
        v = rates * fac[:, None]
        return np.log10(v if no_abs else np.abs(v))


def _set_wss(mp, **kw):
    mp.setattr(tb, "wss_cie_table", lambda: _wss_variant(**kw))


def _scaled_cie_row(row, f):
    orig = tb.cool_table_cie.__wrapped__

    def broken():
        d = dict(orig())
        v, s = d[row]
        d[row] = (v * f, s * f)
        return d
    return broken


def _no_he_fractions(temp, abhe=tb.ABHE):
    return abhe + 0.0 * temp, 0.0 * temp, 0.0 * temp


def _top_interval_used(temp):
    """The last WSS interval interpolated instead of clamped at the second-to-last node."""
    nlt = tb.WSS_NODES - 1
    idx_real = nlt * (jnp.log10(temp) - 2.0) / tb.WSS_LOGT_RANGE + 1.0
    i = jnp.clip(jnp.floor(idx_real), 1, nlt)
    x = jnp.clip(idx_real - i, 0.0, 1.0)[..., None]
    j = i.astype(jnp.int32) - 1
    tab = cooling.wss_table()
    return 10.0 ** ((1.0 - x) * tab[j] + x * tab[jnp.minimum(j + 1, nlt)])


BREAKS = {
    "wss_no_abs": ("wss", lambda mp: _set_wss(mp, no_abs=True)),
    "wss_no_suppression": ("wss", lambda mp: _set_wss(mp, no_fac=True)),
    "wss_swap_C_O": ("wss", lambda mp: _set_wss(mp, swap=(0, 2))),
    "wss_swap_Mg_Si": ("wss", lambda mp: _set_wss(mp, swap=(4, 5))),
    "wss_top_interval": ("wss", lambda mp: mp.setattr(cooling, "wss_cie_elements",
                                                      _top_interval_used)),
    "wss_range_double": ("wss", lambda mp: mp.setattr(tb, "WSS_LOGT_RANGE", 6.98185)),
    "metallicity_ignored": ("cool_terms", lambda mp: mp.setattr(
        cooling, "wss_cie_cooling", _metallicity_ignored)),
    "he_fractions_dropped": ("cool_terms", lambda mp: mp.setattr(
        cooling, "he_cie_fractions", _no_he_fractions)),
    "he_collider_dropped": ("cool_terms", lambda mp: mp.setattr(
        cooling, "he_cie_fractions", _no_neutral_he)),
    "cl59_1pct": ("cool_terms", lambda mp: mp.setattr(
        tb, "cool_table_cie", _scaled_cie_row(59, 1.01))),
    "cl62_1pct": ("rhs", lambda mp: mp.setattr(tb, "cool_table_cie", _scaled_cie_row(62, 1.01))),
    "gaunt_literals_in_double": ("tables", lambda mp: mp.setattr(tb, "f32", lambda x: float(x))),
    "no_cie_in_rhs": ("trajectory_states", lambda mp: mp.setattr(
        rates_net5.cooling, "_cie_terms", lambda *a: {})),
}


def _metallicity_ignored(temp, metallicity):
    return _orig_wss_cooling(temp, 1.0)


def _no_neutral_he(temp, abhe=tb.ABHE):
    x1, x2, x3 = _orig_he(temp, abhe)
    return 0.0 * x1, x2, x3


_orig_wss_cooling = cooling.wss_cie_cooling
_orig_he = cooling.he_cie_fractions


@pytest.mark.parametrize("brk", list(BREAKS))
def test_teeth(brk, monkeypatch):
    check, apply = BREAKS[brk]
    apply(monkeypatch)
    err = CHECKS[check]()
    print(f"broken {brk}: {check} error {err:.3e}")
    assert not err <= RTOL, f"breaking {brk} went unnoticed by the {check} comparison"


# --------------------------------------------------------------------------
# the integrator with cie_cool on the 75 hot-gas DVODE trajectories (bypass off, see docs)
# --------------------------------------------------------------------------
TIGHT = ig.Options(rtol=1e-8, atol=tuple(1e-4 * a for a in networks.NET5.atol), max_steps=20000)
# n = 1 and 1e2 from 1e6, 1e7, 1e8 K, every z: the runaway-cooling runs
TIGHT_CASES = [i * 15 + j * 3 + k for i in (1, 3) for j in (2, 3, 4) for k in range(3)]


def _cie_params(cp, tdust=10.0):
    return val.params(cp, tdust)._replace(metallicity=jnp.asarray(cp[:, 6]))


def _run(net, opt=ig.Options(), cases=None):
    """Every case over every output interval: (T (case, output), {qty: max error}, eq, failed)."""
    names, cp, t, y0, ref = val.load(TRAJ, cases)
    n = y0.shape[0]
    p = ig.broadcast_params(_cie_params(cp), n)

    @jax.jit
    def go(y0, p, dts):
        def step(c, dt):
            y, p = c
            r = ig.integrate(y, p, dt, net=net, opt=opt)
            temp = net.rhs(r.y, r.params)[1]["T"]
            return (r.y, r.params), (r.y, temp, r.failed)
        return jax.lax.scan(step, (y0, p), dts)[1]
    y, temp, failed = (np.moveaxis(np.asarray(a), 0, 1)
                       for a in go(jnp.asarray(y0), p, jnp.asarray(np.diff(t))))
    errs = val.rel_errors(np.concatenate([y, temp[..., None]], -1), ref[:, 1:])
    eq = cp[:, 0] >= 1e3                   # n = 1e3 is in equilibrium at the end
    return (temp, {q: float(e.max()) for q, e in errs.items()},
            {q: float(e[eq, -1].max()) if eq.any() else np.nan for q, e in errs.items()}, failed)


def cooling_times(temp, t, thr=1e5):
    """First time T falls below thr (log-interpolated between outputs); nan if it never does."""
    out = np.full(temp.shape[0], np.nan)
    for c, T in enumerate(temp):
        k = np.where((T[:-1] >= thr) & (T[1:] < thr))[0]
        if len(k) and t[k[0]] > 0:
            k = k[0]
            f = np.log(T[k] / thr) / np.log(T[k] / T[k + 1])
            out[c] = t[k] * (t[k + 1] / t[k]) ** f
    return out


def _cooling_time_error(temp):
    """Max relative error of t(T < 1e5 K) against DVODE at rtol 1e-8, and DVODE's at 1e-4."""
    d = np.load(TRAJ)
    t, ref = d["t"], d["ref"][..., 4].astype(float)
    tc_ref = cooling_times(ref, t)
    tc = cooling_times(np.concatenate([ref[:, :1], temp], 1), t)
    tc_dvode = cooling_times(d["T_dvode_rtol1e-4"].astype(float), t)
    cooled = ~np.isnan(tc_ref)
    assert cooled.sum() >= 30
    err = np.abs(tc / tc_ref - 1)[cooled]
    return (np.inf if np.isnan(err).any() else float(err.max()),
            float(np.max(np.abs(tc_dvode / tc_ref - 1)[cooled])))


def test_default_tolerance_cooling_times_and_equilibria():
    """Default Rodas3: cooling times at least as good as DVODE's at its own rtol 1e-4."""
    temp, rows, eq, failed = _run(networks.NET5_CIE)
    assert not failed.any()
    err, dvode = _cooling_time_error(temp)
    print(f"t(T < 1e5 K): Rodas3 {err:.2e}, DVODE rtol 1e-4 {dvode:.2e}; equilibrium {eq}")
    assert err <= dvode
    for q in val.QTY:
        assert eq[q] < EQ_TARGET, q


def test_tight_tolerance_matches_dvode_at_every_output():
    """Both integrators converged (rtol 1e-8): every output of the runaway-cooling runs."""
    temp, rows, eq, failed = _run(networks.NET5_CIE, TIGHT, TIGHT_CASES)
    assert not failed.any()
    print(rows)
    for q in val.QTY:
        assert rows[q] < TARGET, q


def test_teeth_trajectories_without_cie():
    """The same runs integrated without cie_cool miss the reference (most never cool)."""
    temp, rows, eq, failed = _run(networks.NET5)
    err, dvode = _cooling_time_error(temp)
    worst = max(rows.values())
    print(f"without cie_cool: cooling times {err:.2e}, worst rel err {worst:.2e}")
    assert not err <= dvode and worst > TARGET


# --------------------------------------------------------------------------
# derivatives
# --------------------------------------------------------------------------
HOT = [  # (x_H2, x_H+, T0, n, z): hot and warm gas, every He / WSS regime
    (1e-7, 0.97, 2.03e4, 1.0, 1.0), (1e-8, 0.999, 3.07e5, 0.1, 0.3),
    (1e-9, 0.9999, 2.11e6, 10.0, 1.0), (1e-9, 0.9999, 4.3e7, 1.0, 3.0),
    (1e-4, 0.3, 9.13e3, 100.0, 1.0)]


@pytest.mark.parametrize("k", range(len(HOT)))
def test_jacfwd_matches_finite_differences(k):
    xh2, xhp, T, n, z = HOT[k]
    p = rates_net5.make_params(n, 1.7, tdust=12.0, metallicity=z)
    ntot = (1 + tb.ABHE - xh2 + xhp + 1.41e-4) * n
    y0 = jnp.array([xh2, xhp, 0.0, 1.5 * tb.KBOLTZ * T * ntot])
    f = lambda y: rates_net5.rhs(y, p, cie_cool=True)[0]  # noqa: E731
    J = np.asarray(jax.jacfwd(f)(y0))
    fd = np.zeros_like(J)
    for j in (0, 1, 3):
        h = 1e-7 * abs(float(y0[j]))
        e = np.zeros(4)
        e[j] = h
        fd[:, j] = (np.asarray(f(y0 + e)) - np.asarray(f(y0 - e))) / (2 * h)
    for i in (0, 1, 3):
        cols = [0, 1, 3]
        scale = np.max(np.abs(J[i, cols] * np.asarray(y0)[cols])) + 1e-300
        err = np.max(np.abs((J[i, cols] - fd[i, cols]) * np.asarray(y0)[cols])) / scale
        print(f"state {k} row {i}: |J - FD| / scale = {err:.2e}")
        assert err < 1e-6
