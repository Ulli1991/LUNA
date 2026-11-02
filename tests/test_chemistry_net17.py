"""The JAX port of chemcool network 17 against the Fortran, function by function, in both fixes17 modes.

Golden values: $LUNA_DATA/chemistry/chemistry_net17_golden.npz and _fixed_golden.npz; FP64, 1e-10.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


from luna.chemistry import cooling, photo, rates_net17, thermo  # noqa: E402
from luna.chemistry import tables as tb  # noqa: E402
from tests.test_chemistry_net5 import pix, rel_err, scaled_err  # noqa: E402

RTOL = 1e-10
# mode -> (golden data, full trajectories, rhs keywords); the fixed Fortran keeps the energy
# table (fix 5, the Newton EOS, is LUNA's own), so the comparison runs eos="table"
MODES = {
    "chemcool": (dict(np.load(tb.data_path("chemistry_net17_golden.npz"))),
                 "net17_trajectories.npz", {}),
    "fixed": (dict(np.load(tb.data_path("chemistry_net17_fixed_golden.npz"))),
              "net17_fixed_trajectories.npz",
              {"fixes17": "fixed", "eos": "table"}),
}
G = MODES["chemcool"][0]


@pytest.fixture(autouse=True)
def fresh_tables():
    """Every test starts (and ends) with clean table caches."""
    def clear():
        cooling._CACHE.clear()
        thermo._TEMP = thermo._GAMMA = None
        for f in (tb.chem_table, tb.chem_table17, tb.chem_table17_fixed, tb.cool_table,
                  tb.cool_table_hd, tb.co_tables, tb.energy_table, tb.raw):
            f.cache_clear()
    clear()
    yield
    clear()


# --------------------------------------------------------------------------
# the comparisons (each returns the max error; tests assert <= RTOL)
# --------------------------------------------------------------------------
def _mode(mode):
    g, _, kw = MODES[mode]
    return g, kw, kw.get("fixes17", "chemcool")


def err_tables(mode="chemcool"):
    G, _, fixes17 = _mode(mode)
    every = int(G["table_every"])
    T = tb.temptab()
    ch = {**tb.chem_table(), **tb.chem_table17()}
    if fixes17 == "fixed":
        ch.update(tb.chem_table17_fixed())
    rows = [int(r) for r in G["chtab_rows"]]
    nr = len(rows)
    errs = [rel_err(T[::every], G["temptab"])]
    dT = np.diff(T)[::every][:-1]
    for i, r in enumerate(rows):
        v, s = ch[r][0][::every], ch[r][1][::every]
        errs.append(rel_err(v, G["chtab"][:, i]))
        errs.append(scaled_err(s[:-1] * dT, G["chtab"][:-1, nr + i] * dT,
                               np.abs(v[:-1]) + np.abs(s[:-1] * dT)))
    hd, c64 = tb.cool_table_hd(), tb.cool_table()[64][0][::every]
    for i, r in enumerate(G["cltab_rows"]):
        if r in hd:
            errs.append(rel_err(hd[r][0][::every], G["cltab"][:, i]))
        elif r == 64:
            errs.append(rel_err(c64, G["cltab"][:, i]))
        else:                              # He rows: zero under iflag_atom = 2
            errs.append(rel_err(0.0, G["cltab"][:, i]))
    zeta = 3e-17
    errs.append(rel_err([zeta, 2 * zeta, rates_net17.CR_HE * zeta], G["crtab"]))
    errs.append(rel_err([rates_net17.CST1, rates_net17.CST75], G["cst"]))
    ph = [photo.K_H2_THIN, rates_net17.PH4_THIN, rates_net17.PH5_THIN, 0.0, 0.0,
          photo.K_CO_THIN, rates_net17.PH35_THIN] + [0.0] * 8
    errs.append(rel_err(ph, G["phtab"]))
    return max(errs)


def err_rate_coefficients(mode="chemcool"):
    G, _, fixes17 = _mode(mode)
    ch = rates_net17.chem_coefficients(jnp.asarray(G["chinterp_in"][:, 0]), fixes17)
    return max(rel_err(ch[int(r)], G["chinterp_out"][:, i])
               for i, r in enumerate(G["chtab_rows"]))


def _photo_inputs(x):
    """Params and photo state from columns g0, dgr, NH, NH2, NCO, n, dl, xH2, xHD, xCO, T, D."""
    c = [jnp.asarray(x[:, i]) for i in range(x.shape[1])]
    p = rates_net17.Params(n_h=c[5], n_h_col=pix(c[2]), n_h2_col=pix(c[3]), n_co_col=pix(c[4]),
                           g0=c[0], g0_dust=c[0], zeta=3e-17, dgr=c[1], abundc=1.41e-4,
                           abundo=3.16e-4, abund_d=c[11], dl=c[6])
    return p, c[10], c[7], c[8]


def err_photo(mode="chemcool"):
    G, _, fixes17 = _mode(mode)
    p, t, xh2, xhd = _photo_inputs(G["photo_in"])
    ph = rates_net17.photo_rates(p, t, xh2, xhd, fixes17)
    out = G["photo_out"]
    e = [rel_err(ph[k], out[:, i]) for i, k in
         ((0, "k_h2_dust"), (1, "k_hm"), (2, "k_h2p"), (5, "k_co"), (6, "k_hd"), (15, "g_dust"))]
    e += [rel_err(0.0, out[:, i]) for i in (3, 4) + tuple(range(7, 15))]
    lit = G["photo_in"][:, 0] > 0          # the Fortran leaves chi stale when G0 = 0
    e.append(rel_err(np.asarray(ph["chi"])[lit], out[lit, 16]))
    return max(e)


def err_hd_shielding(mode="chemcool"):
    G, _, fixes17 = _mode(mode)
    x = G["hdshield_in"]
    f = photo.hd_self_shielding(jnp.asarray(x[:, 2] * x[:, 3]), pix(x[:, 4]), jnp.asarray(x[:, 1]),
                                jnp.asarray(x[:, 2]), jnp.asarray(x[:, 5]), jnp.asarray(x[:, 0]),
                                weight_both=fixes17 == "fixed")
    return rel_err(f, G["hdshield_out"][:, 0])


def _cool_func(G):
    x = jnp.asarray(G["coolfunc_in"])
    c = [x[:, i] for i in range(x.shape[1])]
    ab = {"h2": c[5], "o": c[6], "co": c[7], "cp": c[8], "si": c[9], "e": c[10], "hp": c[11],
          "hi": c[12], "hd": c[13], "hei": c[14]}
    return cooling.cool_func(c[0], c[1], c[2], c[3], c[4], ab, c[17], c[18], c[19], c[20],
                             c[21], c[22])


def err_cool_terms(mode="chemcool"):
    G = _mode(mode)[0]
    rates, tdust = _cool_func(G)
    rates = np.array(rates)
    out = G["coolfunc_out"].copy()
    # above ~1.5e7 K the gas-grain table is denormal: XLA flushes it to zero
    T = G["coolfunc_in"][:, 0]
    i, _ = cooling._node_index(jnp.asarray(T), clamp_top=False)
    denorm = np.abs(tb.cool_table()[14][0][np.asarray(i)]) < np.finfo(float).tiny
    flushed = denorm & (rates[:, 0] == 0.0)
    out[flushed, 0] = rates[flushed, 0] = 1.0
    e = [rel_err(rates[:, k], out[:, k]) for k in range(28)]
    e.append(rel_err(tdust, out[:, 28]))
    return max(e)


def _rateeq_params(x, t_photo):
    c = [jnp.asarray(x[:, i]) for i in range(x.shape[1])]
    p = rates_net17.Params(n_h=c[0], n_h_col=pix(c[10]), n_h2_col=pix(c[11]),
                           n_co_col=pix(c[12]), g0=c[9], g0_dust=c[9], zeta=c[15], dgr=c[16],
                           abundc=c[17], abundo=c[18], abund_d=c[19], dl=c[13], divv=c[14],
                           tdust=c[20], photo_temp=jnp.asarray(t_photo), photo_x_h2=c[2],
                           photo_x_hd=c[8])
    y = jnp.stack([c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[1]], axis=-1)
    return y, p


def err_eqb(mode="chemcool"):
    G, kw, _ = _mode(mode)
    y, p = _rateeq_params(G["rateeq_in"], G["rateeq_out"][:, 69])
    _, aux = rates_net17.rhs(y, p, **kw)
    eq, out = aux["eqb"], G["eqb_out"]
    e = [rel_err(eq[k], out[:, i]) for i, k in enumerate(("hm", "h2p", "hehp", "e", "hi", "hei",
                                                          "m"))]
    # where eqtol * x is denormal for some species x, XLA flushes the convergence
    # threshold to zero and the port sweeps on (to the same abundances)
    tiny = np.finfo(float).tiny
    flush = np.any((out[:, :3] > 0) & (rates_net17.EQB_TOL * out[:, :3] < tiny), 1)
    # likewise where the HeH+ formation term (= HeH+ x its destruction rate) is denormal
    ch = rates_net17.chem_coefficients(aux["T"])
    num = out[:, 2] * np.asarray(ch[156] * out[:, 4] + ch[153] * out[:, 3])
    flush |= (out[:, 2] > 0) & (num < tiny)
    differ = (np.asarray(eq["sweeps"]) != out[:, 7]) & ~flush
    e.append(np.inf if np.any(differ) else 0.0)
    e += [rel_err(aux["h2var0"], out[:, 8]), rel_err(aux["ph4"], out[:, 9]),
          rel_err(aux["ph5"], out[:, 10]), rel_err(aux["alpha_h2p"], out[:, 11]),
          rel_err(aux["T"], out[:, 12])]
    return max(e)


def err_rhs(mode="chemcool", fn=None):
    G, kw, _ = _mode(mode)
    y, p = _rateeq_params(G["rateeq_in"], G["rateeq_out"][:, 69])
    ydot, aux = rates_net17.rhs(y, p, **kw) if fn is None else fn(y, p)
    out = G["rateeq_out"]
    ok = out[:, 67] == 0                    # ipar(1): rejected states return ydot = 0
    assert bool(np.all(np.asarray(aux["invalid"]) == ~ok))
    ph = aux["photo"]
    e = [rel_err(np.asarray(ydot)[~ok], 0.0 * out[~ok, :8]),
         scaled_err(ydot, out[:, :8], np.asarray(aux["gross"])),
         rel_err(np.asarray(aux["tdust"])[ok], out[ok, 58]),
         rel_err(np.asarray(aux["T"])[ok], out[ok, 59]),
         rel_err(np.asarray(aux["gamma"])[ok], out[ok, 68])]
    e += [rel_err(np.asarray(aux["cool"])[ok, k], out[ok, 8 + k]) for k in range(28)]
    e += [rel_err(np.asarray(aux["chem"])[ok, k], out[ok, 36 + k]) for k in range(22)]
    e += [rel_err(ph[k], out[:, i]) for i, k in ((60, "k_h2_dust"), (61, "k_hm"), (62, "k_h2p"),
                                                   (63, "k_co"), (64, "k_hd"), (65, "g_dust"))]
    return max(e)


def _traj_states(rows, prev, params, fixed):
    n, _, g0, nh, nh2, nco, z = (params[:, i] for i in range(7))
    a = lambda v: jnp.asarray(v, jnp.float64)  # noqa: E731
    p = rates_net17.Params(
        n_h=a(n), n_h_col=pix(a(nh)), n_h2_col=pix(a(nh2)), n_co_col=pix(a(nco)), g0=a(g0),
        g0_dust=a(g0), zeta=float(fixed["zeta"]), dgr=a(z),
        abundc=a(float(fixed["abundc_solar"]) * z), abundo=a(float(fixed["abundo_solar"]) * z),
        abund_d=float(fixed["abundD"]), tdust=a(rows[:, 10]), photo_temp=a(rows[:, 69]),
        photo_x_h2=a(prev[:, 0]), photo_x_hd=a(prev[:, 1]))
    return a(rows[:, 1:9]), p


def traj_errors(rows, prev, params, fixed, kw):
    y, p = _traj_states(rows, prev, params, fixed)
    ydot, aux = jax.jit(lambda y, p: rates_net17.rhs(y, p, **kw))(y, p)   # fresh trace: sees patches
    ph = aux["photo"]
    return {"ydot": scaled_err(ydot, rows[:, 11:19], np.asarray(aux["gross"])),
            "cool": rel_err(aux["cool"], rows[:, 19:47]),
            "chem": rel_err(aux["chem"], rows[:, 47:69]),
            "photo": max(rel_err(ph[k], rows[:, 70 + i]) for i, k in enumerate(
                ("k_h2_dust", "k_hm", "k_h2p", "k_co", "k_hd", "g_dust")))}


def err_trajectories(mode="chemcool"):
    G, kw, _ = _mode(mode)
    fixed = {"zeta": G["traj_fixed_zeta"], "abundD": G["traj_fixed_abundD"],
             "abundc_solar": G["traj_abundc_solar"], "abundo_solar": G["traj_abundo_solar"]}
    e = traj_errors(G["traj_rows"], G["traj_prev"], G["traj_params"][G["traj_case"]], fixed, kw)
    return max(e.values())


CHECKS = {
    "tables": err_tables, "rate_coefficients": err_rate_coefficients, "photo": err_photo,
    "hd_shielding": err_hd_shielding, "cool_terms": err_cool_terms, "eqb": err_eqb,
    "rhs": err_rhs, "trajectories": err_trajectories,
}


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("name", list(CHECKS))
def test_matches_fortran(name, mode):
    err = CHECKS[name](mode)
    print(f"{mode} {name}: max error {err:.3e}")
    assert err <= RTOL, f"{mode} {name}: {err:.3e} > {RTOL}"


@pytest.mark.parametrize("mode", list(MODES))
def test_all_trajectory_states(mode):
    """The RHS at every output of every one-zone reference run."""
    try:
        full = tb.data_path(MODES[mode][1])
    except FileNotFoundError:
        pytest.skip("full reference not in $LUNA_DATA/chemistry")
    with np.load(full) as z:
        fixed = {"zeta": z["fixed_zeta"], "abundD": z["fixed_abundD"],
                 "abundc_solar": z["abundc_solar"], "abundo_solar": z["abundo_solar"]}
        rows, prev, params = [], [], []
        for i, name in enumerate(z["case_names"]):
            tr = z["traj_" + str(name)]
            rows.append(tr)
            prev.append(tr[np.maximum(np.arange(len(tr)) - 1, 0)][:, [1, 7]])
            params.append(np.repeat(z["case_params"][i][None], len(tr), 0))
    e = traj_errors(np.concatenate(rows), np.concatenate(prev), np.concatenate(params), fixed,
                    MODES[mode][2])
    print(f"{mode}, {sum(len(r) for r in rows)} states:", {k: f"{v:.2e}" for k, v in e.items()})
    assert max(e.values()) <= RTOL


# --------------------------------------------------------------------------
# teeth: break one thing, the matching comparison must fail
# --------------------------------------------------------------------------
def _patched_row(table_fn, row, how):
    orig = table_fn.__wrapped__

    def broken():
        d = dict(orig())
        v, s = d[row]
        d[row] = (np.roll(v, 1), np.roll(s, 1)) if how == "shift" else (v * how, s * how)
        return d
    return broken


def _patch_cache(mp, key, builder):
    mp.setitem(cooling._CACHE, key, cooling._arrays(builder()))


BREAKS = {
    # table rows shifted by one node
    "ch161_shifted": ("rate_coefficients", lambda mp: _patch_cache(
        mp, "ch17", _patched_row(tb.chem_table17, 161, "shift"))),
    "cl53_shifted": ("cool_terms", lambda mp: _patch_cache(
        mp, "hd", _patched_row(tb.cool_table_hd, 53, "shift"))),
    # a rate 1 % off
    "ch98_1pct": ("rhs", lambda mp: _patch_cache(
        mp, "ch17", _patched_row(tb.chem_table17, 98, 1.01))),
    "ch24_1pct": ("eqb", lambda mp: _patch_cache(
        mp, "ch17", _patched_row(tb.chem_table17, 24, 1.01))),
    "ch99_1pct_trajectories": ("trajectories", lambda mp: _patch_cache(
        mp, "ch17", _patched_row(tb.chem_table17, 99, 1.01))),
    "cst1_1pct": ("eqb", lambda mp: mp.setattr(rates_net17, "CST1", 6.4e-10 * 1.01)),
    "ph35_1pct": ("photo", lambda mp: mp.setattr(rates_net17, "PH35_THIN", 3.3e-11 * 1.01)),
    "ph4_1pct": ("photo", lambda mp: mp.setattr(rates_net17, "PH4_THIN", 4.2e-7 * 1.01)),
    "h2_shield_no_local_column": ("rhs", lambda mp: mp.setattr(rates_net17, "NH_LOCAL", 0.0)),
    "uv_pumps_net5": ("rhs", lambda mp: mp.setattr(rates_net17, "UV_PUMPS", 8.5)),
    # single-precision literals read in double
    "f32_literals_in_double": ("tables", lambda mp: mp.setattr(tb, "f32", lambda x: float(x))),
    "abhe_in_double": ("rhs", lambda mp: mp.setattr(rates_net17, "ABHE", 0.079)),
    "msm_shift_in_double": ("rhs", lambda mp: mp.setattr(rates_net17, "MSM_SHIFT",
                                                         (-1.164408, -1.174242))),
    # the equilibrium loop: one sweep only
    "eqb_one_sweep": ("eqb", lambda mp: mp.setattr(rates_net17, "N_EQB_ITER", 1)),
    # HD shielding with the pixel weight on both terms
    "hd_shield_weighted": ("hd_shielding", lambda mp: mp.setattr(
        photo, "hd_self_shielding", _weighted_shield(photo.hd_self_shielding))),
}

# each fix undone, one at a time: the fixed-mode comparison must notice
FIXED_BREAKS = {
    "fix1_no_hd_cooling": ("rhs", lambda mp: mp.setattr(
        cooling, "_hd_cooling", lambda temp, n_h, ynh, x_hd: 0.0 * temp)),
    "fix1_no_hd_cooling_trajectories": ("trajectories", lambda mp: mp.setattr(
        cooling, "_hd_cooling", lambda temp, n_h, ynh, x_hd: 0.0 * temp)),
    "fix2_nh_half": ("rhs", lambda mp: mp.setattr(rates_net17, "NH_LOCAL_FIXED", 0.5)),
    "fix3_first_term_weight_only": ("hd_shielding", lambda mp: mp.setattr(
        photo, "hd_self_shielding", _first_weight_only(photo.hd_self_shielding))),
    "fix4_h2p_with_df05": ("photo", lambda mp: mp.setattr(rates_net17, "DUST_EXP_H2P", 0.5)),
    "fix4_hm_with_df19": ("rhs", lambda mp: mp.setattr(rates_net17, "DUST_EXP_HM", 1.9)),
    "fix6_ch100_unclamped": ("tables", lambda mp: mp.setattr(
        tb, "chem_table17_fixed", lambda: {100: tb.chem_table17()[100]})),
}


def _first_weight_only(orig):
    def f(*a, weight_both=False):
        return orig(*a, weight_both=False)
    return f


def _weighted_shield(orig):
    def f(n_hd_local, n_h2_col, x_h2, x_hd, abund_d, temp, **_):
        n = n_h2_col.shape[-1]
        per = [orig(n_hd_local, n_h2_col[..., i:i + 1], x_h2, x_hd, abund_d, temp)
               for i in range(n)]
        return sum(per) / n
    return f


@pytest.mark.parametrize("brk", list(BREAKS))
def test_teeth(brk, monkeypatch):
    check, apply = BREAKS[brk]
    apply(monkeypatch)
    err = CHECKS[check]()
    print(f"broken {brk}: {check} error {err:.3e}")
    assert err > RTOL, f"breaking {brk} went unnoticed by the {check} comparison"


@pytest.mark.parametrize("brk", list(FIXED_BREAKS))
def test_teeth_fixed(brk, monkeypatch):
    check, apply = FIXED_BREAKS[brk]
    assert CHECKS[check]("fixed") <= RTOL
    cooling._CACHE.clear()
    apply(monkeypatch)
    err = CHECKS[check]("fixed")
    print(f"broken {brk}: fixed {check} error {err:.3e}")
    assert err > RTOL, f"undoing {brk} went unnoticed by the fixed {check} comparison"


# --------------------------------------------------------------------------
# differentiability and jit
# --------------------------------------------------------------------------
# e = 1.5 k T0 n_tot keeps e/(k n_tot) = 1.5 T0 off the integer nodes of the energy
# table (T(e) has kinks there), and x_H2 off the 0.1 blocks
STATES = [  # (x_H2, x_H+, x_CO, x_He+, x_He++, x_D+, x_HD, T0), n, G0, shielding, Z
    ((1e-3, 1e-4, 2e-9, 1e-6, 1e-12, 1e-9, 1e-8, 100.37), 1e2, 1.7, "thin", 1.0),
    ((0.31, 2e-6, 5e-7, 1e-8, 1e-15, 1e-11, 2e-5, 30.21), 1e4, 1.7, "av1", 1e-2),
    ((0.45, 1e-7, 1e-12, 1e-10, 0.0, 1e-12, 2.5e-5, 15.13), 1e5, 0.0, "thin", 1e-4),
    ((1e-6, 5e-2, 1e-14, 3e-3, 1e-5, 1.3e-6, 1e-12, 8000.3), 1.0, 1.7, "thin", 0.0),
    ((0.11, 1e-3, 1e-6, 1e-5, 1e-9, 2.6e-8, 1e-6, 300.29), 1e3, 1.7, "av1", 1.0),
    ((2e-3, 3e-4, 0.0, 2e-6, 0.0, 7e-9, 2e-9, 2200.7), 1e2, 1.7, "thin", 0.0),
]
SHIELD = {"thin": (0.0, 0.0, 0.0), "av1": (1.87e21, 5e20, 1e15)}


def _state(s):
    x, n, g0, sh, z = s
    p = rates_net17.make_params(n, g0, *SHIELD[sh], z=z, tdust=12.0)
    xe = x[1] + x[3] + 2 * x[4] + (p.abundc - x[2]) + x[5]
    e = 1.5 * tb.KBOLTZ * x[7] * (1 + rates_net17.ABHE + xe - x[0]) * n
    y = jnp.array(list(x[:7]) + [e])
    t_ph, xh2, xhd = rates_net17.photo_state(y, p)
    return y, p._replace(photo_temp=t_ph, photo_x_h2=xh2, photo_x_hd=xhd)


@pytest.mark.parametrize("k", range(len(STATES)))
def test_jacfwd_matches_finite_differences(k):
    y0, p = _state(STATES[k])
    f = lambda y: rates_net17.rhs(y, p)[0]  # noqa: E731
    J = np.asarray(jax.jacfwd(f)(y0))
    assert np.all(np.isfinite(J))
    fd = np.zeros_like(J)
    for j in range(8):
        h = 1e-7 * max(abs(float(y0[j])), 1e-14 if j < 7 else 1e-20)
        e = np.zeros(8)
        e[j] = h
        fd[:, j] = (np.asarray(f(y0 + e)) - np.asarray(f(y0 - e))) / (2 * h)
    yv = np.asarray(y0)
    for i in range(8):
        scale = np.max(np.abs(J[i] * yv)) + 1e-300
        err = np.max(np.abs((J[i] - fd[i]) * yv)) / scale
        print(f"state {k} row {i}: |J - FD| / scale = {err:.2e}")
        assert err < 1e-7


@pytest.mark.parametrize("mode", list(MODES))
def test_jit_matches_fortran(mode):
    """The jitted RHS (XLA may reorder sums) against rate_eq."""
    kw = MODES[mode][2]
    err = err_rhs(mode, jax.jit(lambda y, p: rates_net17.rhs(y, p, **kw)))
    print(f"{mode} jitted rhs: max error {err:.3e}")
    assert err <= RTOL
