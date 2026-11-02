"""The GOW17 corrections (fixes="gow17") against the published formulae and tables.

Each check returns an error; the teeth tests break one ingredient and require its check to fail.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


from luna.chemistry import gow17, rates_net5  # noqa: E402
from luna.chemistry import tables as tb  # noqa: E402

GOLDEN = tb.data_path("chemistry_net5_golden.npz")
G = dict(np.load(GOLDEN))
RTOL = 1e-12

# Visser, van Dishoeck & Black 2009, Table 5, 12CO block: rows log N(H2) = 0, 19..23,
# columns log N(12CO) = 0, 13..19 (as printed in the paper)
V09_LOG_NCO = [0, 13, 14, 15, 16, 17, 18, 19]
V09_LOG_NH2 = [0, 19, 20, 21, 22, 23]
V09_TABLE5 = np.array([
    [1.000, 8.080e-1, 5.250e-1, 2.434e-1, 5.467e-2, 1.362e-2, 3.378e-3, 5.240e-4],
    [8.176e-1, 6.347e-1, 3.891e-1, 1.787e-1, 4.297e-2, 1.152e-2, 2.922e-3, 4.662e-4],
    [7.223e-1, 5.624e-1, 3.434e-1, 1.540e-1, 3.515e-2, 9.231e-3, 2.388e-3, 3.899e-4],
    [3.260e-1, 2.810e-1, 1.953e-1, 8.726e-2, 1.907e-2, 4.768e-3, 1.150e-3, 1.941e-4],
    [1.108e-2, 1.081e-2, 9.033e-3, 4.441e-3, 1.102e-3, 2.644e-4, 7.329e-5, 1.437e-5],
    [3.938e-7, 3.938e-7, 3.936e-7, 3.923e-7, 3.901e-7, 3.893e-7, 3.890e-7, 3.875e-7]])


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.abs(b)))


def pix(x, n=12):
    return jnp.broadcast_to(jnp.asarray(x, jnp.float64)[..., None], np.shape(x) + (n,))


# --- independent transcriptions of the published formulae -----------------------------
def wd01_he(T, psi, zd):
    """GOW17 Table 2 #4 (WD01), written out."""
    return 5.572e-14 * zd / (1 + 3.185e-7 * psi**1.512 * (
        1 + 5115 * T**3.903e-7 * psi**(-0.4956 - 5.494e-7 * np.log(T))))


def he_case_b(T):
    lt = np.log10(T)
    return 1e-11 * T**-0.5 * (11.19 - 1.676 * lt - 0.2852 * lt**2 + 0.04433 * lt**3)


def k_cr_co_ref(zeta, T, n, x_hi, x_h2, x_e, x_co, k_gr):
    """CO destruction per CO by cosmic rays, from GOW17 Tables 1-2 and App. A.1."""
    f = 2 * x_h2 / (x_hi + 2 * x_h2)
    xi = zeta * (x_hi + 2 * x_h2) / (1.5 * x_hi + 2.3 * x_h2)     # T2 #6-7: equal total ionisation
    sinks = k_gr + (1.26e-13 * np.exp(-22.5 / T) + 7.2e-15) * x_h2 + he_case_b(T) * x_e + 1.6e-9 * x_co
    x_hep = 1.1 * xi * float(np.float32(0.1)) / (n * sinks)
    return 1.6e-9 * x_hep * n + 90 * xi * f


# --- the checks ------------------------------------------------------------------------
def err_visser_table5():
    lco, lh2 = np.meshgrid(V09_LOG_NCO, V09_LOG_NH2)
    got = gow17.co_shielding_visser09(jnp.asarray(10.0**lco), jnp.asarray(10.0**lh2))
    return rel(got, V09_TABLE5)


def err_visser_interpolation():
    """At log-midpoints of the fine grid: geometric mean of the four corners."""
    ax, ay, lt = gow17.visser09_table()
    i, j = np.meshgrid(np.arange(1, ax.size - 1), np.arange(1, ay.size - 1))
    mx, my = 0.5 * (ax[i] + ax[i + 1]), 0.5 * (ay[j] + ay[j + 1])
    want = 10.0 ** (0.25 * (lt[j, i] + lt[j, i + 1] + lt[j + 1, i] + lt[j + 1, i + 1]))
    got = gow17.co_shielding_visser09(jnp.asarray(10.0**mx), jnp.asarray(10.0**my))
    return rel(got, want)


AV_NODES = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0])


def err_co_photo_dust():
    """No line shielding: 2.4e-10 chi exp(-3.88 A_V), chi = G0 / 1.7, per pixel and as a pixel mean."""
    g0 = jnp.asarray([1.7, 10.0, 1.7, 1.7, 1.7, 1.7])
    n_h = AV_NODES / 5.348e-22
    got = gow17.co_photo_rate(g0, pix(n_h), pix(0.0 * n_h), pix(0.0 * n_h), jnp.ones(6))
    want = 2.4e-10 * np.asarray(g0) / 1.7 * np.exp(-3.88 * AV_NODES)
    two = jnp.stack([jnp.zeros(6), jnp.asarray(n_h)], -1)      # two pixels: thin and A_V
    got2 = gow17.co_photo_rate(g0, two, 0 * two, 0 * two, jnp.ones(6))
    want2 = 2.4e-10 * np.asarray(g0) / 1.7 * 0.5 * (1 + np.exp(-3.88 * AV_NODES))
    return max(rel(got, want), rel(got2, want2))


def err_co_photo_shielded():
    """Visser Theta times dust: N_CO = 1e15, N_H2 = 1e21 (Theta = 8.726e-2) at A_V = 1."""
    got = gow17.co_photo_rate(1.7, pix(1.0 / 5.348e-22), pix(1e21), pix(1e15), 1.0)
    return rel(got, 2.4e-10 * 8.726e-2 * np.exp(-3.88))


T_GRID = np.array([10.0, 20.0, 50.0, 100.0, 300.0, 1e3, 8e3, 1e5])
PSI = np.array([1e-6, 1e-3, 1.0, 1e2, 1e4, 1e6])


def err_he_grain_formula():
    T, psi = np.meshgrid(T_GRID, PSI)
    got = gow17.he_grain_recombination(jnp.asarray(T), jnp.asarray(psi), 0.7)
    low = gow17.he_grain_recombination(20.0, 1e-8, 0.7)          # chemcool's psi < 1e-6 branch
    return max(rel(got, wd01_he(T, psi, 0.7)), rel(low, 5.572e-14 * 0.7))


def err_he_grain_table():
    """chtab(135/136) on chemcool's T grid, linear in T between nodes as rate_eq interpolates."""
    c135, c136 = gow17.he_grain_table()
    T = tb.temptab()
    e = [rel(c135, 5115 * T**3.903e-7), rel(c136, -0.4956 - 5.494e-7 * np.log(T))]
    Tq = np.geomspace(10.0, 1e7, 997)
    co = (jnp.asarray(np.interp(Tq, T, c135)), jnp.asarray(np.interp(Tq, T, c136)))
    psi = 30.0
    tab = gow17.he_grain_recombination(jnp.asarray(Tq), psi, 1.0, coeffs=co)
    e.append(rel(tab, wd01_he(Tq, psi, 1.0)))
    return max(e)


def err_he_grain_hook():
    """he_grain_rate (the network-17 hook) = WD01 at chemcool's phi = G_dust sqrt(T) / (n x_e)."""
    T, g, n, xe = np.meshgrid([15.0, 80.0, 3e3], [0.0, 1.7, 30.0], [1.0, 1e3], [0.0, 1e-4, 0.1])
    got = gow17.he_grain_rate(*(jnp.asarray(v) for v in (g, T, n, xe)), 1.0)
    phi = np.where(xe == 0, 1e20, g * np.sqrt(T) / (n * np.where(xe == 0, 1, xe)))
    want = np.where(phi < 1e-6, 5.572e-14, wd01_he(T, np.maximum(phi, 1e-6), 1.0))
    return rel(got, want)


def err_xi_conversion():
    x_h2 = np.array([0.0, 0.5, 0.25, 0.1, 0.49])
    x_hi = 1.0 - 2.0 * x_h2                   # neutral gas: the conversion matches total ionisation
    zeta = 3e-16
    xi = np.asarray(gow17.xi_from_zeta(zeta, jnp.asarray(x_hi), jnp.asarray(x_h2)))
    gow_total = xi * (1.5 * x_hi + 2.3 * x_h2) * (x_hi + 2 * x_h2)      # T2 #6 + 2 x #7, per H
    chemcool_total = zeta * x_hi + 2 * zeta * x_h2                       # cr1 + cr2, per H
    return max(rel(gow_total, chemcool_total), rel(xi[0], 2e-16), rel(xi[1], 3e-16 / 1.15))


def _golden_rhs(fixes, **kw):
    x = G["rateeq_in"]
    c = [jnp.asarray(x[:, i]) for i in range(17)]
    p = rates_net5.Params(n_h=c[0], n_h_col=pix(c[6]), n_h2_col=pix(c[7]), n_co_col=pix(c[8]),
                          g0=c[5], g0_dust=c[5], zeta=c[11], dgr=c[12], abundc=c[13],
                          abundo=c[14], abundsi=c[15], dl=c[9], divv=c[10], tdust=c[16])
    y = jnp.stack([c[2], c[3], c[4], c[1]], axis=-1)
    return y, p, rates_net5.rhs(y, p, fixes=fixes, **kw)


def err_rhs_co_row(**kw):
    """gow17 minus chemcool dx_CO = -(k_CO,gow17 - k_CO,chemcool) x_CO - k_CR x_CO."""
    y, p, (yd_c, aux_c) = _golden_rhs("chemcool", **kw)
    _, _, (yd_g, aux_g) = _golden_rhs("gow17", **kw)
    ok = ~np.asarray(aux_c["invalid"])
    x_h2, x_hp, x_co = (np.clip(np.asarray(y[:, k]), 0, b) for k, b in ((0, .5), (1, 1), (2, 1)))
    x_hi = np.maximum(1 - 2 * x_h2 - x_hp, 0)
    x_e = x_hp + np.maximum(0, np.asarray(p.abundc) - x_co) + np.asarray(p.abundsi)
    T, n = np.asarray(aux_c["T"]), np.asarray(p.n_h)
    phi = np.where(x_e == 0, 1e20, np.asarray(aux_c["g_dust"]) * np.sqrt(T)
                   / (n * np.where(x_e == 0, 1, x_e)))
    k_gr = np.where(phi < 1e-6, 5.572e-14 * np.asarray(p.dgr),
                    wd01_he(T, np.maximum(phi, 1e-6), np.asarray(p.dgr)))
    k_cr = k_cr_co_ref(np.asarray(p.zeta), T, n, x_hi, x_h2, x_e, x_co, k_gr)
    want = -(np.asarray(aux_g["k_co"]) - np.asarray(aux_c["k_co"])) * x_co - k_cr * x_co
    got = np.asarray(yd_g[:, 2] - yd_c[:, 2])
    scale = np.asarray(aux_g["gross"][:, 2])
    sel = ok & (np.abs(want) > 1e-4 * scale) & (x_hi + x_h2 > 0)
    assert sel.sum() > 100
    return float(np.max(np.abs(got - want)[sel] / np.abs(want)[sel]))


# check -> (function, tolerance); the table one is linear interpolation between T nodes
CHECKS = {"visser_table5": (err_visser_table5, RTOL),
          "visser_interpolation": (err_visser_interpolation, RTOL),
          "co_photo_dust": (err_co_photo_dust, RTOL),
          "co_photo_shielded": (err_co_photo_shielded, RTOL),
          "he_grain_formula": (err_he_grain_formula, RTOL),
          "he_grain_table": (err_he_grain_table, 1e-9),
          "he_grain_hook": (err_he_grain_hook, RTOL),
          "xi_conversion": (err_xi_conversion, RTOL),
          "rhs_co_row": (err_rhs_co_row, 1e-10)}


@pytest.mark.parametrize("name", list(CHECKS))
def test_matches_published(name):
    fn, tol = CHECKS[name]
    err = fn()
    print(f"{name}: max error {err:.3e}")
    assert err <= tol, f"{name}: {err:.3e} > {tol}"


def _roll_visser(mp):
    ax, ay, lt = gow17.visser09_table()
    mp.setattr(gow17, "visser09_table", lambda: (ax, ay, np.roll(lt, 1, axis=1)))


BREAKS = {
    "co_dust_exponent_3.53": ("co_photo_dust", lambda mp: mp.setattr(gow17, "GAMMA_CO", 3.53)),
    "co_rate_vdb88": ("co_photo_dust", lambda mp: mp.setattr(gow17, "K_CO_THIN", 1.235e-10 * 1.7)),
    "visser_table_shift": ("visser_table5", _roll_visser),
    "visser_shift_in_rate": ("co_photo_shielded", _roll_visser),
    "xi_no_conversion": ("xi_conversion", lambda mp: mp.setattr(gow17, "SEC_ATOMIC", 1.0)),
    "xi_atomic_only": ("xi_conversion", lambda mp: mp.setattr(gow17, "SEC_MOLECULAR", 1.5)),
    "xi_no_conversion_rhs": ("rhs_co_row", lambda mp: mp.setattr(gow17, "SEC_ATOMIC", 1.0)),
    "he_cr_rate_1pct": ("rhs_co_row", lambda mp: mp.setattr(gow17, "CR_HE", 1.111)),
    "cr_photo_co_off": ("rhs_co_row", lambda mp: mp.setattr(gow17, "CR_PHOTO_CO", 0.0)),
    "he_grain_off": ("rhs_co_row", lambda mp: mp.setattr(
        gow17, "he_grain_recombination", lambda T, phi, dgr, coeffs=None: 0.0 * T)),
    "he_grain_phi_without_ne": ("he_grain_hook", lambda mp: mp.setattr(
        gow17, "grain_phi", lambda g, T, n, xe: g * jnp.sqrt(T) / n)),
    "he_grain_c135_1pct": ("he_grain_formula", lambda mp: mp.setattr(
        gow17, "he_grain_coefficients",
        lambda T: (5.115e3 * 1.01 * jnp.asarray(T)**3.903e-7,
                   -0.4956 - 5.494e-7 * jnp.log(jnp.asarray(T))))),
}


@pytest.mark.parametrize("brk", list(BREAKS))
def test_teeth(brk, monkeypatch):
    check, apply = BREAKS[brk]
    apply(monkeypatch)
    fn, tol = CHECKS[check]
    err = fn()
    print(f"broken {brk}: {check} error {err:.3e}")
    assert err > tol, f"breaking {brk} went unnoticed by {check}"


def test_chemcool_mode_unchanged():
    """fixes="chemcool" is the default; gow17 changes only the CO row (same state)."""
    y, p, (yd_c, aux_c) = _golden_rhs("chemcool")
    yd_d, aux_d = rates_net5.rhs(y, p)
    _, _, (yd_g, aux_g) = _golden_rhs("gow17")
    assert np.array_equal(np.asarray(yd_c), np.asarray(yd_d))
    for k in ("cool", "chem", "T", "tdust", "k_h2_dis", "g_dust", "k_co"):
        assert np.array_equal(np.asarray(aux_c[k]), np.asarray(aux_d[k]))
    for k in (0, 1, 3):
        assert np.array_equal(np.asarray(yd_c[:, k]), np.asarray(yd_g[:, k]))
    assert not np.array_equal(np.asarray(yd_c[:, 2]), np.asarray(yd_g[:, 2]))


@pytest.mark.parametrize("kw", [{"eos": "newton"}, {"cie_cool": True},
                                {"cie_cool": True, "eos": "newton"}], ids=["newton", "cie", "both"])
def test_gow17_with_newton_eos_and_cie(kw):
    """fixes="gow17" composes with eos="newton" and cie_cool: only the CO row changes, as published."""
    y, p, (yd_c, _) = _golden_rhs("chemcool", **kw)
    _, _, (yd_g, aux_g) = _golden_rhs("gow17", **kw)
    for k in (0, 1, 3):
        assert np.array_equal(np.asarray(yd_c[:, k]), np.asarray(yd_g[:, k]))
    assert np.all(np.isfinite(np.asarray(yd_g)[~np.asarray(aux_g["invalid"])]))
    err = err_rhs_co_row(**kw)
    print(f"{kw}: CO row error {err:.3e}")
    assert err <= CHECKS["rhs_co_row"][1]


def test_mode_switch_rejects_unknown():
    y, p, _ = _golden_rhs("chemcool")
    with pytest.raises(ValueError):
        rates_net5.rhs(y, p, fixes="gow")
    with pytest.raises(ValueError):
        rates_net5.rhs(y, p, co_shield=True, fixes="gow17")


STATES = [  # (x_H2, x_H+, x_CO, T0), n, G0, (N_H, N_H2, N_CO), zeta
    ((0.31, 2e-6, 5e-5, 30.21), 1e4, 1.7, (9.35e21, 4.5e21, 1e17), 3e-17),
    ((0.45, 1e-7, 1.3e-4, 15.13), 1e2, 10.0, (1.87e21, 5e20, 1e15), 3e-16),
    ((0.11, 1e-3, 1e-6, 300.29), 1e3, 0.0, (0.0, 0.0, 0.0), 3e-16),
    ((1e-3, 1e-4, 2e-9, 100.37), 1e2, 1.7, (3e20, 1e19, 3e13), 3e-16),
]


@pytest.mark.parametrize("k", range(len(STATES)))
def test_gow17_jacfwd_matches_finite_differences(k):
    (xh2, xhp, xco, T), n, g0, cols, zeta = STATES[k]
    p = rates_net5.make_params(n, g0, *cols, zeta=zeta, tdust=12.0)
    e = 1.5 * tb.KBOLTZ * T * (1 + tb.ABHE - xh2 + xhp + (1.41e-4 - xco)) * n
    y0 = jnp.array([xh2, xhp, xco, e])
    f = lambda y: rates_net5.rhs(y, p, fixes="gow17")[0]  # noqa: E731
    J = np.asarray(jax.jacfwd(f)(y0))
    assert np.all(np.isfinite(J))
    fd = np.zeros_like(J)
    for j in range(4):
        h = 1e-7 * abs(float(y0[j]))
        d = np.zeros(4)
        d[j] = h
        fd[:, j] = (np.asarray(f(y0 + d)) - np.asarray(f(y0 - d))) / (2 * h)
    i = 2                                     # the CO row is the one the fixes change
    scale = np.max(np.abs(J[i] * np.asarray(y0))) + 1e-300
    err = np.max(np.abs((J[i] - fd[i]) * np.asarray(y0))) / scale
    print(f"state {k}: CO row |J - FD| / scale = {err:.2e}")
    assert err < 1e-7


def test_gow17_jit_matches_eager():
    y, p, (yd, _) = _golden_rhs("gow17")
    yj, _ = jax.jit(lambda y, p: rates_net5.rhs(y, p, fixes="gow17"))(y, p)
    _, aux = rates_net5.rhs(y, p, fixes="gow17")
    sc = np.asarray(aux["gross"])
    assert np.max(np.abs(np.asarray(yj) - np.asarray(yd)) / np.where(sc > 0, sc, 1)) < 1e-12
