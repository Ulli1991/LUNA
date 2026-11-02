"""The SymPy-generated network-5 f and Jacobian against rates_net5.rhs and jax.jacfwd, on the golden states.
f to 1e-12 of the gross rate (sum of |terms|); J to 1e-12 of the row scale; teeth corrupt one generated term.
"""
import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest


from luna.chemistry import cooling, generated as gen, rates_net5, tables as tb  # noqa: E402
from luna.chemistry.generated import net5_p12  # noqa: E402

RTOL = 1e-12
ATOL = np.array([1e-7, 1e-12, 1e-14, 0.0])         # the integrator's atol (integrate.NET5), rtol 1e-4
G = dict(np.load(tb.data_path("chemistry_net5_golden.npz")))
DUST_RTOL = {"implicit": 1e-12, "frozen": 1e-2}    # implicit: converged T_dust, as the integrator runs it


def golden(dust):
    """(y, Params) of the 800 rate_eq golden states (12 TreeCol pixels)."""
    x = G["rateeq_in"]
    c = [jnp.asarray(x[:, i]) for i in range(17)]
    pix = lambda v: jnp.broadcast_to(v[:, None], v.shape + (12,))  # noqa: E731
    p = rates_net5.Params(n_h=c[0], n_h_col=pix(c[6]), n_h2_col=pix(c[7]), n_co_col=pix(c[8]),
                          g0=c[5], g0_dust=c[5], zeta=c[11], dgr=c[12], abundc=c[13],
                          abundo=c[14], abundsi=c[15], dl=c[9], divv=c[10], tdust=c[16],
                          dust_rtol=DUST_RTOL[dust])
    p = rates_net5.Params(*[jnp.asarray(v, jnp.float64) + 0.0 * (c[0] if jnp.ndim(v) == 0 else 0.0)
                            for v in p])                  # every field per cell, for vmap
    return jnp.stack([c[2], c[3], c[4], c[1]], axis=-1), p


def _dust_frozen(n_h, temp, x_hi, chi, g0_dust, dgr, tdust_in, h2_form_ex=0.84, h2_form_kin=0.12,
                 rtol=1e-2):
    """cooling.dust_temperature without the Newton solve: T_dust = tdust_in."""
    ch7 = cooling.interp_rows({7: cooling.chem_rows()[7]}, temp)[7] * dgr
    active = (dgr > 1e-8) & (temp < 1e5)
    temp2 = temp * 1e-2

    def rates(td):
        fa = 1.0 / (1.0 + 1e4 * jnp.exp(-6e2 / td))
        stick = 1.0 / (1.0 + 0.4 * (temp2 + td * 1e-2) ** 0.5 + 0.2 * temp2 + 0.08 * temp2**2.0)
        return fa, stick, ch7 * fa * stick
    td = jnp.where(active, tdust_in, 10.0)
    return cooling._finish(td, td * 0.0 < 1.0, active, temp, x_hi, n_h, rates)


@contextlib.contextmanager
def dust_mode(dust):
    """The existing RHS with T_dust frozen at p.tdust (dust='frozen') or as it is."""
    orig = cooling.dust_temperature
    if dust == "frozen":
        cooling.dust_temperature = _dust_frozen
    try:
        yield
    finally:
        cooling.dust_temperature = orig


def ref_rhs(y, p, dust):
    with dust_mode(dust):
        return rates_net5.rhs(y, p)


def ref_jac(y, p, dust):
    """jax.jacfwd of the existing rhs, cell by cell."""
    with dust_mode(dust):
        return jax.vmap(lambda yy, pp: jax.jacfwd(lambda z: rates_net5.rhs(z, pp)[0])(yy))(y, p)


def scaled_err(a, b, scale):
    a, b, s = np.broadcast_arrays(*(np.asarray(v, float) for v in (a, b, scale)))
    d = np.where(a == b, 0.0, np.abs(a - b))
    r = np.where(d == 0.0, 0.0, np.where(s > 0, d / np.where(s > 0, s, 1.0), np.inf))
    return float(np.max(r))


def jac_err(J, Jref, y, gross):
    """max |dJ_ij| s_j / (max_k |Jref_ik| s_k + gross_i), s_j = max(|y_j|, atol_j / 1e-4)."""
    s = np.maximum(np.abs(np.asarray(y)), ATOL / 1e-4)
    Jr = np.asarray(Jref) * s[:, None, :]
    row = np.max(np.abs(Jr), axis=-1) + np.asarray(gross)
    return scaled_err(np.asarray(J) * s[:, None, :], Jr, row[:, :, None])


def f_err(f, y, p, dust):
    fr, aux = ref_rhs(y, p, dust)
    return scaled_err(f, fr, aux["gross"])


@pytest.fixture(scope="module")
def refs():
    out = {}
    for dust in ("implicit", "frozen"):
        y, p = golden(dust)
        fr, aux = ref_rhs(y, p, dust)
        out[dust] = (y, p, fr, aux, ref_jac(y, p, dust))
    return out


@pytest.mark.parametrize("dust", ["implicit", "frozen"])
def test_f_matches_rhs(dust, refs):
    y, p, fr, aux, _ = refs[dust]
    f, a = gen.rhs(y, p, dust)
    err = scaled_err(f, fr, aux["gross"])
    errT = scaled_err(a["T"], aux["T"], np.abs(aux["T"]))
    errTd = scaled_err(a["tdust"], aux["tdust"], np.abs(aux["tdust"]))
    print(f"{dust}: f {err:.2e}, T {errT:.2e}, T_dust {errTd:.2e}")
    assert max(err, errT, errTd) <= RTOL
    assert np.array_equal(np.asarray(a["invalid"]), np.asarray(aux["invalid"]))
    f2, J, _ = gen.rhs_and_jac(y, p, dust)
    assert scaled_err(f2, fr, aux["gross"]) <= RTOL


@pytest.mark.parametrize("dust", ["implicit", "frozen"])
def test_jac_matches_jacfwd(dust, refs):
    y, p, fr, aux, Jr = refs[dust]
    _, J, _ = gen.rhs_and_jac(y, p, dust)
    err = jac_err(J, Jr, y, aux["gross"])
    big = np.abs(np.asarray(Jr)) > 1e-3 * np.max(np.abs(np.asarray(Jr)), axis=-1, keepdims=True)
    rel = np.abs(np.asarray(J) - np.asarray(Jr))[big] / np.abs(np.asarray(Jr))[big]
    print(f"{dust}: J scaled {err:.2e}; plain relative on entries > 1e-3 of the row max: "
          f"max {rel.max():.2e}, median {np.median(rel):.2e}")
    assert err <= RTOL


@pytest.mark.parametrize("dust", ["implicit", "frozen"])
def test_jacfwd_through_generated(dust, refs):
    """jax.jacfwd of the generated f agrees with jacfwd of the existing rhs (frozen: and with J)."""
    y, p, fr, aux, Jr = refs[dust]
    Jg = jax.vmap(lambda yy, pp: jax.jacfwd(lambda z: gen.rhs(z, pp, dust)[0])(yy))(y, p)
    err = jac_err(Jg, Jr, y, aux["gross"])
    print(f"{dust}: jacfwd(generated f) vs jacfwd(rhs) {err:.2e}")
    assert err <= RTOL
    if dust == "frozen":
        _, J, _ = gen.rhs_and_jac(y, p, dust)
        assert jac_err(Jg, J, y, aux["gross"]) <= RTOL


def test_grad_through_generated(refs):
    """Reverse mode through the generated f (frozen T_dust: no while loop) and the n_H tangent."""
    y, p, fr, aux, Jr = refs["frozen"]
    w = jnp.asarray(np.random.default_rng(0).normal(size=y.shape))

    def grad(fn):
        return np.asarray(jax.vmap(jax.grad(lambda z, pp, ww: jnp.sum(ww * fn(z, pp)[0])))(y, p, w))
    g = grad(lambda z, pp: gen.rhs(z, pp, "frozen"))
    with dust_mode("frozen"):
        g_ref = grad(rates_net5.rhs)
    # reverse mode through the existing rhs is NaN in some states (a masked branch's 0 * inf)
    ok = np.isfinite(g_ref).all(1)
    assert np.array_equal(np.isfinite(g).all(1), ok)
    vjp_ref = np.einsum("ci,cij->cj", np.asarray(w), np.asarray(Jr))
    s = np.maximum(np.abs(np.asarray(y)), ATOL / 1e-4)
    scale = np.sum(np.abs(np.asarray(w))[:, :, None] * (np.abs(np.asarray(Jr)) * s[:, None, :]
                                                        + np.asarray(aux["gross"])[:, :, None]), 1)
    err = scaled_err((g * s)[ok], (vjp_ref * s)[ok], scale[ok])
    print(f"grad vs w^T jacfwd(rhs) on {ok.sum()} of {ok.size} states (rest NaN in both): {err:.2e}")
    assert err <= RTOL

    def dn(fn):
        return jax.jvp(lambda nh: fn(y, p._replace(n_h=nh))[0], (p.n_h,), (p.n_h,))[1]
    d_gen = dn(lambda yy, pp: gen.rhs(yy, pp, "frozen"))
    d_ref = dn(lambda yy, pp: ref_rhs(yy, pp, "frozen"))
    err = scaled_err(d_gen, d_ref, aux["gross"])
    print(f"n_H tangent: {err:.2e}")
    assert err <= RTOL


# --------------------------------------------------------------------------
# teeth: one generated term corrupted -> the matching comparison fails
# --------------------------------------------------------------------------
def _corrupt(fname, old, new, count=None):
    """net5_p12 with one textual change inside function fname (exec'd); returns the corrupted function."""
    src = open(net5_p12.__file__).read()
    start = src.index(f"def {fname}(")
    end = src.find("\ndef ", start + 1)
    end = len(src) if end < 0 else end
    body = src[start:end]
    assert old in body, old
    body = body.replace(old, new) if count is None else body.replace(old, new, count)
    ns = dict(vars(net5_p12))
    exec(compile(body, "<corrupted>", "exec"), ns)
    return ns[fname]


TEETH = {
    # a lookup slope used only by J: dT/d(e/kn) 1e-6 too large
    "slope_dT_dekn": ("rhs_and_jac_frozen", "jac", "tg0_G_d1 = _r",
                      "tg0_G_d1 = _r\n    tg0_T_d0 = 1.000001 * tg0_T_d0"),
    # one Jacobian entry, d f_H2 / d x_H2, off by 1e-9
    "J00_1e-9": ("rhs_and_jac_frozen", "jac", "jnp.stack([jnp.stack([(",
                 "jnp.stack([jnp.stack([1.000000001*("),
    # a term of f and J: the DB96 self-shielding 0.965 / 12 pixels, 1.2e-7 too large
    "db96_coefficient": ("rhs_and_jac_frozen", "f", "0.08041666666666666", "0.08041667666666666"),
}


@pytest.mark.parametrize("name", list(TEETH))
def test_teeth(name, refs):
    fname, check, old, new = TEETH[name]
    y, p, fr, aux, Jr = refs["frozen"]
    f, J, _ = _corrupt(fname, old, new)(y, p)
    err = f_err(f, y, p, "frozen") if check == "f" else jac_err(J, Jr, y, aux["gross"])
    print(f"broken {name}: {check} error {err:.3e}")
    assert err > RTOL, f"corrupting {name} went unnoticed"
    if name == "slope_dT_dekn":                     # J-only term: f is untouched
        assert f_err(f, y, p, "frozen") <= RTOL
