"""The Newton EOS against Glover's energy table and its exact He/H = 0.079 rescale; limits, gradients."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


from luna.chemistry import rates_net5, thermo  # noqa: E402
from luna.chemistry import tables as tb  # noqa: E402

A079 = tb.data_path("energy_table_abhe0079_every25.npz")
XS = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
# the table's own T at 20-30 K and its gamma ramp at 2e4 K disagree with its e(T) at this level
T_MAX, T_MED, G_MAX, G_MED = 1e-4, 3e-8, 5e-2, 1e-5
newton = jax.jit(thermo.temperature_gamma_newton)


def _errors(t, g, t_ref, g_ref):
    et, eg = np.abs(np.asarray(t) / t_ref - 1), np.abs(np.asarray(g) / g_ref - 1)
    return et.max(), np.median(et), eg.max(), np.median(eg)


def err_table_grid(a=0.1):
    _, temp, gamma = tb.energy_table()
    ekn = np.tile(np.arange(1.0, 50001.0), 5)
    t, g = newton(np.repeat(XS, 50000), ekn, a)
    return _errors(t, g, temp.ravel(), gamma.ravel())


def err_rescale(a=0.079):
    z = np.load(A079)
    t, g = newton(z["x"], z["ekn"], a)
    return _errors(t, g, z["temp"], z["gamma"])


def _check(e):
    print(f"T max {e[0]:.2e} median {e[1]:.2e}; gamma max {e[2]:.2e} median {e[3]:.2e}")
    return e[0] < T_MAX and e[1] < T_MED and e[2] < G_MAX and e[3] < G_MED


def test_newton_matches_table_on_its_grid():
    assert _check(err_table_grid())


def test_newton_matches_abhe0079_rescale():
    assert _check(err_rescale())


def test_teeth_rescale_with_the_he01_table():
    """The table path (He/H = 0.1) is what the Newton solve fixes at He/H = 0.079."""
    z = np.load(A079)
    t, g = thermo.temperature_gamma_table(z["x"], z["ekn"])
    assert not _check(_errors(t, g, z["temp"], z["gamma"]))
    assert not _check(err_rescale(0.1))


def _states(n=200000, seed=3):
    rng = np.random.default_rng(seed)
    ekn = 10 ** rng.uniform(-3, 8, n)
    x = rng.uniform(0, 0.5, n) * rng.choice([1.0, 1e-3, 1e-9, 0.0], n, p=[.7, .1, .1, .1])
    return ekn, x, rng.choice([0.1, 0.079, 0.0], n)


def residual():
    """max |ekn(T)/ekn - 1| of the solution over random states, atomic edges included."""
    ekn, x, a = _states()
    t = np.asarray(jax.jit(thermo.temperature_gamma_newton)(x, ekn, a)[0])
    phi, _ = thermo.eh2_over_t(jnp.log(t))
    back = (x * np.asarray(phi) * t + 1.5 * (1 + a - 2 * x) * t) / (1 + a - x)
    return np.max(np.abs(back / ekn - 1))


def test_newton_converges():
    r = residual()
    print(f"max residual {r:.1e}")
    assert r < 1e-13


def test_teeth_newton_converges(monkeypatch):
    monkeypatch.setattr(thermo, "N_NEWTON", 1)
    jax.clear_caches()
    assert residual() > 1e-10
    jax.clear_caches()


def test_limits():
    ekn = np.array([1e-3, 0.5, 0.99, 3.0, 1e2, 1e4, 6e4, 1e7])
    for a in (0.1, 0.079):
        t, g = thermo.temperature_gamma_newton(0.0 * ekn, ekn, a)          # atomic
        assert np.allclose(t, ekn / 1.5, rtol=1e-14, atol=0) and np.allclose(g, 5 / 3, rtol=1e-14)
        t, g = thermo.temperature_gamma_newton(0.3 + 0 * ekn[:3], ekn[:3], a)   # T < 1 K: no rotation
        assert np.allclose(t, ekn[:3] / 1.5, rtol=1e-14, atol=0)
        hot = np.array([1e5, 1e6, 1e8])                                     # e_H2 = 3 T above 2e4 K
        x = 0.2
        t, _ = thermo.temperature_gamma_newton(x + 0 * hot, hot, a)
        assert np.allclose(t, hot * (1 + a - x) / (3 * x + 1.5 * (1 + a - 2 * x)), rtol=1e-12)
    t, g = thermo.temperature_gamma_newton(np.array([0.2, 0.2]), np.array([0.0, -5.0]), 0.079)
    assert np.allclose(t, [0.0, -5.0 / 1.5]) and np.allclose(g, 5 / 3)


GRAD_PTS = [(37.3, 0.31, 0.079), (512.7, 0.05, 0.1), (3.3e3, 0.45, 0.079), (2.4e4, 0.2, 0.1),
            (6e4, 0.1, 0.079), (18.2, 0.02, 0.1), (9.1e3, 0.5, 0.079), (127.4, 0.4, 0.0)]


def gradient_error():
    """Max over points, outputs (T, gamma) and inputs (ekn, x, a) of |AD - FD| * input / max."""
    worst = 0.0
    for i in (0, 1):
        f = lambda e, x, a: thermo.temperature_gamma_newton(x, e, a)[i]  # noqa: E731
        g = jax.jit(jax.grad(f, argnums=(0, 1, 2)))
        for pt in GRAD_PTS:
            ad = np.array(g(*pt))
            fd = np.zeros(3)
            for j in range(3):
                h = 1e-5 * max(abs(pt[j]), 1e-2)
                up, dn = list(pt), list(pt)
                up[j] += h
                dn[j] -= h
                fd[j] = (float(f(*up)) - float(f(*dn))) / (2 * h)
            w = np.maximum(np.abs(pt), 1e-2)
            worst = max(worst, np.max(np.abs(ad - fd) * w) / np.max(np.abs(fd) * w))
    return worst


def test_gradients_match_fd():
    e = gradient_error()
    print(f"gradient vs central differences: {e:.1e}")
    assert e < 1e-6


def test_teeth_gradients_without_implicit_step(monkeypatch):
    monkeypatch.setattr(thermo, "_residual", _stopped(thermo._residual))
    assert gradient_error() > 1e-2


def _stopped(orig):
    def f(u, lnr, x, c1):
        g, dg = orig(u, lnr, x, c1)
        return jax.lax.stop_gradient(g), jax.lax.stop_gradient(dg)
    return f


def test_default_eos_is_the_table():
    """rates_net5.rhs without eos runs the table lookup, bit for bit."""
    from tests.test_chemistry_integrator import val
    names, cp, t, y0, ref = val.load(cases=[5, 70, 140])
    p = val.params(cp)
    y = jnp.asarray(ref[:, 60, :4])
    a = np.asarray(rates_net5.rhs(y, p)[0])
    assert np.array_equal(a, np.asarray(rates_net5.rhs(y, p, eos="table")[0]))
    assert not np.array_equal(a, np.asarray(rates_net5.rhs(y, p, eos="newton")[0]))


def test_unknown_eos_raises():
    with pytest.raises(ValueError):
        thermo.temperature_gamma(0.1, 100.0, eos="tabel")
