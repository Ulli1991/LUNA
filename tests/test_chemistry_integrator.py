"""The stiff chemistry integrator: method, linear algebra, DVODE trajectories, gradients, sharding."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest


from luna.ode import rodas3 as ig  # noqa: E402


from luna.chemistry import networks  # noqa: E402
from luna.chemistry import rates_net5  # noqa: E402
from tools.chem_integrator import validate_net5 as val  # noqa: E402
from tests._helpers import check  # noqa: E402

TARGET, EQ_TARGET = 1e-3, 1e-4         # vs DVODE: every output / the equilibrium end state
ONE_PASS = ig.Options(tiers=((None, 1),))   # one compiled pass, where tiers do not matter


# --------------------------------------------------------------------------
# the method
# --------------------------------------------------------------------------
def _stab(z):
    """Rodas3's stability function on y' = z y, h = 1: (R(z), embedded R(z))."""
    ks = []
    for i in range(4):
        f = z * (1 + sum(a * k for a, k in zip(ig.A[i], ks)))
        ks.append((f + sum(c * k for c, k in zip(ig.C[i], ks))) / (1 / ig.GAMMA - z))
    y1 = 1 + sum(m * k for m, k in zip(ig.M, ks))
    return y1, y1 - ks[3]


def test_rodas3_linear_order_and_l_stability():
    for which, order in ((0, 3), (1, 2)):
        e = [abs(_stab(-z)[which] - np.exp(-z)) for z in (1e-1, 1e-2)]
        assert 0.7 * 10 ** (order + 1) < e[0] / e[1] < 1.3 * 10 ** (order + 1)
        assert abs(_stab(-1e8)[which]) < 1e-7 and abs(_stab(-1e4)[which]) < 1e-3   # L-stable
        assert max(abs(_stab(1j * y)[which]) for y in np.logspace(-3, 8, 300)) <= 1 + 1e-12


class Pair(NamedTuple):
    a: float


def test_rodas3_nonlinear_order_3():
    """Local error of one step on a nonlinear 2x2 system scales as h^4."""
    def rhs(y, p):
        return jnp.stack([-p.a * y[0] * y[1] - y[0] ** 2, y[0] - 0.5 * y[1] ** 3]), {}
    net = ig.Network(rhs=rhs, rtol=1e-6, atol=(1e-12,) * 2, safety=(0.1,) * 2,
                     ymax=lambda p: jnp.full(2, jnp.inf), ctrl_floor=lambda p: jnp.zeros(2))
    p = Pair(a=3.0)

    @jax.jit
    def step(y, h):
        (f, _), jac = ig._f_and_jac(net, y, p)
        return ig._rodas3(net, y, f, jac, p, h, jnp.ones(2))[0]

    def fine(y, h, k=2000):
        for _ in range(k):
            y = step(y, h / k)
        return y
    y0 = jnp.array([1.0, 0.7])
    err = [float(jnp.max(jnp.abs(step(y0, h) - fine(y0, h)))) for h in (0.1, 0.05)]
    assert 12 < err[0] / err[1] < 20, err


def test_lu_matches_numpy():
    rng = np.random.default_rng(1)
    for n in (4, 8):
        for k in range(20):
            m = rng.normal(size=(n, n)) * 10.0 ** rng.uniform(-6, 6, size=(n, 1))
            if k == 0:
                m[0, 0] = 0.0                      # needs a row swap
            rhs = rng.normal(size=(n, 3))
            x = ig.lu_solve(ig.lu_factor(jnp.asarray(m)), jnp.asarray(rhs))
            assert np.allclose(np.asarray(x), np.linalg.solve(m, rhs), rtol=1e-9, atol=0)


CHAIN_SUM = 1e-13         # measured 5.6e-16 (CPU), 1.0e-15 (A100)


class LinParams(NamedTuple):
    k: float


def _chain(n=8):
    """A decay chain y_i -> y_i+1 with rates 1e-3 .. 1e6 (stiff, stays positive), y' = k M y."""
    lam = np.concatenate([np.logspace(-3, 6, n - 1), [0.0]])
    m = np.diag(-lam) + np.diag(lam[:-1], -1)
    mj = jnp.asarray(m)
    net = ig.Network(rhs=lambda y, p: (p.k * (mj @ y), {}), rtol=1e-8, atol=(1e-14,) * n,
                     safety=(0.1,) * n, ymax=lambda p: jnp.full(n, jnp.inf),
                     ctrl_floor=lambda p: jnp.zeros(n))
    return net, m, lam


def test_linear_stiff_chain_8x8_value_and_gradients():
    from scipy.linalg import expm
    net, m, lam = _chain()
    y0 = np.random.default_rng(4).uniform(0.5, 1.5, size=(3, 8))
    for i in range(3, 7):                          # the fast species start quasi-steady
        y0[:, i] = lam[i - 1] * y0[:, i - 1] / lam[i]
    dt, k = 30.0, 1.3
    opt = ig.Options(diff_fields=("k",))
    e = expm(k * m * dt)
    r = ig.integrate(y0, LinParams(k=k), dt, net=net, opt=opt)
    assert not np.any(np.asarray(r.failed))
    assert np.max(np.abs(np.asarray(r.y) - y0 @ e.T) / (y0 @ e.T)) < 1e-6
    # the chain's columns sum to 0: sum y is a linear invariant, kept by a Rosenbrock step up to its linear solves
    check("decay chain |d sum y| / sum y", float(np.max(np.abs(np.asarray(r.y).sum(1) / y0.sum(1) - 1.0))), CHAIN_SUM)

    w = np.random.default_rng(5).normal(size=(3, 8))
    loss = lambda y, kk: jnp.sum(w * ig.integrate(y, LinParams(k=kk), dt, net=net, opt=opt).y)  # noqa: E731
    gy, gk = jax.grad(loss, argnums=(0, 1))(jnp.asarray(y0), jnp.asarray(k))
    assert np.allclose(np.asarray(gy), w @ e, rtol=1e-5, atol=1e-8)
    exact_k = np.sum(w * (y0 @ (m * dt @ e).T))
    assert abs(float(gk) - exact_k) < 1e-5 * abs(exact_k)
    gdt = jax.grad(lambda d: jnp.sum(w * ig.integrate(y0, LinParams(k=k), d, net=net,
                                                         opt=opt).y))(dt)
    exact_dt = np.sum(w * (y0 @ (k * m @ e).T))
    assert abs(float(gdt) - exact_dt) < 1e-5 * abs(exact_dt)


def test_fails_loudly_at_the_step_cap():
    names, cp, t, y0, ref = val.load(cases=[0, 100])
    p = val.params(cp)
    r = ig.integrate(y0, p, 3e13, opt=ig.Options(max_steps=3), net=networks.NET5)
    assert np.all(np.asarray(r.failed)) and np.all(np.asarray(r.t) < 3e13)


# --------------------------------------------------------------------------
# the 168 DVODE trajectories
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def reference():
    return val.load()


@pytest.fixture(scope="module")
def trajectories(reference):
    names, cp, t, y0, ref = reference
    return val.run(cp, t, y0, ig.Options())


def test_trajectories_match_dvode(reference, trajectories):
    names, cp, t, y0, ref = reference
    assert not trajectories["failed"].any()
    rows, eq = val.table(names, cp, trajectories["got"], ref)
    for q, (worst, med, case, k) in rows.items():
        print(f"{q}: max {worst:.2e} (median {med:.1e}) at {case} output {k}; eq {eq[q]:.2e}")
    for q in val.QTY:
        assert rows[q][0] < TARGET, q
        assert eq[q] < EQ_TARGET, q


# cases that span the stiffness range: n = 1e1..1e5, cold/warm, thin/shielded
TEETH_CASES = [9, 37, 70, 103, 140, 167]


def _broken_euler(net, y, f0, jac, p, h, w):
    """Explicit Euler (with Heun's error estimate) in place of the Rosenbrock step."""
    y1 = y + h * f0
    f1, aux = net.rhs(y1, p)
    return y1, 0.5 * h * (f1 - f0), aux["invalid"]


def _broken_jac(net, y, p):
    (f, aux), jac = _orig_f_and_jac(net, y, p)
    return (f, aux), -jac


_orig_f_and_jac = ig._f_and_jac


@pytest.mark.parametrize("brk", ["explicit_euler", "jacobian_sign", "no_error_estimate"])
def test_teeth_trajectories(brk, reference, monkeypatch):
    names, cp, t, y0, ref = reference
    sel = np.asarray(TEETH_CASES)
    if brk == "explicit_euler":
        monkeypatch.setattr(ig, "_rodas3", _broken_euler)
    elif brk == "jacobian_sign":
        monkeypatch.setattr(ig, "_f_and_jac", _broken_jac)
    else:
        monkeypatch.setattr(ig, "_error_norm", lambda err, w: jnp.zeros(()))
    jax.clear_caches()
    res = val.run(cp[sel], t, y0[sel], ONE_PASS._replace(max_steps=300))
    jax.clear_caches()
    rows, _ = val.table(names[sel], cp[sel], res["got"], ref[sel])
    worst = max(r[0] for r in rows.values())
    print(f"{brk}: failed {int(res['failed'].sum())}, worst rel err {worst:.2e}")
    assert res["failed"].any() or not worst < TARGET


# --------------------------------------------------------------------------
# gradients against central differences
# --------------------------------------------------------------------------

GRAD_CASES = [70, 115, 145]         # n = 1e2 (no field), 1e4 (A_V 5), 1e5 (A_V 1): CO matters


def _dust_jacobian_error():
    """jacfwd of the RHS with the dust temperature converged vs central differences."""
    names, cp, t, y0, ref = val.load(cases=GRAD_CASES)
    y = jnp.asarray(ref[:, 120, :4])
    p = ig.broadcast_params(val.params(cp), 3)._replace(dust_rtol=jnp.full(3, 1e-12))
    p = p._replace(tdust=rates_net5.rhs(y, p)[1]["tdust"])     # warm start: already converged
    f = lambda yy, k: rates_net5.rhs(yy, jax.tree_util.tree_map(lambda a: a[k], p))[0]  # noqa: E731
    worst = 0.0
    for k in range(3):
        jac = np.asarray(jax.jacfwd(f)(y[k], k))
        for j in range(4):
            e = np.zeros(4)
            e[j] = 1e-6 * abs(float(y[k, j]))
            fd = (np.asarray(f(y[k] + e, k)) - np.asarray(f(y[k] - e, k))) / (2 * e[j])
            scale = np.abs(jac * np.asarray(y[k])[None, :]).max(1) + 1e-300
            worst = max(worst, float(np.max(np.abs(jac[:, j] - fd) * abs(float(y[k, j])) / scale)))
    return worst


def test_converged_dust_jacobian_matches_fd():
    err = _dust_jacobian_error()
    print(f"converged-dust Jacobian vs central differences: {err:.2e}")
    assert err < 1e-6


def test_teeth_converged_dust_jacobian(monkeypatch):
    """Without the final Newton step the dust temperature carries no derivative."""
    from luna.chemistry import cooling
    monkeypatch.setattr(cooling, "_polish", lambda td, rtol, step: td)
    err = _dust_jacobian_error()
    print(f"without the polish: {err:.2e}")
    assert err > 1e-3


def _objective(y0, zeta, cp, dt, opt):
    p = val.params(cp)._replace(zeta=zeta)
    r = ig.integrate(y0, p, dt, opt=opt, net=networks.NET5)
    temp = rates_net5.rhs(r.y, r.params)[1]["T"]
    return jnp.sum(jnp.log(temp) + jnp.log(r.y[:, 0]) + 1e3 * r.y[:, 1] / (1e-4 + r.y[:, 1]))


def _fd_elasticities(f, y0, zeta):
    """x dF/dx by central differences (1e-3 of x) for every case and every x in (y0, zeta)."""
    out = np.zeros((y0.shape[0], 5))
    for c in range(y0.shape[0]):
        for j in range(5):
            e = np.zeros((y0.shape[0], 5))
            e[c, j] = 1e-3 * (y0[c, j] if j < 4 else float(zeta[c]))
            out[c, j] = float(f(y0 + e[:, :4], zeta + e[:, 4]) - f(y0 - e[:, :4], zeta - e[:, 4])) / 2e-3
    return out


def _elasticity_error(fd, y0, zeta, gy, gz):
    """Per case: max |AD - FD| of x dF/dx, relative to the largest one of that case."""
    ad = np.concatenate([gy * y0, (gz * np.asarray(zeta))[:, None]], 1)
    print(f"x dF/dx AD {ad}\n          FD {fd}")
    return float(np.max(np.abs(ad - fd).max(1) / np.abs(ad).max(1)))


def test_gradient_matches_central_differences(monkeypatch):
    """Over dt = 1e3 and 1e5 yr (one compiled program, dt traced)."""
    names, cp, t, y0, ref = val.load(cases=GRAD_CASES[::2])
    y0 = ref[:, 120, :4]                  # a mid-trajectory state: all four components matter
    opt = ig.Options(rtol=1e-8, atol=tuple(1e-4 * a for a in networks.NET5.atol),
                     diff_fields=("zeta",), tiers=ONE_PASS.tiers)
    zeta = jnp.full(y0.shape[0], 3e-17)
    f = jax.jit(lambda y, z, dt: _objective(y, z, cp, dt, opt))
    grad = jax.jit(jax.grad(f, argnums=(0, 1)))
    fds = {}
    for dt_yr in (1e3, 1e5):
        dt = dt_yr * 3.15576e7
        gy, gz = (np.asarray(g) for g in grad(jnp.asarray(y0), zeta, dt))
        fds[dt] = _fd_elasticities(lambda y, z: f(y, z, dt), y0, zeta)
        err = _elasticity_error(fds[dt], y0, zeta, gy, gz)
        print(f"dt {dt_yr:g} yr: gradient vs central differences {err:.1e}")
        assert err < 1e-4
    # teeth: sensitivities that do not propagate through the steps must fail the same check
    real = ig._cell

    def frozen(net, opt, sens, *a):
        out, (s, fend) = real(net, opt, sens, *a)
        return out, (jnp.eye(*s.shape) if sens else s, fend)
    monkeypatch.setattr(ig, "_cell", frozen)
    jax.clear_caches()
    for dt, fd in fds.items():
        gy, gz = (np.asarray(g) for g in jax.grad(f, argnums=(0, 1))(jnp.asarray(y0), zeta, dt))
        assert _elasticity_error(fd, y0, zeta, gy, gz) > 1e-2
    jax.clear_caches()


# --------------------------------------------------------------------------
# sharded == one device
# --------------------------------------------------------------------------
def test_shard_map_equals_single_device():
    from jax.sharding import Mesh, PartitionSpec as P
    from luna.utils.shmap import shard_map
    if len(jax.devices()) < 2:
        pytest.skip("needs two devices")
    names, cp, t, y0, ref = val.load(cases=list(range(0, 168, 7)))
    y = ref[:, 100, :4]
    p = ig.broadcast_params(val.params(cp), y.shape[0])
    dt = 1e5 * 3.15576e7
    mesh = Mesh(np.array(jax.devices()[:2]), ("x",))
    spec = P("x")
    f = jax.jit(shard_map(lambda y, p: ig.integrate(y, p, dt, opt=ONE_PASS, net=networks.NET5).y, mesh=mesh,
                          in_specs=(spec, jax.tree_util.tree_map(lambda _: spec, p)),
                          out_specs=spec))
    two = np.asarray(f(jnp.asarray(y), p))
    one = np.asarray(ig.integrate(y, p, dt, opt=ONE_PASS, net=networks.NET5).y)
    # XLA:CPU compiles the two programs differently (last-bit changes); bitwise equality
    # of 2 GPUs and 1 is measured by tools/chem_integrator/bench_net5.py --sharded
    print(f"max rel difference sharded vs one device: {np.max(np.abs(one - two) / one):.1e}")
    assert np.allclose(one, two, rtol=1e-8, atol=0)


# --------------------------------------------------------------------------
# the lazily built tables are concrete even when first built under jit
# --------------------------------------------------------------------------
def _first_call_under_jit():
    from luna.chemistry import cooling, thermo
    cooling._CACHE.clear()
    thermo._TEMP = thermo._GAMMA = None
    try:
        names, cp, t, y0, ref = val.load(cases=[5])
        p = val.params(cp)
        jax.jit(lambda y, p: rates_net5.rhs(y, p))(jnp.asarray(y0), p)      # a fresh trace
        return rates_net5.rhs(jnp.asarray(y0), p)[0]          # eager, from the cache
    finally:
        cooling._CACHE.clear()
        thermo._TEMP = thermo._GAMMA = None


def test_tables_first_built_under_jit():
    assert np.all(np.isfinite(np.asarray(_first_call_under_jit())))


def test_teeth_tables_first_built_under_jit(monkeypatch):
    import contextlib
    monkeypatch.setattr(jax, "ensure_compile_time_eval", contextlib.nullcontext)
    with pytest.raises(jax.errors.UnexpectedTracerError):
        _first_call_under_jit()


# --------------------------------------------------------------------------
# skipping a tier with no unfinished cell changes nothing
# --------------------------------------------------------------------------
def test_skipping_empty_tiers_is_bitwise_neutral():
    names, cp, t, y0, ref = val.load(cases=list(range(0, 168, 7)))
    y = ref[:, 1, :4]                    # early, transient states: some cells need every tier
    p = ig.broadcast_params(val.params(cp), y.shape[0])
    dt = 1e5 * 3.15576e7
    a = ig.integrate(y, p, dt, net=networks.NET5)
    b = ig.integrate(y, p, dt, opt=ig.Options(skip_empty_tiers=False), net=networks.NET5)
    assert int(np.max(np.asarray(b.n_steps) + np.asarray(b.n_rejected))) > 8
    for u, v in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_array_equal(np.asarray(u), np.asarray(v))
