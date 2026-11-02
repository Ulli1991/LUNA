"""The fixed-structure hybrid solve (integrate_fixed), its generated network and its fused kernels (GPU).
Against independent code: lu_factor, _rodas3 with _check, both passes on every cell, jax.grad, central differences.
"""
from dataclasses import replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from luna.chemistry import generated, rates_net5
from luna.ode import rodas3 as ig
from luna.chemistry import networks
from luna.hydro.conversions import prim_to_cons
from luna.sources import chemistry as ch
from luna.template_config import Config
from tests.test_chemistry_coupled import GAMMA, UNITS, _ism_box
from tests.test_chemistry_sympy import ATOL, golden


GPU = jax.default_backend() in ("gpu", "rocm", "cuda")
DT = 2e3 * 3.15576e7                  # 2 kyr: the 512^3 run's step
NET = ch.fixed_net(1e-2)
HYB = ig.Options(chunk=128, chunk2=37)  # several second-pass batches, the last one partial


@pytest.fixture(scope="module")
def cells():
    """(y0, shared Params) of an 8^3 ISM box with the stage's inputs (one TreeCol pixel), relaxed 0.3 Myr."""
    n = 8
    cfg = replace(Config(), units=UNITS, Lx=10.0, Ly=10.0, Lz=10.0, Nx=n, Ny=n, Nz=n,
                  chemistry="net5", gamma=GAMMA)
    c = ch.ChemConsts(cfg)
    W, x = _ism_box(n)
    U = prim_to_cons(jnp.asarray(W), GAMMA).reshape(5, -1)
    y0, n_h, cols, _, _ = ch.cell_inputs(U, U[0] * jnp.asarray(x).reshape(3, -1), c)
    p = ch.params(c, n_h, cols, jnp.full(n_h.shape, 10.0), per_cell=False)
    for _ in range(3):                # the made-up abundances relax: fewer cells flagged at DT
        r = ig.integrate_fixed(y0, p, 1e5 * 3.15576e7, net=NET, opt=HYB)
        y0, p = r.y, r.params
    return y0, p


def _partly_flagged(run, dts):
    """The first dt at which between 10 % and 90 % of the cells are flagged."""
    for dt in dts:
        f = float(np.mean(np.asarray(run(dt).flagged)))
        if 0.1 < f < 0.9:
            return dt
    raise AssertionError("no dt flags part of the cells")


@pytest.fixture(scope="module")
def dt_hyb(cells):
    y0, p = cells
    return _partly_flagged(lambda dt: ig.integrate_fixed(y0, p, dt, net=NET, opt=HYB),
                           [f * 3.15576e7 for f in (1e4, 3e4, 1e5, 3e5, 1e6)])


def rel(a, b, atol=0.0):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b) / (np.abs(b) + atol)))


def both_passes(net, opt, y0, p, dt):
    """The hybrid without custom VJP or compaction: k1 and k2 on every cell, then where (AD-able)."""
    y1, h1, emax, bad = ig._vcell(net, opt.k1, y0, p, dt)
    flag = jax.lax.stop_gradient((emax > opt.flag_err) | bad)
    y2, h2, _, _ = ig._vcell(net, opt.k2, y0, p, dt)
    pick = lambda a, b: jnp.where(flag.reshape(flag.shape + (1,) * (a.ndim - 1)), b, a)  # noqa: E731
    return pick(y1, y2), jax.tree_util.tree_map(pick, h1, h2), flag


def test_lu_lists_match_lu_factor():
    """lu_factor_lists / lu_solve_lists give lu_factor / lu_solve's bits, pivoting included."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 4, 4)) * 10.0 ** rng.integers(-8, 8, size=(500, 4, 4))
    b = rng.normal(size=(500, 4))
    ref = jax.vmap(lambda m, v: ig.lu_solve(ig.lu_factor(m), v))(a, b)
    lu = ig.lu_factor_lists([[a[:, i, j] for j in range(4)] for i in range(4)])
    got = jnp.stack(ig.lu_solve_lists(lu, [b[:, j] for j in range(4)]), -1)
    perm = np.stack([np.asarray(v) for v in lu[1]], -1)
    assert (perm != np.arange(4)).any(1).mean() > 0.5             # pivoting exercised
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


def test_core_matches_rhs():
    """net5_core (one pixel, no LVG) equals rates_net5.rhs(lvg=False); its J equals jacfwd of it."""
    y, p = golden("frozen")
    p = p._replace(n_h_col=p.n_h_col[:, :1], n_h2_col=p.n_h2_col[:, :1], n_co_col=p.n_co_col[:, :1])
    for kw in ({"dust_iters": 4}, {}):
        p1 = p if kw else p._replace(dust_rtol=1e-12 + 0.0 * p.dust_rtol)   # converged: J is implicit
        f_ref, aux = rates_net5.rhs(y, p1, lvg=False, **kw)
        f, J, aux_g = generated.rhs_and_jac_nolvg(y, p1, dust_iters=kw.get("dust_iters"))
        scale = np.asarray(aux["gross"])
        err = np.max(np.abs(np.asarray(f) - np.asarray(f_ref)) / np.where(scale > 0, scale, 1.0))
        J_ref = jax.vmap(lambda z, q: jax.jacfwd(
            lambda u: rates_net5.rhs(u, q, lvg=False, **kw)[0])(z))(y, p1)
        s = np.maximum(np.abs(np.asarray(y)), ATOL / 1e-4)
        row = np.max(np.abs(np.asarray(J_ref)) * s[:, None, :], -1) + scale
        errJ = np.max(np.abs(np.asarray(J) - np.asarray(J_ref)) * s[:, None, :]
                      / np.where(row > 0, row, 1.0)[:, :, None])
        print(f"{kw or 'converged dust'}: f {err:.2e}, J {errJ:.2e}")
        assert err <= 1e-12
        np.testing.assert_allclose(np.asarray(aux_g["tdust"]), np.asarray(aux["tdust"]), rtol=1e-12)
        if not kw:                    # 4 masked Newton steps: J is the implicit derivative, not jacfwd's
            assert errJ <= 1e-12


def _reference_step(y0, p, dt, k):
    """k substeps per cell from _rodas3, _check and _error_norm (the adaptive integrator's pieces)."""
    net = NET._replace(ymax=networks._net5_ymax)
    h = dt / k

    def cell(y, q):
        q = q._replace(dust_rtol=jnp.asarray(1e-2))
        emax, bad = 0.0, False
        for _ in range(k):
            f0, J, aux0 = generated.rhs_and_jac_nolvg(y, q, dust_iters=ch.DUST_ITERS)
            q = q._replace(tdust=aux0["tdust"])
            scale = jnp.maximum(net.rtol * jnp.abs(y) + jnp.asarray(net.atol), 1e-300)
            y1, e, b = ig._rodas3(net, y, f0, J, q, h, scale)
            w = jnp.maximum(net.rtol * jnp.maximum(jnp.abs(y), jnp.abs(y1)) + jnp.asarray(net.atol),
                            1e-300)
            y, ok, _ = ig._check(net, net.rtol, jnp.asarray(net.atol), y1, q)
            emax, bad = jnp.maximum(emax, ig._error_norm(e, w)), bad | ~ok | b | aux0["invalid"]
        return y, emax, bad
    return jax.vmap(cell, in_axes=(0, rates_net5.Params(*[0 if jnp.ndim(v) else None for v in p])))(
        y0, p)


@pytest.mark.parametrize("k", [1, 2])
def test_substeps_match_rodas3(cells, dt_hyb, k):
    """k substeps (the second pass the same k, so flags do not matter) equal the array-form Rodas3."""
    y0, p = cells
    r = ig.integrate_fixed(y0, p, dt_hyb, net=NET, opt=ig.Options(k1=k, k2=k))
    y_ref, emax, bad = _reference_step(y0, p, dt_hyb, k)
    err = rel(r.y, y_ref, np.asarray(ATOL))
    print(f"k={k}: {err:.2e}, flagged {int(np.sum(r.flagged))} of {r.flagged.size}")
    assert err <= 1e-13
    if k == 1:                        # flagged: k1's error norm above flag_err, or an invalid step
        np.testing.assert_array_equal(np.asarray(r.flagged), np.asarray((emax > 0.1) | bad))


def test_hybrid_is_both_passes_where_flagged(cells, dt_hyb):
    """The hybrid (compacted second pass) equals k2 on its flagged cells and k1 elsewhere (to rounding)."""
    y0, p = cells
    rh = ig.integrate_fixed(y0, p, dt_hyb, net=NET, opt=HYB)
    y, h, flag = jax.jit(both_passes, static_argnums=(0, 1))(NET, HYB, y0, p, jnp.asarray(dt_hyb))
    nf = int(np.sum(rh.flagged))
    print(f"flagged {nf} of {flag.size}")
    assert HYB.chunk2 < nf < flag.size and nf % HYB.chunk2 != 0
    np.testing.assert_array_equal(np.asarray(rh.flagged), np.asarray(flag))
    # batch shapes differ (compacted vs all cells): the CPU compiler may round 1 ulp differently
    assert rel(rh.y, y, 1e-300) <= 1e-14 and rel(rh.params.tdust, h["tdust"]) <= 1e-14


# --------------------------------------------------------------------------
# gradients: integrate_fixed's custom VJP on a small stiff network (cheap to compile)
# --------------------------------------------------------------------------
class Toy(NamedTuple):
    a: float            # per cell: the stiffness
    b: float            # shared
    w: float = 1.0      # a warm start the RHS returns (hidden)


def _toy_rhs(y, p):
    """A stiff nonlinear 3-species system; aux carries a warm start."""
    r1, r2, r3 = p.a * y[0], p.b * (1.0 + 0.01 * p.w) * y[1] * y[2], 3.0 * y[1] ** 2
    return jnp.stack([-r1 + r2, r1 - r2 - r3, r3]), {"w": 0.5 * (p.w + y[0])}


TOY = ig.Network(rhs=_toy_rhs, rtol=1e-4, atol=(1e-6,) * 3, safety=(0.1,) * 3,
                 ymax=lambda p: [jnp.inf + 0.0 * p.a] * 3, ctrl_floor=lambda p: jnp.zeros(3),
                 hidden=("w",))
TOY_OPT = ig.Options(chunk=200, chunk2=37)


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(5)
    n = 600
    y0 = jnp.asarray(np.stack([rng.uniform(0.5, 1.0, n), np.zeros(n), rng.uniform(0, 0.1, n)], 1))
    p = Toy(a=jnp.asarray(10.0 ** rng.uniform(-2, 2, n)), b=jnp.asarray(1.0), w=jnp.asarray(1.0))
    dt = _partly_flagged(lambda dt: ig.integrate_fixed(y0, p, dt, net=TOY, opt=TOY_OPT),
                         [jnp.asarray(v) for v in (0.01, 0.03, 0.1, 0.3, 1.0)])
    return y0, p, dt


def _toy_loss(solve, y0, p, dt):
    wt = jnp.asarray(np.random.default_rng(6).normal(size=y0.shape))

    def F(y0, a, b, dt):
        r = ig.integrate_fixed(y0, p._replace(a=a, b=b), dt, net=TOY, opt=TOY_OPT, solve=solve)
        return jnp.sum(wt * r.y) + jnp.sum(r.params.w)
    return F


def test_toy_hybrid_is_both_passes(toy):
    y0, p, dt = toy
    r = ig.integrate_fixed(y0, p, dt, net=TOY, opt=TOY_OPT)
    y, h, flag = both_passes(TOY, TOY_OPT, y0, p, dt)
    nf = int(np.sum(flag))
    print(f"flagged {nf} of {flag.size}")
    assert TOY_OPT.chunk2 < nf < flag.size and nf % TOY_OPT.chunk2 != 0
    assert rel(r.y, y, 1e-300) <= 1e-14 and rel(r.params.w, h["w"]) <= 1e-14


def test_toy_grad_matches_jax_grad(toy):
    """jax.grad through the custom VJP equals jax.grad through both_passes (per-cell, shared, dt)."""
    y0, p, dt = toy
    args = (y0, p.a, p.b, dt)
    g = jax.grad(_toy_loss(None, y0, p, dt), argnums=(0, 1, 2, 3))(*args)
    g_ref = jax.grad(_toy_loss(both_passes, y0, p, dt), argnums=(0, 1, 2, 3))(*args)
    for a, b, name in zip(g, g_ref, ("y0", "a", "b", "dt")):
        err = float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(np.asarray(b))))
        print(f"d/d{name}: {err:.2e}")
        assert err <= 1e-12


def test_toy_grad_matches_central_differences(toy):
    y0, p, dt = toy
    F = _toy_loss(None, y0, p, dt)
    g = jax.grad(F, argnums=(2, 3))(y0, p.a, p.b, dt)
    Fj = jax.jit(F)
    for eps in (1e-4, 1e-5, 1e-6):    # truncation (eps^2) against rounding (1 / eps): the best of three
        fd_b = (Fj(y0, p.a, p.b * (1 + eps), dt) - Fj(y0, p.a, p.b * (1 - eps), dt)) / (2 * eps * p.b)
        fd_t = (Fj(y0, p.a, p.b, dt * (1 + eps)) - Fj(y0, p.a, p.b, dt * (1 - eps))) / (2 * eps * dt)
        print(f"eps {eps:g}: b {float(g[0]):.8e} vs {float(fd_b):.8e}; dt {float(g[1]):.8e} vs {float(fd_t):.8e}")
        if rel(g[0], fd_b) <= 1e-6 and rel(g[1], fd_t) <= 1e-6:
            return
    raise AssertionError("no step agrees with the gradient to 1e-6")


# --------------------------------------------------------------------------
# the fused kernels (GPU): forward to 1e-12 of the plain solve, the same gradient
# --------------------------------------------------------------------------
@pytest.mark.skipif(not GPU, reason="the fused kernels run on a GPU")
def test_fused_matches_plain(cells, dt_hyb):
    """The fused kernels' hybrid equals the plain one to 1e-12 relative, with the same flags."""
    from luna.chemistry import fused
    y0, p = cells
    r = ig.integrate_fixed(y0, p, dt_hyb, net=NET, opt=HYB, solve=fused.solve)
    ref = ig.integrate_fixed(y0, p, dt_hyb, net=NET, opt=HYB)
    err = rel(r.y, ref.y, 1e-300)
    print(f"fused vs plain: {err:.2e}, flagged {int(np.sum(ref.flagged))}")
    np.testing.assert_array_equal(np.asarray(r.flagged), np.asarray(ref.flagged))
    assert err <= 1e-12 and rel(r.params.tdust, ref.params.tdust) <= 1e-12


def _net5_loss(solve, cells, dt):
    y0, p = cells
    wt = jnp.asarray(np.random.default_rng(2).normal(size=y0.shape)) / (jnp.abs(y0) + jnp.asarray(ATOL)
                                                                         + 1e-30)

    def F(y0, zeta, n_h):
        r = ig.integrate_fixed(y0, p._replace(zeta=zeta, n_h=n_h), dt, net=NET, opt=HYB, solve=solve)
        return jnp.sum(wt * r.y)
    return F, (y0, p.zeta, p.n_h)


@pytest.mark.skipif(not GPU, reason="the fused kernels run on a GPU")
def test_fused_grad_matches_plain_and_differences(cells, dt_hyb):
    """jax.grad through the fused solve equals jax.grad through the plain one; zeta's also central differences."""
    from luna.chemistry import fused
    F, args = _net5_loss(fused.solve, cells, dt_hyb)
    G, _ = _net5_loss(None, cells, dt_hyb)
    g = jax.jit(jax.grad(F, argnums=(0, 1, 2)))(*args)
    g_ref = jax.jit(jax.grad(G, argnums=(0, 1, 2)))(*args)
    for a, b in zip(g, g_ref):
        assert np.all(np.isfinite(np.asarray(a)))
        assert float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(np.asarray(b)))) <= 1e-12
    y0, zeta, n_h = args
    Gj = jax.jit(G)
    fd = (Gj(y0, zeta * (1 + 1e-6), n_h) - Gj(y0, zeta * (1 - 1e-6), n_h)) / (2e-6 * zeta)
    print(f"d/dzeta: {float(g[1]):.8e}, central differences {float(fd):.8e}")
    assert rel(g[1], fd) <= 1e-5


@pytest.mark.skipif(not GPU, reason="the fused kernels run on a GPU")
def test_stage_fused_matches_plain():
    """The stage's update (shard_map, one GPU) with the fused kernels equals the plain hybrid's."""
    from jax.sharding import Mesh
    from luna.chemistry import fused
    n = 16
    cfg = replace(Config(), units=UNITS, Lx=10.0, Ly=10.0, Lz=10.0, Nx=n, Ny=n, Nz=n,
                  chemistry="net5", gamma=GAMMA)
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1, 1), ("x", "y", "z"))
    c = ch.ChemConsts(cfg)
    W, x = _ism_box(n)
    U = prim_to_cons(jnp.asarray(W), GAMMA)
    carry = {"S": U[0] * jnp.asarray(x), "tdust": jnp.full(U.shape[1:], 10.0),
             "h": jnp.zeros(U.shape[1:])}
    dt = DT / c.t_s
    (Uf, cf), (Up, cp) = (ch.make_apply_fixed(c, mesh, kernel=k)(U, carry, dt)
                          for k in (fused.solve, None))
    assert rel(Uf[4], Up[4]) <= 1e-12 and rel(cf["S"], cp["S"], 1e-300) <= 1e-12
    np.testing.assert_array_equal(np.asarray(cf["n_steps"]), np.asarray(cp["n_steps"]))
    assert np.asarray(cf["n_steps"]).max() == 5 and not np.asarray(cf["failed"]).any()
