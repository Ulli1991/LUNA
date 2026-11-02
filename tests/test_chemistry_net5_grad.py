"""Reverse-mode gradients of the network-5 RHS on the 800 golden states: finite, equal to w^T jacfwd and to
central differences, with T_dust frozen or from a fixed number of Newton steps (no while loop).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


from luna.chemistry import generated as gen, rates_net5  # noqa: E402
from tests.test_chemistry_sympy import ATOL, dust_mode, golden  # noqa: E402

DUST_ITERS = 4
H = 1e-6                     # central-difference step, relative to s_j = max(|y_j|, atol_j / 1e-4)


def rhs_of(mode):
    """The RHS f(y, p) of one dust mode: frozen T_dust, or DUST_ITERS Newton steps at the 1 % stop."""
    if mode == "frozen":
        def f(z, pp):
            with dust_mode("frozen"):
                return rates_net5.rhs(z, pp)[0]
        return f
    if mode == "iters":
        return lambda z, pp: rates_net5.rhs(z, pp, dust_iters=DUST_ITERS)[0]
    return lambda z, pp: gen.rhs(z, pp, "frozen")[0]


@pytest.fixture(scope="module")
def states():
    out = {}
    for mode in ("frozen", "iters", "generated"):
        y, p = golden("frozen")              # dust_rtol 1 %, the stage's stop
        f = rhs_of(mode)
        w = jnp.asarray(np.random.default_rng(1).normal(size=y.shape))
        g = np.asarray(jax.vmap(jax.grad(lambda z, pp, ww: jnp.sum(ww * f(z, pp))))(y, p, w))
        J = np.asarray(jax.vmap(lambda z, pp: jax.jacfwd(lambda q: f(q, pp))(z))(y, p))
        out[mode] = (y, p, w, f, g, J)
    return out


def scale_of(y, w, J, p, f):
    """Per cell and input: sum_i |w_i| (|J_ij| s_j + |f_i|), the size a gradient error is held to."""
    s = np.maximum(np.abs(np.asarray(y)), ATOL / 1e-4)
    fy = np.asarray(jax.vmap(f)(y, p))
    return s, np.sum(np.abs(np.asarray(w))[:, :, None] * (np.abs(J) * s[:, None, :]
                                                          + np.abs(fy)[:, :, None]), 1)


def rel(d, scale):
    """|d| / scale; where the scale is 0 (an invalid state: f = 0) d must be 0."""
    d = np.abs(d)
    return np.where(scale > 0, d / np.where(scale > 0, scale, 1.0), np.where(d == 0, 0.0, np.inf))


@pytest.mark.parametrize("mode", ["frozen", "iters", "generated"])
def test_grad_finite(mode, states):
    y, p, w, f, g, J = states[mode]
    bad = ~np.isfinite(g).all(1)
    print(f"{mode}: non-finite reverse-mode gradients in {bad.sum()} of {bad.size} states")
    assert not bad.any()


@pytest.mark.parametrize("mode", ["frozen", "iters", "generated"])
def test_grad_equals_jacfwd(mode, states):
    """grad of w . f equals w^T jacfwd(f) to 1e-12 of the scale (exact AD both ways)."""
    y, p, w, f, g, J = states[mode]
    s, scale = scale_of(y, w, J, p, f)
    vjp = np.einsum("ci,cij->cj", np.asarray(w), J)
    err = np.max(rel((g - vjp) * s, scale))
    print(f"{mode}: grad vs w^T jacfwd {err:.2e}")
    assert err <= 1e-12


@pytest.mark.parametrize("mode", ["frozen", "iters"])
def test_grad_equals_central_differences(mode, states):
    """grad of w . f equals central differences (step 1e-6 or 1e-8 s_j) to 1e-6 of the scale."""
    # not where the stencil leaves the open domain 0 < y_j < bound: the clip there is a kink,
    # and x_H2 = 0 switches the EOS to atomic gas (a jump in T)
    y, p, w, f, g, J = states[mode]
    s, scale = scale_of(y, w, J, p, f)
    F = jax.jit(jax.vmap(lambda z, pp, ww: jnp.sum(ww * f(z, pp))))
    yy = np.asarray(y)
    bound = np.array([0.5, 1.0, 1.0, np.inf])
    inside = (yy - H * s > 0.0) & (yy + H * s < bound)
    errs = []
    for h in (H, H / 100):        # a stencil across a node of a linear T table sees two slopes
        fd = np.zeros_like(g)
        for j in range(4):
            e = np.zeros(4)
            e[j] = 1.0
            dy = jnp.asarray(h * s * e)
            fd[:, j] = np.asarray(F(y + dy, p, w) - F(y - dy, p, w)) / (2 * h * s[:, j])
        errs.append(np.where(inside, rel((g - fd) * s, scale), 0.0))
    err = np.minimum(*errs)
    print(f"{mode}: grad vs central differences on {inside.sum()} of {inside.size} entries: "
          f"max {err.max():.2e}, median {np.median(err[inside]):.2e}")
    assert inside.sum() > 0.75 * inside.size
    assert err.max() <= 1e-6
