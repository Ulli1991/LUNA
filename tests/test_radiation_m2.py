"""The maximum-entropy M2 closure: does it solve the entropy problem, does
its implicit gradient agree with the honest one, and does it fix the case
M1 gets wrong.

The last of those is the reason for the whole thing. Two equal beams
travelling in opposite directions have m1 = 0, and M1 -- which carries only
(m0, m1) -- cannot tell them from isotropic radiation, so it gives them the
pressure of an isotropic field and they pile up instead of passing through
each other. M2 carries m2 as a variable, sees g = 1 rather than 1/3, and
lets them cross.
"""

import numpy as np
import pytest


import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402

from luna.radiation import m2                                 # noqa: E402

pytestmark = pytest.mark.fast

CHAT = 1.0

_ROCM_SKIP = pytest.mark.skipif(
    jax.default_backend() != "cpu",
    reason="the entropy solve on a grid segfaults ROCm 7.2 unless every "
           "kernel launch is serialized -- more than two evaluations of "
           "it in one executable is enough (docs/RADIATION.md). The maths "
           "is backend-independent and the suite default is CPU, where "
           "this runs; it is the cluster GPU nightly that skips it.")


def _realizable_samples(n=24, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.uniform(-0.95, 0.95, size=n)
    g = f ** 2 + rng.uniform(0.02, 0.98, size=n) * (1.0 - f ** 2)
    return np.stack([f, g], axis=1)


def test_the_entropy_problem_is_actually_solved():
    """The multipliers must reproduce the moments they were asked for. This
    is the whole closure: if the solve is off, everything downstream is."""
    worst = 0.0
    for fg in _realizable_samples():
        ab = m2.solve_multipliers(jnp.asarray(fg))
        got = np.asarray(m2.normalised_moments(ab)[0][:2])
        worst = max(worst, float(np.max(np.abs(got - fg))))
    assert worst < 1e-11, f"worst moment residual {worst:.2e}"


def test_the_closure_hits_the_limits_it_must():
    """Isotropic, free-streaming and two-beam are the three states whose
    answer is known without solving anything."""
    h_iso = float(m2.closure(jnp.array([0.0, 1.0 / 3.0])))
    assert abs(h_iso) < 1e-12                       # symmetric: m3 = 0
    h_beam = float(m2.closure(jnp.array([0.999, 0.999])))
    assert h_beam > 0.99                            # all of it at mu = 1
    h_two = float(m2.closure(jnp.array([0.0, 0.999])))
    assert abs(h_two) < 1e-10                       # +1 and -1 cancel
    # and the parity of the distribution carries to the closure
    for f, g in ((0.4, 0.5), (0.8, 0.9), (0.2, 0.3)):
        a = float(m2.closure(jnp.array([f, g])))
        b = float(m2.closure(jnp.array([-f, g])))
        assert abs(a + b) < 1e-11, (f, g, a, b)


def test_the_closure_stays_inside_the_moment_bounds():
    """|m3| <= m2 for any non-negative intensity, because |mu| <= 1."""
    for fg in _realizable_samples(seed=1):
        h = float(m2.closure(jnp.asarray(fg)))
        assert abs(h) <= fg[1] + 1e-12, (fg, h)


def test_the_implicit_gradient_matches_finite_differences():
    """The point of the exercise: the adjoint of the closure comes from the
    implicit function theorem, not from differentiating the Newton loop."""
    grad = jax.jit(jax.grad(m2.closure))
    worst = 0.0
    for fg in _realizable_samples(n=10, seed=2):
        fg = jnp.asarray(fg)
        g_ad = np.asarray(grad(fg))
        for k in range(2):
            h = 1e-6
            e = jnp.zeros(2).at[k].set(h)
            fd = float(m2.closure(fg + e) - m2.closure(fg - e)) / (2 * h)
            worst = max(worst, abs(g_ad[k] - fd) / max(abs(fd), 1e-6))
    assert worst < 2e-6, f"worst relative difference {worst:.2e}"


def test_the_implicit_gradient_matches_differentiating_the_solver():
    """The other honest route: unroll the Newton iterations and let AD walk
    them. Same answer, much more tape -- which is the argument for the
    implicit one."""
    def unrolled(fg):
        return m2._third(m2.solve_multipliers(fg))

    g_imp = jax.jit(jax.grad(m2.closure))
    g_unr = jax.jit(jax.grad(unrolled))
    for fg in _realizable_samples(n=8, seed=3):
        fg = jnp.asarray(fg)
        a, b = np.asarray(g_imp(fg)), np.asarray(g_unr(fg))
        assert np.allclose(a, b, rtol=1e-7, atol=1e-9), (fg, a, b)


def test_the_closure_transposes():
    """<v, J u> = <J^T v, u> for the linearisation: forward and reverse mode
    of the same closure have to be each other's transpose."""
    rng = np.random.default_rng(4)
    for fg in _realizable_samples(n=6, seed=5):
        fg = jnp.asarray(fg)
        u = jnp.asarray(rng.normal(size=2))
        v = float(rng.normal())
        jv = jax.jvp(m2.closure, (fg,), (u,))[1]          # J u  (scalar)
        jt = jax.grad(m2.closure)(fg)                     # J^T (as a vector)
        lhs = v * float(jv)
        rhs = float(jnp.dot(jt, u)) * v
        assert abs(lhs - rhs) <= 1e-9 * max(abs(lhs), 1e-9)


def _two_beam_state(n=200, sigma=0.05):
    """A packet of radiation that is two opposite beams, not isotropic:
    m1 = 0 but m2 = m0, which is g = 1."""
    x = (np.arange(n) + 0.5) / n - 0.5
    m0 = np.exp(-0.5 * (x / sigma) ** 2)
    return x, jnp.asarray(np.stack([m0, np.zeros_like(m0), m0]))


@_ROCM_SKIP
def test_two_beams_cross_under_m2_and_collide_under_m1():
    """The case that separates the closures. After a time t the two beams
    should be at +-c t; M1, which sees only m1 = 0, keeps one blob in the
    middle spreading at c/sqrt(3)."""
    n, sigma = 200, 0.05
    x, U = _two_beam_state(n, sigma)
    dx = 1.0 / n
    dt = 0.4 * dx / CHAT
    steps = 75                                      # c t = 0.15
    # chunked: a step inlines four entropy solves and more than two in one
    # executable segfaults ROCm 7.2 (this test did, on GPU, in the suite)
    closure = lambda u: m2.third_moment(u, chunk=64)          # noqa: E731
    m2_step = jax.jit(lambda u: m2.step(u, dt, dx, chat=CHAT, boundary=1,
                                        order=2, closure_fn=closure))
    for _ in range(steps):
        U = m2_step(U)
    E2 = np.asarray(U[0])

    from luna.radiation.m1 import m1_step
    R = jnp.asarray(np.stack([np.exp(-0.5 * (x / sigma) ** 2),
                              np.zeros(n)]))       # same energy, F = 0
    m1_j = jax.jit(lambda r: m1_step(r, dt, dx, chat=CHAT, boundary=1, order=2))
    for _ in range(steps):
        R = m1_j(R)
    E1 = np.asarray(R[0])

    ct = CHAT * dt * steps
    w2, w1 = E2 / E2.sum(), E1 / E1.sum()
    spread2 = float(np.sqrt(np.sum(w2 * x ** 2)))
    spread1 = float(np.sqrt(np.sum(w1 * x ** 2)))
    # Both have an answer to compare against, and they are different
    # answers: two beams at +- c t give sqrt((ct)^2 + sigma^2), while an
    # isotropic field spreads at c/sqrt(3) and gives sqrt((ct/sqrt3)^2 +
    # sigma^2). Measured 2026-09-21: M2 0.158 against 0.158 predicted,
    # M1 0.106 against 0.102.
    expect2 = np.sqrt(ct ** 2 + sigma ** 2)
    expect1 = np.sqrt((ct / np.sqrt(3.0)) ** 2 + sigma ** 2)
    assert abs(spread2 - expect2) < 0.10 * expect2, (spread2, expect2)
    assert abs(spread1 - expect1) < 0.10 * expect1, (spread1, expect1)
    assert spread2 > 1.4 * spread1, (spread2, spread1)
    # and the energy has left the middle for M2 and has not for M1
    assert E2[n // 2] < 0.5 * E2.max(), "M2 kept the energy in the middle"
    assert E1[n // 2] > 0.9 * E1.max(), "M1 moved the peak off centre"


@_ROCM_SKIP
def test_a_single_beam_still_streams_at_the_light_speed():
    """M2 must not have broken the case M1 already got right."""
    n, sigma = 200, 0.05
    x = (np.arange(n) + 0.5) / n - 0.3
    m0 = np.exp(-0.5 * (x / sigma) ** 2)
    U = jnp.asarray(np.stack([m0, m0, m0]))          # f = 1, g = 1: one beam
    dx, steps = 1.0 / n, 75
    dt = 0.4 * dx / CHAT
    stepf = jax.jit(lambda u: m2.step(u, dt, dx, chat=CHAT, boundary=1,
                                      order=2))
    c0 = float(np.sum(m0 * x) / np.sum(m0))
    for _ in range(steps):
        U = stepf(U)
    E = np.asarray(U[0])
    c1 = float(np.sum(E * x) / np.sum(E))
    expect = CHAT * dt * steps
    assert abs((c1 - c0) - expect) < 2.0 * dx, (c1 - c0, expect)


@_ROCM_SKIP
def test_the_rollout_differentiates():
    """A gradient through the transport AND the entropy solves inside it:
    this is the state the closure has to be usable in."""
    n = 64
    x = (np.arange(n) + 0.5) / n - 0.5
    m0 = np.exp(-0.5 * (x / 0.08) ** 2)
    U0 = jnp.asarray(np.stack([m0, 0.3 * m0, 0.5 * m0]))
    dx = 1.0 / n
    dt = 0.4 * dx / CHAT

    def loss(U):
        for _ in range(4):
            U = m2.step(U, dt, dx, chat=CHAT, boundary=1, order=2)
        return jnp.sum(U[0] ** 2)

    g = jax.jit(jax.grad(loss))(U0)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0
    # and it agrees with a finite difference in one direction
    d = jnp.zeros_like(U0).at[0, n // 2].set(1.0)
    h = 1e-6
    fd = float(loss(U0 + h * d) - loss(U0 - h * d)) / (2 * h)
    ad = float(jnp.sum(g * d))
    # The rollout has a minmod limiter and two clamps in it, so it is
    # piecewise differentiable and a finite difference straddles kinks:
    # measured 0.2% here, against 6e-7 for the smooth hydro step in
    # tests/test_autodiff_validation.py. The gate is that they agree at all.
    assert abs(ad - fd) < 1e-2 * max(abs(fd), 1.0), (ad, fd)
