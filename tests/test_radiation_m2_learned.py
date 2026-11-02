"""The learned entropy closure: the properties that hold whatever the
weights are.

Nothing here trains. Training the network needs a gradient with respect to
its parameters, and on this ROCm stack that program segfaults (the value of
the network differentiates fine; the multiplier map does not -- the same
class of crash as in docs/AUTODIFF.md, and the CPU backend runs all of it).
Training therefore happens on the CPU backend in
scripts/validation/runs/m2_learned_train.py, and what a test on the GPU can
and should check is that the structure is really structure: an untrained
network must still produce a legal closure.
"""

import numpy as np
import pytest


import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402

from luna.radiation import m2, m2_learned                     # noqa: E402

pytestmark = pytest.mark.fast


def _params(seed=0):
    return m2_learned.init(jax.random.PRNGKey(seed))


def _excited(seed=0):
    """Parameters with the output layer switched on, so the multipliers are
    not the zero the initialisation deliberately starts from."""
    p = _params(seed)
    rng = np.random.default_rng(seed)
    Wz, Wy, b = p[-1]
    return p[:-1] + [(jnp.asarray(rng.normal(size=Wz.shape)),
                      jnp.asarray(rng.normal(size=Wy.shape)), b)]


def test_the_multiplier_map_is_the_gradient_of_the_potential():
    """It is written out by hand instead of taken with jax.grad, so it has
    to be checked against the potential it claims to differentiate -- on a
    network whose output layer is awake, or both sides are zero and the
    comparison says nothing."""
    p = _excited()
    fg = m2_learned.sample_states(jax.random.PRNGKey(1), 6)
    worst = 0.0
    for y in fg:
        a = np.asarray(m2_learned.multipliers(p, y))
        fd = []
        for k in range(2):
            e = jnp.zeros(2).at[k].set(1e-6)
            fd.append(float(m2_learned.potential(p, y + e)
                            - m2_learned.potential(p, y - e)) / 2e-6)
        worst = max(worst, float(np.max(np.abs(a - np.array(fd))
                                        / np.maximum(np.abs(fd), 1e-9))))
    # the bound is the central difference's own accuracy at h = 1e-6, not
    # the gradient's: measured 4.0e-7 here and 8e-9 on a smoother parameter
    # set, both of them finite-difference truncation
    assert worst < 1e-5, f"worst relative difference {worst:.2e}"


def test_the_potential_is_convex_whatever_the_weights_are():
    """Convexity is the architecture's, not the fit's: along any line the
    second difference of h must be non-negative for random parameters."""
    for seed in (0, 1, 2):
        p = _params(seed)
        rng = np.random.default_rng(seed)
        for _ in range(8):
            y = jnp.asarray(rng.uniform(-0.8, 0.8, size=2))
            d = jnp.asarray(rng.normal(size=2))
            d = d / jnp.linalg.norm(d)
            h = 1e-3
            second = float(m2_learned.potential(p, y + h * d)
                           - 2 * m2_learned.potential(p, y)
                           + m2_learned.potential(p, y - h * d)) / h ** 2
            assert second > -1e-6, (seed, second)


def test_the_closure_is_realizable_whatever_the_weights_are():
    """m3 is a moment of exp(a.v), a positive distribution, so |m3| <= m2
    holds by construction -- the reason for learning the dual rather than
    fitting m3 directly."""
    for seed in (0, 3, 7):
        p = _params(seed)
        fg = m2_learned.sample_states(jax.random.PRNGKey(seed + 10), 64)
        h = jax.vmap(lambda y: m2_learned.closure(p, y))(fg)
        assert bool(jnp.all(jnp.abs(h) <= fg[:, 1] + 1e-12))
        assert bool(jnp.all(jnp.isfinite(h)))


def test_an_untrained_network_starts_near_isotropic():
    """The output layer is initialised nearly off, so the multipliers start
    small and the closure starts near zero -- the isotropic answer, and the
    part of the closure where it is still sensitive to its input.

    Starting anywhere else saturates it and training stalls: measured, an
    rms closure error of 0.47 that would not come down. Measured at this
    initialisation: |a| <= 0.10 and |m3| <= 0.03, which is near-isotropic
    rather than exactly isotropic, because the output layer's non-negative
    weight is softplus(-8) ~ 3e-4 and a true zero would never train."""
    p = _params()
    fg = m2_learned.sample_states(jax.random.PRNGKey(2), 16)
    a = jax.vmap(lambda y: m2_learned.multipliers(p, y))(fg)
    h = jax.vmap(lambda y: m2_learned.closure(p, y))(fg)
    assert float(jnp.max(jnp.abs(a))) < 0.2, float(jnp.max(jnp.abs(a)))
    assert float(jnp.max(jnp.abs(h))) < 0.05, float(jnp.max(jnp.abs(h)))


def test_it_drops_into_the_transport_in_place_of_the_exact_closure():
    """`m2.step(..., closure_fn=...)` has to take it, and the step has to
    stay realizable with a closure that is (at initialisation) wrong."""
    n = 64
    x = (np.arange(n) + 0.5) / n - 0.5
    e = np.exp(-0.5 * (x / 0.07) ** 2)
    U = jnp.asarray(np.stack([e, 0.3 * e, 0.5 * e]))
    dx, dt = 1.0 / n, 0.4 / n
    U2 = m2.step(U, dt, dx, chat=1.0, boundary=1, order=2,
                 closure_fn=m2_learned.closure_fn(_params()))
    assert bool(jnp.all(jnp.isfinite(U2)))
    m0 = jnp.maximum(U2[0], 1e-300)
    f, g = U2[1] / m0, U2[2] / m0
    assert bool(jnp.all(g >= f * f - 1e-12)) and bool(jnp.all(g <= 1.0 + 1e-12))
