"""The learned stand-in for the 3D entropy closure.

The point of the construction is that the network is only allowed to
choose the part of m3 that is free. The trace condition m3_iij = m1_j is
imposed algebraically, so it holds for untrained weights, random weights
and trained weights alike -- these tests use random weights on purpose,
because a property that needs training is not a property of the
architecture.
"""

import numpy as np
import pytest


import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402

from luna.radiation import m2_3d, m2_learned_3d as ml         # noqa: E402

pytestmark = pytest.mark.fast

CHAT = 1.0

_SLOW_ON_GPU = pytest.mark.skipif(
    jax.default_backend() != "cpu",
    reason="compiling a transport step -- twelve closure evaluations -- "
           "takes minutes on ROCm and blows the suite's 900 s per-file "
           "budget; 225 s for one at 64^3 was measured. The arithmetic "
           "is backend-independent and runs on the CPU default.")


def _states(n=16, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k = rng.integers(2, 5)
        d = rng.normal(size=(k, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        w = rng.uniform(0.2, 1.0, size=k)
        out.append(np.concatenate([[w.sum()], (w[:, None] * d).sum(0),
                                   [np.einsum("k,ki,kj->ij", w, d, d)[i, j]
                                    for i, j in m2_3d._SYM]]))
    return jnp.asarray(np.stack(out).T)


def _random_params(seed=0, scale=1.0):
    p = ml.init(jax.random.PRNGKey(seed))
    keys = jax.random.split(jax.random.PRNGKey(seed + 100), len(p))
    return [(W + scale * jax.random.normal(k, W.shape),
             b + scale * jax.random.normal(k, b.shape)[:b.shape[0]])
            for (W, b), k in zip(p, keys)]


def test_the_trace_condition_holds_for_any_weights():
    """m3_iij = m1_j is imposed, not learned. If this can be broken by
    the weights, the construction is wrong."""
    U = _states()
    for seed in (0, 1, 2):
        m3 = ml.third_moment(_random_params(seed), U)
        got = jnp.einsum("iijk->jk", m3.reshape(3, 3, 3, -1))
        assert float(jnp.max(jnp.abs(got - U[1:4]))) < 1e-12


def test_the_prediction_is_totally_symmetric():
    """m3 = <n n n> is symmetric in all three indices by definition."""
    m3 = np.asarray(ml.third_moment(_random_params(4), _states())
                    ).reshape(3, 3, 3, -1)
    for perm in ((1, 0, 2, 3), (0, 2, 1, 3), (2, 1, 0, 3)):
        assert np.max(np.abs(m3 - m3.transpose(perm))) < 1e-14


def test_zero_weights_give_exactly_the_trace_part():
    """The network is initialised to output zero, so an untrained closure
    is the pure trace term -- a definite, checkable starting point."""
    U = _states()
    m0 = jnp.maximum(U[0], m2_3d.EPS)
    m1 = (U[1:4] / m0).reshape(3, -1).T
    got = ml.third_moment(ml.init(jax.random.PRNGKey(0)), U)
    want = ml._rebuild(jnp.zeros((m1.shape[0], 7)), m1)
    want = jnp.moveaxis(want, 0, 3).reshape(3, 3, 3, *m0.shape) * m0
    assert float(jnp.max(jnp.abs(got - want))) < 1e-14


def test_the_free_components_round_trip():
    """free_of . trace_free_part . rebuild is the identity: the seven
    numbers really do parametrise what is predicted."""
    rng = np.random.default_rng(5)
    free = jnp.asarray(rng.normal(size=(9, 7)))
    m1 = jnp.asarray(rng.normal(size=(9, 3)) * 0.3)
    back = ml.free_of(ml.trace_free_part(ml._rebuild(free, m1), m1))
    assert float(jnp.max(jnp.abs(back - free))) < 1e-13


def test_the_trace_free_part_really_is_trace_free():
    rng = np.random.default_rng(6)
    free = jnp.asarray(rng.normal(size=(9, 7)))
    m1 = jnp.asarray(rng.normal(size=(9, 3)) * 0.3)
    Q = ml.trace_free_part(ml._rebuild(free, m1), m1)
    assert float(jnp.max(jnp.abs(jnp.einsum("nijj->ni", Q)))) < 1e-13


@_SLOW_ON_GPU
def test_it_drops_into_the_transport_step():
    """The whole reason for the module: `step(..., closure_fn=...)` runs
    and conserves m0 on a periodic grid, exactly as the entropy closure
    does, because the closure only enters through the flux."""
    n = 8
    x = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    E = jnp.asarray(1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Z))
    U = 0.9 * m2_3d.beam(E, (1.0, 0.3, -0.2)) + 0.1 * m2_3d.pack(
        E, jnp.zeros((3, n, n, n)),
        jnp.einsum("ij,xyz->ijxyz", jnp.eye(3) / 3.0, E))
    before = float(jnp.sum(U[0]))
    V = m2_3d.step(U, 0.2 / n / 3.0, 1.0 / n, chat=CHAT, boundary=0,
                   closure_fn=ml.closure_fn(_random_params(7, scale=0.1), 0))
    assert abs(float(jnp.sum(V[0])) / before - 1.0) < 1e-12


@_SLOW_ON_GPU
def test_it_is_differentiable_in_the_weights():
    """It exists to be trained, and to be trained through the solver: the
    gradient of a rolled-out step with respect to the weights must be
    finite and non-zero."""
    n = 6
    E = jnp.ones((n, n, n))
    U = 0.9 * m2_3d.beam(E, (1.0, 0.0, 0.0)) + 0.1 * m2_3d.pack(
        E, jnp.zeros((3, n, n, n)),
        jnp.einsum("ij,xyz->ijxyz", jnp.eye(3) / 3.0, E))
    p0 = _random_params(8, scale=0.05)

    def objective(p):
        V = m2_3d.step(U, 0.05 / n, 1.0 / n, chat=CHAT, boundary=0,
                       closure_fn=ml.closure_fn(p, 0))
        # the second moment, not m0: only the m2 equation's flux carries
        # m3, so after one step the energy does not depend on the closure
        # at all and a gradient taken there is correctly zero
        return jnp.sum(V[4:] ** 2)

    g = jax.grad(objective)(p0)
    flat = jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(g)])
    assert bool(jnp.all(jnp.isfinite(flat)))
    assert float(jnp.max(jnp.abs(flat))) > 0.0
