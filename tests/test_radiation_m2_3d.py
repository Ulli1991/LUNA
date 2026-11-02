"""The maximum-entropy closure in three dimensions.

In slab there is one angle and the closure is a curve. In three there are
eight multipliers and a sphere to integrate over, and the things worth
checking are the ones that would still look plausible if the solve were
wrong: that the moments come back out, that the trace condition holds,
that rotating the state rotates the answer, and -- the strongest test
available -- that on axisymmetric states the 3D solver and the completely
separate slab solver give the same number.

The physics test is the last one. Two beams meeting head-on have m1 = 0,
which M1 cannot distinguish from isotropic; M2 sees m2 and lets them
through.
"""

import numpy as np
import pytest


import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402

from luna.radiation import m2, m2_3d                          # noqa: E402
from luna.radiation.m1 import m1_step                         # noqa: E402
from tests._helpers import check  # noqa: E402

pytestmark = pytest.mark.fast

CHAT = 1.0

_ROCM_SKIP = pytest.mark.skipif(
    jax.default_backend() != "cpu",
    reason="the entropy solve on a grid segfaults ROCm 7.2 unless every "
           "kernel launch is serialized -- more than two evaluations of "
           "it in one executable is enough (docs/RADIATION.md). The maths "
           "is backend-independent and the suite default is CPU, where "
           "this runs; it is the cluster GPU nightly that skips it.")


def _random_states(n=12, seed=0, mix=0.15):
    """Realizable ten-moment states: mixtures of beams, blended `mix` of
    the way towards isotropic.

    The blend is not cosmetic. A handful of beams spans a rank-deficient
    second moment -- three beams give a zero eigenvalue exactly -- and
    those states are on the boundary of the realizable set, where the
    maximum-entropy problem has no finite solution: the multipliers run
    off to infinity. Tests of the solver have to ask it for states it can
    answer. `mix` measures how far inside the set the sample sits; at
    0.15 the smallest eigenvalue is 0.05.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k = rng.integers(2, 5)
        d = rng.normal(size=(k, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        w = rng.uniform(0.2, 1.0, size=k)
        m0 = float(w.sum())
        m1 = (1.0 - mix) * (w[:, None] * d).sum(0)
        m2_ = ((1.0 - mix) * np.einsum("k,ki,kj->ij", w, d, d)
               + mix * np.eye(3) * m0 / 3.0)
        out.append(np.concatenate([[m0], m1, [m2_[i, j] for i, j in
                                              m2_3d._SYM]]))
    return jnp.asarray(np.stack(out).T)                       # (10, n)


def test_the_entropy_problem_is_actually_solved():
    """The multipliers must reproduce the eight moments they were given.
    Everything downstream is a lie if this fails."""
    u = m2_3d.targets(_random_states())
    worst = 0.0
    for i in range(u.shape[1]):
        a = m2_3d.solve_multipliers(u[:, i])
        got = np.asarray(m2_3d.normalised_moments(a)[0])
        worst = max(worst, float(np.max(np.abs(got - np.asarray(u[:, i])))))
    assert worst < 1e-10, f"worst residual {worst:.2e}"


def test_isotropic_gives_zero_multipliers_and_zero_third_moment():
    U = m2_3d.pack(jnp.array(2.0), jnp.zeros(3), jnp.eye(3) * 2.0 / 3.0)
    a = m2_3d.solve_multipliers(m2_3d.targets(U))
    assert float(jnp.max(jnp.abs(a))) < 1e-12
    m3 = m2_3d.closure_one(m2_3d.targets(U))
    assert float(jnp.max(jnp.abs(m3))) < 1e-12


def test_the_trace_condition_holds():
    """n.n = 1, so <n_i n_i n_j> = <n_j>: the closure cannot be free of
    the moments it is closing."""
    U = _random_states()
    m3 = m2_3d.third_moment(U)
    got = jnp.einsum("iijk->jk", m3.reshape(3, 3, 3, -1))
    assert float(jnp.max(jnp.abs(got - U[1:4]))) < 1e-10


def test_the_closure_approaches_the_beam_as_the_state_does():
    """All the radiation at one direction gives m3 = u(x)u(x)u. A pure
    beam is on the boundary and unsolvable, so the statement to test is
    the approach: blend towards it and the closure must follow, without
    ever being told what the answer is."""
    u = np.array([0.3, -0.5, 0.8])
    u /= np.linalg.norm(u)
    iso = m2_3d.pack(jnp.array(1.0), jnp.zeros(3), jnp.eye(3) / 3.0)
    along, err = [], []
    for f in (0.80, 0.90, 0.95, 0.98):
        U = f * m2_3d.beam(jnp.array(1.0), u) + (1.0 - f) * iso
        m3 = np.asarray(m2_3d.closure_one(m2_3d.targets(U)))
        along.append(float(np.einsum("ijk,i,j,k->", m3, u, u, u)))
        err.append(np.max(np.abs(m3 - np.einsum("i,j,k->ijk", u, u, u))))
    assert all(b > a for a, b in zip(along, along[1:])), along
    assert all(b < a for a, b in zip(err, err[1:])), err
    assert err[-1] < 2e-2, f"at 98% beam the closure is {err[-1]:.3f} off"


def test_it_agrees_with_the_independent_slab_solver():
    """The sharpest check there is: on states symmetric about z the 3D
    eight-multiplier solve must reproduce the 1D two-multiplier solve,
    which shares no code with it."""
    worst = 0.0
    for f, g in ((0.0, 1.0 / 3.0), (0.3, 0.3), (-0.5, 0.45),
                 (0.0, 0.8), (0.7, 0.7), (0.2, 0.9)):
        perp = 0.5 * (1.0 - g)
        U = m2_3d.pack(jnp.array(1.0), jnp.array([0.0, 0.0, f]),
                       jnp.diag(jnp.array([perp, perp, g])))
        h3d = float(m2_3d.closure_one(m2_3d.targets(U))[2, 2, 2])
        h1d = float(m2.closure(jnp.array([f, g])))
        worst = max(worst, abs(h3d - h1d))
    assert worst < 1e-10, f"3D vs slab {worst:.2e}"


def test_the_closure_rotates_with_the_state():
    """No axis is special, and nothing in the solver knows about one."""
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    U = _random_states(n=4, seed=7)
    m3 = np.asarray(m2_3d.third_moment(U)).reshape(3, 3, 3, -1)
    Ur = np.stack([np.asarray(U[0])] + list(Q @ np.asarray(U[1:4]))
                  + [(Q @ np.asarray(m2_3d.tensor(U)).transpose(2, 0, 1)
                      @ Q.T)[:, i, j] for i, j in m2_3d._SYM])
    m3r = np.asarray(m2_3d.third_moment(jnp.asarray(Ur))).reshape(3, 3, 3, -1)
    want = np.einsum("ia,jb,kc,abcn->ijkn", Q, Q, Q, m3)
    assert np.max(np.abs(m3r - want)) < 1e-9


def test_realizable_clips_a_negative_eigenvalue():
    """The failure that killed the rollout: a second moment whose trace is
    exactly 1 but whose spectrum is not in [0, 1]. Rescaling the trace
    leaves it untouched -- it is already 1 -- so the projection has to look
    at the eigenvalues."""
    T = jnp.diag(jnp.array([-0.3, 0.6, 0.7]))              # trace 1
    U = m2_3d.pack(jnp.array(1.0), jnp.array([0.1, 0.0, 0.0]), T)
    V = m2_3d.realizable(U)
    w = np.linalg.eigvalsh(np.asarray(m2_3d.tensor(V)))
    assert w.min() > -1e-14, f"still indefinite: {w}"
    assert abs(w.sum() - 1.0) < 1e-12


def test_the_projection_leaves_good_states_alone():
    """It must be a projection, not a smoothing: a state already inside
    the set comes back unchanged."""
    U = _random_states(n=6, seed=11)
    U = m2_3d.realizable(U)
    assert float(jnp.max(jnp.abs(m2_3d.realizable(U) - U))) < 1e-12


@_ROCM_SKIP
def test_transport_conserves_the_energy_it_should():
    """Periodic boundaries: nothing leaves, so m0 is conserved to round-off."""
    n = 8
    x = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    E = jnp.asarray(1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
                    * np.sin(2 * np.pi * Z))
    U = m2_3d.beam(E, (1.0, 0.4, -0.2))
    U = 0.9 * U + 0.1 * m2_3d.pack(E, jnp.zeros((3, n, n, n)),
                                   jnp.einsum("ij,xyz->ijxyz",
                                              jnp.eye(3) / 3.0, E))
    before = float(jnp.sum(U[0]))
    V = m2_3d.step(U, 0.2 / n / 3.0, 1.0 / n, chat=CHAT, boundary=0)
    assert abs(float(jnp.sum(V[0])) / before - 1.0) < 1e-12
    # and m1, m2: the fluxes telescope for every moment (the state stays realizable, no floor)
    dm = np.abs(np.asarray(jnp.sum(V - U, axis=(1, 2, 3)))).max() / before
    check("m2 3d periodic step max_k |d sum m_k| / sum m0", float(dm), 1e-13)       # measured 1.7e-16 (CPU)


@pytest.mark.slow
@_ROCM_SKIP
def test_the_beams_cross_where_m1_piles_them_up():
    """The reason for M2. Two opposed beams: M1 sees m1 = 0 in the overlap
    and applies isotropic pressure, so energy collects there; M2 carries
    m2, sees two beams, and lets them pass.

    The grid is a thin bar, 24 x 4 x 4, not a cube. The beams travel along
    x and the transverse directions need only enough cells to feed the
    stencil, so a cube would spend 96x the entropy solves -- which is an
    eight-variable Newton per cell, twelve times a step -- to say the same
    thing. At 16^3 this test timed out the suite's per-file budget on CPU.
    """
    nx, nt, sig = 24, 4, 0.16
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(nt) + 0.5) / nt
    X, Y, Z = np.meshgrid(x, y, y, indexing="ij")

    def blob(cx):
        return np.exp(-0.5 * ((X - cx) / sig) ** 2) * np.ones_like(Y) * np.ones_like(Z)

    eL, eR = blob(0.3), blob(0.7)
    dx, steps = 1.0 / nx, 8
    dt = 0.3 * dx / (3.0 * CHAT)
    ct = CHAT * dt * steps
    exact = blob(0.3 + ct) + blob(0.7 - ct)

    U = m2_3d.add(m2_3d.beam(jnp.asarray(eL), (1.0, 0.0, 0.0)),
                  m2_3d.beam(jnp.asarray(eR), (-1.0, 0.0, 0.0)))
    R = jnp.asarray(np.stack([eL + eR, CHAT * (eL - eR),
                              np.zeros_like(eL), np.zeros_like(eL)]))
    closure = lambda u: m2_3d.third_moment(u, chunk=256)      # noqa: E731
    for _ in range(steps):
        U = m2_3d.step(U, dt, dx, chat=CHAT, boundary=1, closure_fn=closure)
        R = m1_step(R, dt, dx, chat=CHAT, boundary=1, order=2)

    def err(E):
        return float(np.sqrt(np.mean((np.asarray(E) - exact) ** 2))
                     / np.sqrt(np.mean(exact ** 2)))

    assert err(U[0]) < err(R[0]), (f"M2 {err(U[0]):.3f} not better than "
                                   f"M1 {err(R[0]):.3f}")
    # and the specific failure: M1 leaves a lump where the beams meet
    mid = nx // 2
    assert float(R[0][mid].max()) > float(U[0][mid].max())


def test_the_closure_survives_a_first_import_inside_a_trace():
    """The stage imports this module lazily, inside its jitted transport: in a
    fresh process m2_entropy died with UnexpectedTracerError, its jnp tables
    built in the trace. Re-run the import inside a trace, then use it."""
    import importlib

    @jax.jit
    def first_import(x):
        importlib.reload(m2_3d)
        return x

    first_import(1.0)
    U = m2_3d.pack(jnp.array(2.0), jnp.zeros(3), jnp.eye(3) * 2.0 / 3.0)[:, None]
    assert bool(jnp.all(jnp.isfinite(m2_3d.third_moment(U))))
