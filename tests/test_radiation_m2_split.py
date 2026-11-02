"""M2 on M1's split structure (radiation.m2_split, m2_pallas): the sweep floor,
conservation, a uniform isotropic field at rest, beam crossing against M1,
the learned closure frozen over a step, and the fused kernels against the XLA
sweeps. On a CPU the kernel runs in
Pallas's interpreter, which rounds differently from XLA's fused CPU code, so
there it is compared to round-off."""
import numpy as np
import pytest

import jax

import jax.numpy as jnp  # noqa: E402

from luna.radiation import m2_3d, m2_learned_3d as ml, m2_pallas, m2_split  # noqa: E402
from luna.radiation.m1 import m1_split_step  # noqa: E402
from tests._helpers import check  # noqa: E402

pytestmark = pytest.mark.fast

CHAT = 1.0


def _states(shape, seed=0, mix=0.15):
    """Realizable fields: per cell a mixture of 2-4 beams blended `mix` towards isotropic."""
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    out = np.zeros((10, n))
    for c in range(n):
        k = rng.integers(2, 5)
        d = rng.normal(size=(k, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        w = rng.uniform(0.2, 1.0, size=k)
        m0 = float(w.sum())
        m1 = (1.0 - mix) * (w[:, None] * d).sum(0)
        m2 = (1.0 - mix) * np.einsum("k,ki,kj->ij", w, d, d) + mix * np.eye(3) * m0 / 3.0
        out[:, c] = np.concatenate([[m0], m1, [m2[i, j] for i, j in m2_3d._SYM]])
    return jnp.asarray(out.reshape((10,) + tuple(shape)))


def test_the_floor_leaves_realizable_states_alone():
    U = _states((40,), seed=3)
    V = m2_split.floor(U)
    assert float(jnp.max(jnp.abs(V - U) / jnp.max(jnp.abs(U), axis=0))) < 1e-14


def test_the_floor_repairs_a_negative_eigenvalue_and_keeps_the_trace():
    T = jnp.diag(jnp.array([-0.3, 0.6, 0.7]))                # trace 1, indefinite
    U = m2_3d.pack(jnp.array(2.0), jnp.array([0.1, 0.0, 0.0]), 2.0 * T)[:, None]
    V = m2_split.floor(U)[:, 0]
    w = np.linalg.eigvalsh(np.asarray(m2_3d.tensor(V)) / 2.0)
    assert w.min() > -1e-14, f"still indefinite: {w}"
    assert abs(w.sum() - 1.0) < 1e-14
    assert float(V[0]) == 2.0


def test_the_floor_puts_m1_back_on_the_cone():
    U = m2_3d.pack(jnp.array(1.0), jnp.array([3.0, 4.0, 0.0]), jnp.eye(3) / 3.0)[:, None]
    V = m2_split.floor(U)[:, 0]
    assert abs(float(jnp.linalg.norm(V[1:4])) - 1.0) < 1e-14


def test_a_split_step_conserves_m0():
    """Periodic boundaries: nothing leaves, so m0 is conserved to round-off."""
    U = _states((8, 8, 8), seed=1)
    V = m2_split.split_step(U, 0.3 / 8, 1.0 / 8, chat=CHAT, boundary=0, kernel="xla")
    assert abs(float(jnp.sum(V[0]) / jnp.sum(U[0])) - 1.0) < 1e-13
    # m0 and m1 telescope (measured 1e-17); m2 not conserved by design: the floor puts every cell back on
    # tr(m2) = m0 (sum m2 moves 8e-5 of sum m0, 0 without the floor)
    dm = np.abs(np.asarray(jnp.sum(V[:4] - U[:4], axis=(1, 2, 3)))).max() / float(jnp.sum(U[0]))
    check("m2 split periodic step max |d sum m0, m1| / sum m0", float(dm), 1e-13)


def test_uniform_isotropic_light_stays_put():
    E = jnp.full((6, 6, 6), 2.5)
    U = m2_3d.pack(E, jnp.zeros((3, 6, 6, 6)), jnp.einsum("ij,xyz->ijxyz", jnp.eye(3) / 3.0, E))
    V = m2_split.split_step(U, 0.3 / 6, 1.0 / 6, chat=CHAT, boundary=0, kernel="xla")
    assert float(jnp.max(jnp.abs(V - U))) < 1e-15


@pytest.mark.slow
@pytest.mark.skipif(jax.default_backend() != "cpu",
                    reason="the entropy solve on a grid needs serialized kernel launches on "
                           "ROCm 7.2 (test_radiation_m2_3d._ROCM_SKIP); the suite default is CPU")
def test_the_beams_cross_where_m1_split_piles_them_up():
    """`test_radiation_m2_3d`'s crossing test on the split structure, with the
    entropy closure: two opposed beams along x on a thin bar; M1 sees m1 = 0
    in the overlap and piles energy up, M2 carries m2 and lets them pass."""
    nx, nt, sig = 24, 4, 0.16
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(nt) + 0.5) / nt
    X, Y, Z = np.meshgrid(x, y, y, indexing="ij")

    def blob(cx):
        return np.exp(-0.5 * ((X - cx) / sig) ** 2) * np.ones_like(Y) * np.ones_like(Z)

    eL, eR = blob(0.3), blob(0.7)
    dx, steps = 1.0 / nx, 4
    dt = 0.35 * m2_split.substep_dt(dx, chat=CHAT)
    ct = CHAT * dt * steps
    exact = blob(0.3 + ct) + blob(0.7 - ct)
    U = m2_3d.add(m2_3d.beam(jnp.asarray(eL), (1.0, 0.0, 0.0)),
                  m2_3d.beam(jnp.asarray(eR), (-1.0, 0.0, 0.0)))
    R = jnp.asarray(np.stack([eL + eR, CHAT * (eL - eR), np.zeros_like(eL), np.zeros_like(eL)]))
    closure = lambda u: m2_3d.third_moment(u, chunk=256)      # noqa: E731
    for _ in range(steps):
        U = m2_split.split_step(U, dt, dx, chat=CHAT, boundary=1, closure_fn=closure)
        R = m1_split_step(R, dt, dx, chat=CHAT, boundary=1)

    def err(E):
        return float(np.sqrt(np.mean((np.asarray(E) - exact) ** 2)) / np.sqrt(np.mean(exact ** 2)))

    assert err(U[0]) < err(R[0]), f"M2 {err(U[0]):.3f} not better than M1 {err(R[0]):.3f}"
    assert float(R[0][nx // 2].max()) > float(U[0][nx // 2].max())


@pytest.mark.parametrize("d", [0, 1, 2])
def test_the_fused_sweep_is_the_xla_sweep(d):
    shape = (8, 8, 16)
    U = _states(shape, seed=10 + d)
    Up = m2_split._pad_axis_local(U, 2, 1, d)
    dx = 1.0 / 8
    dt = 0.35 * dx
    ref = m2_split.floor(m2_split.hancock_sweep(U, Up, dt, dx, d, chat=CHAT))
    got = m2_pallas.sweep(Up, dt, dx, d, chat=CHAT, interpret=m2_pallas.interpret_mode())
    scale = jnp.max(jnp.abs(ref), axis=tuple(range(1, 4)), keepdims=True)
    tol = 1e-12 if m2_pallas.interpret_mode() else 1e-15
    assert float(jnp.max(jnp.abs(got - ref) / scale)) < tol


def test_the_fused_split_step_is_the_xla_split_step():
    U = _states((8, 8, 8), seed=4)
    a = m2_split.split_step(U, 0.3 / 8, 1.0 / 8, chat=CHAT, boundary=1, kernel="xla")
    b = m2_split.split_step(U, 0.3 / 8, 1.0 / 8, chat=CHAT, boundary=1, kernel="pallas")
    scale = jnp.max(jnp.abs(a), axis=tuple(range(1, 4)), keepdims=True)
    assert float(jnp.max(jnp.abs(a - b) / scale)) < 1e-12


@pytest.mark.parametrize("boundary", [0, 1])
@pytest.mark.parametrize("learned", [False, True])
def test_the_unpadded_fused_split_step_is_the_padded_sweeps(boundary, learned):
    """No pad on x and y (an edge or wrapped plane index in the kernel) and in-place z ghosts:
    bitwise the fused sweeps with an XLA pad before each, on a grid whose axes differ."""
    U = _states((4, 8, 16), seed=11)
    dt, dx = 0.3 / 16, 1.0 / 16
    Vp, params = None, _learned_params() if learned else None
    if learned:
        V = m2_pallas.free_components(U, params, interpret=m2_pallas.interpret_mode())
        Vp = m2_split._pad_local(V, 1, boundary, (0, 1, 2))

    def padded(U):
        for d in (0, 1, 2, 2, 1, 0):
            Up = m2_split._pad_axis_local(U, 2, boundary, d)
            U = m2_pallas.sweep(Up, 0.5 * dt, dx, d, chat=CHAT, Vp=Vp, interpret=m2_pallas.interpret_mode())
        return U
    a = np.asarray(jax.jit(padded)(U))
    b = np.asarray(jax.jit(lambda U: m2_split.split_step(
        U, dt, dx, chat=CHAT, boundary=boundary, params=params, kernel="pallas"))(U))
    np.testing.assert_array_equal(b, a)


def _learned_params(seed=5, scale=0.3, width=32):
    """Nonzero weights everywhere, the head included (init() starts it at zero)."""
    keys = jax.random.split(jax.random.PRNGKey(seed), 6)
    shapes = [(width, 8), (width, width), (7, width)]
    return [(scale * jax.random.normal(keys[2 * i], s) / s[1] ** 0.5,
             0.1 * jax.random.normal(keys[2 * i + 1], (s[0],))) for i, s in enumerate(shapes)]


def test_the_frozen_closure_is_the_learned_closure_at_the_centres():
    """What the frozen sweep reconstructs from -- the trace part of the cell's
    m1 plus the free components times m0 -- is `m2_learned_3d.third_moment`."""
    params = _learned_params()
    U = _states((4, 4, 8), seed=6)
    m3 = ml.third_moment(params, U, 0, dtype=None)
    V = m2_split.free_components(U, params, dtype=None)
    c = [U[k] for k in range(10)]
    for d in range(3):
        m0 = jnp.maximum(U[0], m2_3d.EPS)
        row = [m2_split.trace_m3(c, d, i, j) + q * m0
               for (i, j), q in zip(m2_3d._SYM, m2_split.free_row(V, d))]
        for (i, j), r in zip(m2_3d._SYM, row):
            assert float(jnp.max(jnp.abs(r - m3[d, i, j]))) < 1e-13 * float(jnp.max(U[0]))


def test_a_learned_split_step_conserves_m0():
    U = _states((8, 8, 8), seed=2)
    V = m2_split.split_step(U, 0.3 / 8, 1.0 / 8, chat=CHAT, boundary=0,
                            params=_learned_params(), kernel="xla")
    assert bool(jnp.all(jnp.isfinite(V)))
    assert abs(float(jnp.sum(V[0]) / jnp.sum(U[0])) - 1.0) < 1e-13
    # m0 and m1 telescope (measured 1e-17); m2 not conserved by design: the floor puts every cell back on
    # tr(m2) = m0 (sum m2 moves 8e-5 of sum m0, 0 without the floor)
    dm = np.abs(np.asarray(jnp.sum(V[:4] - U[:4], axis=(1, 2, 3)))).max() / float(jnp.sum(U[0]))
    check("m2 learned split periodic step max |d sum m0, m1| / sum m0", float(dm), 1e-13)


@pytest.mark.parametrize("d", [0, 1, 2])
def test_the_fused_frozen_sweep_is_the_xla_frozen_sweep(d):
    """The frozen learned closure in the kernel against the XLA sweep, from the
    same free components; and the closure matters: the trace closure is far
    from it."""
    params = _learned_params()
    shape = (8, 8, 16)
    U = _states(shape, seed=20 + d)
    Up = m2_split._pad_axis_local(U, 2, 1, d)
    Vp = m2_split._pad_local(m2_split.free_components(U, params), 1, 1, (0, 1, 2))
    dx = 1.0 / 8
    dt = 0.35 * dx
    ref = m2_split.floor(m2_split.frozen_sweep(U, Up, Vp, dt, dx, d, chat=CHAT))
    got = m2_pallas.sweep(Up, dt, dx, d, chat=CHAT, Vp=Vp, interpret=m2_pallas.interpret_mode())
    scale = jnp.max(jnp.abs(ref), axis=tuple(range(1, 4)), keepdims=True)
    err = float(jnp.max(jnp.abs(got - ref) / scale))
    tol = 1e-12 if m2_pallas.interpret_mode() else 1e-15
    assert err < tol, err
    trace = m2_split.floor(m2_split.hancock_sweep(U, Up, dt, dx, d, chat=CHAT))
    assert float(jnp.max(jnp.abs(trace - ref) / scale)) > 1e-3


def test_the_network_kernel_is_the_network():
    """`m2_pallas.free_components` (the weights compiled in, FP32 multiply-adds)
    against the network in FP64: FP32 round-off, on outputs of order 0.1."""
    params = _learned_params()
    U = _states((4, 8, 16), seed=8)
    ref = m2_split.free_components(U, params, dtype=None)
    got = m2_pallas.free_components(U, params, interpret=m2_pallas.interpret_mode())
    assert got.dtype == jnp.float32 and got.shape == ref.shape
    err = float(jnp.max(jnp.abs(got - ref)))
    assert err < 1e-6, err
    assert float(jnp.max(jnp.abs(ref))) > 0.1


def test_the_fused_learned_split_step_is_the_xla_one():
    """The whole step: the network kernel, the ghosts of the frozen closure,
    six fused sweeps. The sweeps agree to round-off (above); what separates
    the two steps is the two FP32 networks: XLA's matmul and the kernel's
    multiply-adds, ~1e-7 apart on a CPU; on an MI300A XLA's FP32 matmul is
    off by 6e-4 (docs/RADIATION.md), measured 2.1e-5 here."""
    U = _states((8, 8, 8), seed=4)
    params = _learned_params()
    a = m2_split.split_step(U, 0.3 / 8, 1.0 / 8, chat=CHAT, boundary=1, params=params,
                            kernel="xla")
    b = m2_split.split_step(U, 0.3 / 8, 1.0 / 8, chat=CHAT, boundary=1, params=params,
                            kernel="pallas")
    trace = m2_split.split_step(U, 0.3 / 8, 1.0 / 8, chat=CHAT, boundary=1, kernel="xla")
    scale = jnp.max(jnp.abs(a), axis=tuple(range(1, 4)), keepdims=True)
    err = float(jnp.max(jnp.abs(a - b) / scale))
    assert err < (1e-6 if jax.default_backend() == "cpu" else 1e-4), err
    assert float(jnp.max(jnp.abs(a - trace) / scale)) > 1e-3


def _unfloored(S=1e61):
    """Edge and face states a Hancock predictor makes at STARBENCH's photon scale (1e61): m0 at or
    below zero, or 1e-300, next to large m1 and m2; then a realizable and a mildly unrealizable one."""
    cols = [[0.0, S, 0, 0, S, 0, 0, 0, 0, 0],
            [-1e40, S, 0, 0, S, 1e50, 0, 0, 0, 0],
            [1e-300, 1e10, 0, 0, 1e10, -1e10, 0, 0, 0, 0],
            [1.0, 0.3, 0, 0, 0.5, 0.25, 0.25, 0, 0, 0],
            [S, 1.02 * S, 0, 0, 0.9 * S, 0.05 * S, 0.05 * S, 0, 0, 0]]
    return jnp.asarray(np.array(cols).T)


def test_the_floor_and_the_networks_stay_finite_where_m0_vanishes():
    """m2/m0 at m0 <= 0 overflowed to inf - inf = NaN in the floor and in the
    network's inputs (FP32 overflows at 3e38), and the NaN ran on silently
    (STARBENCH 128^3, substep ~43). The floor of such a cell is realizable."""
    U = _unfloored()
    V = m2_split.floor(U)
    assert bool(jnp.all(jnp.isfinite(V)))
    w = np.linalg.eigvalsh(np.moveaxis(np.asarray(m2_3d.tensor(V)), (0, 1), (-2, -1)))
    assert (w >= -1e-14 * np.asarray(V[0])[:, None]).all()
    assert (np.asarray(jnp.sqrt(jnp.sum(V[1:4] ** 2, axis=0))) <= np.asarray(V[0]) * (1 + 1e-14)).all()
    params = _learned_params()
    for out in (m2_split.free_components(U, params), ml.third_moment(params, U),
                m2_pallas.free_components(U.reshape(10, 1, 1, 5), params,
                                          interpret=m2_pallas.interpret_mode())):
        assert bool(jnp.all(jnp.isfinite(out)))


def test_the_closure_inputs_are_unchanged_where_the_state_is_realizable():
    """The overflow guard divides by m0 wherever the state is realizable: bitwise the old inputs."""
    U = _states((64,), seed=11)
    m0 = jnp.maximum(U[0], m2_3d.EPS)
    t = [U[4 + k] / m0 for k in range(6)]
    old = jnp.stack([U[1 + i] / m0 for i in range(3)] + [t[0] - t[1], 3.0 * t[2] - 1.0, t[3], t[4], t[5]])
    np.testing.assert_array_equal(np.asarray(m2_3d.targets(U)), np.asarray(old))
    V = m2_split.floor(U)
    np.testing.assert_array_equal(np.asarray(m2_3d.density_scale([V[k] for k in range(10)])),
                                  np.asarray(jnp.maximum(V[0], m2_3d.EPS)))
