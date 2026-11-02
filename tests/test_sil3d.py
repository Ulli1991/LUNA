"""The 3D solver-in-the-loop corrector (tools/sil3d.py, ml_corrector.py):

  1. the corrected update is conservative: the correction sums to zero per
     field to round-off, and the corrected step keeps the bare step's totals
  2. zero weights are exactly the bare solver (the corrector starts off)
  3. the parameter gradient of the SiL loss on a 2x2x2 mesh equals the one on
     1 device (replicated parameters, gradient summed over the shards)
  4. the parameter gradient passes a Taylor test, and the revolve gradient
     equals the per-step-checkpoint one
  5. with sil_weights unset (the default) there is no corrector stage, and the
     production hydro step (behind its optimization barrier) is the raw
     solver's bit for bit, alone and fused with the kick before it
on the production coarse step (isothermal driven turbulence, replayed
forcing, minmod, FP64), 16^3, with a generic state.
"""
from contextlib import contextmanager
from functools import lru_cache

import numpy as np
import pytest

import jax
import jax.numpy as jnp                                    # noqa: E402

NX, MACH = 16, 2.0
DT = 0.3 / NX / (1.0 + 3.0 * MACH)
META = dict(rho0=1.0, c=1.0, amp=1.0)
TOL = 1.0e-11          # the identity tests' bound for a sharded vs 1-device derivative


def _smooth(shape, seed):
    rng = np.random.default_rng(seed)
    f = np.fft.rfftn(rng.standard_normal(shape))
    k = np.sqrt(sum(np.meshgrid(*[np.fft.fftfreq(n) ** 2 for n in shape[:-1]],
                                np.fft.rfftfreq(shape[-1]) ** 2, indexing="ij")))
    f *= np.exp(-(k / 0.2) ** 2)
    d = np.fft.irfftn(f, s=shape)
    return d / d.std()


def _host_state():
    g = 5.0 / 3.0
    rho = np.exp(0.5 * _smooth((NX,) * 3, 1))
    v = MACH * np.stack([_smooth((NX,) * 3, 2 + i) for i in range(3)]) / np.sqrt(3)
    E = rho / (g - 1.0) + 0.5 * rho * np.sum(v * v, axis=0)
    return np.concatenate([rho[None], rho * v, E[None]])


KINDS = ["sep", "dense"]


def _params(seed=3, last=0.05, kind="sep"):
    from luna.sources.ml_corrector import cnn3d_init
    from tools.sil3d import randomize_last
    return randomize_last(cnn3d_init(jax.random.PRNGKey(seed), width=8, depth=3, kind=kind),
                          last, seed + 1)


@contextmanager
def _on_devices(mesh):
    """Build on the first prod(mesh) devices; skip when there are fewer (the
    2x2x2 cases run on CPU-8, not on a 2-GPU node)."""
    from luna.utils import distributed
    if jax.device_count() < np.prod(mesh):
        pytest.skip(f"needs {int(np.prod(mesh))} devices (CPU-8 runs it)")
    devs = jax.devices()[:int(np.prod(mesh))]
    real = jax.devices, distributed.mesh_utils.create_device_mesh
    jax.devices = lambda *a, **k: devs
    distributed.mesh_utils.create_device_mesh = \
        lambda shape, *a, **k: np.array(devs).reshape(shape)
    try:
        yield
    finally:
        jax.devices, distributed.mesh_utils.create_device_mesh = real


@lru_cache(maxsize=None)
def _problem(mesh):
    """(init, with_params, U0, targets, table) on the first prod(mesh) devices."""
    from tools import sil3d
    with _on_devices(mesh):
        cfg, init, with_params = sil3d.coarse_problem(sil3d.config(NX, MACH, 7, mesh), META, 3)
    sh = init.U.sharding
    host = _host_state()
    U0 = jax.make_array_from_callback(host.shape, sh, lambda idx: host[idx])
    # targets: the observed fields of a nearby state, three of them
    tg_host = np.stack([np.asarray(sil3d.observed(jnp.asarray(host)))
                        + 0.01 * j * _smooth((4,) + (NX,) * 3, 9) for j in range(3)])
    tsh = jax.sharding.NamedSharding(sh.mesh, jax.sharding.PartitionSpec(None, None, *sil3d.SPEC))
    tg = jax.make_array_from_callback(tg_host.shape, tsh, lambda idx: tg_host[idx])
    from luna.hydro.turbulence import OUDriver
    drv = OUDriver(cfg.ou_config, dtype=float)
    a = drv.init_state(jax.random.PRNGKey(5)).astype(jnp.complex128)
    table = jnp.stack([a * (1.0 + 0.1 * k) for k in range(5)])
    return init, with_params, U0, tg, table


def _loss(mesh, ckpt="revolve", snaps=2):
    from tools import sil3d
    init, with_params, U0, tg, table = _problem(mesh)
    loss = sil3d.make_loss(with_params, init, n_steps=4, every=2, dt=DT, norm=(0.3, 1.5),
                           ckpt=ckpt, snaps=snaps)
    return loss, U0, tg, table


# ---------------------------------------------------------------------------
# 0. the matmul convolution (ROCm has no FP64 conv) is XLA's convolution
# ---------------------------------------------------------------------------

def test_conv3_equals_the_direct_sum():
    from luna.sources.ml_corrector import _conv3
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    h = jax.random.normal(k1, (5, 9, 7, 8))
    W = jax.random.normal(k2, (6, 5, 3, 3, 3))
    b = jax.random.normal(k3, (6,))
    # the reference on the host: ROCm has no FP64 lax.conv to compare against
    hn, Wn, bn = np.asarray(h), np.asarray(W), np.asarray(b)
    ref = bn[:, None, None, None] + sum(
        np.einsum("oc,cxyz->oxyz", Wn[:, :, i, j, k], hn[:, i:i + 7, j:j + 5, k:k + 6])
        for i in range(3) for j in range(3) for k in range(3))
    got = _conv3(h, W, b)
    assert got.shape == ref.shape == (6, 7, 5, 6)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=0, atol=1e-12)


def test_separable_layer_is_a_factored_convolution():
    """(D, P, b) is the dense convolution with W[o, c] = P[o, c] D[c]."""
    from luna.sources.ml_corrector import _conv3, _layer
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(1), 4)
    h = jax.random.normal(k1, (5, 9, 7, 8))
    D = jax.random.normal(k2, (5, 3, 3, 3))
    P = jax.random.normal(k3, (6, 5))
    b = jax.random.normal(k4, (6,))
    W = P[:, :, None, None, None] * D[None]
    np.testing.assert_allclose(np.asarray(_layer(h, (D, P, b))),
                               np.asarray(_conv3(h, W, b)), rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# 1-2. conservation, and zero weights = the bare solver
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 2, 2)], ids=["1dev", "2x2x2"])
def test_correction_is_conservative(mesh, kind):
    init, with_params, U0, _, table = _problem(mesh)
    corr = with_params(None).stages[-1]
    theta = _params(kind=kind)
    dU = np.asarray(jax.jit(lambda th, U: corr._apply(th, U, DT) - U)(theta, U0))
    assert np.max(np.abs(dU[0])) > 1e-4 * DT, "the correction does nothing"
    for k in range(5):
        scale = np.sum(np.abs(dU[k]))
        assert abs(np.sum(dU[k])) <= 1e-13 * scale, (k, np.sum(dU[k]), scale)


def test_corrector_is_one_compiled_program(caplog):
    """Called eagerly, as the time loop calls it: one program, compiled once for any dt."""
    import logging
    init, with_params, U0, _, _ = _problem((1, 1, 1))
    corr = with_params(None).stages[-1]
    theta = _params()
    jax.clear_caches()
    with jax.log_compiles(True), caplog.at_level(logging.WARNING):
        for dt in (DT, 0.7 * DT):
            jax.block_until_ready(corr._apply(theta, U0, dt))
    n = sum("Compiling" in r.getMessage() for r in caplog.records)
    assert n == 1, f"{n} programs compiled for two corrector calls"


def test_corrected_step_keeps_the_totals():
    from luna.setup.evolve import step
    from tools import sil3d
    init, with_params, U0, _, table = _problem((2, 2, 2))
    # the forcing off: its kick rho F h after the corrector depends on the
    # corrected density, so with it on the momentum total legitimately moves
    table = jnp.zeros_like(table)
    theta = _params()
    run = jax.jit(lambda th: step(sil3d.state0(with_params(th), init, U0, table), DT, DT,
                                  with_params(th)).U)
    bare = np.asarray(run(jax.tree.map(jnp.zeros_like, theta)))
    corr = np.asarray(run(theta))
    assert np.max(np.abs(corr[:4] - bare[:4])) > 1e-6
    for k in range(4):                       # mass, momentum (E is the isothermal projection's)
        assert abs(np.sum(corr[k]) - np.sum(bare[k])) <= 1e-13 * np.sum(np.abs(bare[k])), k


@pytest.mark.parametrize("kind", KINDS)
def test_fresh_network_is_the_bare_solver(kind):
    """A freshly initialised network (random hidden layers, zero last layer)
    changes nothing, bit for bit."""
    from dataclasses import replace
    from luna.setup.evolve import step
    from luna.sources.ml_corrector import cnn3d_init
    from tools import sil3d
    init, with_params, U0, _, table = _problem((1, 1, 1))
    fresh = cnn3d_init(jax.random.PRNGKey(4), width=8, depth=3, kind=kind)
    ph = with_params(fresh)
    bare = replace(ph, stages=ph.stages[:-1])
    st = sil3d.state0(ph, init, U0, table)
    # the weights as an argument, as training and eval pass them: closed over,
    # they are constants and ROCm's GEMM autotuner fails on the constant-
    # folded FP64 matmul ("Failed to get configs", jax 0.10.2)
    a = np.asarray(jax.jit(lambda th, s: step(s, DT, DT, with_params(th)).U)(fresh, st))
    b = np.asarray(jax.jit(lambda s: step(s, DT, DT, bare).U)(st))
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# 3-4. the parameter gradient
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _grad(mesh, ckpt="revolve", snaps=2, kind="sep"):
    loss, U0, tg, table = _loss(mesh, ckpt, snaps)
    L, g = jax.jit(jax.value_and_grad(loss))(_params(kind=kind), U0, tg, table)
    return float(L), [np.asarray(x) for x in jax.tree.leaves(g)]


def _flat(g):
    return np.concatenate([x.ravel() for x in g])


@pytest.mark.parametrize("kind", KINDS)
def test_sharded_parameter_gradient_equals_one_device(kind):
    L1, g1 = _grad((1, 1, 1), kind=kind)
    L8, g8 = _grad((2, 2, 2), kind=kind)
    a = _flat(g1)
    assert np.all(np.isfinite(a)) and np.max(np.abs(a)) > 0
    assert abs(L8 - L1) <= TOL * abs(L1), (L1, L8)
    for x, y in zip(g1, g8):                 # every layer, weights and biases
        err = np.max(np.abs(x - y)) / np.max(np.abs(a))
        assert err <= TOL, err


def test_revolve_gradient_equals_per_step():
    _, g_rev = _grad((1, 1, 1), "revolve", 2)
    _, g_step = _grad((1, 1, 1), "step")
    a, b = _flat(g_rev), _flat(g_step)
    assert np.max(np.abs(a - b)) <= 1e-13 * np.max(np.abs(b)), np.max(np.abs(a - b))


@pytest.mark.parametrize("kind", KINDS)
def test_parameter_gradient_taylor(kind):
    """|L(th + e v) - L(th) - e <g, v>| falls as e^2 (the remainder of a
    correct gradient), not as e (the remainder of a wrong one)."""
    loss, U0, tg, table = _loss((1, 1, 1))
    f = jax.jit(loss)
    th = _params(kind=kind)
    L0, g = _grad((1, 1, 1), kind=kind)
    rng = np.random.default_rng(11)
    leaves, tdef = jax.tree.flatten(th)
    # every leaf moves, the zero-initialised biases too
    v = [rng.standard_normal(x.shape) * (np.std(np.asarray(x)) or 1e-2) for x in leaves]
    gv = sum(float(np.sum(a * b)) for a, b in zip(g, v))
    assert abs(gv) > 0
    assert all(float(np.sum(a * b)) != 0.0 for a, b in zip(g, v))
    rem = []
    eps = [1e-2, 5e-3, 2.5e-3, 1.25e-3]
    for e in eps:
        th_e = jax.tree.unflatten(tdef, [x + e * y for x, y in zip(leaves, v)])
        rem.append(abs(float(f(th_e, U0, tg, table)) - L0 - e * gv))
    orders = [np.log2(rem[i] / rem[i + 1]) for i in range(len(eps) - 1)]
    assert all(o > 1.8 for o in orders), (rem, orders)
    # and the first-order term is the gradient's: the central difference at a
    # step that crosses no minmod / min(rho) / HLLC kink (measured sweep: the
    # error is +1.9e-5 at 3e-4, -2.9e-6 at 1e-4, -1.3e-9 at 1e-6)
    e = 1e-6
    th_p = jax.tree.unflatten(tdef, [x + e * y for x, y in zip(leaves, v)])
    th_m = jax.tree.unflatten(tdef, [x - e * y for x, y in zip(leaves, v)])
    fd = (float(f(th_p, U0, tg, table)) - float(f(th_m, U0, tg, table))) / (2 * e)
    assert abs(fd - gv) <= 1e-6 * abs(gv), (fd, gv)


# ---------------------------------------------------------------------------
# 5. SiL off is the default solver, bit for bit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 2, 2)], ids=["1dev", "2x2x2"])
def test_sil_off_is_the_default_solver(mesh):
    from dataclasses import replace
    from luna.setup.hydro import create_solver_distributed
    from luna.setup.evolve import initial_state
    from luna.template_config import Config
    from tools import sil3d
    assert Config().sil_weights == ""
    with _on_devices(mesh):
        cfg, init, physics = sil3d.assemble_problem(sil3d.config(NX, MACH, 7, mesh))
        raw = create_solver_distributed(cfg, physics.hydro.grid, physics.mesh)[1]
    assert [s.name for s in physics.stages] == ["forcing"]
    host = _host_state()
    U = jax.make_array_from_callback(host.shape, init.U.sharding, lambda idx: host[idx])
    np.testing.assert_array_equal(np.asarray(physics.hydro.step(U, DT)),
                                  np.asarray(raw(U, DT)))
    # fused with the kick before it, as in the autodiff scans
    st = initial_state(replace(init, U=U), physics)
    kick = physics.stages[0]

    def fused(hydro_step):
        return jax.jit(lambda s: hydro_step(kick.before(s, 0.5 * DT, DT).U, DT))(st)
    np.testing.assert_array_equal(np.asarray(fused(physics.hydro.step)),
                                  np.asarray(fused(raw)))
