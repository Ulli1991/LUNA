"""Each gravity backend solves with its own multigrid options and kick kernel: building a second
backend with other settings in the same process must not change the first one's solves."""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jax.sharding import Mesh

from luna.gravity.backend import PROPER, PoissonOpts, ShardedGravity
from luna.setup.cosmo_common import GravityPrecision


N = 16


def _mesh(ndev):
    return Mesh(np.asarray(jax.devices()[:ndev]).reshape(ndev, 1, 1), ("x", "y", "z"))


def _src(seed, n=N):
    rng = np.random.default_rng(seed)
    rho = 1.0 + 0.5 * rng.random((n, n, n))
    return jnp.asarray(4.0 * np.pi * (rho - rho.mean()))


def _build(ndev, mode, n=N, **opts):
    o = PoissonOpts(warm_cycles=2, **opts)
    return ShardedGravity(_mesh(ndev), (n,) * 3, 1.0 / n, 1.0, 0, o, GravityPrecision(mode),
                          PROPER, 4.0 * np.pi)


def _warm(backend):
    """A warm solve from a cold seed: the path the mixed cycle changes."""
    n = backend.N3[0]
    seed = backend.solve(_src(0, n), None)
    return np.asarray(backend.solve(_src(1, n), seed))


@pytest.mark.parametrize("ndev", [1, 2])
def test_a_second_backend_does_not_change_the_first(ndev):
    if jax.device_count() < ndev:
        pytest.skip(f"needs {ndev} devices")
    A = _build(ndev, "double")
    a_alone = _warm(A)
    B = _build(ndev, "mixed")
    b_alone = _warm(B)
    assert a_alone.tobytes() != b_alone.tobytes()          # the mixed cycle is a different solve
    assert _warm(A).tobytes() == a_alone.tobytes()          # A after B was built: still FP64
    _build(ndev, "double")
    assert _warm(B).tobytes() == b_alone.tobytes()          # B after another A: still mixed


def test_smoother_options_are_per_backend():
    if jax.device_count() < 2:
        pytest.skip("needs 2 devices")
    # 64 with agg_n 32: a level above the all_gather coarse solve, where smoothing runs
    A = _build(2, "double", n=64, mg_agg_n=32)
    a_alone = _warm(A)
    B = _build(2, "double", n=64, mg_agg_n=32, mg_ca="1", mg_smoother="cheb")
    b = _warm(B)
    assert b.tobytes() != a_alone.tobytes()                 # the options reach the solve
    assert _warm(A).tobytes() == a_alone.tobytes()


def test_options_come_from_the_config(monkeypatch):
    from dataclasses import replace
    from luna.template_config import Config
    monkeypatch.setenv("LUNA_MG_MIXED", "0")                # retired: no effect
    o = PoissonOpts.from_config(replace(Config(), gravity_mg_ca="1",
                                        gravity_mg_smoother="cheb"), 0)
    assert (o.mg_ca, o.mg_smoother) == ("1", "cheb")
    B = _build(1, "mixed", mg_ca="1", mg_smoother="cheb")
    assert B._mg_kwargs() == dict(mixed=True, ca="1", smoother="cheb", agg_n=128)


def test_kick_kernel_is_a_constructor_argument():
    from luna.gravity.backend import _fused_sink_kick
    assert _build(1, "double").kernel == "auto"
    o = PoissonOpts()
    B = ShardedGravity(_mesh(1), (N,) * 3, 1.0 / N, 1.0, 0, o, GravityPrecision("double"),
                       PROPER, 1.0, kernel="xla")
    assert B.kernel == "xla"
    assert _fused_sink_kick(jnp.zeros((5, 4, 4, 4)), ((None, ()),), "xla") is None
    with pytest.raises(ValueError):
        ShardedGravity(_mesh(1), (N,) * 3, 1.0 / N, 1.0, 0, o, GravityPrecision("double"),
                       PROPER, 1.0, kernel="cuda")
