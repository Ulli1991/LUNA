"""The gravity answer does not depend on the device mesh: at default multigrid depth, however thin the shards,
the sharded solve and the coupled bridge give one device's g to round-off (isolated and periodic)."""
import numpy as np
import pytest

import jax
import jax.numpy as jnp


from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from luna.gravity.backend import PROPER, LocalGravity, PoissonOpts, ShardedGravity
from luna.gravity.particle_mesh import ParticleMeshGravity
from luna.gravity.particle_mesh_shardmap import ShardedParticleBridge
from luna.setup.cosmo_common import GravityPrecision
from tests._helpers import _build_cfg_for_test

N = 32
FOURPIG = 4.0 * np.pi
SHAPES = [(8, 1, 1), (4, 1, 1), (2, 2, 2), (1, 1, 8)]
TOL = 2.5e-12            # round-off: ~1.5x the largest measured (1.7e-12, isolated cold on (2,2,2))


def _cfg(boundary, n=N):
    return _build_cfg_for_test(test_problem="Soundwave-3D", SPATIAL_DIMENSION=3, Nx=n, Ny=n, Nz=n,
                               Lx=1.0, Ly=1.0, Lz=1.0, BOUNDARY=0 if boundary == 0 else 4,
                               POISSON_BOUNDARY=boundary, SELF_GRAVITY=True, enable_x64=True)


def _rho(n=N):
    """An off-centre n = 1 polytrope (sinc profile) in a thin ambient."""
    x = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r = np.sqrt((X - 0.47) ** 2 + (Y - 0.53) ** 2 + (Z - 0.5) ** 2)
    return np.where(r < 0.35, np.sinc(r / 0.35), 1e-3)


def _mesh(shape):
    n = int(np.prod(shape))
    if jax.device_count() < n:
        pytest.skip(f"needs {n} devices")
    return Mesh(np.asarray(jax.devices()[:n]).reshape(shape), ("x", "y", "z"))


def _put(a, mesh):
    return jax.device_put(jnp.asarray(a), NamedSharding(mesh, P("x", "y", "z")))


def _g(backend, psi):
    return np.stack([np.asarray(c) for c in backend.grad(psi, 2)])


def _cold_and_warm(backend, src):
    """g from the cold solve and after one warm solve of a slightly heavier source."""
    psi = backend.solve(src, None)
    return _g(backend, psi), _g(backend, backend.solve(1.01 * src, psi))


@pytest.mark.parametrize("boundary", [3, 0])
@pytest.mark.parametrize("shape,n", [(s, N) for s in SHAPES] + [((2, 2, 2), 128), ((8, 1, 1), 128)])
def test_sharded_solve_is_one_device_at_default_depth(shape, n, boundary):
    """The run's backends at default options: cold and warm g on a mesh equal one device's (128: shard levels + root grid)."""
    mesh = _mesh(shape)
    cfg = _cfg(boundary, n)
    opts = PoissonOpts.from_config(cfg, cfg.gravity_warm_vcycles)
    if n == 128:        # keep shard levels below the finest: the default agg_n=128 gathers the whole 128^3
        from dataclasses import replace
        opts = replace(opts, mg_agg_n=32)
    prec = GravityPrecision("double")
    rho = _rho(n)
    src = FOURPIG * (rho - rho.mean() if boundary == 0 else rho)
    one = LocalGravity(ParticleMeshGravity(cfg), 1.0, opts, prec, PROPER, FOURPIG)
    sh = ShardedGravity(mesh, (n,) * 3, 1.0 / n, 1.0, boundary, opts, prec, PROPER, FOURPIG)
    ref = _cold_and_warm(one, jnp.asarray(src))
    got = _cold_and_warm(sh, _put(src, mesh))
    d = [float(np.max(np.abs(a - b)) / np.max(np.abs(a))) for a, b in zip(ref, got)]
    print(f"MEASURED {shape} n={n} bnd={boundary}: g sharded vs one device, cold {d[0]:.2e} warm {d[1]:.2e}")
    assert max(d) < TOL


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="open bug: isolated 128^3 gathered whole (agg_n 128) is 2.7e-11 off one device, also before 2026-10-11")
@pytest.mark.parametrize("shape", [(2, 2, 2), (8, 1, 1)])
def test_isolated_whole_grid_gathered_is_one_device(shape):
    """At the default agg_n=128 a 128^3 isolated grid is gathered whole: g should equal one device's to round-off."""
    mesh = _mesh(shape)
    cfg = _cfg(3, 128)
    opts = PoissonOpts.from_config(cfg, cfg.gravity_warm_vcycles)
    assert opts.mg_agg_n == 128
    prec = GravityPrecision("double")
    src = FOURPIG * _rho(128)
    one = LocalGravity(ParticleMeshGravity(cfg), 1.0, opts, prec, PROPER, FOURPIG)
    sh = ShardedGravity(mesh, (128,) * 3, 1.0 / 128, 1.0, 3, opts, prec, PROPER, FOURPIG)
    ref = _cold_and_warm(one, jnp.asarray(src))
    got = _cold_and_warm(sh, _put(src, mesh))
    d = [float(np.max(np.abs(a - b)) / np.max(np.abs(a))) for a, b in zip(ref, got)]
    print(f"MEASURED whole-grid {shape}: g sharded vs one device, cold {d[0]:.2e} warm {d[1]:.2e}")
    assert max(d) < TOL


@pytest.mark.parametrize("boundary", [3, 0])
@pytest.mark.parametrize("shape", SHAPES)
def test_bridge_is_one_device_at_default_depth(shape, boundary):
    """The coupled worker's sharded bridge at default depth: its gas field equals the one-device bridge's."""
    mesh = _mesh(shape)
    cfg = _cfg(boundary)
    rho = _rho()
    one = np.stack([np.asarray(c) for c in ParticleMeshGravity(cfg).gas_accel_grid(jnp.asarray(rho))])
    sh = np.stack([np.asarray(c) for c in ShardedParticleBridge(cfg, mesh).accel_field(_put(rho, mesh))])
    d = float(np.max(np.abs(sh - one)) / np.max(np.abs(one)))
    print(f"MEASURED bridge {shape} bnd={boundary}: g sharded vs one device {d:.2e}")
    assert d < TOL
