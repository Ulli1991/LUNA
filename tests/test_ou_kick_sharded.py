"""The sharded OU kick (luna/sources/ou_forcing.py) on a 2x2x2 mesh and on
one device, with axes of different length, equals the driver applied with
the global coordinates built on the host, on every shard.

Written while hunting the 1024^3 U[4] corruption (docs/AUTODIFF.md, "What
the reference found"): the kick was a suspect and is exact; the fix was the
hydro's optimization barrier.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp                                    # noqa: E402

SHAPE = (16, 8, 12)


def _setup(mesh):
    from dataclasses import replace
    from luna.utils import distributed
    from tools import sil3d
    devs = jax.devices()[:int(np.prod(mesh))]
    real = jax.devices, distributed.mesh_utils.create_device_mesh
    jax.devices = lambda *a, **k: devs
    distributed.mesh_utils.create_device_mesh = \
        lambda shape, *a, **k: np.array(devs).reshape(shape)
    try:
        cfg = replace(sil3d.config(SHAPE[0], 2.0, 7, mesh), Ny=SHAPE[1], Nz=SHAPE[2],
                      Ly=0.5, Lz=0.75)
        cfg, init, physics = sil3d.assemble_problem(cfg)
    finally:
        jax.devices, distributed.mesh_utils.create_device_mesh = real
    return cfg, init, physics


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 2, 2)], ids=["1dev", "2x2x2"])
def test_kick_equals_the_driver_on_global_coordinates(mesh):
    from luna.hydro.turbulence import OUDriver
    if jax.device_count() < np.prod(mesh):
        pytest.skip(f"needs {int(np.prod(mesh))} devices (CPU-8 runs it)")
    cfg, init, physics = _setup(mesh)
    forcing = next(s for s in physics.stages if s.name == "forcing")
    a = forcing.carry0[0]
    rng = np.random.default_rng(0)
    host = np.concatenate([1.0 + 0.1 * rng.random((1,) + SHAPE),
                           0.3 * rng.standard_normal((3,) + SHAPE),
                           3.0 + rng.random((1,) + SHAPE)])
    U = jax.make_array_from_callback(host.shape, init.U.sharding, lambda idx: host[idx])
    h = 0.013
    got = forcing._kick(U, a, h)
    drv = OUDriver(cfg.ou_config, dtype=float)
    c = [(np.arange(n) + 0.5) * L / n for n, L in zip(SHAPE, (cfg.Lx, cfg.Ly, cfg.Lz))]
    ref = np.asarray(drv.apply(jnp.asarray(host), a, h, *(jnp.asarray(ci) for ci in c)))
    assert np.max(np.abs(ref[1:4] - host[1:4])) > 1e-4           # the kick does something
    for s in got.addressable_shards:
        np.testing.assert_allclose(np.asarray(s.data), ref[s.index], rtol=1e-13, atol=1e-15,
                                   err_msg=str(s.index))
