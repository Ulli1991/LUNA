"""shard_map distribution of the AMR block step.

Runs on whatever devices are present (1 in plain CI). For a real multi-device
check run with simulated devices:
    XLA_FLAGS="--xla_force_host_platform_device_count=4" \
    JAX_PLATFORMS=cpu ~/jax-env/bin/python -m pytest tests/test_amr_shard.py
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from amr.config import AMRConfig
from amr import driver, shard
from tests.test_amr_multilevel import _kh_ic

GAMMA = 5.0 / 3.0


def test_shard_map_multilevel_subcycle_matches_single_device():
    """The full sharded paths (synchronized + subcycled, with coarse-fine) match
    the single-device driver on the active blocks, for however many devices."""
    devs = jax.devices()
    D = len(devs)
    cfg = AMRConfig(ndim=2, box=(1.0, 1.0), base_grid=(32, 32), block_shape=(8, 8), max_levels=2)
    U = _kh_ic(cfg, GAMMA)
    h = driver.build_hierarchy(U, cfg, thresholds=[0.05, 0.05], pad_to=D)
    mesh = Mesh(np.asarray(devs), ("blocks",))

    dt = float(driver.compute_dt(h, cfg, gamma=GAMMA, CFL=0.4))
    ref = driver.advance(h, dt, cfg, gamma=GAMMA)
    out = jax.jit(lambda hh: shard.advance_shmap(hh, dt, cfg, mesh, gamma=GAMMA))(h)
    assert all(jnp.allclose(o.data[: int(o.n_active)], r.data[: int(r.n_active)], atol=1e-10)
               for o, r in zip(out, ref))

    dtb = float(driver.compute_dt_subcycle(h, cfg, gamma=GAMMA, CFL=0.4))
    ref_s = driver.advance_subcycle(h, dtb, cfg, gamma=GAMMA)
    out_s = jax.jit(lambda hh: shard.advance_subcycle_shmap(hh, dtb, cfg, mesh, gamma=GAMMA))(h)
    assert all(jnp.allclose(o.data[: int(o.n_active)], r.data[: int(r.n_active)], atol=1e-10)
               for o, r in zip(out_s, ref_s))


def test_shard_subcycle_gravity_reflux_matches_single_device():
    """The sharded Berger-Oliger step WITH per-level gravity + time-accumulated
    reflux matches the single-device driver (the GSPMD reflux trap is handled
    inside advance_subcycle_shmap via the replicated-shard_map reflux_fn).
    3D: the dense FMG that solve_phi_level uses at L0 is 3D-only (all validated
    AMR-gravity applications are 3D)."""
    from amr.poisson import _ghost_phi, kick_level, solve_phi_level
    from amr.state import assemble_base_level

    devs = jax.devices()
    D = len(devs)
    cfg = AMRConfig(ndim=3, box=(1.0,) * 3, base_grid=(32,) * 3,
                    block_shape=(16,) * 3, max_levels=1, boundary=0)
    x = (np.arange(32) + 0.5) / 32.0
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2
    U = np.zeros((5, 32, 32, 32))
    U[0] = 1.0 + 3.0 * np.exp(-r2 / (2 * 0.08 ** 2))
    U[4] = 1.0 / (GAMMA - 1.0)
    U = jnp.asarray(U)
    h = driver.build_hierarchy(U, cfg, thresholds=[1.5], criterion="overdensity",
                               pad_to=D, periodic=True)
    assert int(h[1].n_active) > 0
    mesh = Mesh(np.asarray(devs), ("blocks",))

    fourpiG = 4.0 * jnp.pi
    n_base = int(np.prod(cfg.base_blocks))
    st = {"m": jnp.asarray(0.0)}

    def gravity(L, lvl, parent_phig, dt, t_sub):
        if L == 0:
            st["m"] = jnp.mean(assemble_base_level(lvl.data[:n_base, 0:1], cfg)[0])
        phi = solve_phi_level(lvl, parent_phig, cfg, fourpiG=fourpiG, boundary=0,
                              mean_rho=st["m"], level=L, n_vcycles=2)
        return (kick_level(lvl, phi, parent_phig, dt, L, cfg),
                _ghost_phi(phi, lvl, parent_phig, cfg))

    dtb = float(driver.compute_dt_subcycle(h, cfg, gamma=GAMMA, CFL=0.4))
    ref = driver.advance_subcycle(h, dtb, cfg, gamma=GAMMA,
                                  gravity=gravity, reflux=True)
    out = jax.jit(lambda hh: shard.advance_subcycle_shmap(
        hh, dtb, cfg, mesh, gamma=GAMMA, gravity=gravity, reflux=True))(h)
    diffs = [float(jnp.max(jnp.abs(o.data[: int(o.n_active)]
                                   - r.data[: int(r.n_active)])))
             for o, r in zip(out, ref)]
    assert max(diffs) < 1e-10, \
        f"shard vs single-device per-level max|diff| = {diffs}"
