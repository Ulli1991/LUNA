"""Conservation of the AMR advance — the core invariant the whole reflux +
fine-authoritative-restrict machinery exists to guarantee.

- A block-uniform (max_levels=0) periodic advance is a plain conservative FV
  scheme: total mass/energy are conserved to machine precision.
- With refinement + reflux, the coarse level (fully overwritten by restrict) sees
  exactly the transverse-averaged fine fluxes, so its total mass telescopes and
  stays conserved across a coarse-fine interface.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from amr.config import AMRConfig
from amr import driver

GAMMA = 5.0 / 3.0


def _blob(n):
    x = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r2 = (X - .5) ** 2 + (Y - .5) ** 2 + (Z - .5) ** 2
    rho = 1.0 + 0.5 * np.exp(-r2 / (2 * 0.09 ** 2))
    p = 1.0 + 0.3 * np.exp(-r2 / (2 * 0.09 ** 2))
    u = 0.2 * np.sin(2 * np.pi * X)                   # nonzero flow so fluxes are real
    U = np.zeros((5, n, n, n))
    U[0] = rho
    U[1] = rho * u
    U[4] = p / (GAMMA - 1.0) + 0.5 * rho * u * u
    return jnp.asarray(U)


def _total_mass(level, dx):
    na = int(level.n_active)
    return float(np.nansum(np.asarray(jax.device_get(level.data))[:na, 0])) * dx ** 3


def test_block_uniform_conserves_mass_periodic():
    """max_levels=0 periodic advance conserves total mass to machine precision."""
    n = 32
    cfg = AMRConfig(ndim=3, box=(1.,) * 3, base_grid=(n,) * 3, block_shape=(16,) * 3,
                    max_levels=0, boundary=0, gamma=GAMMA)
    h = driver.build_hierarchy(_blob(n), cfg, thresholds=[0.0], criterion="static",
                               periodic=True, pad_to=1)
    dx = 1.0 / n
    m0 = _total_mass(h[0], dx)
    step = jax.jit(lambda hh: driver.advance(hh, jnp.asarray(2e-3), cfg, gamma=GAMMA))
    for _ in range(5):
        h = step(h)
    assert abs(_total_mass(h[0], dx) - m0) / m0 < 1e-12


def test_two_level_reflux_conserves_coarse_mass_periodic():
    """2-level periodic advance with reflux: total mass on the (restrict-covered)
    coarse level is conserved -> the C/F fluxes telescope."""
    n = 32
    cfg = AMRConfig(ndim=3, box=(1.,) * 3, base_grid=(n,) * 3, block_shape=(16,) * 3,
                    max_levels=1, boundary=0, gamma=GAMMA,
                    static_region=((0.25, 0.75), (0.25, 0.75), (0.25, 0.75)))
    h = driver.build_hierarchy(_blob(n), cfg, thresholds=[0.0], criterion="static",
                               periodic=True, pad_to=1)
    dx0 = 1.0 / n
    m0 = _total_mass(h[0], dx0)
    step = jax.jit(lambda hh: driver.advance(hh, jnp.asarray(2e-3), cfg, gamma=GAMMA,
                                             reflux=True))
    for _ in range(5):
        h = step(h)
    assert abs(_total_mass(h[0], dx0) - m0) / m0 < 1e-10
