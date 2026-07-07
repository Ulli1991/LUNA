"""Berger-Oliger subcycle advance under a NON-periodic (outflow) boundary.

Regression for the subcycle physical-BC fix: ``ghost_prim`` now applies
``fill_ghost_physical`` after the same-level + coarse-underlay fill, so domain-
edge ghosts are filled instead of left NaN. Before the fix a boundary!=0
subcycled step left the edge blocks NaN (the subcycle was periodic-only).
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
    U = np.zeros((5, n, n, n))
    U[0] = rho
    U[4] = p / (GAMMA - 1.0)
    return jnp.asarray(U)


def test_subcycle_nonperiodic_stays_finite():
    """A 2-level outflow (boundary=1) subcycled step leaves NO NaN in the active
    blocks (edge ghosts get the physical BC, not NaN)."""
    n = 32
    cfg = AMRConfig(ndim=3, box=(1.,) * 3, base_grid=(n,) * 3, block_shape=(16,) * 3,
                    max_levels=1, boundary=1, gamma=GAMMA)
    h = driver.build_hierarchy(_blob(n), cfg, thresholds=[1.1],
                               criterion="overdensity", periodic=False, pad_to=1)
    out = driver.advance_subcycle(h, jnp.asarray(2e-3), cfg, gamma=GAMMA)
    for L, lvl in enumerate(out):
        na = int(h[L].n_active)
        d = np.asarray(jax.device_get(lvl.data))[:na]
        assert np.all(np.isfinite(d)), f"level {L} has non-finite cells"
