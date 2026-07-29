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

from amr import driver
from tests._helpers import GAMMA, blob_U as _blob, cfg3d


def test_subcycle_nonperiodic_stays_finite():
    """A 2-level outflow (boundary=1) subcycled step leaves NO NaN in the active
    blocks (edge ghosts get the physical BC, not NaN)."""
    n = 32
    cfg = cfg3d(n, max_levels=1, boundary=1, gamma=GAMMA)
    h = driver.build_hierarchy(_blob(n), cfg, thresholds=[1.1],
                               criterion="overdensity", periodic=False, pad_to=1)
    out = driver.advance_subcycle(h, jnp.asarray(2e-3), cfg, gamma=GAMMA)
    for L, lvl in enumerate(out):
        na = int(h[L].n_active)
        d = np.asarray(jax.device_get(lvl.data))[:na]
        assert np.all(np.isfinite(d)), f"level {L} has non-finite cells"
