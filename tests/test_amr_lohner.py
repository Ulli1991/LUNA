"""Löhner (1987) normalized-2nd-derivative refinement estimator (criterion="lohner").

The defining property vs a raw gradient: it fires on discontinuities / curvature
and VANISHES on a smooth linear ramp (2nd derivative = 0), so its threshold is
resolution-independent. These pin that behaviour.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from amr.config import AMRConfig
from amr import driver

GAMMA = 5.0 / 3.0


def _U(rho):
    """3D conserved state from a density field, uniform pressure, zero velocity."""
    E = np.ones_like(rho) / (GAMMA - 1.0)
    z = np.zeros_like(rho)
    return jnp.asarray(np.stack([rho, z, z, z, E], axis=0))


def _flags(rho, criterion, thr, bs=8):
    n = rho.shape[0]
    cfg = AMRConfig(ndim=3, box=(1.,) * 3, base_grid=(n,) * 3, block_shape=(bs,) * 3,
                    max_levels=0, boundary=1, gamma=GAMMA)
    h = driver.build_hierarchy(_U(rho), cfg, thresholds=[thr], criterion=criterion,
                               periodic=False, pad_to=1)
    f = np.asarray(jax.device_get(driver._flag_blocks(h[0], cfg, thr, 0, criterion)))
    return int(f[:int(h[0].n_active)].sum())


def test_lohner_ignores_smooth_ramp():
    """A linear density ramp has zero 2nd derivative -> Löhner flags nothing,
    while the gradient criterion (nonzero constant slope) does."""
    n = 32
    x = (np.arange(n) + 0.5) / n
    rho = 1.0 + 0.8 * np.broadcast_to(x[:, None, None], (n, n, n)).copy()
    assert _flags(rho, "lohner", 0.2) == 0
    assert _flags(rho, "density", 0.001) > 0        # gradient IS fooled by the ramp


def test_lohner_fires_on_discontinuity():
    """A sharp step (large 2nd derivative) triggers Löhner refinement."""
    n = 32
    x = (np.arange(n) + 0.5) / n
    s = 1.0 + 0.8 * (x > 0.5).astype(float)
    rho = np.broadcast_to(s[:, None, None], (n, n, n)).copy()
    assert _flags(rho, "lohner", 0.2) > 0
