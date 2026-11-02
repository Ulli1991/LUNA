"""Löhner (1987) normalized-2nd-derivative refinement estimator (criterion="lohner").

The defining property vs a raw gradient: it fires on discontinuities / curvature
and VANISHES on a smooth linear ramp (2nd derivative = 0), so its threshold is
resolution-independent. These pin that behaviour.
"""
import numpy as np
import jax

from luna.amr import driver
from tests._helpers import GAMMA, cfg3d, uniform_U as _U


def _flags(rho, criterion, thr, bs=8):
    n = rho.shape[0]
    cfg = cfg3d(n, bs=bs, max_levels=0, boundary=1, gamma=GAMMA)
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
