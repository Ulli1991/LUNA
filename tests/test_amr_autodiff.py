"""Reverse-mode AD through the AMR advance must stay alive.

Guards the NaN-sentinel AD barriers in amr/driver.py (_fill_ghosts +
the subcycle ghost fill): without them, 0-cotangent x NaN-partial poisons
every gradient. Loss = sum of squares over active blocks after NSTEP fixed-dt
steps; differentiated w.r.t. the FINEST level's data (never overwritten by
restriction, so sensitivity is guaranteed) and checked against a central
finite difference.
"""
import pytest
pytestmark = pytest.mark.slow

import jax
import jax.numpy as jnp
import numpy as np


from luna.amr.config import AMRConfig
from luna.amr import driver
from tests.test_amr_multilevel import _kh_ic

GAMMA = 5.0 / 3.0
NSTEP = 2


def _setup():
    cfg = AMRConfig(ndim=2, box=(1.0, 1.0), base_grid=(32, 32),
                    block_shape=(8, 8), max_levels=2)
    h = driver.build_hierarchy(_kh_ic(cfg, GAMMA), cfg, thresholds=[0.05, 0.05])
    dt = 0.5 * float(driver.compute_dt(h, cfg, gamma=GAMMA, CFL=0.8))
    n_act = [int(lv.n_active) for lv in h]
    return cfg, h, dt, n_act


def _check(advance_fn):
    cfg, h, dt, n_act = _setup()
    top = len(h) - 1

    def loss(data_f):
        hh = tuple(lv._replace(data=data_f) if L == top else lv
                   for L, lv in enumerate(h))
        for _ in range(NSTEP):
            hh = advance_fn(hh, dt, cfg, gamma=GAMMA)
        return sum(jnp.sum(lv.data[:na] ** 2) for lv, na in zip(hh, n_act))

    g = np.asarray(jax.grad(loss)(h[top].data))
    assert np.isfinite(g[:n_act[top]]).all(), "NaN in AMR gradient (AD barrier lost?)"
    assert np.abs(g[:n_act[top]]).max() > 1e-3, "gradient unexpectedly zero"

    idx = next((i, 0, 4, 4) for i in range(n_act[top])
               if abs(g[i, 0, 4, 4]) > 1e-3)
    eps = 1e-6
    d = h[top].data
    fd = (float(loss(d.at[idx].add(eps))) - float(loss(d.at[idx].add(-eps)))) / (2 * eps)
    assert abs(float(g[idx]) - fd) / (abs(fd) + 1e-30) < 1e-3, (float(g[idx]), fd)


def test_grad_through_advance():
    _check(driver.advance)


def test_grad_through_advance_subcycle():
    _check(driver.advance_subcycle)
