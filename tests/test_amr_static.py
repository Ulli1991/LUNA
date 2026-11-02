"""Static (geometric) refinement criterion (criterion="static").

Solution-INDEPENDENT: refines exactly the blocks whose center lies in
``cfg.static_region``, regardless of the field. That fixed coarse-fine boundary
is what the SMR convergence-order tests rely on.
"""
import numpy as np
import jax

from luna.amr import driver
from tests._helpers import GAMMA, cfg3d, uniform_U as _U


def _n_children(criterion, region, n=64, bs=16):
    cfg = cfg3d(n, bs=bs, max_levels=1, boundary=1, gamma=GAMMA, static_region=region)
    h = driver.build_hierarchy(_U(np.ones((n, n, n))), cfg, thresholds=[0.1],
                               criterion=criterion, periodic=False, pad_to=1)
    return int(h[1].n_active), h


def test_static_is_solution_independent():
    """On uniform data a gradient criterion refines nothing; 'static' refines
    its region regardless."""
    assert _n_children("density", None)[0] == 0
    assert _n_children("static", ((0.25, 0.75), (0.0, 1.0), (0.0, 1.0)))[0] > 0


def test_static_region_is_the_central_x_slab():
    """region x in [0.25,0.75) on a 4-blocks/dim base -> only the 2 central
    x-columns of blocks refine (2*4*4 parents -> 32*8 children), and every child
    lies in the covered x-range."""
    na, h = _n_children("static", ((0.25, 0.75), (0.0, 1.0), (0.0, 1.0)), n=64, bs=16)
    assert na == 2 * 4 * 4 * 8                       # 256 children
    coords = np.asarray(jax.device_get(h[1].coords))[:na]
    # level-1 x-coords span children of parent columns 1,2 -> fine cols 2..5
    assert coords[:, 0].min() == 2 and coords[:, 0].max() == 5
