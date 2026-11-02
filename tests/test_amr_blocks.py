"""Tests for block topology + halo (amr/blocks.py, amr/halo.py), 2D and 3D."""

import jax.numpy as jnp
import numpy as np
import pytest


from luna.amr.config import AMRConfig
from luna.amr import blocks as B
from luna.amr import halo as H


def _cfg(ndim, base=64, blk=8, ghost=2, levels=2):
    return AMRConfig(ndim=ndim, box=(1.0,) * ndim, base_grid=(base,) * ndim,
                     block_shape=(blk,) * ndim, max_levels=levels, n_ghost=ghost)


# --------------------------------------------------------------------------
# refinement_block_mask
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ndim", [2, 3])
def test_refinement_block_mask(ndim):
    cfg = _cfg(ndim)
    gm = jnp.zeros((3,) + (cfg.block_shape))
    hot = (1,) + (0,) * ndim  # one cell in block 1
    gm = gm.at[hot].set(1.0)
    flags = np.asarray(B.refinement_block_mask(gm, ndim))
    assert flags.tolist() == [False, True, False]


# --------------------------------------------------------------------------
# find_face_neighbors
# --------------------------------------------------------------------------
def test_neighbors_2d_patch_interior_and_boundary():
    cfg = _cfg(2)  # n_blocks_per_dim(0) = (8,8)
    # 3x3 patch of blocks at coords (r,c) for r,c in {2,3,4}; +1 NaN sentinel.
    patch = [(r, c) for r in (2, 3, 4) for c in (2, 3, 4)]
    coords = jnp.array(patch + [(-1, -1)])  # 9 active + sentinel
    n_active = 9
    nb = np.asarray(B.find_face_neighbors(coords, n_active, level=0, cfg=cfg,
                                          periodic=False))
    nan_slot = coords.shape[0] - 1

    def slot_of(rc):
        return patch.index(rc)

    center = slot_of((3, 3))
    # order [+x,-x,+y,-y]
    assert nb[center, 0] == slot_of((4, 3))
    assert nb[center, 1] == slot_of((2, 3))
    assert nb[center, 2] == slot_of((3, 4))
    assert nb[center, 3] == slot_of((3, 2))

    corner = slot_of((2, 2))
    assert nb[corner, 0] == slot_of((3, 2))   # +x inside patch
    assert nb[corner, 1] == nan_slot          # -x outside patch -> sentinel
    assert nb[corner, 2] == slot_of((2, 3))   # +y inside patch
    assert nb[corner, 3] == nan_slot          # -y outside patch -> sentinel


def test_neighbors_2d_periodic_wrap():
    cfg = _cfg(2, base=16, blk=8)  # n_blocks_per_dim(0) = (2,2)
    coords = jnp.array([(0, 0), (1, 0), (0, 1), (1, 1), (-1, -1)])
    n_active = 4
    nb = np.asarray(B.find_face_neighbors(coords, n_active, level=0, cfg=cfg,
                                          periodic=True))

    def slot_of(rc):
        return [(0, 0), (1, 0), (0, 1), (1, 1)].index(rc)

    b00 = slot_of((0, 0))
    assert nb[b00, 0] == slot_of((1, 0))  # +x
    assert nb[b00, 1] == slot_of((1, 0))  # -x wraps to (1,0)
    assert nb[b00, 2] == slot_of((0, 1))  # +y
    assert nb[b00, 3] == slot_of((0, 1))  # -y wraps to (0,1)


@pytest.mark.parametrize("ndim", [2, 3])
def test_neighbors_symmetry(ndim):
    # If A is B's +d neighbour, B is A's -d neighbour.
    cfg = _cfg(ndim)
    rng = np.random.default_rng(0)
    base = rng.integers(2, 5, size=ndim)
    patch = [tuple(base + np.array(o)) for o in
             __import__("itertools").product((0, 1, 2), repeat=ndim)]
    coords = jnp.array([list(p) for p in patch] + [[-1] * ndim])
    n_active = len(patch)
    nb = np.asarray(B.find_face_neighbors(coords, n_active, level=1, cfg=cfg))
    nan_slot = coords.shape[0] - 1
    for s in range(n_active):
        for d in range(ndim):
            plus = nb[s, 2 * d]
            if plus != nan_slot:
                assert nb[plus, 2 * d + 1] == s  # its -d neighbour is us


# --------------------------------------------------------------------------
# build_children
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ndim", [2, 3])
def test_build_children_count_coords_conservation(ndim):
    cfg = _cfg(ndim)
    n_vars = 5
    bs = cfg.block_shape
    rng = np.random.default_rng(1)
    parent_data = jnp.asarray(rng.standard_normal((3, n_vars) + bs))
    parent_coords = jnp.array([[1] * ndim, [2] * ndim, [-1] * ndim])
    flags = jnp.array([True, False, False])
    max_parents = 2
    cc, cd, nactive = B.build_children(parent_coords, parent_data, flags,
                                       level=0, cfg=cfg, max_parents=max_parents)
    C = cfg.children_per_block
    assert int(nactive) == C  # one flagged parent -> 2**ndim children
    assert cd.shape == (max_parents * C, n_vars) + bs

    # The C children of parent at coords (1,1,..) have coords 2*(1..)+offset.
    active_coords = np.asarray(cc[:C])
    expected = np.array([[2 + o for o in off] for off in
                         __import__("itertools").product((0, 1), repeat=ndim)])
    np.testing.assert_array_equal(np.sort(active_coords, axis=0),
                                  np.sort(expected, axis=0))

    # Conservation: averaging each child back down reconstructs parent halves;
    # the mean over all C children equals the parent block mean (per var).
    child_means = np.asarray(cd[:C]).mean(axis=tuple(range(2, 2 + ndim)))  # (C, n_vars)
    parent_mean = np.asarray(parent_data[0]).mean(axis=tuple(range(1, 1 + ndim)))
    np.testing.assert_allclose(child_means.mean(axis=0), parent_mean, rtol=1e-12)


# --------------------------------------------------------------------------
# halo fill
# --------------------------------------------------------------------------
def test_halo_fill_2d_face_and_missing():
    cfg = _cfg(2, base=64, blk=8, ghost=2)
    g_ = cfg.n_ghost
    bs = cfg.block_shape
    n_vars = 1
    # Two blocks: A at slot 0, B at slot 1 (B is A's +x neighbour). Sentinel = 2.
    data = jnp.zeros((3, n_vars) + bs)
    data = data.at[0].set(1.0)          # A
    data = data.at[1].set(2.0)          # B
    data = data.at[2].set(jnp.nan)      # NaN sentinel
    nan_slot = 2
    # A: +x -> B(1); other faces missing -> sentinel. B: -x -> A(0); rest missing.
    neighbors = jnp.array([
        [1, nan_slot, nan_slot, nan_slot],  # A
        [nan_slot, 0, nan_slot, nan_slot],  # B
        [nan_slot, nan_slot, nan_slot, nan_slot],
    ])
    out = np.asarray(H.fill_ghost_same_level(data, neighbors, cfg))
    nx, ny = bs
    # A interior intact
    assert np.allclose(out[0, 0, g_:g_ + nx, g_:g_ + ny], 1.0)
    # A's +x ghost band (just past +x face) comes from B == 2.0
    assert np.allclose(out[0, 0, g_ + nx:g_ + nx + g_, g_:g_ + ny], 2.0)
    # A's -x ghost band has no neighbour -> NaN
    assert np.all(np.isnan(out[0, 0, 0:g_, g_:g_ + ny]))
    # corners stay NaN
    assert np.all(np.isnan(out[0, 0, 0:g_, 0:g_]))


@pytest.mark.parametrize("ndim", [2, 3])
def test_halo_fill_shape(ndim):
    cfg = _cfg(ndim, ghost=2)
    bs = cfg.block_shape
    data = jnp.zeros((4, 5) + bs)
    nan_slot = 3
    neighbors = jnp.full((4, 2 * ndim), nan_slot)
    out = H.fill_ghost_same_level(data, neighbors, cfg)
    assert out.shape == (4, 5) + tuple(s + 2 * cfg.n_ghost for s in bs)


# --------------------------------------------------------------------------
# coarse-fine ghost interpolation (fill NaN ghosts from a dense parent field)
# --------------------------------------------------------------------------
def test_fill_from_coarse_2d_interior_block():
    # Level-1 block; parent = dense level-0 ramp value = coarse-x index.
    cfg = _cfg(2, base=16, blk=8, ghost=2)  # base_blocks (2,2); L1 grid (4,4)
    g, bs = cfg.n_ghost, cfg.block_shape[0]
    Nc = 16
    cx = jnp.arange(Nc).reshape(Nc, 1) * jnp.ones((1, Nc))  # ramp in x
    parent = cx[None]                                        # (1, 16, 16)

    fc = jnp.array([[1, 1], [-1, -1]])                       # one block + sentinel
    fg = jnp.full((2, 1) + (bs + 2 * g,) * 2, jnp.nan)
    fg = fg.at[0, 0, g:g + bs, g:g + bs].set(0.0)            # interior = 0
    out = np.asarray(H.fill_ghost_from_coarse(fg, fc, parent, level=1, cfg=cfg,
                                              boundary=0))
    # Block fc=(1,1) covers coarse x in [4,8); lower-x ghost maps to coarse x=3.
    assert np.allclose(out[0, 0, 0:g, g:g + bs], 3.0)
    # upper-x ghost maps to coarse x=8.
    assert np.allclose(out[0, 0, g + bs:g + bs + g, g:g + bs], 8.0)
    # interior preserved (not overwritten by the constant underlay).
    assert np.allclose(out[0, 0, g:g + bs, g:g + bs], 0.0)


def test_fill_from_coarse_2d_periodic_wrap():
    cfg = _cfg(2, base=16, blk=8, ghost=2)
    g, bs = cfg.n_ghost, cfg.block_shape[0]
    Nc = 16
    cx = jnp.arange(Nc).reshape(Nc, 1) * jnp.ones((1, Nc))
    parent = cx[None]
    fc = jnp.array([[0, 0], [-1, -1]])                       # block at x-origin
    fg = jnp.full((2, 1) + (bs + 2 * g,) * 2, jnp.nan)
    fg = fg.at[0, 0, g:g + bs, g:g + bs].set(0.0)
    out = np.asarray(H.fill_ghost_from_coarse(fg, fc, parent, level=1, cfg=cfg,
                                              boundary=0))
    # lower-x ghost of the origin block wraps to coarse x = -1 -> 15 (periodic).
    assert np.allclose(out[0, 0, 0:g, g:g + bs], 15.0)


def test_fill_from_coarse_3d_shape():
    cfg = _cfg(3, base=16, blk=8, ghost=2)
    g, bs = cfg.n_ghost, cfg.block_shape[0]
    parent = jnp.ones((1, 16, 16, 16))
    fc = jnp.array([[1, 1, 1], [-1, -1, -1]])
    fg = jnp.full((2, 1) + (bs + 2 * g,) * 3, jnp.nan)
    fg = fg.at[0, 0, g:g + bs, g:g + bs, g:g + bs].set(0.0)
    out = H.fill_ghost_from_coarse(fg, fc, parent, level=1, cfg=cfg, boundary=1)
    assert out.shape == (2, 1) + (bs + 2 * g,) * 3
    # all ghosts filled from the constant parent -> value 1 everywhere ghost.
    assert np.allclose(np.asarray(out[0, 0, 0:g, g:g + bs, g:g + bs]), 1.0)
