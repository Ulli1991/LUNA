"""Tests for AMRState partitioning + re-adaptation helpers (2D and 3D)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from amr.config import AMRConfig
from amr import state as S
from amr import blocks as B


def _cfg(ndim, base=32, blk=8, ghost=2):
    return AMRConfig(ndim=ndim, box=(1.0,) * ndim, base_grid=(base,) * ndim,
                     block_shape=(blk,) * ndim, n_ghost=ghost)


# --------------------------------------------------------------------------
# partition / assemble round-trip
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ndim", [2, 3])
def test_partition_assemble_roundtrip(ndim):
    cfg = _cfg(ndim)
    n_vars = 5
    rng = np.random.default_rng(0)
    field = jnp.asarray(rng.standard_normal((n_vars,) + cfg.base_grid))
    blk = S.partition_to_blocks(field, cfg)
    assert blk.shape[0] == int(np.prod(cfg.base_blocks))
    assert blk.shape[1:] == (n_vars,) + cfg.block_shape
    back = S.assemble_base_level(blk, cfg)
    np.testing.assert_allclose(np.asarray(back), np.asarray(field), rtol=1e-12)


@pytest.mark.parametrize("ndim", [2, 3])
def test_block_coords_match_partition(ndim):
    # Block i's data must equal the field slice at base_block_coords[i].
    cfg = _cfg(ndim, base=16, blk=8)  # base_blocks = (2,)*ndim
    rng = np.random.default_rng(1)
    field = jnp.asarray(rng.standard_normal((1,) + cfg.base_grid))
    blk = np.asarray(S.partition_to_blocks(field, cfg))
    coords = np.asarray(S.base_block_coords(cfg))
    bs = cfg.block_shape
    fnp = np.asarray(field)
    for i in range(blk.shape[0]):
        sl = (slice(None),) + tuple(
            slice(coords[i, d] * bs[d], (coords[i, d] + 1) * bs[d]) for d in range(ndim)
        )
        np.testing.assert_allclose(blk[i], fnp[sl], rtol=1e-12)


@pytest.mark.parametrize("ndim", [2, 3])
def test_build_base_level(ndim):
    cfg = _cfg(ndim)
    field = jnp.ones((5,) + cfg.base_grid)
    lvl = S.build_base_level(field, cfg)
    n_blocks = int(np.prod(cfg.base_blocks))
    assert lvl.data.shape == (n_blocks + 1, 5) + cfg.block_shape
    assert int(lvl.n_active) == n_blocks
    assert np.all(np.isnan(np.asarray(lvl.data[-1])))         # NaN sentinel
    assert np.asarray(lvl.coords[-1]).tolist() == [-1] * ndim


# --------------------------------------------------------------------------
# unaltered-block preservation (Morton set-diff)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ndim", [2, 3])
def test_match_and_preserve(ndim):
    # old = {A,B,C}; new = {B,C,D}. B,C carried over; D fresh.
    A = [1] * ndim
    Bc = [2] * ndim
    Cc = [3] * ndim
    D = [5] * ndim
    old_coords = jnp.array([A, Bc, Cc, [-1] * ndim])
    new_coords = jnp.array([Bc, Cc, D, [-1] * ndim])
    old_slot = np.asarray(B.match_blocks(new_coords, 3, old_coords, 3, ndim))
    # new slot 0 (B) -> old slot 1; new slot 1 (C) -> old slot 2; D -> -1
    assert old_slot[0] == 1
    assert old_slot[1] == 2
    assert old_slot[2] == -1

    old_data = jnp.arange(4 * 2).reshape(4, 2).astype(float)  # (n_slots, n_vars)
    new_data = jnp.zeros((4, 2))
    merged = np.asarray(B.preserve_unaltered(new_data, jnp.asarray(old_slot), old_data))
    np.testing.assert_allclose(merged[0], np.asarray(old_data[1]))  # B kept
    np.testing.assert_allclose(merged[1], np.asarray(old_data[2]))  # C kept
    np.testing.assert_allclose(merged[2], [0.0, 0.0])               # D fresh


# --------------------------------------------------------------------------
# cross-block buffer dilation
# --------------------------------------------------------------------------
def test_cross_block_dilate_2d():
    cfg = _cfg(2, base=64, blk=4, ghost=2)  # n_blocks_per_dim(0) = (16,16)
    nan_slot = 2
    # A=slot0, B=slot1 (B is A's +x neighbour); sentinel=2.
    neighbors = jnp.array([
        [1, nan_slot, nan_slot, nan_slot],  # A: +x=B
        [nan_slot, 0, nan_slot, nan_slot],  # B: -x=A
        [nan_slot, nan_slot, nan_slot, nan_slot],
    ])
    gm = jnp.zeros((3, 4, 4))
    gm = gm.at[1, 0, 2].set(1.0)  # block B, its -x-face cell (x=0), y=2
    out = np.asarray(B.cross_block_dilate(gm, neighbors, cfg, buffer=1))
    # A's +x edge cell (x=3) at the same y should now be flagged across the face.
    assert out[0, 3, 2] == 1.0
    # A interior far from the face stays unflagged.
    assert out[0, 0, 0] == 0.0
    # B's own flagged cell + its intra-block dilation present.
    assert out[1, 0, 2] == 1.0
