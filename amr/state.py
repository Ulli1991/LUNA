"""AMR hierarchy state container + uniform<->block partitioning.

``AMRState`` is a plain tuple of ``LevelData`` (one per level, level 0 = base).
Both are JAX pytrees (NamedTuple fields / tuple elements are children), so an
``AMRState`` flows through ``jit``/``scan``/``shard_map`` transparently. The
*number* of levels and each level's *slot count* are static (fixed shapes);
re-adaptation changes contents (coords/data/n_active), not shapes — so it does
not force a recompile unless a level's ``max_blocks`` is grown.

The static geometry lives in ``AMRConfig`` (passed separately), never inside the
traced state.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class LevelData(NamedTuple):
    """One refinement level's blocks. Last slot is the NaN sentinel."""

    data: jnp.ndarray       # (n_slots, n_vars, *block_shape)
    coords: jnp.ndarray     # (n_slots, ndim) int32 block-coords; inactive = -1
    neighbors: jnp.ndarray  # (n_slots, 2*ndim) int32 face-neighbour slot ids
    n_active: jnp.ndarray   # scalar int32: number of leading active slots


# ---------------------------------------------------------------------------
# Uniform field <-> level-0 blocks (dimension-generic reshape/transpose)
# ---------------------------------------------------------------------------
def partition_to_blocks(field, cfg):
    """Uniform ``(n_vars, *base_grid)`` field -> ``(n_base_blocks, n_vars, *bs)``.

    Row-major (C-order) block ordering, matching :func:`base_block_coords`.
    """
    ndim = cfg.ndim
    nb = cfg.base_blocks
    bs = cfg.block_shape
    n_vars = field.shape[0]

    rshape = [n_vars]
    for d in range(ndim):
        rshape += [nb[d], bs[d]]
    f = field.reshape(rshape)

    nb_axes = [1 + 2 * d for d in range(ndim)]   # block-count axes
    bs_axes = [2 + 2 * d for d in range(ndim)]   # in-block cell axes
    f = f.transpose(nb_axes + [0] + bs_axes)      # (nb0,nb1,..,var,bs0,bs1,..)

    n_blocks = 1
    for x in nb:
        n_blocks *= x
    return f.reshape(n_blocks, n_vars, *bs)


def assemble_base_level(blk_data, cfg):
    """Inverse of :func:`partition_to_blocks` for the dense level-0 grid.

    ``blk_data`` is ``(n_base_blocks, n_vars, *bs)`` (NaN sentinel, if present,
    must be sliced off by the caller). Returns ``(n_vars, *base_grid)``.
    """
    ndim = cfg.ndim
    nb = cfg.base_blocks
    bs = cfg.block_shape
    n_vars = blk_data.shape[1]

    # (nb0,nb1,..,var,bs0,bs1,..)
    f = blk_data.reshape(*nb, n_vars, *bs)
    # -> (var, nb0,bs0, nb1,bs1, ..)
    perm = [ndim]  # var axis currently at position ndim
    for d in range(ndim):
        perm += [d, ndim + 1 + d]
    f = f.transpose(perm)
    out_shape = [n_vars] + [nb[d] * bs[d] for d in range(ndim)]
    return f.reshape(out_shape)


def base_block_coords(cfg):
    """Row-major integer block-coords for the dense level-0 grid.

    Returns int32 ``(n_base_blocks, ndim)`` matching :func:`partition_to_blocks`.
    """
    ndim = cfg.ndim
    nb = cfg.base_blocks
    grids = jnp.meshgrid(*[jnp.arange(nb[d]) for d in range(ndim)], indexing="ij")
    return jnp.stack([g.reshape(-1).astype(jnp.int32) for g in grids], axis=1)


def build_base_level(field, cfg):
    """Construct the level-0 ``LevelData`` from a uniform field.

    Appends a single NaN sentinel block (the invalid-ghost source). Neighbours
    are left unset (-1); the caller fills them with
    ``blocks.find_face_neighbors(..., periodic=...)`` once the BC is known.
    """
    blk = partition_to_blocks(field, cfg)           # (n_blocks, n_vars, *bs)
    coords = base_block_coords(cfg)                 # (n_blocks, ndim)
    n_blocks = blk.shape[0]
    n_vars = blk.shape[1]

    nan_blk = jnp.full((1, n_vars) + tuple(cfg.block_shape), jnp.nan, dtype=blk.dtype)
    data = jnp.concatenate([blk, nan_blk], axis=0)
    coords = jnp.concatenate([coords, -jnp.ones((1, cfg.ndim), jnp.int32)], axis=0)
    neighbors = -jnp.ones((n_blocks + 1, 2 * cfg.ndim), jnp.int32)
    n_active = jnp.asarray(n_blocks, jnp.int32)
    return LevelData(data=data, coords=coords, neighbors=neighbors, n_active=n_active)
