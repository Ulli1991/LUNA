"""Block topology: refinement flags, Morton-based neighbour finding, and
child-block generation. Dimension-generic (one path for 2D/3D).

Conventions
-----------
* A *level* holds a fixed-size array of block slots; the **last slot is the NaN
  sentinel** (invalid-ghost source). Inactive slots also hold NaN data and
  ``coords = -1``.
* Face-neighbour order is canonical: for axis ``d`` in ``range(ndim)`` the pair
  ``[+d, -d]``. So 2D = ``[+x,-x,+y,-y]``; 3D appends ``[+z,-z]``. A missing
  neighbour maps to the NaN-sentinel slot (``n_slots-1``) so downstream gathers
  need no masking — pulling a missing neighbour yields NaN, exactly the upstream
  ``-1 -> last block`` trick, but explicit.
* Block identity is the Morton code of integer block-coords (see geometry). All
  matching is O(N log N) via sort + ``searchsorted`` (upstream used an O(N^2)
  pairwise compare).
"""

from __future__ import annotations

import itertools
from functools import partial

import jax
import jax.numpy as jnp

from amr.geometry import (_morton_params, morton_encode, prolong,
                          prolong_centered_cons, prolong_linear_cons)


def _offset_table(ndim):
    """Static (2**ndim, ndim) table of child offsets, rows in {0,1}**ndim."""
    return jnp.asarray(list(itertools.product((0, 1), repeat=ndim)), dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Refinement flags at the block level
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnums=(1,))
def refinement_block_mask(grid_mask, ndim):
    """Per-block refine flag: True if the block holds any flagged cell.

    grid_mask : (n_blocks, *block_shape) of 0/1 (from geometry.refinement_grid_mask)
    returns   : (n_blocks,) bool. Whole-block refinement -> a flagged block is
                split into 2**ndim children.
    """
    axes = tuple(range(1, ndim + 1))
    return grid_mask.sum(axis=axes) > 0


# ---------------------------------------------------------------------------
# Neighbour topology (Morton sort + searchsorted)
# ---------------------------------------------------------------------------
def find_face_neighbors(coords, n_active, level, cfg, periodic=False):
    """Slot id of each block's 2*ndim face neighbours at the same level.

    Parameters
    ----------
    coords : int array (n_slots, ndim)
        Integer block-coords; inactive/sentinel slots may hold anything (they
        are excluded via ``n_active``).
    n_active : int (scalar, traced ok)
        Number of leading active slots. Slots ``>= n_active`` are inactive.
    periodic : bool
        Wrap neighbour coords modulo ``n_blocks_per_dim`` (base level under
        periodic cosmology). Non-base/refined levels use ``False``.

    Returns
    -------
    int array (n_slots, 2*ndim) of neighbour slot ids; missing -> ``n_slots-1``
    (the NaN-sentinel slot).
    """
    ndim = cfg.ndim
    n_slots = coords.shape[0]
    nan_slot = n_slots - 1
    nb_per_dim = jnp.asarray(cfg.n_blocks_per_dim(level))

    slot_ids = jnp.arange(n_slots)
    active = slot_ids < n_active

    # Inactive slots get the max code for this dtype so they sort last and never
    # match a real neighbour query (dtype tracks the active x64 setting).
    code_dtype, _ = _morton_params(ndim)
    invalid_morton = jnp.array(jnp.iinfo(code_dtype).max, dtype=code_dtype)
    code = morton_encode(jnp.where(active[:, None], coords, 0))
    code = jnp.where(active, code, invalid_morton)
    order = jnp.argsort(code)
    sorted_code = code[order]

    def neighbor_slot(nc, in_domain):
        nb_code = morton_encode(jnp.maximum(nc, 0))
        pos = jnp.clip(jnp.searchsorted(sorted_code, nb_code), 0, n_slots - 1)
        found = (sorted_code[pos] == nb_code) & in_domain
        return jnp.where(found, order[pos], nan_slot)

    cols = []
    for d in range(ndim):
        for side in (+1, -1):
            nc = coords.at[:, d].add(side)
            if periodic:
                nc = nc % nb_per_dim
                in_domain = active
            else:
                in_bounds = (nc[:, d] >= 0) & (nc[:, d] < nb_per_dim[d])
                in_domain = active & in_bounds
            cols.append(neighbor_slot(nc, in_domain))
    return jnp.stack(cols, axis=1)


def find_parent_slots(fine_coords, coarse_coords, coarse_n_active, cfg):
    """Slot id, in the coarse-level array, of each fine block's parent.

    The parent of a fine block at coords ``cf`` is the coarse block at
    ``cf // 2`` (whole-block refinement: children coords are ``2*parent + {0,1}``).
    O(N log N) Morton sort + ``searchsorted``, same machinery as
    :func:`find_face_neighbors`. A fine block whose parent is absent (should not
    happen under proper nesting) maps to the coarse NaN-sentinel slot.

    Returns int array ``(n_fine_slots,)`` of coarse slot ids.
    """
    ndim = cfg.ndim
    n_coarse = coarse_coords.shape[0]
    sentinel = n_coarse - 1

    code_dtype, _ = _morton_params(ndim)
    cslots = jnp.arange(n_coarse)
    cactive = cslots < coarse_n_active
    invalid = jnp.array(jnp.iinfo(code_dtype).max, dtype=code_dtype)
    ccode = morton_encode(jnp.where(cactive[:, None], coarse_coords, 0))
    ccode = jnp.where(cactive, ccode, invalid)
    order = jnp.argsort(ccode)
    sorted_code = ccode[order]

    parent_coords = jnp.maximum(fine_coords // 2, 0)
    pcode = morton_encode(parent_coords)
    pos = jnp.clip(jnp.searchsorted(sorted_code, pcode), 0, n_coarse - 1)
    found = sorted_code[pos] == pcode
    return jnp.where(found, order[pos], sentinel)


# ---------------------------------------------------------------------------
# Child-block generation (refine flagged parents -> 2**ndim children each)
# ---------------------------------------------------------------------------
def build_children(parent_coords, parent_data, flags, level, cfg, max_parents,
                   parent_ghosted=None):
    """Refine flagged parent blocks into child blocks at ``level+1``.

    Each flagged parent emits ``2**ndim`` children; child ``o`` covers the
    parent's spatial half selected by offset ``o in {0,1}**ndim`` and is the
    conservative prolongation of that half.

    ``parent_ghosted`` (optional, ``(n_slots, n_vars, *(bs + 2*g))`` with any halo
    width ``g >= 1``, NaN-clamped at missing neighbours) makes the prolongation
    GHOST-AWARE: the parent block's OUTER-face slopes are taken from the halo
    (same-level neighbours) instead of clamping to one-sided/PC, so child
    initialization is 2nd order at block boundaries too (bare-block prolongation
    is 1st order there -- the residual after the centered-limiter fix). When
    ``None`` the bare parent block is prolonged (backward-compatible).

    Returns
    -------
    child_coords : (max_parents * 2**ndim, ndim) int  (last slot = sentinel -1)
    child_data   : (max_parents * 2**ndim, n_vars, *block_shape)  (last = NaN)
    n_active     : int scalar = min(#flagged, max_parents - 1) * 2**ndim
    """
    ndim = cfg.ndim
    C = cfg.children_per_block
    bs = cfg.block_shape
    hs = tuple(s // 2 for s in bs)
    for s in bs:
        if s % 2 != 0:
            raise ValueError(f"block_shape must be even per dim to refine; got {bs}")
    # Always need >=1 parent slot for the NaN sentinel group; a 0 capacity
    # (amr_capacities[L]=0 -> "don't refine this level") would otherwise make an
    # empty child_data whose `.at[-1]` sentinel write is out of bounds.
    max_parents = max(int(max_parents), 1)

    # At most max_parents - 1 parents may refine: the last slot group hosts the
    # NaN sentinel. Uncapped, _sized_parents allocates flagged+1 so this never
    # binds; under an amr_max_blocks cap the overflow parents are NOT refined --
    # and their children must be fully invalidated (coords -1 / NaN data), or
    # they'd sit outside the neighbor map, advance against unfilled ghosts and
    # restrict NaN into the parent (n_active alone doesn't guard restriction,
    # which keys on coords >= 0; found by the capacity test).
    n_par = jnp.minimum(jnp.sum(flags), max_parents - 1)
    pidx = jnp.nonzero(flags, size=max_parents, fill_value=-1)[0]
    valid = (pidx >= 0) & (jnp.arange(max_parents) < n_par)
    safe = jnp.where(valid, pidx, 0)
    pc = parent_coords[safe]                       # (P, ndim)

    # Prolong from the ghosted parent (2nd-order outer faces) when provided, else
    # the bare block (1st-order outer faces). ``gsrc`` = halo width of the source.
    src = parent_data if parent_ghosted is None else parent_ghosted
    src_bs = tuple(src.shape[2:])                    # bs, or bs + 2*gsrc if ghosted
    gsrc = (src_bs[0] - bs[0]) // 2
    pd = src[safe]                                   # (P, n_vars, *src_bs)
    n_vars = pd.shape[1]

    offsets = _offset_table(ndim)                   # (C, ndim) static rows

    _prolong = (prolong_centered_cons
                if getattr(cfg, "prolong_limiter", "minmod") == "centered"
                else prolong_linear_cons)
    # after prolonging the (halo-padded) source, keep the central 2*bs fine cells
    crop = (slice(None), slice(None)) + tuple(
        slice(2 * gsrc, 2 * gsrc + 2 * bs[d]) for d in range(ndim))

    def _octants(block_data):
        """Conservative prolongation of a batch of parents, sliced into the C child
        octants and interleaved parent-major: (B, nv, *src_bs) -> (B*C, nv, *bs).
        Limiter per ``cfg.prolong_limiter``; ghost-aware if a halo was supplied."""
        big = _prolong(block_data, ndim)             # (B, n_vars, *(2*src_bs))
        big = big[crop]                              # (B, n_vars, *(2*bs))
        outs = []
        for o in itertools.product((0, 1), repeat=ndim):
            sl = [slice(None), slice(None)]
            for d in range(ndim):
                sl.append(slice(o[d] * bs[d], (o[d] + 1) * bs[d]))
            outs.append(big[tuple(sl)])              # (B, n_vars, *bs)
        return jnp.stack(outs, axis=1).reshape(block_data.shape[0] * C, n_vars, *bs)

    # Memory: the whole-batch prolongation materializes (P, n_vars, *(2*bs)) plus
    # its minmod-slope temporaries at once -- 17.8 GiB at P=2048 (measured, the
    # deep-hierarchy OOM). Chunk the parent axis so the transient is only one
    # chunk's worth; the stacked output is bit-identical (scan is a deterministic
    # replay of the same ops). Only kicks in when it helps (P a multiple of the
    # chunk and large enough); otherwise the direct whole-batch path.
    _CH = 256
    if max_parents >= 2 * _CH and max_parents % _CH == 0:
        n_ch = max_parents // _CH
        _, cds = jax.lax.scan(
            lambda _c, blk: (None, _octants(blk)),
            None, pd.reshape(n_ch, _CH, n_vars, *src_bs))
        child_data = cds.reshape(max_parents * C, n_vars, *bs)
    else:
        child_data = _octants(pd)

    coords_per_off = []
    for o in itertools.product((0, 1), repeat=ndim):
        off = jnp.asarray(o, dtype=jnp.int32)
        coords_per_off.append(2 * pc + off)          # (P, ndim)
    child_coords = jnp.stack(coords_per_off, axis=1).reshape(max_parents * C, ndim)

    # Validity per child follows its parent; invalidate coords + NaN the data.
    child_valid = jnp.repeat(valid, C)
    child_coords = jnp.where(child_valid[:, None], child_coords, -1)
    child_data = jnp.where(
        child_valid[:, None, *([None] * ndim)], child_data, jnp.nan
    )
    # Force the very last slot to be the NaN sentinel.
    child_data = child_data.at[-1].set(jnp.nan)
    child_coords = child_coords.at[-1].set(-1)

    n_active = n_par * C
    return child_coords, child_data, n_active


# ---------------------------------------------------------------------------
# Preserve unaltered blocks across re-adaptation (Morton set-diff, O(N log N))
# ---------------------------------------------------------------------------
def match_blocks(new_coords, new_n_active, old_coords, old_n_active, ndim):
    """For each new block slot, the old slot holding the same coords, else -1.

    Morton sort + searchsorted, replacing upstream's O(N^2) ``compare_coords``.
    Used to carry already-computed data for blocks that exist in both the old
    and the re-adapted hierarchy (no need to re-prolong them).
    """
    code_dtype, _ = _morton_params(ndim)
    invalid = jnp.array(jnp.iinfo(code_dtype).max, dtype=code_dtype)

    n_new = new_coords.shape[0]
    n_old = old_coords.shape[0]
    new_active = jnp.arange(n_new) < new_n_active
    old_active = jnp.arange(n_old) < old_n_active

    new_code = jnp.where(new_active, morton_encode(jnp.where(new_active[:, None], new_coords, 0)), invalid)
    old_code = jnp.where(old_active, morton_encode(jnp.where(old_active[:, None], old_coords, 0)), invalid)

    order = jnp.argsort(old_code)
    sorted_old = old_code[order]
    pos = jnp.clip(jnp.searchsorted(sorted_old, new_code), 0, n_old - 1)
    found = (sorted_old[pos] == new_code) & new_active
    return jnp.where(found, order[pos], -1)


def preserve_unaltered(new_data, old_slot, old_data):
    """Copy old block data into matched new slots (``old_slot >= 0``)."""
    keep = old_slot >= 0
    safe = jnp.where(keep, old_slot, 0)
    gathered = old_data[safe]
    bcast = keep.reshape((-1,) + (1,) * (new_data.ndim - 1))
    return jnp.where(bcast, gathered, new_data)


# ---------------------------------------------------------------------------
# Cross-block buffer dilation (extend the refine margin across block faces)
# ---------------------------------------------------------------------------
def cross_block_dilate(grid_mask, neighbors, cfg, buffer=None):
    """Grow the per-cell refine mask by ``buffer`` cells, INCLUDING across block
    faces, using same-level neighbour data.

    Composes the existing same-level halo fill (treating the 0/1 mask as a
    1-variable block field, missing neighbours -> 0) with the box dilation, then
    crops back to block interiors. This makes the refine margin correct at block
    boundaries (upstream did this with a bespoke ``get_ghost_mask``).
    """
    from amr.geometry import _dilate
    from amr.halo import fill_ghost_same_level

    if buffer is None:
        buffer = cfg.n_ghost
    if buffer > cfg.n_ghost:
        raise ValueError(f"buffer ({buffer}) exceeds ghost width ({cfg.n_ghost})")

    ndim = cfg.ndim
    g = cfg.n_ghost
    bs = cfg.block_shape

    m = grid_mask[:, None]                                  # (n_slots, 1, *bs)
    ghosted = jnp.nan_to_num(fill_ghost_same_level(m, neighbors, cfg), nan=0.0)
    dilated = _dilate(ghosted, buffer, ndim)                # SAME-padded, same shape
    interior = (slice(None), slice(None)) + tuple(slice(g, g + bs[d]) for d in range(ndim))
    return jnp.sign(dilated[interior])[:, 0]                # (n_slots, *bs)
