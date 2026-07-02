"""Ghost-halo fill for blocks (dimension-generic).

Same-level fill: each block's ``n_ghost``-wide face bands are copied from the
opposite face of its same-level neighbour. Missing neighbours map to the NaN
sentinel slot, so their bands come out NaN (to be overwritten later by the
coarse-parent interpolation or a physical BC). Corner/edge ghost regions are
left NaN — matching upstream JAX-AMR, whose sequential-concat fill also leaves
corners invalid; the per-axis hydro stencils only need axis-aligned face ghosts.

Coarse-parent fallback and physical-BC fill are layered on top of this in the
Poisson/hydro grafting steps (DESIGN steps 3-4).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from amr.geometry import prolong, prolong_linear


def _spatial_slices(ndim, per_axis):
    """Build an index tuple ``(blocks, vars, *spatial)`` from a per-axis slice
    factory ``per_axis(d) -> slice``."""
    return (slice(None), slice(None)) + tuple(per_axis(d) for d in range(ndim))


def fill_ghost_same_level(blk_data, neighbors, cfg):
    """Fill same-level face ghosts.

    Parameters
    ----------
    blk_data : (n_slots, n_vars, *block_shape)
        Block field data; last slot is the NaN sentinel.
    neighbors : (n_slots, 2*ndim) int
        Face-neighbour slot ids from ``blocks.find_face_neighbors`` (order
        ``[+d,-d]`` per axis). Missing -> NaN-sentinel slot.

    Returns
    -------
    (n_slots, n_vars, *(block_shape + 2*n_ghost)) with interior set, face bands
    filled from neighbours, corners NaN.
    """
    g = cfg.n_ghost
    ndim = cfg.ndim
    bs = cfg.block_shape
    n_slots, n_vars = blk_data.shape[0], blk_data.shape[1]

    ghosted_shape = (n_slots, n_vars) + tuple(s + 2 * g for s in bs)
    out = jnp.full(ghosted_shape, jnp.nan, dtype=blk_data.dtype)

    interior = _spatial_slices(ndim, lambda d: slice(g, g + bs[d]))
    out = out.at[interior].set(blk_data)

    # Sequential axis sweep. Each pass copies a face band from the neighbour, but
    # gathers the neighbour's *full* current ghosted extent along the transverse
    # axes -- so a neighbour's already-filled ghost bands (from earlier passes)
    # propagate into this block's edge and corner ghosts. This matters because
    # LUNA's UNSPLIT MUSCL-Hancock predictor reads edge/corner ghosts (its
    # transverse terms), unlike the axis-split upstream JAX-AMR. A corner touched
    # by every axis resolves to the diagonal block's cell regardless of sweep
    # order, so the fill is reflection-symmetric. Ghosts with no same-level
    # neighbour stay NaN (gathered from the sentinel), for the coarse-parent /
    # physical-BC fill layered on top.
    for d in range(ndim):
        sax = 2 + d  # array axis for spatial dim d
        plus_nb = out[neighbors[:, 2 * d]]       # running array: axes < d filled
        minus_nb = out[neighbors[:, 2 * d + 1]]

        # +d ghost band <- +d neighbour's first g interior cells along d
        src_plus = [slice(None)] * (2 + ndim)
        src_plus[sax] = slice(g, 2 * g)
        dst_plus = [slice(None)] * (2 + ndim)
        dst_plus[sax] = slice(g + bs[d], g + bs[d] + g)
        out = out.at[tuple(dst_plus)].set(plus_nb[tuple(src_plus)])

        # -d ghost band <- -d neighbour's last g interior cells along d
        src_minus = [slice(None)] * (2 + ndim)
        src_minus[sax] = slice(bs[d], bs[d] + g)
        dst_minus = [slice(None)] * (2 + ndim)
        dst_minus[sax] = slice(0, g)
        out = out.at[tuple(dst_minus)].set(minus_nb[tuple(src_minus)])

    return out


# A boundary mode -> jnp.pad mode for the coarse-underlay padding. Periodic
# wraps; Neumann/reflective use 'edge' (the Poisson-side reflective handling
# treats reflective as Neumann for the scalar potential; hydro reflective gets
# proper sign flips in the dedicated hydro halo, layered later).
_PAD_MODE = {0: "wrap", 1: "edge", 2: "edge"}


def fill_ghost_physical(ghosted, blk_data, cfg):
    """Fill remaining NaN ghosts at the domain edge with a physical BC.

    After the same-level and coarse-parent fills, the only ghosts still NaN are
    domain-boundary faces/edges/corners of non-periodic runs (no same-level
    neighbour, no coarse parent). LUNA's UNSPLIT MUSCL-Hancock reads these, so
    leaving them NaN corrupts the boundary blocks (and, via NaN-through-limiter,
    the interior). We fill them by extending each block's own interior
    (zero-gradient / outflow / Neumann) — exact for ``boundary==1`` and used as a
    Neumann approximation for reflective (``boundary==2``; proper velocity sign
    flips are layered later). Periodic runs (``boundary==0``) never reach a NaN
    domain ghost — their same-level halo wraps — so this is a no-op there.
    """
    if cfg.boundary == 0:
        return ghosted
    g = cfg.n_ghost
    ndim = cfg.ndim
    pad = [(0, 0), (0, 0)] + [(g, g)] * ndim
    edge = jnp.pad(blk_data, pad, mode=_PAD_MODE.get(cfg.boundary, "edge"))
    return jnp.where(jnp.isnan(ghosted), edge, ghosted)


def fill_ghost_from_coarse(fine_ghosted, fine_coords, parent_field, level, cfg,
                           boundary=0):
    """Fill remaining NaN ghosts of fine blocks from a prolonged coarse underlay.

    After :func:`fill_ghost_same_level`, ghost cells with no same-level
    neighbour are NaN (coarse-fine faces, corners, and domain edges). This fills
    *all* of them at once from the parent (level ``level-1``) field:

      1. pad ``parent_field`` with the physical BC (so edge patches and periodic
         wrap are handled by the pad mode);
      2. for each fine block, slice the coarse patch covering the block's
         extended (block+ghost) footprint and prolong it by 2;
      3. write the prolonged underlay only where ``fine_ghosted`` is NaN.

    First cut: ``parent_field`` is a *dense* uniform field at level ``level-1``
    (the natural representation for L=1, whose parent is the base grid). Deeper
    levels reuse the identical interpolation with a gathered patch instead of a
    dense slice (later optimisation).

    Requires ``n_ghost`` even (ghost maps to a whole number of coarse cells).

    Parameters
    ----------
    fine_ghosted : (n_slots, n_vars, *(block_shape+2*n_ghost))
        Output of :func:`fill_ghost_same_level` (NaN where coarse fill needed).
    fine_coords : (n_slots, ndim) int
        Fine block integer coords at ``level``.
    parent_field : (n_vars, *parent_grid)
        Dense coarse field at ``level-1``.
    boundary : int
        LUNA boundary code (0 periodic, 1 Neumann, 2 reflective).

    Returns
    -------
    (n_slots, n_vars, *(block_shape+2*n_ghost)) with NaN ghosts filled.
    """
    ndim = cfg.ndim
    g = cfg.n_ghost
    bs = cfg.block_shape
    if g % 2 != 0:
        raise ValueError(f"fill_ghost_from_coarse needs even n_ghost; got {g}")
    gc = g // 2                                  # ghost width in coarse cells
    patch = tuple((bs[d] + 2 * g) // 2 for d in range(ndim))  # coarse patch size
    n_vars = parent_field.shape[0]

    # Pad parent generously so every patch slice (incl. negative starts and
    # periodic wrap) is in-bounds; the pad mode encodes the physical BC.
    pad_w = [(0, 0)] + [(p, p) for p in patch]
    parent_pad = jnp.pad(parent_field, pad_w, mode=_PAD_MODE.get(boundary, "edge"))

    def one_block(fc):
        # coarse start of the block's extended footprint, in padded coords.
        # The leading var-axis index must share fc's int dtype: dynamic_slice
        # requires uniform index dtype, and real LevelData coords are int32
        # while a bare Python 0 would be int64 under x64.
        zero = jnp.zeros((), fc.dtype)
        start = [zero] + [fc[d] * (bs[d] // 2) - gc + patch[d] for d in range(ndim)]
        size = [n_vars] + list(patch)
        coarse_patch = jax.lax.dynamic_slice(parent_pad, start, size)
        return prolong(coarse_patch, ndim)        # (n_vars, *(bs+2g))

    underlay = jax.vmap(one_block)(fine_coords)   # (n_slots, n_vars, *(bs+2g))
    return jnp.where(jnp.isnan(fine_ghosted), underlay, fine_ghosted)


def fill_ghost_from_coarse_sparse(fine_ghosted, fine_coords, parent_slots,
                                  coarse_ghosted, cfg, interp="const"):
    """Sparse coarse-fine ghost fill: gather each fine block's coarse underlay
    from its parent *block* (not a dense field) and prolong.

    Unlike :func:`fill_ghost_from_coarse` (dense parent), this serves deeper
    refinement levels whose parent level is itself block-structured. Each fine
    block lives in one quadrant of its parent coarse block; the fine block's
    extended (block + ghost) footprint, measured in coarse cells, fits inside the
    parent's *ghosted* block (interior + ``n_ghost`` halo), because the overhang
    past the parent boundary is at most ``n_ghost//2`` coarse cells <= the halo
    width. So we slice that footprint straight from ``coarse_ghosted[parent]``,
    prolong by 2, and fill the fine block's NaN ghosts.

    Requires the coarse level's ghosts to be filled first (``coarse_ghosted`` =
    output of the same-level + coarse-parent fill at the coarser level). Needs
    even ``n_ghost``. Corner ghosts that trace back to the parent's own NaN
    corners stay NaN — those fine corners are never read by the per-axis hydro.

    Parameters
    ----------
    fine_ghosted : (n_fine, n_vars, *(block_shape + 2*n_ghost))
        Fine blocks after same-level fill (NaN where coarse fill is needed).
    fine_coords : (n_fine, ndim) int
        Fine block coords at the fine level.
    parent_slots : (n_fine,) int
        Coarse slot of each fine block's parent (:func:`blocks.find_parent_slots`).
    coarse_ghosted : (n_coarse, n_vars, *(block_shape + 2*n_ghost))
        Ghost-filled coarse blocks (the parent level).
    """
    ndim = cfg.ndim
    g = cfg.n_ghost
    bs = cfg.block_shape
    if g % 2 != 0:
        raise ValueError(f"fill_ghost_from_coarse_sparse needs even n_ghost; got {g}")
    gc = g // 2
    # interp="linear": cell-centred (tri)linear underlay for smooth scalars
    # (phi Dirichlet ghosts) -- gather ONE extra coarse cell of margin so the
    # interpolation stencil never clamps inside the footprint, then crop the
    # prolonged patch back. "const" (default) is the conservative hydro fill.
    if interp == "linear" and gc + 1 > g:
        raise ValueError(f"linear coarse fill needs n_ghost >= {2 * (gc + 1) - g + g}")
    margin = gc + 1 if interp == "linear" else gc
    patch = tuple(bs[d] // 2 + 2 * margin for d in range(ndim))
    n_vars = coarse_ghosted.shape[1]

    def one_block(fc, pslot):
        parent = coarse_ghosted[pslot]                  # (n_vars, *(bs+2g))
        q = jnp.maximum(fc, 0) % 2                       # quadrant per dim
        # extended footprint start in the parent's *ghosted* coords.
        start = [jnp.asarray(0, jnp.int32)] + [
            jnp.asarray(g + q[d] * (bs[d] // 2) - margin, jnp.int32) for d in range(ndim)
        ]
        size = [n_vars] + list(patch)
        coarse_patch = jax.lax.dynamic_slice(parent, start, size)
        if interp == "linear":
            up = prolong_linear(coarse_patch, ndim)      # (n_vars, *(bs+2g+4))
            crop = (slice(None),) + (slice(2, -2),) * ndim
            return up[crop]                              # (n_vars, *(bs+2g))
        return prolong(coarse_patch, ndim)               # (n_vars, *(bs+2g))

    underlay = jax.vmap(one_block)(fine_coords, parent_slots)
    return jnp.where(jnp.isnan(fine_ghosted), underlay, fine_ghosted)
