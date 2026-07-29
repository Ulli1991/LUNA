"""Ghost-halo fill for blocks (dimension-generic).

Same-level face bands are copied from neighbours; missing neighbours resolve
to the NaN sentinel, so their bands come out NaN for the coarse-parent or
physical-BC fill layered on top. Corners with no source stay NaN (the per-axis
hydro stencils never read them).
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
    """Fill same-level face ghosts: ``blk_data`` (n_slots, n_vars, *block_shape)
    -> (n_slots, n_vars, *(block_shape + 2*n_ghost)) with face bands gathered
    from neighbours (order ``[+d, -d]`` per axis) and unfillable ghosts NaN."""
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


def fill_ghost_physical(ghosted, blk_data, cfg, coords=None, level=0):
    """Fill remaining NaN domain-edge ghosts by per-axis zero-gradient
    extension of the already-filled ghosted array (outflow for ``boundary==1``,
    Neumann approximation for reflective; no-op for periodic). ``coords``/
    ``level`` restrict the fill to true domain-edge bands so it can run BEFORE
    the coarse underlay."""
    if cfg.boundary == 0:
        return ghosted
    g = cfg.n_ghost
    ndim = cfg.ndim
    # Sequential per-axis zero-gradient extension of the GHOSTED array (not an
    # edge-pad of the block's own interior): each axis pass replaces NaN ghost
    # layers with the nearest already-filled layer along that axis, so edge and
    # corner ghosts inherit from the same-level-neighbour-filled face ghosts.
    # The old blk_data edge-pad gave corner ghosts the block's OWN corner cell
    # even when a face neighbour exists there -- the wrong upstream position --
    # which broke planar symmetry at boundary-block corners and CREATED mass
    # (caught by the Sod AMR benchmark: outflow grew drho/rho ~ 5e-3 while
    # periodic stayed exact; max_y_std 3.6e-3 vs 4e-16).
    #
    # ``coords``/``level`` (optional): fill ONLY true DOMAIN-edge bands (block at
    # the level's domain boundary on that side). This lets the caller run the
    # physical fill BEFORE the coarse-fine underlay: domain-wall ghosts of fine
    # blocks must be the zero-gradient extension of the FINE data -- the underlay
    # otherwise fills them with coarse-prolonged values whose interpolation error
    # near steep features injects a fake wall-normal gradient (v_perp kick ->
    # boundary layer -> the levels>=2 Sod mass leak, seeded at ULP scale at the
    # C/F-edge x wall corner and amplified by the flow).
    out = ghosted
    dom = None
    if coords is not None:
        r = cfg.refinement_ratio
        dom = []
        for d in range(ndim):
            nb_d = (cfg.base_grid[d] // cfg.block_shape[d]) * (r ** level)
            dom.append((coords[:, d] == 0, coords[:, d] == nb_d - 1))
    for d in range(ndim):
        ax = 2 + d
        n = out.shape[ax]
        lo = jax.lax.slice_in_dim(out, g, g + 1, axis=ax)
        hi = jax.lax.slice_in_dim(out, n - g - 1, n - g, axis=ax)
        lo_band = jax.lax.slice_in_dim(out, 0, g, axis=ax)
        hi_band = jax.lax.slice_in_dim(out, n - g, n, axis=ax)
        mid = jax.lax.slice_in_dim(out, g, n - g, axis=ax)
        lo_m = jnp.isnan(lo_band)
        hi_m = jnp.isnan(hi_band)
        if dom is not None:
            sh_lo = (lo_band.shape[0],) + (1,) * (lo_band.ndim - 1)
            lo_m = lo_m & dom[d][0].reshape(sh_lo)
            hi_m = hi_m & dom[d][1].reshape(sh_lo)
        lo_fill = jnp.where(lo_m, jnp.broadcast_to(lo, lo_band.shape), lo_band)
        hi_fill = jnp.where(hi_m, jnp.broadcast_to(hi, hi_band.shape), hi_band)
        out = jnp.concatenate([lo_fill, mid, hi_fill], axis=ax)
    return out


def fill_ghost_from_coarse(fine_ghosted, fine_coords, parent_field, level, cfg,
                           boundary=0):
    """Fill remaining NaN fine-block ghosts from a prolonged DENSE coarse
    underlay ``parent_field`` (n_vars, *parent_grid at ``level-1``); the pad
    mode encodes the physical BC (``boundary``: 0 periodic, 1 Neumann,
    2 reflective). Requires even ``n_ghost``. Returns ``fine_ghosted`` with
    NaN ghosts filled."""
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
    """Sparse coarse-fine ghost fill: gather each fine block's underlay from
    its parent BLOCK (``coarse_ghosted``, which must already be ghost-filled)
    and prolong. Needs even ``n_ghost``; corner ghosts tracing to parent NaN
    corners stay NaN. ``interp="linear"`` gives a (tri)linear underlay for
    smooth scalars; ``"const"`` (default) is the conservative hydro fill."""
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
