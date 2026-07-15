"""Explicit ``jax.shard_map`` distribution of the AMR block step.

Blocks are the shard unit (the ``n_slots`` axis is partitioned across devices),
mirroring how the dense LUNA path shards spatial axes. The one AMR-specific twist:
block neighbours are *irregular* (Morton ids, any block can neighbour any other),
so the halo / seam exchange is an explicit ``all_gather`` of the small boundary
data rather than the dense path's nearest-neighbour ``ppermute``.

Per-device work: each device fills ghosts and computes fluxes/divergence for its
LOCAL blocks (the expensive part, sharded); only the block-boundary reconstructed
states are all-gathered so a block can pull its neighbour's reconstruction even
when that neighbour lives on another device.

First target: one periodic level (the base grid) — the core cross-device
same-level seam exchange. Coarse-fine (parent gather) extends the same pattern.
"""

from __future__ import annotations

import itertools
import os

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from amr.halo import _spatial_slices
from amr.hydro import (
    block_boundary_states,
    block_face_fluxes,
    cons_to_prim_blocks,
    _inner_ghost_index,
    _set_boundary_face,
)
from hydro.muscl_hancock import (
    divergence_from_face_fluxes,
    muscl_hancock_boundary_states,
    muscl_hancock_face_fluxes,
    riemann_face_flux,
)

_shard_map = getattr(jax, "shard_map", None)
if _shard_map is None:  # older jax
    from jax.experimental.shard_map import shard_map as _shard_map


def _use_ring(n_dev, gathered_bytes):
    """Pick the halo exchange strategy. LUNA_AMR_HALO_RING = "1" (force ring) /
    "0" (force all_gather) / "auto" (default): use the memory-lean ring ONLY when
    the all_gather peak (the full block array on every device) would exceed
    LUNA_AMR_HALO_RING_GB (default 16 GB). So shallow hierarchies keep the fast
    single-pass all_gather; deep ones that would otherwise OOM get the total/D
    ring automatically."""
    if n_dev <= 1:
        return False
    mode = os.environ.get("LUNA_AMR_HALO_RING", "auto")
    if mode == "1":
        return True
    if mode == "0":
        return False
    thresh_gb = float(os.environ.get("LUNA_AMR_HALO_RING_GB", "16"))
    return gathered_bytes > thresh_gb * 1e9


def _fill_same_level_2src(local_data, global_data, neighbors, cfg):
    """``fill_ghost_same_level`` with two sources: interior from this device's
    ``local_data`` blocks, neighbour ghost bands gathered from the all-gathered
    ``global_data`` (neighbour indices are global). Returns the local blocks'
    ghosted array ``(Nloc, n_vars, *(bs+2g))``."""
    g, ndim, bs = cfg.n_ghost, cfg.ndim, cfg.block_shape
    Nloc, n_vars = local_data.shape[0], local_data.shape[1]
    out = jnp.full((Nloc, n_vars) + tuple(s + 2 * g for s in bs), jnp.nan, local_data.dtype)
    out = out.at[_spatial_slices(ndim, lambda d: slice(g, g + bs[d]))].set(local_data)
    for d in range(ndim):
        sax = 2 + d
        plus_nb = global_data[neighbors[:, 2 * d]]
        minus_nb = global_data[neighbors[:, 2 * d + 1]]
        nb_first = [slice(None), slice(None)] + [slice(None)] * ndim
        nb_first[sax] = slice(0, g)
        nb_last = [slice(None), slice(None)] + [slice(None)] * ndim
        nb_last[sax] = slice(bs[d] - g, bs[d])
        out_plus = list(_spatial_slices(ndim, lambda dd: slice(g, g + bs[dd])))
        out_plus[sax] = slice(g + bs[d], g + bs[d] + g)
        out = out.at[tuple(out_plus)].set(plus_nb[tuple(nb_first)])
        out_minus = list(_spatial_slices(ndim, lambda dd: slice(g, g + bs[dd])))
        out_minus[sax] = slice(0, g)
        out = out.at[tuple(out_minus)].set(minus_nb[tuple(nb_last)])
    return out


# --------------------------------------------------------------------------- #
# Distributed ghost fill (the replicated ``driver._fill_ghosts`` was ~half of
# the 6-level advance's OOM). Same-level fill is made corner-EXACT and sharded:
# the replicated ``fill_ghost_same_level`` runs a sequential axis sweep whose
# LAST-writing pass sets each ghost region to a *diagonal* neighbour (the unsplit
# MUSCL-Hancock predictor reads edges/corners). We reproduce that diagonal by
# composing face-hops on the small (replicated) neighbour table -- highest axis
# hopped first (the sweep order z,y,x makes the highest nonzero axis the outer
# read), then gather each region's tensor-product sub-cube from the all-gathered
# UNGHOSTED data, so the big ghosted array is born sharded (1/D).
# --------------------------------------------------------------------------- #
def _same_level_offsets(ndim):
    """Ghost-region offsets in a fixed order (matches ``_same_level_stencil``)."""
    return list(itertools.product((-1, 0, 1), repeat=ndim))


def _same_level_stencil(neighbors, cfg):
    """Global source slot for every ghost region, per block. ``neighbors`` is the
    (replicated, small) face table (order ``[+d,-d]`` per axis, sentinel = last
    slot). Returns ``(n_slots, 3**ndim)`` int32: column ``o`` is the block reached
    by composing the face-hops of offset ``offsets[o]`` from HIGHEST nonzero axis
    to lowest (the exact neighbour the sweep's last pass reads). Any missing hop
    collapses to the sentinel -> that region fills NaN, matching the sweep."""
    ndim = cfg.ndim
    n = neighbors.shape[0]
    sentinel = n - 1
    base = jnp.arange(n, dtype=jnp.int32)

    def hop(slots, d, sign):
        col = 2 * d + (0 if sign > 0 else 1)            # [+d, -d]
        nxt = neighbors[slots, col].astype(jnp.int32)
        return jnp.where(slots == sentinel, sentinel, nxt)

    cols = []
    for off in _same_level_offsets(ndim):
        s = base
        for d in reversed(range(ndim)):                 # highest axis first
            if off[d] != 0:
                s = hop(s, d, off[d])
        cols.append(s)
    return jnp.stack(cols, axis=1)                      # (n_slots, 3**ndim)


def _fill_same_level_diag(W, stencil_idx, mesh, cfg):
    """Corner-exact same-level ghost fill, sharded on the block axis. ``W`` is the
    (sharded) unghosted primitive data; ``stencil_idx`` (:func:`_same_level_stencil`,
    sharded rows) gives each local block's global source slot per ghost region.
    Returns the local blocks' ghosted array, sharded -- bit-identical to
    ``halo.fill_ghost_same_level`` on the active blocks.

    Two exchange strategies (bit-identical; each of the 3**ndim ghost regions is
    written by exactly ONE source block, so order is irrelevant):
      * all_gather (default): gather the FULL global block array on every device,
        then index. Simple, one pass, O(total) MEMORY per device -- the sharding
        win evaporates during the fill (was ~half the 6-level OOM).
      * ring (LUNA_AMR_HALO_RING=1): ppermute the block segments around the device
        ring; at each of the D steps a device holds ONE segment (total/D) and
        extracts the bands whose global source lands in that segment. Peak memory
        total/D (the win). Trades memory for ~D x the (cheap) band-copy compute.
    """
    g, ndim, bs = cfg.n_ghost, cfg.ndim, cfg.block_shape
    offsets = _same_level_offsets(ndim)
    W_spec = P("blocks", *([None] * (W.ndim - 1)))
    idx_spec = P("blocks", None)
    out_spec = P("blocks", *([None] * (ndim + 1)))
    n_dev = int(mesh.shape["blocks"])
    ring = _use_ring(n_dev, W.size * W.itemsize)         # all_gather peak = full W

    def _place(out, off, src, valid):
        """Copy source block ``src``'s band for ghost-region ``off`` into ``out``,
        masked by ``valid`` (None = unconditional)."""
        ssl = [slice(None), slice(None)]
        dsl = [slice(None), slice(None)]
        for d in range(ndim):
            if off[d] == -1:                              # -face: src last g -> dst first g
                ssl.append(slice(bs[d] - g, bs[d])); dsl.append(slice(0, g))
            elif off[d] == 0:                             # interior
                ssl.append(slice(0, bs[d]));         dsl.append(slice(g, g + bs[d]))
            else:                                         # +face: src first g -> dst last g
                ssl.append(slice(0, g));             dsl.append(slice(g + bs[d], g + bs[d] + g))
        band = src[tuple(ssl)]
        if valid is None:
            return out.at[tuple(dsl)].set(band)
        cur = out[tuple(dsl)]
        vv = valid.reshape((valid.shape[0],) + (1,) * (cur.ndim - 1))
        return out.at[tuple(dsl)].set(jnp.where(vv, band, cur))

    def local_gather(W_loc, idx_loc):
        Wg = jax.lax.all_gather(W_loc, "blocks", axis=0, tiled=True)   # global unghosted
        Nloc, n_vars = W_loc.shape[0], W_loc.shape[1]
        out = jnp.full((Nloc, n_vars) + tuple(s + 2 * g for s in bs),
                       jnp.nan, W_loc.dtype)
        for o, off in enumerate(offsets):
            out = _place(out, off, Wg[idx_loc[:, o]], None)
        return out

    def local_ring(W_loc, idx_loc):
        bpd = W_loc.shape[0]                              # blocks per device (== Nloc)
        Nloc, n_vars = W_loc.shape[0], W_loc.shape[1]
        dev = jax.lax.axis_index("blocks")
        out = jnp.full((Nloc, n_vars) + tuple(s + 2 * g for s in bs),
                       jnp.nan, W_loc.dtype)
        cur = W_loc                                       # segment currently held
        perm = [(i, (i + 1) % n_dev) for i in range(n_dev)]   # shift ring by 1
        for k in range(n_dev):
            seg_lo = ((dev - k) % n_dev) * bpd            # global range of the held segment
            for o, off in enumerate(offsets):
                li = idx_loc[:, o] - seg_lo               # local index into cur
                valid = (li >= 0) & (li < bpd)            # neighbour is in THIS segment
                src = cur[jnp.clip(li, 0, bpd - 1)]
                out = _place(out, off, src, valid)
            if k < n_dev - 1:
                cur = jax.lax.ppermute(cur, "blocks", perm)
        return out

    fn = local_ring if ring else local_gather
    try:
        return _shard_map(fn, mesh=mesh, in_specs=(W_spec, idx_spec),
                          out_specs=out_spec, check_vma=False)(W, stencil_idx)
    except TypeError:                                     # older jax: check_rep
        return _shard_map(fn, mesh=mesh, in_specs=(W_spec, idx_spec),
                          out_specs=out_spec, check_rep=False)(W, stencil_idx)


def build_exchange_plan(src_slots_np, n_src, n_dev):
    """Host-side sparse exchange plan (Stage 2: O(surface) communication) for a
    general block gather ``dst_row <- src_slots[dst_row, :]``. Topology is static
    between regrids, so we precompute, for every ordered device pair (s -> r), the
    sorted unique list of REMOTE source blocks r's rows need from s. Local sources
    (most of them, under the Morton-contiguous partition) are read directly and
    never touch the wire -- that is what makes the traffic O(surface) instead of
    the all_gather/ring's O(total).

    ``src_slots_np`` (n_rows, R) int -- global source slot per destination row and
    stencil column. ``n_src`` -- size of the SOURCE block array (may differ from
    n_rows, e.g. coarse-fine: rows = fine blocks, sources = coarse blocks). Both
    n_rows and n_src must be divisible by n_dev (guaranteed by pad_to).

    Returns (send_idx, recv_pos, K):
      send_idx (n_dev, n_dev, K) int32 -- sender-LOCAL source indices; row [s, r]
          is what s packs for r, padded with 0 (padded entries are never
          referenced by any recv_pos).
      recv_pos (n_rows, R) int32 -- for each row and column, the flat position
          (s*K + j) of its source in the receiver's all_to_all buffer, or -1 if
          the source is LOCAL to the row's owner (read directly).
      K -- per-pair slot budget, pow2 bucket (bounds recompiles across regrids).
    """
    import numpy as _np
    n_rows, R = src_slots_np.shape
    bpd_row = n_rows // n_dev
    bpd_src = n_src // n_dev
    row_dev = _np.arange(n_rows) // bpd_row
    owner = src_slots_np // bpd_src
    uniq = {}                                        # (s, r) -> sorted unique globals
    kmax = 1
    for r in range(n_dev):
        st = src_slots_np[row_dev == r]              # (bpd_row, R)
        ow = owner[row_dev == r]
        for s in range(n_dev):
            if s == r:
                continue
            u = _np.unique(st[ow == s])
            uniq[(s, r)] = u
            kmax = max(kmax, len(u))
    K = 1 << (kmax - 1).bit_length()                 # pow2 bucket
    send_idx = _np.zeros((n_dev, n_dev, K), _np.int32)
    recv_pos = _np.full((n_rows, R), -1, _np.int32)
    for r in range(n_dev):
        rows = slice(r * bpd_row, (r + 1) * bpd_row)
        st = src_slots_np[rows]
        ow = owner[rows]
        for s in range(n_dev):
            if s == r:
                continue
            u = uniq[(s, r)]
            if len(u) == 0:
                continue
            send_idx[s, r, : len(u)] = u - s * bpd_src   # sender-local indices
            m = ow == s
            j = _np.searchsorted(u, st[m])
            block = recv_pos[rows]
            block[m] = s * K + j
            recv_pos[rows] = block
    return send_idx, recv_pos, K


def build_same_level_plan(stencil_np, n_dev):
    """Same-level ghost-fill plan: rows and sources are the SAME block array."""
    return build_exchange_plan(stencil_np, stencil_np.shape[0], n_dev)


def _fill_same_level_sparse(W, stencil_idx, send_idx, recv_pos, mesh, cfg):
    """Same-level ghost fill with the SPARSE all_to_all exchange (Stage 2).
    Each device packs only the blocks its peers actually need
    (``send_idx[dev]``), one ``all_to_all`` delivers them, and each ghost band
    fills from a two-source select: local block (direct read) vs received
    remote block (``recv_pos``). Communication and remote-buffer memory are
    O(surface blocks) instead of O(total). Bit-identical to the gather/ring
    fills: same source block per region, same band copy."""
    g, ndim, bs = cfg.n_ghost, cfg.ndim, cfg.block_shape
    offsets = _same_level_offsets(ndim)
    n_dev = int(mesh.shape["blocks"])
    K = send_idx.shape[-1]
    W_spec = P("blocks", *([None] * (W.ndim - 1)))
    idx_spec = P("blocks", None)
    out_spec = P("blocks", *([None] * (ndim + 1)))

    def local(W_loc, idx_loc, rp_loc, send_tab):
        bpd = W_loc.shape[0]
        dev = jax.lax.axis_index("blocks")
        # pack: (n_dev*K, n_vars, *bs) -- chunk r = what this device sends to r
        buf = W_loc[send_tab[dev].reshape(-1)]
        got = jax.lax.all_to_all(buf, "blocks", 0, 0, tiled=True)
        Nloc, n_vars = W_loc.shape[0], W_loc.shape[1]
        out = jnp.full((Nloc, n_vars) + tuple(s + 2 * g for s in bs),
                       jnp.nan, W_loc.dtype)
        for o, off in enumerate(offsets):
            li = idx_loc[:, o] - dev * bpd
            is_loc = (li >= 0) & (li < bpd)
            src_l = W_loc[jnp.clip(li, 0, bpd - 1)]
            src_r = got[jnp.clip(rp_loc[:, o], 0, n_dev * K - 1)]
            vv = is_loc.reshape((is_loc.shape[0],) + (1,) * (src_l.ndim - 1))
            src = jnp.where(vv, src_l, src_r)
            ssl = [slice(None), slice(None)]
            dsl = [slice(None), slice(None)]
            for d in range(ndim):
                if off[d] == -1:
                    ssl.append(slice(bs[d] - g, bs[d])); dsl.append(slice(0, g))
                elif off[d] == 0:
                    ssl.append(slice(0, bs[d]));         dsl.append(slice(g, g + bs[d]))
                else:
                    ssl.append(slice(0, g));             dsl.append(slice(g + bs[d], g + bs[d] + g))
            out = out.at[tuple(dsl)].set(src[tuple(ssl)])
        return out

    ins = (W_spec, idx_spec, idx_spec, P())          # send table replicated
    try:
        return _shard_map(local, mesh=mesh, in_specs=ins, out_specs=out_spec,
                          check_vma=False)(W, stencil_idx, recv_pos, send_idx)
    except TypeError:
        return _shard_map(local, mesh=mesh, in_specs=ins, out_specs=out_spec,
                          check_rep=False)(W, stencil_idx, recv_pos, send_idx)


def _coarse_underlay_sharded(fine_ghosted, fine_coords, parent_slots,
                             coarse_ghosted, mesh, cfg, interp="linear",
                             plan=None):
    """Sparse coarse-fine ghost fill (``halo.fill_ghost_from_coarse_sparse``) with
    the fine blocks sharded. Each device all-gathers the (smaller) parent ghosted
    level, then runs the identical per-block gather+prolong for its local fine
    blocks. Output sharded on the fine block axis.

    ``plan`` (``(send_idx, recv_pos)`` from :func:`build_exchange_plan` on
    ``parent_slots[:, None]``): switches to the SPARSE all_to_all exchange -- pack
    only the coarse parents peers need, one all_to_all, then each fine block reads
    its single parent (local coarse or received) and prolongs. O(surface) comm."""
    from amr.halo import fill_ghost_from_coarse_sparse
    fg_spec = P("blocks", *([None] * (fine_ghosted.ndim - 1)))
    coord_spec = P("blocks", None)
    ps_spec = P("blocks")
    cg_spec = P("blocks", *([None] * (coarse_ghosted.ndim - 1)))
    n_dev = int(mesh.shape["blocks"])
    ring = _use_ring(n_dev, coarse_ghosted.size * coarse_ghosted.itemsize)

    if plan is not None:
        send_idx, recv_pos = plan
        K = send_idx.shape[-1]
        rp_spec = P("blocks")                        # recv_pos is (n_fine,)

        def local_sparse(fg_loc, coord_loc, ps_loc, cg_loc, rp_loc, send_tab):
            bpd_c = cg_loc.shape[0]                   # coarse blocks per device
            dev = jax.lax.axis_index("blocks")
            buf = cg_loc[send_tab[dev].reshape(-1)]
            got = jax.lax.all_to_all(buf, "blocks", 0, 0, tiled=True)
            li = ps_loc - dev * bpd_c
            is_loc = (li >= 0) & (li < bpd_c)
            parent_local = cg_loc[jnp.clip(li, 0, bpd_c - 1)]
            parent_remote = got[jnp.clip(rp_loc, 0, n_dev * K - 1)]
            vv = is_loc.reshape((is_loc.shape[0],) + (1,) * (parent_local.ndim - 1))
            parent_per_fine = jnp.where(vv, parent_local, parent_remote)
            # reuse the exact prolong: parent_slots = arange -> each fine block's
            # source is parent_per_fine[i], bit-identical to the gather path.
            idx = jnp.arange(fg_loc.shape[0], dtype=jnp.int32)
            return fill_ghost_from_coarse_sparse(fg_loc, coord_loc, idx,
                                                 parent_per_fine, cfg, interp=interp)

        ins = (fg_spec, coord_spec, ps_spec, cg_spec, rp_spec, P())
        try:
            return _shard_map(local_sparse, mesh=mesh, in_specs=ins, out_specs=fg_spec,
                              check_vma=False)(fine_ghosted, fine_coords, parent_slots,
                                               coarse_ghosted, recv_pos, send_idx)
        except TypeError:
            return _shard_map(local_sparse, mesh=mesh, in_specs=ins, out_specs=fg_spec,
                              check_rep=False)(fine_ghosted, fine_coords, parent_slots,
                                               coarse_ghosted, recv_pos, send_idx)

    def local_gather(fg_loc, coord_loc, ps_loc, cg_loc):
        cg = jax.lax.all_gather(cg_loc, "blocks", axis=0, tiled=True)  # global parent ghosted
        return fill_ghost_from_coarse_sparse(fg_loc, coord_loc, ps_loc, cg,
                                             cfg, interp=interp)

    def local_ring(fg_loc, coord_loc, ps_loc, cg_loc):
        # Ring the coarse (parent) segments; at each step fill only the fine
        # blocks whose parent slot lands in the held segment. Peak = one coarse
        # segment (total_coarse/D) + local fine. Bit-identical: each fine block is
        # filled exactly once, at its parent's segment step (outer where guards).
        bpd_c = cg_loc.shape[0]
        dev = jax.lax.axis_index("blocks")
        out = fg_loc
        cur = cg_loc
        perm = [(i, (i + 1) % n_dev) for i in range(n_dev)]
        for k in range(n_dev):
            seg_lo = ((dev - k) % n_dev) * bpd_c
            ps_local = ps_loc - seg_lo
            valid = (ps_local >= 0) & (ps_local < bpd_c)
            filled = fill_ghost_from_coarse_sparse(
                out, coord_loc, jnp.clip(ps_local, 0, bpd_c - 1), cur, cfg, interp=interp)
            vv = valid.reshape((valid.shape[0],) + (1,) * (out.ndim - 1))
            out = jnp.where(vv, filled, out)
            if k < n_dev - 1:
                cur = jax.lax.ppermute(cur, "blocks", perm)
        return out

    fn = local_ring if ring else local_gather
    ins = (fg_spec, coord_spec, ps_spec, cg_spec)
    try:
        return _shard_map(fn, mesh=mesh, in_specs=ins, out_specs=fg_spec,
                          check_vma=False)(fine_ghosted, fine_coords, parent_slots, coarse_ghosted)
    except TypeError:
        return _shard_map(fn, mesh=mesh, in_specs=ins, out_specs=fg_spec,
                          check_rep=False)(fine_ghosted, fine_coords, parent_slots, coarse_ghosted)


def _fill_ghosts_sharded(levels, cfg, mesh, gamma, plans=None):
    """Sharded twin of ``driver._fill_ghosts``: per-level ghost-filled PRIMITIVE
    blocks, each level's ghosted array born SHARDED on the block axis (the big
    array), with only the smaller unghosted / parent gathers transient. Same-level
    fill is corner-exact (:func:`_fill_same_level_diag`); finer levels add the
    sparse coarse underlay; a physical BC (per-block, sharding-preserving) fills
    domain-edge NaNs. Bit-identical to the replicated ``_fill_ghosts``.

    ``plans`` (optional, per level ``(same_plan, cf_plan)`` or None; each sub-plan
    ``(send_idx, recv_pos)`` or None) switches the same-level and/or coarse-fine
    fill to the SPARSE all_to_all exchange (Stage 2, O(surface) comm) — built at
    (re)grid time by :func:`build_exchange_plan` and threaded through the jitted
    step as arguments (never closure-captured: the plan VALUES change at every
    regrid even when shapes don't, so a captured constant would go stale)."""
    from amr.halo import fill_ghost_physical
    D = mesh.devices.size
    ghosted = []
    for L, lvl in enumerate(levels):
        W = cons_to_prim_blocks(lvl.data, gamma)
        if W.shape[0] % D != 0:                          # not shardable -> replicated fallback
            from amr.halo import fill_ghost_same_level, fill_ghost_from_coarse_sparse
            gw = fill_ghost_same_level(W, lvl.neighbors, cfg)
            if L > 0:
                gw = fill_ghost_from_coarse_sparse(gw, lvl.coords, lvl.parent_slots,
                                                   ghosted[L - 1], cfg, interp="linear")
            gw = fill_ghost_physical(gw, W, cfg)
            ghosted.append(gw)
            continue
        same_plan, cf_plan = (plans[L] if (plans is not None and plans[L] is not None)
                              else (None, None))
        stencil = _same_level_stencil(lvl.neighbors, cfg)
        if same_plan is not None:
            gw = _fill_same_level_sparse(W, stencil, same_plan[0], same_plan[1],
                                         mesh, cfg)
        else:
            gw = _fill_same_level_diag(W, stencil, mesh, cfg)
        # physical DOMAIN bands BEFORE the coarse underlay (see driver._fill_ghosts)
        gw = fill_ghost_physical(gw, W, cfg, coords=lvl.coords, level=L)
        if L > 0:
            gw = _coarse_underlay_sharded(gw, lvl.coords, lvl.parent_slots,
                                          ghosted[L - 1], mesh, cfg, interp="linear",
                                          plan=cf_plan)
        gw = fill_ghost_physical(gw, W, cfg)
        ghosted.append(gw)
    return ghosted


# --------------------------------------------------------------------------- #
# Batched single-launch divergence across ALL levels (task #3/#69). The 6-level
# advance compiled 6 distinct MUSCL divergence kernels (one per level's block
# count) -> the wall was COMPILE time, not memory. Every level shares the same
# per-block ghosted shape, so we concatenate all levels onto ONE block axis and
# run a SINGLE kernel with a per-block ``inv_dx`` (the only per-level variation
# under a synchronized dt) and globally-offset neighbour indices -> one compile,
# one dispatch. Bit-identical to summing the per-level ``divergence_shmap`` (the
# seam Riemann is just gather-by-global-index; a block's neighbours stay within
# its own level so no cross-level coupling is introduced).
# --------------------------------------------------------------------------- #
def _inv_kw(iv, ndim):
    kw = dict(inv_dx=iv[0], inv_dy=iv[1] if ndim >= 2 else 1.0)
    kw["inv_dz"] = iv[2] if ndim == 3 else 1.0
    return kw


def _block_fluxes_vec(W_ghosted, dt, inv, cfg, *, gamma, limiter, riemann_solver):
    """``block_face_fluxes`` with a PER-BLOCK ``inv`` (n, ndim) instead of a scalar
    level -- the inner MUSCL kernel is unchanged, ``inv`` is just vmapped."""
    ndim, g = cfg.ndim, cfg.n_ghost
    W_pad = W_ghosted[_inner_ghost_index(ndim, g)]

    def one_block(Wp, iv):
        return muscl_hancock_face_fluxes(
            Wp, dt, gamma=gamma, limiter=limiter, riemann_solver=riemann_solver,
            spatial_dim=ndim, boundary=1, exact_wrap=False, **_inv_kw(iv, ndim))

    return jax.vmap(one_block)(W_pad, inv)


def _block_bstates_vec(W_ghosted, dt, inv, cfg, *, gamma, limiter):
    """``block_boundary_states`` with a per-block ``inv`` (n, ndim)."""
    ndim, g = cfg.ndim, cfg.n_ghost
    W_pad = W_ghosted[_inner_ghost_index(ndim, g)]

    def one_block(Wp, iv):
        return muscl_hancock_boundary_states(
            Wp, dt, gamma=gamma, limiter=limiter, spatial_dim=ndim, **_inv_kw(iv, ndim))

    return jax.vmap(one_block)(W_pad, inv)


def _seam_couple(fluxes, states, nbr_loc, mask_loc, ndim, n_dev, ring, seam,
                 seam_send=None, seam_recv=None, dom_loc=None):
    """Replace each block-block face flux with the SHARED seam Riemann flux built
    from this block's reconstruction and its (possibly remote) neighbour's.

    sparse (seam_send/seam_recv given): O(surface) all_to_all. Each block's
    payload stacks its per-axis boundary states in neighbour-column order
    (slot 2d = Ld_d wanted by a -side neighbour via col 2d, slot 2d+1 = Rd_d
    wanted by a +side neighbour via col 2d+1); one all_to_all delivers the
    remote payloads, and each face reads its neighbour's plane local-or-received.
    Requires cubic blocks (equal per-axis face shape to stack) -- the runner only
    builds the plan then. Bit-identical to the gather path.

    ring=False: all_gather the per-axis boundary states (O(total) memory/device).
    ring=True : ppermute the state segments around the device ring; at each of
    the n_dev steps couple only the faces whose neighbour slot lands in the held
    segment. Peak = one segment (total/D). Bit-identical: each face's neighbour
    lives in exactly one segment, so it is coupled exactly once (masked set);
    faces with no real neighbour (mask False) keep the ghost-based flux, as in
    the gather path.

    ``dom_loc`` (Nloc, 2*ndim) marks PHYSICAL (outflow) domain faces: their flux
    is replaced by the consistent zero-gradient flux f(pred) = seam(own, own)
    (see conservative_block_divergence -- the raw ghost-based flux pairs a
    predicted interior state with an un-predicted ghost copy and pumps mass
    through the wall). Applied AFTER the seam coupling, before returning."""

    def _domain_faces(fluxes):
        if dom_loc is None:
            return fluxes
        for d in range(ndim):
            Ld, Rd = states[d]
            F = fluxes[d]
            F = _set_boundary_face(F, d, -1, seam(Rd, Rd, d), dom_loc[:, 2 * d], ndim)
            F = _set_boundary_face(F, d, 0, seam(Ld, Ld, d), dom_loc[:, 2 * d + 1], ndim)
            fluxes[d] = F
        return fluxes
    if seam_send is not None:
        bpd = states[0][0].shape[0]
        dev = jax.lax.axis_index("blocks")
        K = seam_send.shape[-1]
        planes = []
        for d in range(ndim):                        # slot 2d=Ld_d, 2d+1=Rd_d
            Ld, Rd = states[d]
            planes.append(Ld)
            planes.append(Rd)
        payload = jnp.stack(planes, axis=1)          # (bpd, 2ndim, n_vars, *face)
        buf = payload[seam_send[dev].reshape(-1)]
        got = jax.lax.all_to_all(buf, "blocks", 0, 0, tiled=True)

        def nbr_state(c):                            # neighbour's plane for column c
            li = nbr_loc[:, c] - dev * bpd
            is_loc = (li >= 0) & (li < bpd)
            loc = payload[jnp.clip(li, 0, bpd - 1)][:, c]
            rem = got[jnp.clip(seam_recv[:, c], 0, n_dev * K - 1)][:, c]
            vv = is_loc.reshape((is_loc.shape[0],) + (1,) * (loc.ndim - 1))
            return jnp.where(vv, loc, rem)

        for d in range(ndim):
            Ld, Rd = states[d]
            mpl, mmi = mask_loc[:, 2 * d], mask_loc[:, 2 * d + 1]
            F = fluxes[d]
            F = _set_boundary_face(F, d, -1, seam(Rd, nbr_state(2 * d), d), mpl, ndim)
            F = _set_boundary_face(F, d, 0, seam(nbr_state(2 * d + 1), Ld, d), mmi, ndim)
            fluxes[d] = F
        return _domain_faces(fluxes)

    if not ring:
        for d in range(ndim):
            Ld, Rd = states[d]
            Ld_g = jax.lax.all_gather(Ld, "blocks", axis=0, tiled=True)
            Rd_g = jax.lax.all_gather(Rd, "blocks", axis=0, tiled=True)
            npl, nmi = nbr_loc[:, 2 * d], nbr_loc[:, 2 * d + 1]
            mpl, mmi = mask_loc[:, 2 * d], mask_loc[:, 2 * d + 1]
            F = fluxes[d]
            F = _set_boundary_face(F, d, -1, seam(Rd, Ld_g[npl], d), mpl, ndim)
            F = _set_boundary_face(F, d, 0, seam(Rd_g[nmi], Ld, d), mmi, ndim)
            fluxes[d] = F
        return _domain_faces(fluxes)

    bpd = states[0][0].shape[0]                      # blocks per device
    dev = jax.lax.axis_index("blocks")
    perm = [(i, (i + 1) % n_dev) for i in range(n_dev)]
    cur = [states[d] for d in range(ndim)]           # held (Ld, Rd) segments
    for k in range(n_dev):
        seg_lo = ((dev - k) % n_dev) * bpd
        for d in range(ndim):
            Ld, Rd = states[d]                       # LOCAL blocks' own states
            Ld_seg, Rd_seg = cur[d]                  # held remote segment
            npl, nmi = nbr_loc[:, 2 * d], nbr_loc[:, 2 * d + 1]
            mpl, mmi = mask_loc[:, 2 * d], mask_loc[:, 2 * d + 1]
            lp = npl - seg_lo
            lm = nmi - seg_lo
            vp = mpl & (lp >= 0) & (lp < bpd)
            vm = mmi & (lm >= 0) & (lm < bpd)
            F = fluxes[d]
            F = _set_boundary_face(F, d, -1,
                                   seam(Rd, Ld_seg[jnp.clip(lp, 0, bpd - 1)], d),
                                   vp, ndim)
            F = _set_boundary_face(F, d, 0,
                                   seam(Rd_seg[jnp.clip(lm, 0, bpd - 1)], Ld, d),
                                   vm, ndim)
            fluxes[d] = F
        if k < n_dev - 1:
            cur = [(jax.lax.ppermute(l, "blocks", perm),
                    jax.lax.ppermute(r, "blocks", perm)) for (l, r) in cur]
    return _domain_faces(fluxes)


def _states_bytes(W_all, cfg):
    """Rough aggregate size of the 2*ndim boundary-state planes (the seam
    all_gather payload) for the ring-vs-gather decision."""
    ndim, bs = cfg.ndim, cfg.block_shape
    N, n_vars = W_all.shape[0], W_all.shape[1]
    tot = 0
    for d in range(ndim):
        face = 1
        for dd in range(ndim):
            if dd != d:
                face *= bs[dd]
        tot += 2 * N * n_vars * face                  # Ld + Rd
    return tot * W_all.dtype.itemsize


def divergence_batched_shmap(W_all, inv_all, nbr_all, mask_all, mesh, dt, cfg, *,
                             gamma, limiter=1, riemann_solver="hllc", seam_plan=None,
                             dom_all=None):
    """Single sharded divergence over the concatenated all-levels block axis.
    ``W_all`` (N, n_vars, *ghosted) is every level's ghosted blocks stacked;
    ``inv_all`` (N, ndim) the per-block 1/dx; ``nbr_all`` (N, 2*ndim) GLOBAL
    same-level neighbour slots (invalid -> that level's NaN sentinel slot);
    ``mask_all`` (N, 2*ndim) bool marks real neighbours. Returns (dU_all, fluxes)
    sharded on the block axis. Bit-matches the per-level ``divergence_shmap``.

    ``seam_plan`` (``(send_idx, recv_pos)`` from :func:`build_exchange_plan` on
    ``nbr_all``): the seam boundary-state coupling uses the SPARSE all_to_all
    (O(surface) comm) instead of the all_gather/ring."""
    ndim = cfg.ndim
    blk = P("blocks", *([None] * (W_all.ndim - 1)))
    vec = P("blocks", None)
    out_specs = (blk, tuple(blk for _ in range(ndim)))
    n_dev = int(mesh.shape["blocks"])
    ring = _use_ring(n_dev, _states_bytes(W_all, cfg))
    sparse = seam_plan is not None
    seam_send = seam_plan[0] if sparse else None

    def local(W_loc, inv_loc, nbr_loc, mask_loc, dom_loc, seam_recv, seam_send_tab):
        fluxes = list(_block_fluxes_vec(W_loc, dt, inv_loc, cfg, gamma=gamma,
                                        limiter=limiter, riemann_solver=riemann_solver))
        states = _block_bstates_vec(W_loc, dt, inv_loc, cfg, gamma=gamma, limiter=limiter)

        def seam(a, b, d):
            return jax.vmap(lambda x, y: riemann_face_flux(
                x, y, axis=d, gamma=gamma, riemann_solver=riemann_solver))(a, b)

        fluxes = _seam_couple(fluxes, states, nbr_loc, mask_loc, ndim, n_dev,
                              ring, seam,
                              seam_send=(seam_send_tab if sparse else None),
                              seam_recv=(seam_recv if sparse else None),
                              dom_loc=dom_loc)

        def dv(iv, *fs):
            return divergence_from_face_fluxes(fs, spatial_dim=ndim, **_inv_kw(iv, ndim))

        dU = jax.vmap(dv)(inv_loc, *fluxes)
        return dU, tuple(fluxes)

    # seam_recv sharded like nbr_all; seam_send replicated. Dummies when dense so
    # the shard_map signature is uniform (never read unless sparse).
    seam_recv_in = nbr_all if not sparse else seam_plan[1]
    seam_send_in = jnp.zeros((1,), jnp.int32) if not sparse else seam_send
    dom_in = dom_all if dom_all is not None else jnp.zeros_like(mask_all)
    ins = (blk, vec, vec, vec, vec, vec, P())
    args = (W_all, inv_all, nbr_all, mask_all, dom_in, seam_recv_in, seam_send_in)
    try:
        return _shard_map(local, mesh=mesh, in_specs=ins, out_specs=out_specs,
                          check_vma=False)(*args)
    except TypeError:
        return _shard_map(local, mesh=mesh, in_specs=ins, out_specs=out_specs,
                          check_rep=False)(*args)


def _batch_levels(ghosted, hierarchy, cfg):
    """Stack per-level ghosted blocks + topology into the concatenated-block-axis
    arrays for :func:`divergence_batched_shmap`. Returns (W_all, inv_all, nbr_all,
    mask_all, offsets) where offsets[L] is level L's start in the stacked axis."""
    ndim = cfg.ndim
    bs = cfg.block_shape
    r = cfg.refinement_ratio
    W_list, inv_list, nbr_list, mask_list, dom_list, offsets = [], [], [], [], [], []
    acc = 0
    for L, lvl in enumerate(hierarchy):
        gL = ghosted[L]
        nL = gL.shape[0]
        dxL = cfg.level_dx(L)
        invL = jnp.broadcast_to(
            jnp.asarray([1.0 / dxL[d] for d in range(ndim)], gL.dtype), (nL, ndim))
        nbrL = lvl.neighbors                              # (nL, 2ndim) local, sentinel = nL-1
        maskL = nbrL != (nL - 1)
        # physical (outflow) domain faces: block at the level's domain edge AND
        # no same-level neighbour there (consistent-BC flux, see _seam_couple).
        if cfg.boundary == 1:
            cols = []
            for d in range(ndim):
                nb_d = (cfg.base_grid[d] // bs[d]) * (r ** L)
                cols.append((lvl.coords[:, d] == nb_d - 1) & ~maskL[:, 2 * d])
                cols.append((lvl.coords[:, d] == 0) & ~maskL[:, 2 * d + 1])
            domL = jnp.stack(cols, axis=1)
        else:
            domL = jnp.zeros_like(maskL)
        offsets.append(acc)
        W_list.append(gL)
        inv_list.append(invL)
        nbr_list.append((nbrL + acc).astype(jnp.int32))   # invalid -> level's NaN sentinel slot
        mask_list.append(maskL)
        dom_list.append(domL)
        acc += nL
    return (jnp.concatenate(W_list, axis=0),
            jnp.concatenate(inv_list, axis=0),
            jnp.concatenate(nbr_list, axis=0),
            jnp.concatenate(mask_list, axis=0),
            jnp.concatenate(dom_list, axis=0),
            offsets)


def divergence_shmap(W_ghosted, neighbors, mesh, dt, level, cfg, *,
                     gamma, limiter=1, riemann_solver="hllc",
                     return_face_fluxes=False, coords=None):
    """Conservative seam-flux divergence for one level, with the per-block flux
    compute (the bottleneck) **sharded over devices** and the same-level seam
    coupling done by an ``all_gather`` of the boundary states. ``W_ghosted`` /
    ``neighbors`` are replicated; each device computes its local blocks' fluxes,
    all-gathers the boundary states + its dU, and returns the full (replicated)
    divergence — so it drops into the existing (replicated) driver loop. Reuses
    the existing ghost fill and restrict; only this hot kernel is distributed."""
    ndim = cfg.ndim
    W_spec = P("blocks", *([None] * (W_ghosted.ndim - 1)))
    nbr_spec = P("blocks", None)
    dx = cfg.level_dx(level)
    inv = dict(inv_dx=1.0 / dx[0], inv_dy=1.0 / dx[1],
               inv_dz=1.0 / dx[2] if ndim == 3 else 1.0)

    n_dev = int(mesh.shape["blocks"])
    ring = _use_ring(n_dev, _states_bytes(W_ghosted, cfg))
    sentinel = W_ghosted.shape[0] - 1                     # global NaN sentinel slot
    if coords is not None and cfg.boundary == 1:
        bs_, r_ = cfg.block_shape, cfg.refinement_ratio
        cols = []
        for d in range(ndim):
            nb_d = (cfg.base_grid[d] // bs_[d]) * (r_ ** level)
            cols.append((coords[:, d] == nb_d - 1) & (neighbors[:, 2 * d] == sentinel))
            cols.append((coords[:, d] == 0) & (neighbors[:, 2 * d + 1] == sentinel))
        dom = jnp.stack(cols, axis=1)
    else:
        dom = jnp.zeros((W_ghosted.shape[0], 2 * ndim), bool)

    def local(W_loc, nbr_loc, dom_loc):
        fluxes = list(block_face_fluxes(W_loc, dt, level, cfg, gamma=gamma,
                                        limiter=limiter, riemann_solver=riemann_solver))
        states = block_boundary_states(W_loc, dt, level, cfg, gamma=gamma, limiter=limiter)

        def seam(a, b, d):
            return jax.vmap(lambda x, y: riemann_face_flux(
                x, y, axis=d, gamma=gamma, riemann_solver=riemann_solver))(a, b)

        fluxes = _seam_couple(fluxes, states, nbr_loc, nbr_loc != sentinel,
                              ndim, n_dev, ring, seam, dom_loc=dom_loc)

        dU = jax.vmap(lambda *fs: divergence_from_face_fluxes(fs, spatial_dim=ndim, **inv))(*fluxes)
        # Step 2: return the LOCAL (per-device) divergence + seam-corrected
        # fluxes SHARDED on the block axis -- no output all_gather. The flux
        # compute above is already per-device (W_loc is this shard); only the
        # small face-state halo (Ld_g/Rd_g) is gathered. So the divergence peak
        # is now total/n_dev instead of replicated. dU stays sharded through
        # ``new_data = data + dt*dU`` in advance_shmap.
        if return_face_fluxes:
            return dU, tuple(fluxes)
        return dU

    blk_spec = P("blocks", *([None] * (W_ghosted.ndim - 1)))
    out_specs = ((blk_spec, tuple(blk_spec for _ in range(ndim)))
                 if return_face_fluxes else blk_spec)
    try:
        return _shard_map(local, mesh=mesh, in_specs=(W_spec, nbr_spec, nbr_spec),
                          out_specs=out_specs, check_vma=False)(W_ghosted, neighbors, dom)
    except TypeError:                                    # older jax: check_rep
        return _shard_map(local, mesh=mesh, in_specs=(W_spec, nbr_spec, nbr_spec),
                          out_specs=out_specs, check_rep=False)(W_ghosted, neighbors, dom)


def advance_shmap(hierarchy, dt, cfg, mesh, *, gamma, limiter=1,
                  riemann_solver="hllc", use_batched=True, halo_plans=None):
    """Synchronized multi-level step with the per-level divergence sharded across
    devices. Ghost fill is sharded (:func:`_fill_ghosts_sharded`); the divergence
    is a SINGLE batched kernel over all levels (:func:`divergence_batched_shmap`,
    ``use_batched=True``) -- collapsing the 6 per-level compiles into one -- or
    the per-level :func:`divergence_shmap` loop (``use_batched=False``, kept as a
    bit-identical fallback). Restrict/reflux stay per-level. Matches
    :func:`amr.driver.advance`."""
    from amr.driver import _restrict_into_parent
    from amr.reflux import apply_reflux_blocks

    from jax.sharding import NamedSharding

    repl = NamedSharding(mesh, P())

    # halo_plans = (level_plans, seam_plan): per-level fill plans + the global
    # batched-seam plan. None -> gather/ring path everywhere.
    level_plans, seam_plan = (halo_plans if halo_plans is not None else (None, None))

    # Sharded ghost fill: each level's (big) ghosted array is born sharded on the
    # block axis instead of replicated -- this was ~half of the 6-level advance's
    # ~90 GB peak. Bit-identical to driver._fill_ghosts (validated 4xA100).
    ghosted = _fill_ghosts_sharded(hierarchy, cfg, mesh, gamma, plans=level_plans)
    new_data = []
    fluxes = []
    if use_batched:
        # ONE divergence launch across all levels (per-block inv_dx, global
        # neighbour offsets) -> one XLA compile instead of one per level.
        W_all, inv_all, nbr_all, mask_all, dom_all, offsets = _batch_levels(ghosted, hierarchy, cfg)
        dU_all, fl_all = divergence_batched_shmap(
            W_all, inv_all, nbr_all, mask_all, mesh, dt, cfg,
            gamma=gamma, limiter=limiter, riemann_solver=riemann_solver,
            seam_plan=seam_plan, dom_all=dom_all)
        for L, lvl in enumerate(hierarchy):
            nL = ghosted[L].shape[0]
            s = offsets[L]
            new_data.append(lvl.data + dt * dU_all[s:s + nL])
            fluxes.append(tuple(f[s:s + nL] for f in fl_all))
    else:
        for L, lvl in enumerate(hierarchy):
            dU, fl = divergence_shmap(ghosted[L], lvl.neighbors, mesh, dt, L, cfg,
                                      gamma=gamma, limiter=limiter,
                                      riemann_solver=riemann_solver,
                                      return_face_fluxes=True, coords=lvl.coords)
            # dU / fl come back SHARDED on the block axis (no output all_gather);
            # new_data stays sharded so the divergence peak is total/n_dev.
            new_data.append(lvl.data + dt * dU)
            fluxes.append(fl)
    # Berger-Colella reflux. Two modes:
    #  * replicated (default): the whole correction runs INSIDE a shard_map with
    #    fully-replicated specs -- every device redundantly computes the identical
    #    correction, so the SPMD partitioner has zero freedom to shard the
    #    segment-sums over the block axis (which it silently did under an outer
    #    jit, dropping shard-remote contributions).
    #  * sharded (LUNA_AMR_REFLUX=sharded): apply_reflux_sharded -- fine fluxes
    #    stay sharded, partial corrections psum'd EXPLICITLY (root-cause fix for
    #    the GSPMD landmine; the compute distributes too).
    n_dev_ = int(mesh.shape["blocks"])
    _refl_sharded = os.environ.get("LUNA_AMR_REFLUX", "repl") == "sharded"
    for L in range(len(hierarchy) - 1, 0, -1):
        if (_refl_sharded
                and new_data[L - 1].shape[0] % n_dev_ == 0
                and hierarchy[L].neighbors.shape[0] % n_dev_ == 0):
            new_data[L - 1] = apply_reflux_sharded(
                new_data[L - 1], fluxes[L - 1], fluxes[L],
                hierarchy[L].coords, hierarchy[L].parent_slots,
                hierarchy[L].neighbors, hierarchy[L].n_active,
                hierarchy[L - 1].neighbors, dt, L - 1, cfg, mesh)
            continue
        args = (new_data[L - 1], fluxes[L - 1], fluxes[L],
                hierarchy[L].coords, hierarchy[L].parent_slots,
                hierarchy[L].neighbors, hierarchy[L].n_active,
                hierarchy[L - 1].neighbors)

        def _rf(cd, fc, ff, cc, ps, nb, na, cnb, _L=L):
            return apply_reflux_blocks(cd, fc, ff, cc, ps, nb, na, cnb,
                                       dt, _L - 1, cfg)

        try:
            new_data[L - 1] = _shard_map(_rf, mesh=mesh, in_specs=P(),
                                         out_specs=P(), check_vma=False)(*args)
        except TypeError:                                # older jax: check_rep
            new_data[L - 1] = _shard_map(_rf, mesh=mesh, in_specs=P(),
                                         out_specs=P(), check_rep=False)(*args)
    for L in range(len(hierarchy) - 1, 0, -1):
        fine = hierarchy[L]._replace(data=new_data[L])
        new_data[L - 1] = _restrict_into_parent(new_data[L - 1], fine, cfg)
    new_data = _reshard_blocks(new_data, mesh)
    return tuple(lvl._replace(data=nd) for lvl, nd in zip(hierarchy, new_data))


def _reshard_blocks(datas, mesh):
    """Pin each level's updated data back to block-sharded storage (step 1 of the
    sharded-memory plan). The all_gather in :func:`divergence_shmap` makes the
    step's intermediates replicated; without this the returned hierarchy would be
    replicated too and between-step storage would not be distributed. This is a
    cheap slice of the already-materialized replicated result -- it does NOT
    reduce the in-step PEAK (that is step 2: removing the gather). A no-op when
    the block axis is not divisible by the device count (leaves it replicated)."""
    from jax.sharding import NamedSharding
    D = mesh.devices.size
    blk = NamedSharding(mesh, P("blocks"))
    out = []
    for nd in datas:
        if nd.shape[0] % D == 0:
            out.append(jax.lax.with_sharding_constraint(nd, blk))
        else:
            out.append(nd)
    return out


def advance_subcycle_shmap(hierarchy, dt_base, cfg, mesh, *,
                           gamma, limiter=1, riemann_solver="hllc",
                           gravity=None, reflux=True, t0=0.0, halo_plans=None):
    """Subcycled (Berger-Oliger) step with each per-level divergence sharded via
    :func:`divergence_shmap`. Reuses :func:`amr.driver.advance_subcycle`'s
    recursion + time-interpolated coarse ghosts, swapping the divergence kernel
    for the distributed one. Matches the single-device subcycled step.

    ``gravity``/``reflux``/``t0`` mirror the driver hooks. The gravity solve and
    kick run on the REPLICATED hierarchy (same as the synchronized shard path --
    every device redundantly computes the identical phi; distribution buys
    memory/flops only in the divergence). Fluxes coming out of the sharded
    divergence are pinned explicitly replicated and the Berger-Colella
    correction executes INSIDE a fully-replicated shard_map -- the same GSPMD
    trap + fix as :func:`advance_shmap` (an outer jit otherwise silently
    re-partitions the segment-sums to shard-local scope)."""
    from jax.sharding import NamedSharding

    from amr.driver import advance_subcycle
    from amr.reflux import apply_reflux_blocks

    repl = NamedSharding(mesh, P())
    # halo_plans = (level_plans, seam_plan); the subcycle uses the per-level
    # divergence (its seam still gathers), so only level_plans is consumed here.
    level_plans = (halo_plans[0] if halo_plans is not None else None)

    def div(gw, neighbors, level, dt_l, want_flux=False):
        out = divergence_shmap(gw, neighbors, mesh, dt_l, level, cfg,
                               gamma=gamma, limiter=limiter,
                               riemann_solver=riemann_solver,
                               return_face_fluxes=want_flux,
                               coords=hierarchy[level].coords)
        if not want_flux:
            return out
        dU, fl = out
        # dU stays SHARDED (data carries sharded through the sub-step recursion ->
        # the between-substep storage is distributed, not just the fill). Fluxes
        # are pinned replicated for the GSPMD-safe reflux (its segment-sums must
        # not be silently re-partitioned; it all-gathers the small flux arrays).
        fl = tuple(jax.lax.with_sharding_constraint(f, repl) for f in fl)
        return dU, fl

    def fill_shard(L, data, parent_gprim):
        """Distributed twin of driver.advance_subcycle's ``ghost_prim``: same-level
        halo (corner-exact diagonal fill) + (L>0) the piecewise-CONSTANT coarse
        underlay, each level's ghosted array born SHARDED. Bit-identical to the
        replicated ghost_prim (no physical BC, interp='const' -- the subcycle's
        C/F fluxes telescope against restriction with PC ghosts)."""
        from amr.halo import fill_ghost_physical
        lvl = hierarchy[L]
        W = cons_to_prim_blocks(data, gamma)
        same_plan, cf_plan = (level_plans[L] if (level_plans is not None
                              and level_plans[L] is not None) else (None, None))
        stencil = _same_level_stencil(lvl.neighbors, cfg)
        if same_plan is not None:
            gw = _fill_same_level_sparse(W, stencil, same_plan[0], same_plan[1],
                                         mesh, cfg)
        else:
            gw = _fill_same_level_diag(W, stencil, mesh, cfg)
        # physical DOMAIN bands BEFORE the coarse underlay (see driver._fill_ghosts)
        gw = fill_ghost_physical(gw, W, cfg, coords=lvl.coords, level=L)
        if L > 0:
            gw = _coarse_underlay_sharded(gw, lvl.coords, lvl.parent_slots,
                                          parent_gprim, mesh, cfg, interp="const",
                                          plan=cf_plan)
        gw = fill_ghost_physical(gw, W, cfg)   # domain-edge BC (no-op if periodic)
        return gw

    def reflux_repl(cd, fc, ff, cc, ps, nb, na, cnb, dt_l, L, _cfg):
        def _rf(cd_, fc_, ff_, cc_, ps_, nb_, na_, cnb_):
            return apply_reflux_blocks(cd_, fc_, ff_, cc_, ps_, nb_, na_,
                                       cnb_, dt_l, L, _cfg)
        args = (cd, fc, ff, cc, ps, nb, na, cnb)
        try:
            return _shard_map(_rf, mesh=mesh, in_specs=P(),
                              out_specs=P(), check_vma=False)(*args)
        except TypeError:                                # older jax: check_rep
            return _shard_map(_rf, mesh=mesh, in_specs=P(),
                              out_specs=P(), check_rep=False)(*args)

    return advance_subcycle(hierarchy, dt_base, cfg, gamma=gamma, limiter=limiter,
                            riemann_solver=riemann_solver, divergence_fn=div,
                            gravity=gravity, reflux=reflux, t0=t0,
                            reflux_fn=reflux_repl, fill_fn=fill_shard)


def advance_level_shmap(data, neighbors, mesh, dt, level, cfg, *,
                        gamma, limiter=1, riemann_solver="hllc"):
    """One conservative same-level (seam-flux) step on a single periodic level,
    sharded over devices via ``shard_map`` + ``all_gather``. ``data`` and
    ``neighbors`` are sharded on the block axis; returns updated ``data``,
    sharded the same way. Bit-matches the single-device pipeline
    (``fill_ghost_same_level`` -> ``conservative_block_divergence`` -> update).
    """
    ndim = cfg.ndim
    data_spec = P("blocks", *([None] * (data.ndim - 1)))
    nbr_spec = P("blocks", None)
    dx = cfg.level_dx(level)
    inv = dict(inv_dx=1.0 / dx[0], inv_dy=1.0 / dx[1],
               inv_dz=1.0 / dx[2] if ndim == 3 else 1.0)

    # NOTE: test-only legacy kernel (test_amr_shard level-step); the production
    # paths are advance_shmap/advance_subcycle_shmap, whose gathers are all
    # ring-capable (_use_ring). This one keeps the simple O(total) all_gather.
    def local(data_loc, nbr_loc):
        data_g = jax.lax.all_gather(data_loc, "blocks", axis=0, tiled=True)   # global blocks
        Wg = cons_to_prim_blocks(data_g, gamma)
        gw = _fill_same_level_2src(cons_to_prim_blocks(data_loc, gamma), Wg, nbr_loc, cfg)

        fluxes = list(block_face_fluxes(gw, dt, level, cfg, gamma=gamma,
                                        limiter=limiter, riemann_solver=riemann_solver))
        states = block_boundary_states(gw, dt, level, cfg, gamma=gamma, limiter=limiter)
        sentinel = data_g.shape[0] - 1

        def seam(a, b, d):
            return jax.vmap(lambda x, y: riemann_face_flux(
                x, y, axis=d, gamma=gamma, riemann_solver=riemann_solver))(a, b)

        for d in range(ndim):
            Ld, Rd = states[d]
            Ld_g = jax.lax.all_gather(Ld, "blocks", axis=0, tiled=True)       # neighbour states
            Rd_g = jax.lax.all_gather(Rd, "blocks", axis=0, tiled=True)
            npl, nmi = nbr_loc[:, 2 * d], nbr_loc[:, 2 * d + 1]
            F = fluxes[d]
            F = _set_boundary_face(F, d, -1, seam(Rd, Ld_g[npl], d), npl != sentinel, ndim)
            F = _set_boundary_face(F, d, 0, seam(Rd_g[nmi], Ld, d), nmi != sentinel, ndim)
            fluxes[d] = F

        dU = jax.vmap(lambda *fs: divergence_from_face_fluxes(fs, spatial_dim=ndim, **inv))(*fluxes)
        return data_loc + dt * dU

    return _shard_map(local, mesh=mesh, in_specs=(data_spec, nbr_spec),
                      out_specs=data_spec)(data, neighbors)


def apply_reflux_sharded(coarse_data, coarse_fluxes, fine_fluxes, fine_coords,
                         fine_parent_slots, fine_neighbors, fine_n_active,
                         coarse_neighbors, dt, coarse_level, cfg, mesh):
    """Distributed Berger-Colella reflux: robust-by-construction replacement for
    the fully-replicated shard_map wrapper around ``apply_reflux_blocks``.

    Why: the replicated wrapper exists because a raw ``segment_sum`` under an
    outer jit lets GSPMD re-partition the scatter over the block axis and
    silently drop shard-remote flux contributions (mass loss). Here the
    cross-device reduction is an EXPLICIT ``psum`` inside the shard_map, so the
    partitioner has no freedom to break it -- and the work distributes: each
    device builds correction patches only for its LOCAL fine blocks (fine
    fluxes stay sharded; no fine-level-size gather), partial-segment-sums them
    into a coarse-level-size buffer, psums (coarse is ~1/2**ndim the fine
    level), and applies the summed correction to its local coarse rows.
    Bit-comparable to ``apply_reflux_blocks`` up to floating-point summation
    order in the psum (identical masked patches, same targets).

    Coarse fluxes and the small topology tables are replicated (the correction
    reads the coarse flux at arbitrary remote targets); fine per-block arrays
    are sharded on the fine block axis; ``coarse_data`` is sharded on the
    coarse block axis and returned sharded."""
    from amr.reflux import (_fine_boundary_face,
                            _restrict_transverse, _coarse_face_patch, _i32)
    ndim = cfg.ndim
    bs = cfg.block_shape
    hs = tuple(s // 2 for s in bs)
    n_vars = coarse_data.shape[1]
    Nc = coarse_data.shape[0]
    Nf = fine_neighbors.shape[0]
    dxc = cfg.level_dx(coarse_level)
    n_dev = int(mesh.shape["blocks"])
    bpd_c = Nc // n_dev
    bpd_f = Nf // n_dev
    c_sent = Nc - 1
    f_sent = Nf - 1                                          # GLOBAL fine sentinel

    cd_spec = P("blocks", *([None] * (coarse_data.ndim - 1)))
    ffl_spec = tuple(P("blocks", *([None] * (f.ndim - 1))) for f in fine_fluxes)
    fine_vec = P("blocks", None)
    fine_scalar = P("blocks")

    def local(cd_loc, cfx, ffx_loc, coord_loc, ps_loc, nbr_loc, na, cnb):
        dev = jax.lax.axis_index("blocks")
        # coarse_fine_face_mask with GLOBAL row indices/sentinel (its replicated
        # form uses arange(n_slots)/n_slots-1, both wrong on a local shard).
        gidx = dev * bpd_f + jnp.arange(bpd_f)
        cf = (nbr_loc == f_sent) & (gidx < na)[:, None]       # (Nf_loc, 2*ndim)
        q = jnp.maximum(coord_loc, 0) % 2
        parent = ps_loc
        parent_safe = jnp.maximum(parent, 0)

        out = cd_loc
        for d in range(ndim):
            for s, col, f_idx in ((+1, 2 * d, -1), (-1, 2 * d + 1, 0)):
                face = jax.vmap(lambda F: _fine_boundary_face(F, d, f_idx, ndim))(
                    ffx_loc[d])
                ff = jax.vmap(lambda f: _restrict_transverse(f, ndim))(face)
                qd = q[:, d]
                if s > 0:
                    target = jnp.where(qd == 0, parent_safe,
                                       cnb[parent_safe, 2 * d])
                    cell_d = jnp.where(qd == 0, hs[d], 0)
                    face_d = cell_d
                else:
                    target = jnp.where(qd == 1, parent_safe,
                                       cnb[parent_safe, 2 * d + 1])
                    cell_d = jnp.where(qd == 1, hs[d] - 1, bs[d] - 1)
                    face_d = cell_d + 1
                valid = cf[:, col] & (parent >= 0) & (target >= 0) & (target != c_sent)
                tsafe = jnp.where(valid, target, c_sent)

                def one(tslot, qq, cell, facei, ffp):
                    Fc_d = cfx[d][tslot]                       # replicated coarse flux
                    o = [qq[t] * hs[t] for t in range(ndim)]
                    Fcp = _coarse_face_patch(Fc_d, d, facei, o, hs, ndim)
                    corr = (dt / dxc[d]) * float(s) * (ffp - Fcp)
                    shape = [n_vars] + [1 if t == d else hs[t] for t in range(ndim)]
                    start = [_i32(0)] + [_i32(cell) if t == d else _i32(o[t])
                                         for t in range(ndim)]
                    zero = jnp.zeros((n_vars,) + tuple(bs), cd_loc.dtype)
                    return jax.lax.dynamic_update_slice(zero, corr.reshape(shape), start)

                patches = jax.vmap(one)(tsafe, q, cell_d, face_d, ff)
                patches = jnp.where(valid.reshape((-1,) + (1,) * (1 + ndim)),
                                    patches, 0.0)
                seg = jax.ops.segment_sum(patches, tsafe, num_segments=Nc)
                seg = seg.at[c_sent].set(0.0)                  # keep sentinel NaN
                seg = jax.lax.psum(seg, "blocks")              # EXPLICIT reduction
                out = out + jax.lax.dynamic_slice_in_dim(seg, dev * bpd_c, bpd_c, 0)
        return out

    ins = (cd_spec, P(), ffl_spec, fine_vec, fine_scalar, fine_vec, P(), P())
    args = (coarse_data, coarse_fluxes, fine_fluxes, fine_coords,
            fine_parent_slots, fine_neighbors, fine_n_active, coarse_neighbors)
    try:
        return _shard_map(local, mesh=mesh, in_specs=ins, out_specs=cd_spec,
                          check_vma=False)(*args)
    except TypeError:
        return _shard_map(local, mesh=mesh, in_specs=ins, out_specs=cd_spec,
                          check_rep=False)(*args)


def _preserve_sharded(new_data, old_slot, old_data, mesh):
    """Sharded twin of :func:`amr.blocks.preserve_unaltered`: copy matched OLD
    block data (``old_slot >= 0``) into the new level, with both data arrays
    sharded on the block axis. gather mode replicates ``old_data`` transiently
    (the regrid's last big transient); ring mode ppermutes the old segments so
    the peak is one segment (old/D). Bit-identical: each new slot's source lives
    in exactly one segment (masked copy); unmatched slots keep ``new_data``."""
    n_dev = int(mesh.shape["blocks"])
    ring = _use_ring(n_dev, old_data.size * old_data.itemsize)
    nd_spec = P("blocks", *([None] * (new_data.ndim - 1)))
    os_spec = P("blocks")
    od_spec = P("blocks", *([None] * (old_data.ndim - 1)))

    if (new_data.shape[0] % n_dev) or (old_data.shape[0] % n_dev):
        # capacity not divisible by the device count (pre-_pad_blocks bucket):
        # keep the existing replicated-gather path.
        from amr.blocks import preserve_unaltered
        return jax.jit(preserve_unaltered)(new_data, old_slot, old_data)

    def local_gather(nd_loc, os_loc, od_loc):
        og = jax.lax.all_gather(od_loc, "blocks", axis=0, tiled=True)
        keep = os_loc >= 0
        got = og[jnp.where(keep, os_loc, 0)]
        bc = keep.reshape((-1,) + (1,) * (nd_loc.ndim - 1))
        return jnp.where(bc, got, nd_loc)

    def local_ring(nd_loc, os_loc, od_loc):
        bpd = od_loc.shape[0]
        dev = jax.lax.axis_index("blocks")
        perm = [(i, (i + 1) % n_dev) for i in range(n_dev)]
        out, cur = nd_loc, od_loc
        for k in range(n_dev):
            seg_lo = ((dev - k) % n_dev) * bpd
            li = os_loc - seg_lo
            valid = (os_loc >= 0) & (li >= 0) & (li < bpd)
            got = cur[jnp.clip(li, 0, bpd - 1)]
            bc = valid.reshape((-1,) + (1,) * (out.ndim - 1))
            out = jnp.where(bc, got, out)
            if k < n_dev - 1:
                cur = jax.lax.ppermute(cur, "blocks", perm)
        return out

    fn = local_ring if ring else local_gather
    ins = (nd_spec, os_spec, od_spec)
    try:
        return _shard_map(fn, mesh=mesh, in_specs=ins, out_specs=nd_spec,
                          check_vma=False)(new_data, old_slot, old_data)
    except TypeError:
        return _shard_map(fn, mesh=mesh, in_specs=ins, out_specs=nd_spec,
                          check_rep=False)(new_data, old_slot, old_data)


# ---------------------------------------------------------------------------
# Shard-aware regrid (#2): the child DATA built sharded, topology replicated
# ---------------------------------------------------------------------------
def build_children_sharded(parent_coords, parent_data, flags, level, cfg, mesh,
                           max_parents, parent_ghosted=None):
    """Sharded twin of :func:`amr.blocks.build_children`. Refine flagged parents
    into child blocks, but produce the (big) child DATA array SHARDED on the
    block axis while all TOPOLOGY (coords / valid / n_active -- small int arrays)
    stays replicated. Bit-identical to build_children on the active blocks.

    Why: the regrid rebuild is the last replicated big-data op -- measured 70 GiB
    at 6 levels (2048^3-eff) -> OOM even on 4xA100. The prolongation is
    per-parent-independent, so it shards cleanly: gather the flagged parents'
    data replicated (transient, ~1.3 GB at the coarse parent level, << the 70 GiB
    child output), then prolong under a shard_map so each device owns 1/D of the
    children. Topology stays replicated because it is tiny (~2 MB/level).
    """
    import itertools
    from amr.geometry import prolong_centered_cons, prolong_linear_cons
    _prolong = (prolong_centered_cons
                if getattr(cfg, "prolong_limiter", "minmod") == "centered"
                else prolong_linear_cons)

    ndim = cfg.ndim
    C = cfg.children_per_block
    bs = cfg.block_shape
    n_vars = parent_data.shape[1]
    D = mesh.devices.size

    # --- topology (replicated) ---
    n_par = jnp.minimum(jnp.sum(flags), max_parents - 1)
    pidx = jnp.nonzero(flags, size=max_parents, fill_value=-1)[0]
    valid = (pidx >= 0) & (jnp.arange(max_parents) < n_par)
    safe = jnp.where(valid, pidx, 0)
    pc = parent_coords[safe]                            # (P, ndim)
    # ghost-aware source (2nd-order outer faces) when provided, else the bare block
    src = parent_data if parent_ghosted is None else parent_ghosted
    src_bs = tuple(src.shape[2:])
    gsrc = (src_bs[0] - bs[0]) // 2
    pd = src[safe]                                      # (P, n_vars, *src_bs) gathered
    crop = (slice(None), slice(None)) + tuple(
        slice(2 * gsrc, 2 * gsrc + 2 * bs[d]) for d in range(ndim))

    coords_per_off = []
    for o in itertools.product((0, 1), repeat=ndim):
        coords_per_off.append(2 * pc + jnp.asarray(o, dtype=jnp.int32))
    child_coords = jnp.stack(coords_per_off, axis=1).reshape(max_parents * C, ndim)

    # --- data (sharded): prolong each device's shard of parents ---
    def _octants(block_data):
        big = _prolong(block_data, ndim)[crop]          # (B, n_vars, *(2*bs))
        outs = []
        for o in itertools.product((0, 1), repeat=ndim):
            sl = [slice(None), slice(None)]
            for d in range(ndim):
                sl.append(slice(o[d] * bs[d], (o[d] + 1) * bs[d]))
            outs.append(big[tuple(sl)])
        return jnp.stack(outs, axis=1).reshape(block_data.shape[0] * C, n_vars, *bs)

    blk = P("blocks", *([None] * (pd.ndim - 1)))
    if max_parents % D == 0:
        try:
            child_data = _shard_map(_octants, mesh=mesh, in_specs=blk,
                                    out_specs=blk, check_vma=False)(pd)
        except TypeError:                               # older jax: check_rep
            child_data = _shard_map(_octants, mesh=mesh, in_specs=blk,
                                    out_specs=blk, check_rep=False)(pd)
    else:
        child_data = _octants(pd)                       # fallback: replicated

    # --- invalidate + sentinel (mask replicated, data sharded -> sharded) ---
    child_valid = jnp.repeat(valid, C)
    child_coords = jnp.where(child_valid[:, None], child_coords, -1)
    child_data = jnp.where(child_valid[:, None, *([None] * ndim)], child_data,
                           jnp.nan)
    child_data = child_data.at[-1].set(jnp.nan)
    child_coords = child_coords.at[-1].set(-1)
    n_active = n_par * C
    return child_coords, child_data, n_active


def regrid_sharded(levels, cfg, mesh, *, thresholds, periodic=True, bucket=None,
                   pad_to=1, criterion="density", max_blocks=0):
    """Shard-aware twin of :func:`amr.driver.regrid`: child DATA built sharded
    (:func:`build_children_sharded`), all TOPOLOGY on the existing replicated
    kernels. Removes the regrid's 70 GiB replicated child-data peak (6 levels)
    so 2048^3-eff fits across devices. The one remaining transient is
    ``_preserve_j``'s gather of the OLD level's data (carries evolved fine state
    across the regrid); it briefly replicates the old level (~its own size, not
    the new 8x-larger one). Bit-identical to driver.regrid on active blocks."""
    from amr.driver import (Level, _flag_blocks, _ghosted_parent, _match_blocks_j,
                            _neighbors_j, _pad_blocks, _parent_slots_j, _preserve_j,
                            _sized_parents)

    import os as _os
    _dbg = _os.environ.get("LUNA_AMR_DBG_COLLECTIVE") or _os.environ.get("LUNA_AMR_LOOP_DBG")
    _mp = jax.process_count() > 1
    _pi = jax.process_index()
    def _mark(L, msg, x=None):
        # Multi-process lockstep: regrid chains several independently-jitted
        # collective kernels (build_children_sharded -> match_blocks -> neighbors
        # -> parent_slots). Their all_gathers are dispatched lazily; without a
        # barrier between kernels the two processes' collective streams drift and
        # deadlock. block_until_ready forces each kernel's collectives to complete
        # before the next is dispatched, keeping every rank in lockstep. Fires in
        # multi-process (correctness) or under a debug flag (tracing).
        if x is not None and (_mp or _dbg):
            jax.block_until_ready(x)
        if _dbg:
            print(f"[dbg-regrid] proc {_pi} L{L}: {msg}", flush=True)

    new = [levels[0]]
    rho0 = jnp.nanmean(levels[0].data[:, 0])
    for L in range(cfg.max_levels):
        parent = new[L]
        old_child = levels[L + 1]
        flags = _flag_blocks(parent, cfg, thresholds[L], L, criterion, rho_ref=rho0)
        _mark(L, "flags OK", flags)
        prev_parents = old_child.data.shape[0] // cfg.children_per_block
        max_parents = _sized_parents(flags, cfg.children_per_block, bucket,
                                     floor=prev_parents,
                                     cap=max_blocks // cfg.children_per_block
                                     if max_blocks else 0)
        _mark(L, f"max_parents={max_parents} (prev={prev_parents})")
        cc, cd, na = build_children_sharded(parent.coords, parent.data, flags, L,
                                            cfg, mesh, max_parents,
                                            parent_ghosted=_ghosted_parent(parent, cfg))
        _mark(L, "build_children_sharded OK", cd)
        old_slot = _match_blocks_j(cc, na, old_child.coords, old_child.n_active, cfg.ndim)
        _mark(L, "match_blocks OK", old_slot)
        cd = _preserve_sharded(cd, old_slot, old_child.data, mesh)
        _mark(L, "preserve OK", cd)
        cd, cc = _pad_blocks(cd, cc, pad_to)
        nbrs = _neighbors_j(cc, na, L + 1, cfg, periodic=False)
        _mark(L, "neighbors OK", nbrs)
        pslots = _parent_slots_j(cc, parent.coords, parent.n_active, cfg)
        _mark(L, "parent_slots OK", pslots)
        new.append(Level(data=cd, coords=cc, neighbors=nbrs,
                         parent_slots=pslots, n_active=na))
    return tuple(new)
