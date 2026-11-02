"""Explicit ``jax.shard_map`` distribution of the AMR block step; blocks are
the shard unit (the ``n_slots`` axis is partitioned across devices).

Neighbours are irregular, so halo/seam exchange is ``sparse`` (all_to_all
routing plans rebuilt at regrid time and passed through the jitted step),
``gather`` or ``ring`` -- via ``LUNA_AMR_HALO_MODE`` / ``LUNA_AMR_HALO_RING``;
reflux modes via ``cfg.amr_reflux``. Entry points: :func:`advance_shmap`,
:func:`advance_subcycle_shmap`, :func:`regrid_sharded`."""

from __future__ import annotations

import itertools
import os

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from luna.amr.hydro import (
    block_boundary_states,
    block_face_fluxes,
    cons_to_prim_blocks,
    _inner_ghost_index,
    _make_seam,
    _set_boundary_face,
    report_amr_faces,
)
from luna.hydro.muscl_hancock import (
    divergence_from_face_fluxes,
    muscl_hancock_boundary_states,
    muscl_hancock_face_fluxes,
)

_shard_map = getattr(jax, "shard_map", None)
if _shard_map is None:  # older jax
    from luna.utils.shmap import shard_map as _shard_map


# --------------------------------------------------------------------------- #
# small shared helpers: shard_map compat, ring permutation, block specs,
# broadcast/band-slice/domain-face idioms, ring-vs-gather policy
# --------------------------------------------------------------------------- #


def _smap(fn, mesh, in_specs, out_specs):
    """``shard_map`` with the replication check off, across jax versions
    (``check_vma`` on new jax, ``check_rep`` on old)."""
    try:
        return _shard_map(fn, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
                          check_vma=False)
    except TypeError:                                     # older jax: check_rep
        return _shard_map(fn, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
                          check_rep=False)


def _ring_perm(n_dev):
    """ppermute permutation shifting the "blocks"-axis device ring by one."""
    return [(i, (i + 1) % n_dev) for i in range(n_dev)]


def _blk_spec(x):
    """PartitionSpec sharding the leading (block) axis of ``x`` (array or rank)."""
    nd = x if isinstance(x, int) else x.ndim
    return P("blocks", *([None] * (nd - 1)))


def _bcast(v, arr):
    """Reshape per-block vector ``v`` to broadcast against ``arr``."""
    return v.reshape((v.shape[0],) + (1,) * (arr.ndim - 1))


def _band_slices(off, ndim, bs, g):
    """(src, dst) index tuples for ghost-region ``off`` of (block, var, *space)
    arrays: per axis, -1 -> src last g / dst first g, 0 -> interior,
    +1 -> src first g / dst last g."""
    ssl = [slice(None), slice(None)]
    dsl = [slice(None), slice(None)]
    for d in range(ndim):
        if off[d] == -1:                              # -face: src last g -> dst first g
            ssl.append(slice(bs[d] - g, bs[d])); dsl.append(slice(0, g))
        elif off[d] == 0:                             # interior
            ssl.append(slice(0, bs[d]));         dsl.append(slice(g, g + bs[d]))
        else:                                         # +face: src first g -> dst last g
            ssl.append(slice(0, g));             dsl.append(slice(g + bs[d], g + bs[d] + g))
    return tuple(ssl), tuple(dsl)


def _domain_face_cols(coords, no_nbr, level, cfg):
    """(N, 2*ndim) bool: face sits on the level's domain edge AND has no
    same-level neighbour (column order [+d, -d] per axis) -- the physical
    (outflow) faces that get the consistent-BC seam flux."""
    ndim, bs, r = cfg.ndim, cfg.block_shape, cfg.refinement_ratio
    cols = []
    for d in range(ndim):
        nb_d = (cfg.base_grid[d] // bs[d]) * (r ** level)
        cols.append((coords[:, d] == nb_d - 1) & no_nbr[:, 2 * d])
        cols.append((coords[:, d] == 0) & no_nbr[:, 2 * d + 1])
    return jnp.stack(cols, axis=1)


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


# --------------------------------------------------------------------------- #
# same-level ghost fill: corner-exact diagonal stencil, gather/ring and
# sparse all_to_all exchanges (+ the legacy two-source fill)
# --------------------------------------------------------------------------- #


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
    """Corner-exact same-level ghost fill, sharded on the block axis: ``W``
    (sharded, unghosted primitives) + ``stencil_idx`` (per-block global source
    slot per ghost region) -> the local blocks' ghosted array, bit-identical to
    ``halo.fill_ghost_same_level``. Exchange strategy per :func:`_use_ring`:
    all_gather (one pass, O(total) memory/device) or ring (ppermute the block
    segments; peak one segment = total/D, ~D x the cheap band-copy compute)."""
    g, ndim, bs = cfg.n_ghost, cfg.ndim, cfg.block_shape
    offsets = _same_level_offsets(ndim)
    W_spec = _blk_spec(W)
    idx_spec = P("blocks", None)
    out_spec = _blk_spec(ndim + 2)                        # (Nloc, n_vars, *ghosted)
    n_dev = int(mesh.shape["blocks"])
    ring = _use_ring(n_dev, W.size * W.itemsize)         # all_gather peak = full W

    def _place(out, off, src, valid):
        """Copy source block ``src``'s band for ghost-region ``off`` into ``out``,
        masked by ``valid`` (None = unconditional)."""
        ssl, dsl = _band_slices(off, ndim, bs, g)
        band = src[ssl]
        if valid is None:
            return out.at[dsl].set(band)
        cur = out[dsl]
        return out.at[dsl].set(jnp.where(_bcast(valid, cur), band, cur))

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
        perm = _ring_perm(n_dev)
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
    return _smap(fn, mesh, (W_spec, idx_spec), out_spec)(W, stencil_idx)


def _fill_same_level_sparse(W, stencil_idx, send_idx, recv_pos, mesh, cfg):
    """Same-level ghost fill with the SPARSE all_to_all exchange: pack only the
    blocks peers need (``send_idx[dev]``), one ``all_to_all``, then each ghost
    band selects local block (direct read) vs received remote (``recv_pos``).
    O(surface) comm/memory; bit-identical to the gather/ring fills."""
    g, ndim, bs = cfg.n_ghost, cfg.ndim, cfg.block_shape
    offsets = _same_level_offsets(ndim)
    n_dev = int(mesh.shape["blocks"])
    K = send_idx.shape[-1]
    W_spec = _blk_spec(W)
    idx_spec = P("blocks", None)
    out_spec = _blk_spec(ndim + 2)                        # (Nloc, n_vars, *ghosted)

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
            src = jnp.where(_bcast(is_loc, src_l), src_l, src_r)
            ssl, dsl = _band_slices(off, ndim, bs, g)
            out = out.at[dsl].set(src[ssl])
        return out

    ins = (W_spec, idx_spec, idx_spec, P())          # send table replicated
    return _smap(local, mesh, ins, out_spec)(W, stencil_idx, recv_pos, send_idx)


# --------------------------------------------------------------------------- #
# coarse-fine underlay: prolonged-parent ghost fill + the per-level
# sharded ghost-fill driver
# --------------------------------------------------------------------------- #


def _coarse_underlay_sharded(fine_ghosted, fine_coords, parent_slots,
                             coarse_ghosted, mesh, cfg, interp="linear",
                             plan=None):
    """Coarse-fine ghost fill (``halo.fill_ghost_from_coarse_sparse``) with the
    fine blocks sharded; output sharded on the fine block axis. Parent exchange:
    all_gather / ring per :func:`_use_ring`, or -- when ``plan`` (``(send_idx,
    recv_pos)`` from :func:`build_exchange_plan` on ``parent_slots[:, None]``) is
    given -- the SPARSE all_to_all of only the parents peers need (O(surface))."""
    from luna.amr.halo import fill_ghost_from_coarse_sparse
    fg_spec = _blk_spec(fine_ghosted)
    coord_spec = P("blocks", None)
    ps_spec = P("blocks")
    cg_spec = _blk_spec(coarse_ghosted)
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
            parent_per_fine = jnp.where(_bcast(is_loc, parent_local),
                                        parent_local, parent_remote)
            # reuse the exact prolong: parent_slots = arange -> each fine block's
            # source is parent_per_fine[i], bit-identical to the gather path.
            idx = jnp.arange(fg_loc.shape[0], dtype=jnp.int32)
            return fill_ghost_from_coarse_sparse(fg_loc, coord_loc, idx,
                                                 parent_per_fine, cfg, interp=interp)

        ins = (fg_spec, coord_spec, ps_spec, cg_spec, rp_spec, P())
        return _smap(local_sparse, mesh, ins, fg_spec)(
            fine_ghosted, fine_coords, parent_slots, coarse_ghosted,
            recv_pos, send_idx)

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
        perm = _ring_perm(n_dev)
        for k in range(n_dev):
            seg_lo = ((dev - k) % n_dev) * bpd_c
            ps_local = ps_loc - seg_lo
            valid = (ps_local >= 0) & (ps_local < bpd_c)
            filled = fill_ghost_from_coarse_sparse(
                out, coord_loc, jnp.clip(ps_local, 0, bpd_c - 1), cur, cfg, interp=interp)
            out = jnp.where(_bcast(valid, out), filled, out)
            if k < n_dev - 1:
                cur = jax.lax.ppermute(cur, "blocks", perm)
        return out

    fn = local_ring if ring else local_gather
    ins = (fg_spec, coord_spec, ps_spec, cg_spec)
    return _smap(fn, mesh, ins, fg_spec)(fine_ghosted, fine_coords,
                                         parent_slots, coarse_ghosted)


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
    from luna.amr.halo import fill_ghost_physical
    D = mesh.devices.size
    ghosted = []
    for L, lvl in enumerate(levels):
        W = cons_to_prim_blocks(lvl.data, gamma)
        if W.shape[0] % D != 0:                          # not shardable -> replicated fallback
            from luna.amr.halo import fill_ghost_same_level, fill_ghost_from_coarse_sparse
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
# seam/divergence kernels: per-block MUSCL fluxes, seam Riemann coupling,
# batched all-levels and per-level sharded divergence
# --------------------------------------------------------------------------- #


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
    """Replace each block-block face flux with the SHARED seam Riemann flux from
    this block's reconstruction and its (possibly remote) neighbour's. Neighbour
    exchange (all bit-identical): sparse all_to_all when ``seam_send``/``seam_recv``
    given (O(surface); payload stacks per-axis states in neighbour-column order,
    needs cubic blocks), else all_gather (``ring=False``, O(total) memory) or the
    ppermute ring (``ring=True``, peak one segment). ``dom_loc`` (Nloc, 2*ndim)
    marks PHYSICAL (outflow) faces, overwritten AFTER coupling with the
    consistent zero-gradient flux seam(own, own) = f(pred) (see
    conservative_block_divergence: the raw ghost flux pumps mass through walls)."""

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
            return jnp.where(_bcast(is_loc, loc), loc, rem)

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
    perm = _ring_perm(n_dev)
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
    ``W_all`` (N, n_vars, *ghosted) stacked ghosted blocks; ``inv_all`` (N, ndim)
    per-block 1/dx; ``nbr_all`` (N, 2*ndim) GLOBAL neighbour slots; ``mask_all``
    marks real neighbours. Returns (dU_all, fluxes) sharded on the block axis;
    bit-matches the per-level ``divergence_shmap``. ``seam_plan`` (``(send_idx,
    recv_pos)`` on ``nbr_all``) switches the seam coupling to the sparse
    all_to_all (O(surface) comm) instead of the all_gather/ring."""
    ndim = cfg.ndim
    blk = _blk_spec(W_all)
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
        seam = _make_seam(gamma, riemann_solver)
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
    return _smap(local, mesh, ins, out_specs)(*args)


def _batch_levels(ghosted, hierarchy, cfg):
    """Stack per-level ghosted blocks + topology into the concatenated-block-axis
    arrays for :func:`divergence_batched_shmap`. Returns (W_all, inv_all, nbr_all,
    mask_all, offsets) where offsets[L] is level L's start in the stacked axis."""
    ndim = cfg.ndim
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
        # physical (outflow) domain faces get the consistent-BC flux (_seam_couple)
        domL = (_domain_face_cols(lvl.coords, ~maskL, L, cfg)
                if cfg.boundary == 1 else jnp.zeros_like(maskL))
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
    """Conservative seam-flux divergence for ONE level: per-block flux compute
    sharded over devices, same-level seam coupling via all_gather/ring of the
    small boundary states. Returns dU (and optionally the seam-corrected face
    fluxes) SHARDED on the block axis. Reuses the existing ghost fill/restrict;
    only this hot kernel is distributed."""
    ndim = cfg.ndim
    W_spec = _blk_spec(W_ghosted)
    nbr_spec = P("blocks", None)
    dx = cfg.level_dx(level)
    inv = dict(inv_dx=1.0 / dx[0], inv_dy=1.0 / dx[1],
               inv_dz=1.0 / dx[2] if ndim == 3 else 1.0)

    n_dev = int(mesh.shape["blocks"])
    ring = _use_ring(n_dev, _states_bytes(W_ghosted, cfg))
    sentinel = W_ghosted.shape[0] - 1                     # global NaN sentinel slot
    if coords is not None and cfg.boundary == 1:
        dom = _domain_face_cols(coords, neighbors == sentinel, level, cfg)
    else:
        dom = jnp.zeros((W_ghosted.shape[0], 2 * ndim), bool)

    crd = (coords if coords is not None
           else jnp.full((W_ghosted.shape[0], ndim), -1, jnp.int32)).astype(jnp.int32)

    def local(W_loc, nbr_loc, dom_loc, crd_loc):
        fluxes, own = block_face_fluxes(W_loc, dt, level, cfg, gamma=gamma, limiter=limiter,
                                        riemann_solver=riemann_solver, report=True)
        fluxes = list(fluxes)
        report_amr_faces(own, level, cfg, crd_loc, slot0=jax.lax.axis_index("blocks") * W_loc.shape[0])
        states = block_boundary_states(W_loc, dt, level, cfg, gamma=gamma, limiter=limiter)
        seam = _make_seam(gamma, riemann_solver)
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

    out_specs = ((W_spec, tuple(W_spec for _ in range(ndim)))
                 if return_face_fluxes else W_spec)
    return _smap(local, mesh, (W_spec, nbr_spec, nbr_spec, nbr_spec),
                 out_specs)(W_ghosted, neighbors, dom, crd)


# --------------------------------------------------------------------------- #
# reflux: interface-local (plan-driven) and psum-sharded Berger-Colella
# corrections (the replicated wrapper lives inline in advance_shmap;
# host-side plan builders in amr.shard_plans)
# --------------------------------------------------------------------------- #


def apply_reflux_iface(coarse_data, coarse_fluxes, fine_fluxes,
                       fine_parent_slots, fine_n_active, coarse_neighbors,
                       dt, coarse_level, cfg, mesh, plan):
    """INTERFACE-LOCAL Berger-Colella reflux, O(C/F interface) communication
    (the prior modes ship volume-scale flux arrays every step). Correction
    split U_c += (dt/dx_c) * s * (<F_f> - F_c) into: pass A (coarse-local, zero
    comm) applies -F_c to boundary layers whose opposite-side neighbour is
    covered by children; pass B (+<F_f>) routes each fine C/F face's restricted
    flux patch via the sparse all_to_all ``plan`` = (send_idx, meta) from
    :func:`build_reflux_plan`, scatter-adding through precomputed meta rows."""
    from luna.amr.reflux import _fine_boundary_face, _restrict_transverse
    ndim = cfg.ndim
    bs = cfg.block_shape
    hs = tuple(s2 // 2 for s2 in bs)
    n_vars = coarse_data.shape[1]
    Nc = coarse_data.shape[0]
    Nf = fine_parent_slots.shape[0]
    dxc = cfg.level_dx(coarse_level)
    n_dev = int(mesh.shape["blocks"])
    bpd_c = Nc // n_dev
    c_sent = Nc - 1

    send_idx, meta = plan

    # replicated: covered[c] = coarse slot has active children
    covered = jnp.zeros((Nc,), bool).at[
        jnp.where(fine_parent_slots >= 0, fine_parent_slots, c_sent)
    ].max(jnp.arange(Nf) < fine_n_active)
    covered = covered.at[c_sent].set(False)

    cd_spec = _blk_spec(coarse_data)
    cfl_spec = tuple(_blk_spec(f) for f in coarse_fluxes)
    ffl_spec = tuple(_blk_spec(f) for f in fine_fluxes)

    def local(cd_loc, cfx_loc, ffx_loc, cov, cnb_repl, meta_all, send_tab):
        dev = jax.lax.axis_index("blocks")
        out = cd_loc
        row0 = dev * bpd_c
        gidx = row0 + jnp.arange(bpd_c)

        # ---- pass A: -F_c on boundary layers facing a covered neighbour ----
        # Reference mapping (octet-complete): s=+1 corrections land on the
        # coarse block whose -d neighbour is covered, at layer 0, using its
        # OWN -d face flux (face index 0); s=-1 mirrored (layer bs-1, face bs).
        for d in range(ndim):
            for s2, layer, face_i, opp_col in ((+1.0, 0, 0, 2 * d + 1),
                                               (-1.0, bs[d] - 1, bs[d], 2 * d)):
                nb = cnb_repl[gidx, opp_col]
                m = (cov[jnp.clip(nb, 0, Nc - 1)] & (nb >= 0) & (nb != c_sent)
                     & ~cov[gidx])          # covered targets: restriction owns them
                Fc = jax.lax.index_in_dim(cfx_loc[d], face_i, axis=2 + d,
                                          keepdims=False)
                corr = -(dt / dxc[d]) * s2 * Fc
                sl = [slice(None), slice(None)] + [slice(None)] * ndim
                sl[2 + d] = layer
                cur = out[tuple(sl)]
                out = out.at[tuple(sl)].set(
                    cur + jnp.where(_bcast(m, cur), corr, 0.0))

        # ---- pass B: +<F_f> patches via sparse plan --------------------------
        patches = []
        for d in range(ndim):
            for f_idx in (-1, 0):                       # +d face, then -d face
                face = jax.vmap(lambda F: _fine_boundary_face(F, d, f_idx, ndim))(
                    ffx_loc[d])
                patches.append(jax.vmap(
                    lambda f: _restrict_transverse(f, ndim))(face))
        payload = jnp.stack(patches, axis=1)            # (bpd_f, 2ndim, nv, *hs_t)
        flat = payload.reshape((-1,) + payload.shape[2:])
        buf = flat[send_tab[dev].reshape(-1)]
        got = jax.lax.all_to_all(buf, "blocks", 0, 0, tiled=True)

        mrows = meta_all[dev]                            # (n_dev*K, 4+ndim)
        tgt, md_, sgn, cell = mrows[:, 0], mrows[:, 1], mrows[:, 2], mrows[:, 3]
        offs = mrows[:, 4:4 + ndim]
        ok = tgt >= 0
        for d in range(ndim):
            sel = ok & (md_ == d)
            corr = (dt / dxc[d]) * _bcast(sgn.astype(coarse_data.dtype), got) * got

            def place(c, cl, o, d=d):
                shape = [n_vars] + [1 if t == d else hs[t] for t in range(ndim)]
                start = [jnp.asarray(0, jnp.int32)] + [
                    (cl if t == d else o[t]).astype(jnp.int32)
                    for t in range(ndim)]
                z = jnp.zeros((n_vars,) + tuple(bs), coarse_data.dtype)
                return jax.lax.dynamic_update_slice(z, c.reshape(shape), start)

            blocks = jax.vmap(place)(corr, cell, offs)
            blocks = jnp.where(_bcast(sel, blocks), blocks, 0.0)
            seg = jax.ops.segment_sum(blocks, jnp.where(sel, tgt, bpd_c),
                                      num_segments=bpd_c + 1)[:bpd_c]
            out = out + seg
        return out

    ins = (cd_spec, cfl_spec, ffl_spec, P(), P(), P(), P())
    args = (coarse_data, coarse_fluxes, fine_fluxes, covered, coarse_neighbors,
            meta, send_idx)
    return _smap(local, mesh, ins, cd_spec)(*args)


def apply_reflux_sharded(coarse_data, coarse_fluxes, fine_fluxes, fine_coords,
                         fine_parent_slots, fine_neighbors, fine_n_active,
                         coarse_neighbors, dt, coarse_level, cfg, mesh):
    """Distributed Berger-Colella reflux: each device builds correction patches
    for its LOCAL fine blocks (fine fluxes stay sharded), partial-segment-sums
    into a coarse-level-size buffer, then reduces with an EXPLICIT ``psum`` --
    GSPMD cannot re-partition it and drop shard-remote contributions (the bug
    the replicated wrapper guards against). Coarse fluxes + topology replicated;
    ``coarse_data`` sharded in and out. Bit-comparable to ``apply_reflux_blocks``
    up to summation order in the psum."""
    from luna.amr.reflux import (_fine_boundary_face,
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

    cd_spec = _blk_spec(coarse_data)
    ffl_spec = tuple(_blk_spec(f) for f in fine_fluxes)
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
                patches = jnp.where(_bcast(valid, patches), patches, 0.0)
                seg = jax.ops.segment_sum(patches, tsafe, num_segments=Nc)
                seg = seg.at[c_sent].set(0.0)                  # keep sentinel NaN
                seg = jax.lax.psum(seg, "blocks")              # EXPLICIT reduction
                out = out + jax.lax.dynamic_slice_in_dim(seg, dev * bpd_c, bpd_c, 0)
        return out

    ins = (cd_spec, P(), ffl_spec, fine_vec, fine_scalar, fine_vec, P(), P())
    args = (coarse_data, coarse_fluxes, fine_fluxes, fine_coords,
            fine_parent_slots, fine_neighbors, fine_n_active, coarse_neighbors)
    return _smap(local, mesh, ins, cd_spec)(*args)


# --------------------------------------------------------------------------- #
# regrid: sharded block preserve/rebuild
# --------------------------------------------------------------------------- #


def _preserve_sharded(new_data, old_slot, old_data, mesh):
    """Sharded twin of :func:`amr.blocks.preserve_unaltered`: copy matched OLD
    block data (``old_slot >= 0``) into the new level, both arrays sharded on
    the block axis; unmatched slots keep ``new_data``. Exchange per
    :func:`_use_ring`: transient replicated gather of ``old_data`` or the
    ppermute ring (peak one segment). Bit-identical either way."""
    n_dev = int(mesh.shape["blocks"])
    ring = _use_ring(n_dev, old_data.size * old_data.itemsize)
    nd_spec = _blk_spec(new_data)
    os_spec = P("blocks")
    od_spec = _blk_spec(old_data)

    if (new_data.shape[0] % n_dev) or (old_data.shape[0] % n_dev):
        # capacity not divisible by the device count (pre-_pad_blocks bucket):
        # keep the existing replicated-gather path.
        from luna.amr.blocks import preserve_unaltered
        return jax.jit(preserve_unaltered)(new_data, old_slot, old_data)

    def local_gather(nd_loc, os_loc, od_loc):
        og = jax.lax.all_gather(od_loc, "blocks", axis=0, tiled=True)
        keep = os_loc >= 0
        got = og[jnp.where(keep, os_loc, 0)]
        return jnp.where(_bcast(keep, nd_loc), got, nd_loc)

    def local_ring(nd_loc, os_loc, od_loc):
        bpd = od_loc.shape[0]
        dev = jax.lax.axis_index("blocks")
        perm = _ring_perm(n_dev)
        out, cur = nd_loc, od_loc
        for k in range(n_dev):
            seg_lo = ((dev - k) % n_dev) * bpd
            li = os_loc - seg_lo
            valid = (os_loc >= 0) & (li >= 0) & (li < bpd)
            got = cur[jnp.clip(li, 0, bpd - 1)]
            out = jnp.where(_bcast(valid, out), got, out)
            if k < n_dev - 1:
                cur = jax.lax.ppermute(cur, "blocks", perm)
        return out

    fn = local_ring if ring else local_gather
    ins = (nd_spec, os_spec, od_spec)
    return _smap(fn, mesh, ins, nd_spec)(new_data, old_slot, old_data)


def build_children_sharded(parent_coords, parent_data, flags, level, cfg, mesh,
                           max_parents, parent_ghosted=None):
    """Sharded twin of :func:`amr.blocks.build_children`: refine flagged parents
    with the (big) child DATA array born SHARDED on the block axis, all TOPOLOGY
    (small int arrays) replicated. The per-parent-independent prolongation runs
    under a shard_map over a transient replicated gather of the (much smaller)
    flagged parents. Bit-identical to build_children on the active blocks."""
    from luna.amr.geometry import prolong_centered_cons, prolong_linear_cons
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

    blk = _blk_spec(pd)
    if max_parents % D == 0:
        child_data = _smap(_octants, mesh, blk, blk)(pd)
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

    from luna.amr.driver import regrid as _regrid_shared
    return _regrid_shared(
        levels, cfg, thresholds=thresholds, periodic=periodic, bucket=bucket,
        pad_to=pad_to, criterion=criterion, max_blocks=max_blocks, mark=_mark,
        children_fn=lambda co, da, fl, lv, cf_, mp, pg: build_children_sharded(
            co, da, fl, lv, cf_, mesh, mp, parent_ghosted=pg),
        preserve_fn=lambda cd, old_slot, old_data: _preserve_sharded(
            cd, old_slot, old_data, mesh))


# --------------------------------------------------------------------------- #
# advance entry points: synchronized (advance_shmap), subcycled
# (advance_subcycle_shmap) and the test-only single-level step
# --------------------------------------------------------------------------- #


def advance_shmap(hierarchy, dt, cfg, mesh, *, gamma, limiter=1,
                  riemann_solver="hllc", use_batched=True, halo_plans=None):
    """Synchronized multi-level step, sharded: ghost fill via
    :func:`_fill_ghosts_sharded`, divergence as ONE batched kernel over all
    levels (``use_batched=True``; one compile instead of one per level) or the
    per-level :func:`divergence_shmap` loop (bit-identical fallback).
    Restrict/reflux stay per-level. Matches :func:`amr.driver.advance`."""
    from luna.amr.driver import _restrict_into_parent
    from luna.amr.reflux import apply_reflux_blocks

    # halo_plans = (level_plans, seam_plan[, reflux_plans]): per-level fill
    # plans + the global batched-seam plan + optional per-level-pair
    # interface-reflux plans. None -> gather/ring/replicated paths.
    reflux_plans = None
    if halo_plans is None:
        level_plans, seam_plan = None, None
    elif len(halo_plans) == 3:
        level_plans, seam_plan, reflux_plans = halo_plans
    else:
        level_plans, seam_plan = halo_plans

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
    #  * sharded (amr_reflux=sharded): apply_reflux_sharded -- fine fluxes
    #    stay sharded, partial corrections psum'd EXPLICITLY (root-cause fix for
    #    the GSPMD landmine; the compute distributes too).
    n_dev_ = int(mesh.shape["blocks"])
    _refl_mode = cfg.reflux                     # cfg here is the AMRConfig
    _refl_sharded = _refl_mode == "sharded"
    for L in range(len(hierarchy) - 1, 0, -1):
        if (_refl_mode == "iface" and reflux_plans is not None
                and reflux_plans[L - 1] is not None):
            new_data[L - 1] = apply_reflux_iface(
                new_data[L - 1], fluxes[L - 1], fluxes[L],
                hierarchy[L].parent_slots, hierarchy[L].n_active,
                hierarchy[L - 1].neighbors, dt, L - 1, cfg, mesh,
                reflux_plans[L - 1])
            continue
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

        new_data[L - 1] = _smap(_rf, mesh, P(), P())(*args)
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
    """Subcycled (Berger-Oliger) step: :func:`amr.driver.advance_subcycle`'s
    recursion with the divergence swapped for :func:`divergence_shmap` and the
    ghost fill for the sharded one. ``gravity``/``reflux``/``t0`` mirror the
    driver hooks; gravity runs on the REPLICATED hierarchy. Fluxes are pinned
    replicated and reflux runs inside a fully-replicated shard_map -- same GSPMD
    trap + fix as :func:`advance_shmap`. Matches the single-device step."""
    from jax.sharding import NamedSharding

    from luna.amr.driver import advance_subcycle
    from luna.amr.reflux import apply_reflux_blocks

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
        from luna.amr.halo import fill_ghost_physical
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
        return _smap(_rf, mesh, P(), P())(cd, fc, ff, cc, ps, nb, na, cnb)

    return advance_subcycle(hierarchy, dt_base, cfg, gamma=gamma, limiter=limiter,
                            riemann_solver=riemann_solver, divergence_fn=div,
                            gravity=gravity, reflux=reflux, t0=t0,
                            reflux_fn=reflux_repl, fill_fn=fill_shard)
