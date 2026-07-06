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
    ``halo.fill_ghost_same_level`` on the active blocks."""
    g, ndim, bs = cfg.n_ghost, cfg.ndim, cfg.block_shape
    offsets = _same_level_offsets(ndim)
    W_spec = P("blocks", *([None] * (W.ndim - 1)))
    idx_spec = P("blocks", None)
    out_spec = P("blocks", *([None] * (ndim + 1)))

    def local(W_loc, idx_loc):
        Wg = jax.lax.all_gather(W_loc, "blocks", axis=0, tiled=True)   # global unghosted
        Nloc, n_vars = W_loc.shape[0], W_loc.shape[1]
        out = jnp.full((Nloc, n_vars) + tuple(s + 2 * g for s in bs),
                       jnp.nan, W_loc.dtype)
        for o, off in enumerate(offsets):
            src = Wg[idx_loc[:, o]]                       # (Nloc, n_vars, *bs)
            ssl = [slice(None), slice(None)]
            dsl = [slice(None), slice(None)]
            for d in range(ndim):
                if off[d] == -1:                          # -face: src last g -> dst first g
                    ssl.append(slice(bs[d] - g, bs[d])); dsl.append(slice(0, g))
                elif off[d] == 0:                         # interior
                    ssl.append(slice(0, bs[d]));         dsl.append(slice(g, g + bs[d]))
                else:                                     # +face: src first g -> dst last g
                    ssl.append(slice(0, g));             dsl.append(slice(g + bs[d], g + bs[d] + g))
            out = out.at[tuple(dsl)].set(src[tuple(ssl)])
        return out

    try:
        return _shard_map(local, mesh=mesh, in_specs=(W_spec, idx_spec),
                          out_specs=out_spec, check_vma=False)(W, stencil_idx)
    except TypeError:                                     # older jax: check_rep
        return _shard_map(local, mesh=mesh, in_specs=(W_spec, idx_spec),
                          out_specs=out_spec, check_rep=False)(W, stencil_idx)


def _coarse_underlay_sharded(fine_ghosted, fine_coords, parent_slots,
                             coarse_ghosted, mesh, cfg, interp="linear"):
    """Sparse coarse-fine ghost fill (``halo.fill_ghost_from_coarse_sparse``) with
    the fine blocks sharded. Each device all-gathers the (smaller) parent ghosted
    level, then runs the identical per-block gather+prolong for its local fine
    blocks. Output sharded on the fine block axis."""
    from amr.halo import fill_ghost_from_coarse_sparse
    fg_spec = P("blocks", *([None] * (fine_ghosted.ndim - 1)))
    coord_spec = P("blocks", None)
    ps_spec = P("blocks")
    cg_spec = P("blocks", *([None] * (coarse_ghosted.ndim - 1)))

    def local(fg_loc, coord_loc, ps_loc, cg_loc):
        cg = jax.lax.all_gather(cg_loc, "blocks", axis=0, tiled=True)  # global parent ghosted
        return fill_ghost_from_coarse_sparse(fg_loc, coord_loc, ps_loc, cg,
                                             cfg, interp=interp)

    ins = (fg_spec, coord_spec, ps_spec, cg_spec)
    try:
        return _shard_map(local, mesh=mesh, in_specs=ins, out_specs=fg_spec,
                          check_vma=False)(fine_ghosted, fine_coords, parent_slots, coarse_ghosted)
    except TypeError:
        return _shard_map(local, mesh=mesh, in_specs=ins, out_specs=fg_spec,
                          check_rep=False)(fine_ghosted, fine_coords, parent_slots, coarse_ghosted)


def _fill_ghosts_sharded(levels, cfg, mesh, gamma):
    """Sharded twin of ``driver._fill_ghosts``: per-level ghost-filled PRIMITIVE
    blocks, each level's ghosted array born SHARDED on the block axis (the big
    array), with only the smaller unghosted / parent gathers transient. Same-level
    fill is corner-exact (:func:`_fill_same_level_diag`); finer levels add the
    sparse coarse underlay; a physical BC (per-block, sharding-preserving) fills
    domain-edge NaNs. Bit-identical to the replicated ``_fill_ghosts``."""
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
        stencil = _same_level_stencil(lvl.neighbors, cfg)
        gw = _fill_same_level_diag(W, stencil, mesh, cfg)
        if L > 0:
            gw = _coarse_underlay_sharded(gw, lvl.coords, lvl.parent_slots,
                                          ghosted[L - 1], mesh, cfg, interp="linear")
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


def divergence_batched_shmap(W_all, inv_all, nbr_all, mask_all, mesh, dt, cfg, *,
                             gamma, limiter=1, riemann_solver="hllc"):
    """Single sharded divergence over the concatenated all-levels block axis.
    ``W_all`` (N, n_vars, *ghosted) is every level's ghosted blocks stacked;
    ``inv_all`` (N, ndim) the per-block 1/dx; ``nbr_all`` (N, 2*ndim) GLOBAL
    same-level neighbour slots (invalid -> that level's NaN sentinel slot);
    ``mask_all`` (N, 2*ndim) bool marks real neighbours. Returns (dU_all, fluxes)
    sharded on the block axis. Bit-matches the per-level ``divergence_shmap``."""
    ndim = cfg.ndim
    blk = P("blocks", *([None] * (W_all.ndim - 1)))
    vec = P("blocks", None)
    out_specs = (blk, tuple(blk for _ in range(ndim)))

    def local(W_loc, inv_loc, nbr_loc, mask_loc):
        fluxes = list(_block_fluxes_vec(W_loc, dt, inv_loc, cfg, gamma=gamma,
                                        limiter=limiter, riemann_solver=riemann_solver))
        states = _block_bstates_vec(W_loc, dt, inv_loc, cfg, gamma=gamma, limiter=limiter)

        def seam(a, b, d):
            return jax.vmap(lambda x, y: riemann_face_flux(
                x, y, axis=d, gamma=gamma, riemann_solver=riemann_solver))(a, b)

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

        def dv(iv, *fs):
            return divergence_from_face_fluxes(fs, spatial_dim=ndim, **_inv_kw(iv, ndim))

        dU = jax.vmap(dv)(inv_loc, *fluxes)
        return dU, tuple(fluxes)

    ins = (blk, vec, vec, vec)
    try:
        return _shard_map(local, mesh=mesh, in_specs=ins, out_specs=out_specs,
                          check_vma=False)(W_all, inv_all, nbr_all, mask_all)
    except TypeError:
        return _shard_map(local, mesh=mesh, in_specs=ins, out_specs=out_specs,
                          check_rep=False)(W_all, inv_all, nbr_all, mask_all)


def _batch_levels(ghosted, hierarchy, cfg):
    """Stack per-level ghosted blocks + topology into the concatenated-block-axis
    arrays for :func:`divergence_batched_shmap`. Returns (W_all, inv_all, nbr_all,
    mask_all, offsets) where offsets[L] is level L's start in the stacked axis."""
    ndim = cfg.ndim
    W_list, inv_list, nbr_list, mask_list, offsets = [], [], [], [], []
    acc = 0
    for L, lvl in enumerate(hierarchy):
        gL = ghosted[L]
        nL = gL.shape[0]
        dxL = cfg.level_dx(L)
        invL = jnp.broadcast_to(
            jnp.asarray([1.0 / dxL[d] for d in range(ndim)], gL.dtype), (nL, ndim))
        nbrL = lvl.neighbors                              # (nL, 2ndim) local, sentinel = nL-1
        maskL = nbrL != (nL - 1)
        offsets.append(acc)
        W_list.append(gL)
        inv_list.append(invL)
        nbr_list.append((nbrL + acc).astype(jnp.int32))   # invalid -> level's NaN sentinel slot
        mask_list.append(maskL)
        acc += nL
    return (jnp.concatenate(W_list, axis=0),
            jnp.concatenate(inv_list, axis=0),
            jnp.concatenate(nbr_list, axis=0),
            jnp.concatenate(mask_list, axis=0),
            offsets)


def divergence_shmap(W_ghosted, neighbors, mesh, dt, level, cfg, *,
                     gamma, limiter=1, riemann_solver="hllc",
                     return_face_fluxes=False):
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

    def local(W_loc, nbr_loc):
        fluxes = list(block_face_fluxes(W_loc, dt, level, cfg, gamma=gamma,
                                        limiter=limiter, riemann_solver=riemann_solver))
        states = block_boundary_states(W_loc, dt, level, cfg, gamma=gamma, limiter=limiter)

        def seam(a, b, d):
            return jax.vmap(lambda x, y: riemann_face_flux(
                x, y, axis=d, gamma=gamma, riemann_solver=riemann_solver))(a, b)

        for d in range(ndim):
            Ld, Rd = states[d]
            Ld_g = jax.lax.all_gather(Ld, "blocks", axis=0, tiled=True)
            Rd_g = jax.lax.all_gather(Rd, "blocks", axis=0, tiled=True)
            sentinel = Ld_g.shape[0] - 1
            npl, nmi = nbr_loc[:, 2 * d], nbr_loc[:, 2 * d + 1]
            F = fluxes[d]
            F = _set_boundary_face(F, d, -1, seam(Rd, Ld_g[npl], d), npl != sentinel, ndim)
            F = _set_boundary_face(F, d, 0, seam(Rd_g[nmi], Ld, d), nmi != sentinel, ndim)
            fluxes[d] = F

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
        return _shard_map(local, mesh=mesh, in_specs=(W_spec, nbr_spec),
                          out_specs=out_specs, check_vma=False)(W_ghosted, neighbors)
    except TypeError:                                    # older jax: check_rep
        return _shard_map(local, mesh=mesh, in_specs=(W_spec, nbr_spec),
                          out_specs=out_specs, check_rep=False)(W_ghosted, neighbors)


def advance_shmap(hierarchy, dt, cfg, mesh, *, gamma, limiter=1,
                  riemann_solver="hllc", use_batched=True):
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

    # Sharded ghost fill: each level's (big) ghosted array is born sharded on the
    # block axis instead of replicated -- this was ~half of the 6-level advance's
    # ~90 GB peak. Bit-identical to driver._fill_ghosts (validated 4xA100).
    ghosted = _fill_ghosts_sharded(hierarchy, cfg, mesh, gamma)
    new_data = []
    fluxes = []
    if use_batched:
        # ONE divergence launch across all levels (per-block inv_dx, global
        # neighbour offsets) -> one XLA compile instead of one per level.
        W_all, inv_all, nbr_all, mask_all, offsets = _batch_levels(ghosted, hierarchy, cfg)
        dU_all, fl_all = divergence_batched_shmap(
            W_all, inv_all, nbr_all, mask_all, mesh, dt, cfg,
            gamma=gamma, limiter=limiter, riemann_solver=riemann_solver)
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
                                      return_face_fluxes=True)
            # dU / fl come back SHARDED on the block axis (no output all_gather);
            # new_data stays sharded so the divergence peak is total/n_dev.
            new_data.append(lvl.data + dt * dU)
            fluxes.append(fl)
    # Berger-Colella reflux on the replicated fluxes (cheap, O(face area)),
    # executed INSIDE a shard_map with fully-replicated specs: every device
    # redundantly computes the identical correction, so the SPMD partitioner
    # has zero freedom to shard the segment-sums/gathers over the block axis
    # (which it silently did under an outer jit, dropping shard-remote
    # contributions -- even with an explicit sharding constraint upstream).
    for L in range(len(hierarchy) - 1, 0, -1):
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
                           gravity=None, reflux=False, t0=0.0):
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

    def div(gw, neighbors, level, dt_l, want_flux=False):
        out = divergence_shmap(gw, neighbors, mesh, dt_l, level, cfg,
                               gamma=gamma, limiter=limiter,
                               riemann_solver=riemann_solver,
                               return_face_fluxes=want_flux)
        if not want_flux:
            return out
        dU, fl = out
        dU = jax.lax.with_sharding_constraint(dU, repl)
        fl = tuple(jax.lax.with_sharding_constraint(f, repl) for f in fl)
        return dU, fl

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
                            reflux_fn=reflux_repl)


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

    new = [levels[0]]
    rho0 = jnp.nanmean(levels[0].data[:, 0])
    for L in range(cfg.max_levels):
        parent = new[L]
        old_child = levels[L + 1]
        flags = _flag_blocks(parent, cfg, thresholds[L], L, criterion, rho_ref=rho0)
        prev_parents = old_child.data.shape[0] // cfg.children_per_block
        max_parents = _sized_parents(flags, cfg.children_per_block, bucket,
                                     floor=prev_parents,
                                     cap=max_blocks // cfg.children_per_block
                                     if max_blocks else 0)
        cc, cd, na = build_children_sharded(parent.coords, parent.data, flags, L,
                                            cfg, mesh, max_parents,
                                            parent_ghosted=_ghosted_parent(parent, cfg))
        old_slot = _match_blocks_j(cc, na, old_child.coords, old_child.n_active, cfg.ndim)
        cd = _preserve_j(cd, old_slot, old_child.data)
        cd, cc = _pad_blocks(cd, cc, pad_to)
        nbrs = _neighbors_j(cc, na, L + 1, cfg, periodic=False)
        pslots = _parent_slots_j(cc, parent.coords, parent.n_active, cfg)
        new.append(Level(data=cd, coords=cc, neighbors=nbrs,
                         parent_slots=pslots, n_active=na))
    return tuple(new)
