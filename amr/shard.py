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

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from amr.halo import _spatial_slices
from amr.hydro import (
    block_boundary_states,
    block_face_fluxes,
    cons_to_prim_blocks,
    _set_boundary_face,
)
from hydro.muscl_hancock import divergence_from_face_fluxes, riemann_face_flux

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


def advance_shmap(hierarchy, dt, cfg, mesh, *, gamma, limiter=1, riemann_solver="hllc"):
    """Synchronized multi-level step with the per-level divergence sharded across
    devices (:func:`divergence_shmap`). Ghost fill (same-level + sparse coarse-
    fine) and restrict run on the replicated hierarchy via the existing driver
    helpers; only the hot flux/divergence kernel is distributed. Matches
    :func:`amr.driver.advance`."""
    from amr.driver import _fill_ghosts, _restrict_into_parent
    from amr.reflux import apply_reflux_blocks

    from jax.sharding import NamedSharding

    repl = NamedSharding(mesh, P())

    ghosted = _fill_ghosts(hierarchy, cfg, gamma)
    new_data = []
    fluxes = []
    for L, lvl in enumerate(hierarchy):
        dU, fl = divergence_shmap(ghosted[L], lvl.neighbors, mesh, dt, L, cfg,
                                  gamma=gamma, limiter=limiter,
                                  riemann_solver=riemann_solver,
                                  return_face_fluxes=True)
        # Step 2: dU / fl now come back SHARDED on the block axis (no output
        # all_gather), so new_data stays sharded -> the divergence peak is
        # total/n_dev. The reflux below still needs the fluxes replicated for its
        # cross-device coarse-fine scatter; it all_gathers them itself inside its
        # own replicated shard_map (the GSPMD-safe path -- so the peak there is
        # only the small flux/coarse arrays, not the big dU). No upstream repl
        # constraint here: pinning dU replicated is exactly what we are removing.
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
