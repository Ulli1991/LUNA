"""Particles on the block-AMR hierarchy (DM phase of the cosmo port).

Design contract (verified against the block/patch-AMR literature -- Enzo,
Bryan+2014 / Passy & Bryan 2014 -- see memory note project-amr-dm-design):

  * particles are NEVER split or refined; the MESH refines (per-cell mass
    flags, whole-block granularity, Athena++-MeshBlock-style);
  * each particle is OWNED by the finest active block covering its position;
  * mass deposits by CIC at the owning level's dx into a GHOSTED per-block
    buffer whose ghost bands are folded back into same-level neighbours /
    the coarse parent (Enzo's oversized GravitatingMassField);
  * forces gather by inverse-CIC from the owning block's ghosted phi.

This module: kernel 1 -- the particle -> block map. Pure jnp, O(N log N)
per level (Morton sort + searchsorted, the same machinery as
blocks.find_face_neighbors), vectorised over particles, jit/scan-safe
(static level count, fixed particle count).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from amr.geometry import morton_encode, prolong


def _level_block_width(cfg, level):
    """Physical block width per dim at ``level``."""
    return tuple(cfg.box[d] / (cfg.base_blocks[d] * 2 ** level)
                 for d in range(cfg.ndim))


def locate_particles(pos, levels, cfg):
    """Finest active block covering each particle.

    Parameters
    ----------
    pos : (n_p, ndim) positions in the domain (periodic-wrapped internally).
    levels : hierarchy tuple of Level namedtuples (coords, n_active, ...).
    cfg : AMRConfig.

    Returns
    -------
    lev  : (n_p,) int32 -- owning level index (finest covering; 0 = base,
           which always covers, so every particle is owned).
    slot : (n_p,) int32 -- block slot in that level's arrays.
    """
    ndim = cfg.ndim
    n_p = pos.shape[0]

    # periodic wrap into the primary box
    box = jnp.asarray([cfg.box[d] for d in range(ndim)])
    p = jnp.mod(pos, box)

    lev = jnp.zeros((n_p,), jnp.int32)
    slot = jnp.zeros((n_p,), jnp.int32)
    found_any = jnp.zeros((n_p,), bool)

    # coarse -> fine: later (finer) matches overwrite, leaving the FINEST owner.
    for L, lvl in enumerate(levels):
        w = _level_block_width(cfg, L)
        c = jnp.stack([jnp.clip((p[:, d] / w[d]).astype(jnp.int32), 0,
                                cfg.base_blocks[d] * 2 ** L - 1)
                       for d in range(ndim)], axis=1)             # (n_p, ndim)
        pcode = morton_encode(c)

        n_slots = lvl.coords.shape[0]
        active = jnp.arange(n_slots) < lvl.n_active
        bcode = morton_encode(jnp.maximum(lvl.coords, 0))
        invalid = jnp.array(jnp.iinfo(bcode.dtype).max, dtype=bcode.dtype)
        bcode = jnp.where(active, bcode, invalid)                 # inactive sort last
        order = jnp.argsort(bcode)
        scode = bcode[order]

        i = jnp.clip(jnp.searchsorted(scode, pcode), 0, n_slots - 1)
        hit = scode[i] == pcode
        s = order[i].astype(jnp.int32)

        lev = jnp.where(hit, jnp.int32(L), lev)
        slot = jnp.where(hit, s, slot)
        found_any = found_any | hit

    # level 0 tiles the domain -> every particle must be owned
    return lev, slot


def accel_blocks(phi_ghosted, level, cfg):
    """Per-block acceleration -grad(phi) from a GHOSTED scalar phi.

    Input  (n_slots, 1, *(bs+2g)); output (n_slots, ndim, *(bs+2)) -- the
    2nd-order central difference is valid on the interior PLUS one ghost layer
    (with g=2 the ghosted phi has exactly the extra cell the edge diff needs),
    which is precisely the extent the CIC gather touches for a particle
    anywhere inside the block (i0 in [-1, bs-1])."""
    ndim = cfg.ndim
    g = cfg.n_ghost
    bs = cfg.block_shape
    dx = cfg.level_dx(level)
    comps = []
    for d in range(ndim):
        # window: [1, bs+2g-1) on every axis (extent bs+2), diff along d
        hi = [slice(None), slice(0, 1)] + [slice(2, bs[k] + 2 * g) if k == d
                                           else slice(1, bs[k] + 2 * g - 1)
                                           for k in range(ndim)]
        lo = [slice(None), slice(0, 1)] + [slice(0, bs[k] + 2 * g - 2) if k == d
                                           else slice(1, bs[k] + 2 * g - 1)
                                           for k in range(ndim)]
        comps.append(-(phi_ghosted[tuple(hi)] - phi_ghosted[tuple(lo)])
                     * (0.5 / dx[d]))
    return jnp.concatenate(comps, axis=1)                 # (n_slots, ndim, *(bs+2))


def gather_cic(fields, levels, pos, lev, slot, cfg):
    """Inverse-CIC: sample per-level block fields at particle positions.

    ``fields[L]`` is ``(n_slots, n_comp, *(bs+2))`` -- interior plus ONE ghost
    layer per side (:func:`accel_blocks` output). Each particle samples its
    OWNING block (from :func:`locate_particles`) with the standard cell-centred
    trilinear weights (matching gravity.particle_mesh._interp_one: cell centres
    at (i+0.5)dx). The 2**ndim corner gather is a static loop of fancy gathers
    -- O(n_p) memory, no per-particle block materialisation.

    Returns (n_p, n_comp).
    """
    import itertools

    ndim = cfg.ndim
    bs = cfg.block_shape
    n_p = pos.shape[0]
    n_comp = fields[0].shape[1]
    box = jnp.asarray([cfg.box[d] for d in range(ndim)])
    p = jnp.mod(pos, box)

    out = jnp.zeros((n_p, n_comp), dtype=fields[0].dtype)
    for L, (F, lvl) in enumerate(zip(fields, levels)):
        dx = cfg.level_dx(L)
        sel = lev == L
        s = jnp.where(sel, slot, 0)
        cb = jnp.maximum(lvl.coords[s], 0)                # (n_p, ndim)
        i0 = []
        t = []
        for d in range(ndim):
            gl = p[:, d] / dx[d] - 0.5 - cb[:, d] * bs[d]
            i = jnp.clip(jnp.floor(gl).astype(jnp.int32), -1, bs[d] - 1)
            i0.append(i)
            t.append(jnp.clip(gl - i, 0.0, 1.0))
        acc = jnp.zeros((n_p, n_comp), dtype=fields[0].dtype)
        for off in itertools.product((0, 1), repeat=ndim):
            w = jnp.ones((n_p,), dtype=fields[0].dtype)
            for d in range(ndim):
                w = w * (t[d] if off[d] else 1.0 - t[d])
            idx = tuple(i0[d] + 1 + off[d] for d in range(ndim))
            for c in range(n_comp):
                acc = acc.at[:, c].add(w * F[(s, c) + idx])
        out = jnp.where(sel[:, None], acc, out)
    return out


# ---------------------------------------------------------------------------
# Kernel 2: CIC mass deposit onto the hierarchy (Enzo semantics)
# ---------------------------------------------------------------------------
def _cic_fractions(p, dx, coords_of_slot, bs, ndim):
    """Block-local CIC anchor + fractions for cell-centred deposit/gather:
    ``i0`` in [-1, bs-1] per dim (one ghost layer), weights ``(1-t, t)``."""
    i0, t = [], []
    for d in range(ndim):
        gl = p[:, d] / dx[d] - 0.5 - coords_of_slot[:, d] * bs[d]
        i = jnp.clip(jnp.floor(gl).astype(jnp.int32), -1, bs[d] - 1)
        i0.append(i)
        t.append(jnp.clip(gl - i, 0.0, 1.0))
    return i0, t


def _deposit_ghosted(p, mass, lvl, level, lev, slot, cfg):
    """Stage (a): scatter this level's OWNED particles into a ghosted per-block
    density buffer ``(n_slots, 1, *(bs+2))``. Non-owned particles park on the
    sentinel slot with zero weight."""
    import itertools

    ndim = cfg.ndim
    bs = cfg.block_shape
    dx = cfg.level_dx(level)
    dV = 1.0
    for d in range(ndim):
        dV = dV * dx[d]
    n_slots = lvl.data.shape[0]

    sel = lev == level
    w0 = jnp.where(sel, mass, 0.0) / dV
    s = jnp.where(sel, slot, n_slots - 1)
    cb = jnp.maximum(lvl.coords[s], 0)
    i0, t = _cic_fractions(p, dx, cb, bs, ndim)

    buf = jnp.zeros((n_slots, 1) + tuple(b + 2 for b in bs))
    zero_c = jnp.zeros_like(s)
    for off in itertools.product((0, 1), repeat=ndim):
        w = w0
        for d in range(ndim):
            w = w * (t[d] if off[d] else 1.0 - t[d])
        idx = (s, zero_c) + tuple(i0[d] + 1 + off[d] for d in range(ndim))
        buf = buf.at[idx].add(w)
    return buf


def _fold_same_level(buf, neighbors, cfg):
    """Stage (b): fold ghost bands into face neighbours -- the TRANSPOSE of the
    sequential corner-filling ghost sweep. Each axis pass moves the FULL-EXTENT
    ghost slab (transverse ghost corners included) into the neighbour's GHOSTED
    buffer edge; corner mass thereby cascades through neighbours across passes,
    exactly mirroring how the fill sweep propagates corner ghosts (an
    interior-extent fold silently DROPPED corner deposits -- measured as a 1.4%
    mass leak). Returns the interior density ``(n_slots, 1, *bs)`` and the
    swept buffer, whose remaining ghost mass sits only at faces with no
    same-level neighbour (coarse-fine / domain), for stage (c)."""
    ndim = cfg.ndim
    n_slots = neighbors.shape[0]
    sentinel = n_slots - 1

    for d in range(ndim):
        sax = 2 + d
        for s_dir, col in ((+1, 2 * d), (-1, 2 * d + 1)):
            band_sl = [slice(None), slice(None)] + [slice(None)] * ndim
            band_sl[sax] = slice(-1, None) if s_dir > 0 else slice(0, 1)
            band = buf[tuple(band_sl)]                    # full extent transverse
            nb = neighbors[:, col]
            has = nb != sentinel
            nb_safe = jnp.where(has, nb, sentinel)
            mask = has.reshape((-1,) + (1,) * (1 + ndim))
            place = jnp.zeros_like(buf)
            dst = [slice(None), slice(None)] + [slice(None)] * ndim
            # neighbour's edge INTERIOR cell layer (ghosted coords: 1 / -2)
            dst[sax] = slice(1, 2) if s_dir > 0 else slice(-2, -1)
            place = place.at[tuple(dst)].set(jnp.where(mask, band, 0.0))
            buf = buf + jax.ops.segment_sum(place, nb_safe, num_segments=n_slots)
            # claim: zero the folded band on blocks that had a neighbour
            buf = buf.at[tuple(band_sl)].set(
                jnp.where(mask, jnp.zeros_like(band), band))

    interior = (slice(None), slice(None)) + (slice(1, -1),) * ndim
    return buf[interior], buf


def _fold_to_parent(rho_coarse, buf, fine_lvl, coarse_lvl, level, cfg):
    """Stage (c): the still-unclaimed ghost bands (coarse-fine faces) restrict
    onto the parent-side coarse cell across the face -- Enzo's oversized
    GravitatingMassField summed into the parent. Density rescales by 2**-ndim
    (fine->coarse cell volume); transverse fine cells sum onto their covering
    coarse cell. Target geometry mirrors amr.reflux.apply_reflux_blocks."""
    from amr.reflux import _i32, coarse_fine_face_mask

    ndim = cfg.ndim
    bs = cfg.block_shape
    hs = tuple(b // 2 for b in bs)
    Nc = rho_coarse.shape[0]
    csent = Nc - 1
    cf = coarse_fine_face_mask(fine_lvl.neighbors, fine_lvl.n_active)
    q = jnp.maximum(fine_lvl.coords, 0) % 2
    parent_safe = jnp.maximum(fine_lvl.parent_slots, 0)

    for d in range(ndim):
        for s_dir, col in ((+1, 2 * d), (-1, 2 * d + 1)):
            band_sl = [slice(None), slice(None)] + [slice(None)] * ndim
            band_sl[2 + d] = slice(-1, None) if s_dir > 0 else slice(0, 1)
            band = buf[tuple(band_sl)]                   # FULL transverse extent
            band_raw = band                              # for the claim-zero below
            # clamp transverse-ghost (corner) mass into the nearest in-face
            # cell: exactly conservative; placement off by <= 1 fine cell, and
            # only at C/F corners (same-level corners already cascaded away).
            for dd in range(ndim):
                if dd == d:
                    continue
                ax = 2 + dd
                lo = [slice(None)] * band.ndim
                lo[ax] = slice(0, 1)
                lo1 = [slice(None)] * band.ndim
                lo1[ax] = slice(1, 2)
                hi = [slice(None)] * band.ndim
                hi[ax] = slice(-1, None)
                hi1 = [slice(None)] * band.ndim
                hi1[ax] = slice(-2, -1)
                band = band.at[tuple(lo1)].add(band[tuple(lo)])
                band = band.at[tuple(hi1)].add(band[tuple(hi)])
                keep = [slice(None)] * band.ndim
                keep[ax] = slice(1, -1)
                band = band[tuple(keep)]                 # -> interior extent
            if s_dir > 0:
                tgt = jnp.where(q[:, d] == 0, parent_safe,
                                coarse_lvl.neighbors[parent_safe, 2 * d])
                cell_d = jnp.where(q[:, d] == 0, hs[d], 0)
            else:
                tgt = jnp.where(q[:, d] == 1, parent_safe,
                                coarse_lvl.neighbors[parent_safe, 2 * d + 1])
                cell_d = jnp.where(q[:, d] == 1, hs[d] - 1, bs[d] - 1)
            valid = cf[:, col] & (fine_lvl.parent_slots >= 0) & (tgt >= 0) & (tgt != csent)
            tsafe = jnp.where(valid, tgt, csent)

            # transverse 2->1 sum, then the fine->coarse density volume factor
            b = band
            ax = 2
            for dd in range(ndim):
                if dd == d:
                    ax += 1
                    continue
                shp = b.shape
                b = b.reshape(shp[:ax] + (shp[ax] // 2, 2) + shp[ax + 1:]).sum(axis=ax + 1)
                ax += 1
            b = b / (2.0 ** ndim)

            def one(tt, qq, cd, bb):
                zero = jnp.zeros((1,) + tuple(bs))
                start = [_i32(0)] + [_i32(cd) if dd == d else _i32(qq[dd] * hs[dd])
                                     for dd in range(ndim)]
                shape = [1] + [1 if dd == d else hs[dd] for dd in range(ndim)]
                return jax.lax.dynamic_update_slice(zero, bb.reshape(shape), start)

            patches = jax.vmap(one)(tsafe, q, cell_d, b)
            patches = jnp.where(valid.reshape((-1,) + (1,) * (1 + ndim)), patches, 0.0)
            seg = jax.ops.segment_sum(patches, tsafe, num_segments=Nc)
            seg = seg.at[csent].set(0.0)
            rho_coarse = rho_coarse + seg
            # claim-zero the consumed band: with FULL-extent slabs a ghost
            # CORNER cell belongs to ndim different bands -- without zeroing,
            # its mass folds to the parent once per axis (measured +4.9e-5
            # over-count). Unconsumed (invalid-target) bands stay for later
            # passes; on properly-nested hierarchies (dilation buffer) every
            # C/F band has a valid target, so conservation is exact.
            vmask = valid.reshape((-1,) + (1,) * (1 + ndim))
            buf = buf.at[tuple(band_sl)].set(
                jnp.where(vmask, jnp.zeros_like(band_raw), band_raw))
    return rho_coarse


def deposit_cic(pos, mass, levels, lev, slot, cfg):
    """CIC mass deposit onto the hierarchy: per-level ghosted scatter at the
    OWNING level's dx, same-level fold-back, coarse-fine fold-to-parent, then
    fine-authoritative restriction so every level's rho_dm includes its
    children's particles. Total composite mass == sum(mass) to machine
    precision (the conservation test). Returns per-level (n_slots, 1, *bs)."""
    from amr.driver import _restrict_into_parent

    ndim = cfg.ndim
    box = jnp.asarray([cfg.box[d] for d in range(ndim)])
    p = jnp.mod(pos, box)

    bufs = [_deposit_ghosted(p, mass, lvl, L, lev, slot, cfg)
            for L, lvl in enumerate(levels)]
    rhos, rem = [], []
    for lvl, buf in zip(levels, bufs):
        rho, buf2 = _fold_same_level(buf, lvl.neighbors, cfg)
        rhos.append(rho)
        rem.append(buf2)
    for L in range(1, len(levels)):
        rhos[L - 1] = _fold_to_parent(rhos[L - 1], rem[L], levels[L],
                                      levels[L - 1], L, cfg)

    # Coarse-owned particles near a C/F face spill CIC mass into COVERED coarse
    # cells; the fine-authoritative restriction below would overwrite (destroy)
    # it. Prolong-ADD each fine block's parent-octant coarse deposit into the
    # fine field first -- then restrict(fine) reproduces the spill on the
    # coarse level instead of erasing it (measured leak without this: 1.4%).
    hs = tuple(b // 2 for b in cfg.block_shape)
    for L in range(1, len(levels)):
        lvl = levels[L]
        parent_rho = rhos[L - 1]
        q = jnp.maximum(lvl.coords, 0) % 2
        ps = jnp.maximum(lvl.parent_slots, 0)

        def octant(pslot, qq):
            start = [jnp.asarray(0, jnp.int32)] + [
                jnp.asarray(qq[d] * hs[d], jnp.int32) for d in range(ndim)]
            return jax.lax.dynamic_slice(parent_rho[pslot], start, [1] + list(hs))

        add = jax.vmap(lambda s_, q_: prolong(octant(s_, q_), ndim))(ps, q)
        valid = (lvl.parent_slots >= 0).reshape((-1,) + (1,) * (1 + ndim))
        rhos[L] = rhos[L] + jnp.where(valid, add, 0.0)

    for L in range(len(levels) - 1, 0, -1):
        fine = levels[L]._replace(data=rhos[L])
        rhos[L - 1] = _restrict_into_parent(rhos[L - 1], fine, cfg)
    return rhos
