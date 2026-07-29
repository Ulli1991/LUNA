"""Coarse-fine reflux (Berger-Colella flux correction).

Corrects the coarse cell just outside a fine patch so it sees the
transverse-averaged fine subface flux: ``U_c += +/-(dt/dx_c) * (<F_f> - F_c)``.
Assumes NO time subcycling (coarse and fine share ``dt``). A coarse-fine face
is any fine-block face lacking a same-level neighbour.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from amr.geometry import restrict


def _i32(x):
    """Cast a scalar (Python int or traced) to int32 so all dynamic_slice start
    indices share a dtype (a bare Python int is int64 under x64, which
    dynamic_(update_)slice rejects when mixed with int32 block coords)."""
    return jnp.asarray(x, jnp.int32)


def coarse_fine_face_mask(neighbors, n_active):
    """Boolean ``(n_slots, 2*ndim)``: a fine-block face is a coarse-fine
    interface iff it has no same-level neighbour (maps to the NaN sentinel) and
    the slot is active. Face column order matches
    ``blocks.find_face_neighbors`` (``[+d, -d]`` per axis)."""
    n_slots = neighbors.shape[0]
    sentinel = n_slots - 1
    active = jnp.arange(n_slots) < n_active
    return (neighbors == sentinel) & active[:, None]


def _restrict_transverse(face, ndim):
    """Average a fine block-boundary face (``(n_vars, *(ndim-1 axes of size
    block))``) down by 2 along every transverse axis -> one value per coarse
    face cell."""
    out = face
    ax = 1
    for _ in range(ndim - 1):
        shp = out.shape
        out = out.reshape(shp[:ax] + (shp[ax] // 2, 2) + shp[ax + 1:]).mean(axis=ax + 1)
        ax += 1
    return out


def _fine_boundary_face(Ff_block, d, idx, ndim):
    """The fine block's boundary face on axis ``d`` (``idx`` = 0 for -d, -1 for
    +d), dropping the face axis -> ``(n_vars, *(block transverse axes))``. The
    face axis sits at array position ``1+d`` (after the n_vars axis)."""
    sl = [slice(None)] * (ndim + 1)
    sl[1 + d] = idx
    return Ff_block[tuple(sl)]


def _coarse_face_patch(Fc_d, d, face_idx, o, h, ndim):
    """Coarse flux on axis ``d`` at face ``face_idx``, over the transverse coarse
    cells covered by the fine patch -> ``(n_vars, *h_transverse)``. The face axis
    is at array position ``1+d`` (Fx -> axis 1, Fy -> axis 2, Fz -> axis 3)."""
    n_vars = Fc_d.shape[0]
    start = [_i32(0)]
    size = [n_vars]
    for dd in range(ndim):
        if dd == d:
            start.append(_i32(jnp.maximum(face_idx, 0)))
            size.append(1)
        else:
            start.append(_i32(o[dd]))
            size.append(h[dd])
    return jnp.squeeze(jax.lax.dynamic_slice(Fc_d, start, size), axis=1 + d)


def apply_reflux_dense(
    U_coarse,
    coarse_fluxes,
    fine_fluxes,
    fine_coords,
    cf_mask,
    fine_level,
    cfg,
    dt,
):
    """Correct the dense coarse field (``(n_vars, *coarse_grid)``, AFTER its own
    hydro update) at coarse-fine interfaces. ``coarse_fluxes``/``fine_fluxes``
    are per-axis face-flux tuples; ``cf_mask`` is from
    :func:`coarse_fine_face_mask`. Returns the refluxed coarse field."""
    ndim = cfg.ndim
    bs = cfg.block_shape
    h = tuple(s // 2 for s in bs)                 # patch extent in coarse cells
    dx_c = cfg.level_dx(fine_level - 1)
    n_slots = fine_coords.shape[0]
    n_vars = U_coarse.shape[0]

    acc = U_coarse
    for s in range(n_slots):                       # static unroll
        cf = fine_coords[s]
        o = [jnp.maximum(cf[d], 0) * h[d] for d in range(ndim)]   # coarse origin
        for d in range(ndim):
            Fc_d = coarse_fluxes[d]
            Ff_d = fine_fluxes[d][s]
            for side, col in ((+1, 2 * d), (-1, 2 * d + 1)):
                mask = cf_mask[s, col].astype(U_coarse.dtype)
                if side == +1:
                    fbar = _restrict_transverse(_fine_boundary_face(Ff_d, d, -1, ndim), ndim)
                    face_idx = o[d] + h[d]
                    cell_d = o[d] + h[d]            # coarse cell outside on +d
                else:
                    fbar = _restrict_transverse(_fine_boundary_face(Ff_d, d, 0, ndim), ndim)
                    face_idx = o[d]
                    cell_d = o[d] - 1              # coarse cell outside on -d

                fc = _coarse_face_patch(Fc_d, d, face_idx, o, h, ndim)
                sgn = 1.0 if side == +1 else -1.0
                delta = mask * (dt / dx_c[d]) * sgn * (fbar - fc)
                delta_full = jnp.expand_dims(delta, axis=1 + d)   # size-1 along d

                start = [_i32(0)] + [
                    _i32(jnp.maximum(cell_d, 0) if dd == d else o[dd]) for dd in range(ndim)
                ]
                size = [n_vars] + [(1 if dd == d else h[dd]) for dd in range(ndim)]
                region = jax.lax.dynamic_slice(acc, start, size)
                acc = jax.lax.dynamic_update_slice(acc, region + delta_full, start)
    return acc


def restrict_fine_into_coarse_dense(U_coarse, fine_data, fine_coords, fine_level, cfg):
    """Overwrite the coarse cells under each fine block with restrict(fine)
    (fine authoritative). ``U_coarse`` is the dense level-``(fine_level-1)``
    field; inactive slots (coord ``-1``) are skipped."""
    ndim = cfg.ndim
    bs = cfg.block_shape
    h = tuple(s // 2 for s in bs)
    n_slots = fine_coords.shape[0]

    acc = U_coarse
    for s in range(n_slots):
        cf = fine_coords[s]
        valid = cf[0] >= 0
        rc = restrict(fine_data[s], ndim)                  # (n_vars, *h)
        start = [_i32(0)] + [_i32(jnp.maximum(cf[d], 0) * h[d]) for d in range(ndim)]
        updated = jax.lax.dynamic_update_slice(acc, rc, start)
        acc = jnp.where(valid, updated, acc)
    return acc


def apply_reflux_blocks(coarse_data, coarse_fluxes, fine_fluxes, fine_coords,
                        fine_parent_slots, fine_neighbors, fine_n_active,
                        coarse_neighbors, dt, coarse_level, cfg):
    """Block-sparse Berger-Colella reflux (production version): vectorised
    vmap + segment_sum correction of the unrefined coarse cells outside every
    coarse-fine face. ``coarse_fluxes``/``fine_fluxes`` must be the per-axis
    seam-corrected face fluxes the updates actually used; sentinel/inactive
    slots carry NaN fluxes and are masked to zero before the scatter."""
    ndim = cfg.ndim
    bs = cfg.block_shape
    hs = tuple(s // 2 for s in bs)
    n_vars = coarse_data.shape[1]
    Nc = coarse_data.shape[0]
    dxc = cfg.level_dx(coarse_level)
    cf = coarse_fine_face_mask(fine_neighbors, fine_n_active)    # (Nf, 2*ndim)
    q = jnp.maximum(fine_coords, 0) % 2                           # (Nf, ndim)
    parent = fine_parent_slots                                    # (Nf,)
    parent_safe = jnp.maximum(parent, 0)
    c_sent = Nc - 1

    out = coarse_data
    for d in range(ndim):
        for s, col, f_idx in ((+1, 2 * d, -1), (-1, 2 * d + 1, 0)):
            # fine boundary flux on this face, transverse-averaged to coarse
            face = jax.vmap(lambda F: _fine_boundary_face(F, d, f_idx, ndim))(
                fine_fluxes[d])                                   # (Nf, n_vars, *bs_trans)
            ff = jax.vmap(lambda f: _restrict_transverse(f, ndim))(face)

            qd = q[:, d]
            if s > 0:
                target = jnp.where(qd == 0, parent_safe,
                                   coarse_neighbors[parent_safe, 2 * d])
                cell_d = jnp.where(qd == 0, hs[d], 0)
                face_d = cell_d                                   # cell's -d face
            else:
                target = jnp.where(qd == 1, parent_safe,
                                   coarse_neighbors[parent_safe, 2 * d + 1])
                cell_d = jnp.where(qd == 1, hs[d] - 1, bs[d] - 1)
                face_d = cell_d + 1                               # cell's +d face
            valid = cf[:, col] & (parent >= 0) & (target >= 0) & (target != c_sent)
            tsafe = jnp.where(valid, target, c_sent)

            def one(tslot, qq, cell, facei, ffp):
                Fc_d = coarse_fluxes[d][tslot]
                o = [qq[t] * hs[t] for t in range(ndim)]          # o[d] unused
                Fcp = _coarse_face_patch(Fc_d, d, facei, o, hs, ndim)
                corr = (dt / dxc[d]) * float(s) * (ffp - Fcp)     # (n_vars, *h_trans)
                shape = [n_vars] + [1 if t == d else hs[t] for t in range(ndim)]
                start = [_i32(0)] + [_i32(cell) if t == d else _i32(o[t])
                                     for t in range(ndim)]
                zero = jnp.zeros((n_vars,) + tuple(bs), coarse_data.dtype)
                return jax.lax.dynamic_update_slice(zero, corr.reshape(shape), start)

            patches = jax.vmap(one)(tsafe, q, cell_d, face_d, ff)
            patches = jnp.where(valid.reshape((-1,) + (1,) * (1 + ndim)),
                                patches, 0.0)
            seg = jax.ops.segment_sum(patches, tsafe, num_segments=Nc)
            seg = seg.at[c_sent].set(0.0)                         # keep sentinel NaN
            out = out + seg
    return out
