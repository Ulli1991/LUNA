"""Coarse-fine reflux (Berger-Colella flux correction).

When a fine block borders an unrefined coarse region, the two levels disagree
on the flux through that shared face: the coarse cell was updated with a single
coarse flux ``F_c``, while the fine side resolves it as ``r**(ndim-1)`` subface
fluxes ``F_f``. The mismatch ``F_c - <F_f>`` is the conservation error at the
coarse-fine interface. Reflux removes it by correcting the *coarse* cell just
outside the fine patch (the fine solution is taken as truth):

    dx_c side, + face of the patch:  U_c += (dt/dx_c) * (<F_f> - F_c)
    dx_c side, - face of the patch:  U_c += (dt/dx_c) * (F_c - <F_f>)

with ``<F_f>`` the transverse average of the fine subface fluxes over the coarse
face. After reflux the coarse-outside cell sees exactly the flux the fine block
applied to itself, so mass/momentum/energy telescope across the interface.

First cut — matches the dense-parent assumption already used by
:func:`amr.halo.fill_ghost_from_coarse`:

  * the coarse level is a *dense* ``(n_vars, *grid)`` field (the natural L0
    representation; its parent is the base grid);
  * **no time subcycling** — coarse and fine advance with the same ``dt`` (so
    ``<F_f>`` is a plain spatial average, no time accumulation);
  * a coarse-fine face is any fine-block face lacking a same-level fine
    neighbour (``coarse_fine_face_mask``); under the periodic cosmology BC the
    coarse underlay always exists there.

The per-slot Python loops below unroll under ``jit`` (slot count is static); they
are written for clarity over a scatter and are a first-cut hot-path cost to
revisit when wiring the full multi-level subcycled step.
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
    """Correct the dense coarse field at coarse-fine interfaces.

    Parameters
    ----------
    U_coarse : (n_vars, *coarse_grid)
        Dense coarse field AFTER its own hydro update (level ``fine_level-1``).
    coarse_fluxes : tuple of per-axis ``Fc_d`` (n_vars, coarse_grid[d]+1, *transverse)
        Coarse face fluxes (``muscl_hancock_face_fluxes`` on the dense coarse).
    fine_fluxes : tuple of per-axis ``Ff_d`` (n_slots, n_vars, block[d]+1, *transverse)
        Fine block face fluxes (``amr.hydro.block_face_fluxes``).
    fine_coords : (n_slots, ndim) int
        Fine block coords at ``fine_level`` (inactive slots ignored via mask).
    cf_mask : (n_slots, 2*ndim) bool
        From :func:`coarse_fine_face_mask`.

    Returns
    -------
    (n_vars, *coarse_grid) refluxed coarse field.
    """
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
    """Overwrite the coarse cells under each fine block with the conservative
    restriction of the fine data (the fine solution is authoritative there).

    ``U_coarse`` is the dense level-``(fine_level-1)`` field; ``fine_data`` is
    ``(n_slots, n_vars, *block_shape)``. Inactive slots (coord ``-1``) are
    skipped.
    """
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
