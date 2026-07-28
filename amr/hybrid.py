"""HYBRID dense-L0 block-AMR driver: align the block path with the dense path.

Motivation (measured, 2026-07-20): the pure block solver runs ~250x slower per
GPU than the dense fixed-grid solver at the same cell count -- the per-step
27-region gather ghost fill + ghosted-copy materialization is XLA-hostile,
while the AMR logic itself (C/F, reflux, 2 levels) only costs 1.75x. Level 0
always tiles the domain, so this driver stores L0 as ONE DENSE ARRAY stepped by
the proven dense MUSCL kernels (gigacell/s), and keeps the (validated) block
machinery only for the refined levels, which are small.

v1: single-device, synchronized same-dt, hydro-only. The dense/block bridges
all pre-exist: ``fill_ghost_from_coarse`` (dense-parent C/F fill -- the
original design), ``apply_reflux_dense`` (Berger-Colella into a dense coarse
field), and restriction as a batched ``dynamic_update_slice`` scatter.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from amr import driver
from amr.geometry import restrict
from amr.halo import (fill_ghost_from_coarse, fill_ghost_physical,
                      fill_ghost_same_level)
from amr.hydro import cons_to_prim_blocks, conservative_block_divergence
from amr.reflux import apply_reflux_blocks, apply_reflux_dense, coarse_fine_face_mask
from hydro.conversions import cons_to_prim
from hydro.muscl_hancock import divergence_from_face_fluxes, muscl_hancock_face_fluxes


def _pad_dense(W, g, boundary, ndim):
    """Ghost-pad the dense L0 primitives: periodic wrap or zero-gradient."""
    mode = "wrap" if boundary == 0 else "edge"
    pad = [(0, 0)] + [(g, g)] * ndim
    return jnp.pad(W, pad, mode=mode)


def dense_l0_step(U0, dt, cfg, *, gamma, limiter, riemann_solver):
    """One dense MUSCL-Hancock update of L0; returns (U0_new, face_fluxes).
    The face fluxes are kept for ``apply_reflux_dense`` (Berger-Colella)."""
    ndim = cfg.ndim
    g = cfg.n_ghost
    dx = cfg.level_dx(0)
    inv = dict(inv_dx=1.0 / dx[0],
               inv_dy=1.0 / dx[1] if ndim >= 2 else 1.0,
               inv_dz=1.0 / dx[2] if ndim == 3 else 1.0)
    W = cons_to_prim(U0, gamma)
    W_pad = _pad_dense(W, g, cfg.boundary, ndim)
    fluxes = muscl_hancock_face_fluxes(
        W_pad, dt, gamma=gamma, limiter=limiter, riemann_solver=riemann_solver,
        spatial_dim=ndim, boundary=cfg.boundary, **inv)
    dU = divergence_from_face_fluxes(fluxes, spatial_dim=ndim, **inv)
    # the kernels cover pad-minus-one-ring (N + 2(g-1) cells): slice one ghost
    # ring off the divergence AND each face-flux array to land on the interior
    # (faces: N+2g-1 -> N+1 along the face axis, N along transverse).
    ring = (slice(None),) + (slice(1, -1),) * ndim
    dU = dU[ring]
    fluxes = tuple(F[ring] for F in fluxes)
    return U0 + dt * dU, fluxes


def _restrict_blocks_into_dense(U0, fine, cfg):
    """Overwrite covered dense-L0 cells with restrict(fine blocks)."""
    ndim = cfg.ndim
    bs = cfg.block_shape
    h = tuple(s // 2 for s in bs)
    n = fine.data.shape[0]
    restricted = jax.vmap(lambda d: restrict(d, ndim))(fine.data)   # (n, nv, *h)
    valid = fine.coords[:, 0] >= 0

    def scatter(U, i):
        c = jnp.maximum(fine.coords[i], 0)
        start = (jnp.asarray(0, jnp.int32),) + tuple(
            (c[d] * h[d]).astype(jnp.int32) for d in range(ndim))
        blk = jnp.where(valid[i], restricted[i],
                        jax.lax.dynamic_slice(U, start, (U.shape[0],) + h))
        return jax.lax.dynamic_update_slice(U, blk, start), None

    U0, _ = jax.lax.scan(scatter, U0, jnp.arange(n))
    return U0


def advance_hybrid_sharded(U0, levels, dt, cfg, *, gamma, limiter=1,
                           riemann_solver="hllc", reflux=True,
                           l0_sharding=None, blk_sharding=None):
    """Multi-GPU hybrid step (v2, GSPMD): the dense L0 carries a SPATIAL
    sharding constraint (GSPMD lowers the ghost pad to halo exchange -- the
    fixed-grid autoshard mode; the explicit-ppermute dense step is the v3
    upgrade), refined-level blocks carry the block-axis sharding. One jitted
    graph, mixed shardings."""
    if l0_sharding is not None:
        U0 = jax.lax.with_sharding_constraint(U0, l0_sharding)
    if blk_sharding is not None:
        levels = tuple(l._replace(data=jax.lax.with_sharding_constraint(
            l.data, blk_sharding)) for l in levels)
    U0_new, new_levels = advance_hybrid(U0, levels, dt, cfg, gamma=gamma,
                                        limiter=limiter,
                                        riemann_solver=riemann_solver,
                                        reflux=reflux)
    if l0_sharding is not None:
        U0_new = jax.lax.with_sharding_constraint(U0_new, l0_sharding)
    if blk_sharding is not None:
        new_levels = tuple(l._replace(data=jax.lax.with_sharding_constraint(
            l.data, blk_sharding)) for l in new_levels)
    return U0_new, new_levels


def advance_hybrid(U0, levels, dt, cfg, *, gamma, limiter=1,
                   riemann_solver="hllc", reflux=True):
    """Synchronized hybrid step: dense L0 + block levels >= 1.

    ``levels`` holds ONLY the refined levels (index 0 = physical level 1).
    Bit-compatibility with the all-blocks driver is NOT expected (different FP
    order on L0); the gates are conservation + the physics benchmarks."""
    ndim = cfg.ndim
    g = cfg.n_ghost

    # 1) dense L0 update (the fast path), keeping face fluxes for reflux
    U0_new, flux0 = dense_l0_step(U0, dt, cfg, gamma=gamma, limiter=limiter,
                                  riemann_solver=riemann_solver)

    # 2) refined-level ghost fills: L1 from the DENSE parent, deeper from blocks
    W0 = cons_to_prim(U0, gamma)
    ghosted = []
    for i, lvl in enumerate(levels):
        L = i + 1
        Wb = cons_to_prim_blocks(lvl.data, gamma)
        gw = fill_ghost_same_level(Wb, lvl.neighbors, cfg)
        gw = fill_ghost_physical(gw, Wb, cfg, coords=lvl.coords, level=L)
        if i == 0:
            gw = fill_ghost_from_coarse(gw, lvl.coords, W0, L, cfg,
                                        boundary=cfg.boundary)
        else:
            from amr.halo import fill_ghost_from_coarse_sparse
            gw = fill_ghost_from_coarse_sparse(gw, lvl.coords, lvl.parent_slots,
                                               ghosted[i - 1], cfg, interp="linear")
        gw = fill_ghost_physical(gw, Wb, cfg)
        ghosted.append(gw)

    # 3) refined-level updates (existing seam-consistent block divergence)
    new_data, fluxes = [], []
    for i, lvl in enumerate(levels):
        dU, fl = conservative_block_divergence(
            ghosted[i], dt, i + 1, cfg, lvl.neighbors, gamma=gamma,
            limiter=limiter, riemann_solver=riemann_solver,
            return_face_fluxes=True, coords=lvl.coords)
        new_data.append(lvl.data + dt * dU)
        fluxes.append(fl)

    # 4) Berger-Colella reflux, finest -> coarsest; L1 -> dense L0 last
    if reflux:
        for i in range(len(levels) - 1, 0, -1):
            new_data[i - 1] = apply_reflux_blocks(
                new_data[i - 1], fluxes[i - 1], fluxes[i], levels[i].coords,
                levels[i].parent_slots, levels[i].neighbors,
                levels[i].n_active, levels[i - 1].neighbors, dt, i, cfg)
        if levels:
            cf = coarse_fine_face_mask(levels[0].neighbors, levels[0].n_active)
            U0_new = apply_reflux_dense(U0_new, flux0, fluxes[0],
                                        levels[0].coords, cf, 1, cfg, dt)

    # 5) restriction, finest -> coarsest; L1 -> dense scatter last
    for i in range(len(levels) - 1, 0, -1):
        fine = levels[i]._replace(data=new_data[i])
        new_data[i - 1] = driver._restrict_into_parent(new_data[i - 1], fine, cfg)
    if levels:
        U0_new = _restrict_blocks_into_dense(
            U0_new, levels[0]._replace(data=new_data[0]), cfg)

    new_levels = tuple(l._replace(data=d) for l, d in zip(levels, new_data))
    return U0_new, new_levels
