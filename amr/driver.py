"""Multi-level AMR hydro driver (single device, first cut).

A hierarchy of block levels: level 0 fully tiles the domain (periodic); levels
>= 1 are sparse refined blocks. One synchronized step advances every level with
the SAME dt (no subcycling yet):

  1. fill ghosts coarse->fine: same-level seam halo per level, then coarse-fine
     ghosts gathered from the parent level (sparse, :func:`amr.halo`);
  2. advance each level with conservative same-level seam fluxes
     (:func:`amr.hydro.conservative_block_divergence`);
  3. restrict fine results into the covered coarse cells (fine is authoritative).

Topology (which blocks exist, neighbours, parent slots) is fixed between
regrids, so the hierarchy is a static-shape pytree and :func:`advance` jits.
This first cut uses **static refinement** (flag once at build time) and omits
coarse-fine reflux; both are layered on next. Targeted at 2D (KH demo) but the
block ops are dimension-generic.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from amr.blocks import (
    build_children,
    find_face_neighbors,
    find_parent_slots,
    match_blocks,
    preserve_unaltered,
    refinement_block_mask,
)
from amr.geometry import prolong, restrict

# ---------------------------------------------------------------------------
# Jitted topology kernels for build/regrid. These are pure-jnp argsort /
# searchsorted / gather pipelines whose EAGER per-op dispatch (~3 s per regrid;
# find_face_neighbors alone ~1.7 s, roughly independent of block count)
# dominated the whole AMR run wall-time. Their shapes live on the geometric
# capacity ladder (_sized_parents), so each (shape, static-arg) combination
# compiles once and replays in milliseconds thereafter.
# ---------------------------------------------------------------------------
_neighbors_j = jax.jit(find_face_neighbors, static_argnames=("level", "cfg", "periodic"))
_parent_slots_j = jax.jit(find_parent_slots, static_argnames=("cfg",))
_match_blocks_j = jax.jit(match_blocks, static_argnames=("ndim",))
_children_j = jax.jit(build_children, static_argnames=("level", "cfg", "max_parents"))
_preserve_j = jax.jit(preserve_unaltered)
from amr.halo import fill_ghost_from_coarse_sparse, fill_ghost_physical, fill_ghost_same_level
from amr.hydro import conservative_block_divergence, cons_to_prim_blocks
from amr.state import build_base_level
from hydro.conversions import cons_to_prim


class Level(NamedTuple):
    """One refinement level (block array + topology). Last slot is the NaN
    sentinel; inactive slots have coords -1."""

    data: jnp.ndarray        # (n_slots, n_vars, *block_shape) conservative
    coords: jnp.ndarray      # (n_slots, ndim) int32
    neighbors: jnp.ndarray   # (n_slots, 2*ndim) int32 same-level face slots
    parent_slots: jnp.ndarray  # (n_slots,) int32 parent slot in the coarser level (-1 at L0)
    n_active: jnp.ndarray    # scalar int32


# ---------------------------------------------------------------------------
# Refinement flagging
# ---------------------------------------------------------------------------
def flag_density_gradient(data, neighbors, n_active, cfg, threshold, buffer=None):
    """Flag a block when any cell's density gradient magnitude exceeds
    ``threshold``, computed on **ghost-filled** density so cells at a block edge
    see across the boundary. A cross-block buffer dilation refines a margin around
    the feature. Returns (n_slots,) bool.
    """
    ndim = cfg.ndim
    g = cfg.n_ghost
    bs = cfg.block_shape
    rho = data[:, 0:1]                                 # (n_slots, 1, *bs)
    gh = fill_ghost_same_level(rho, neighbors, cfg)    # (n_slots, 1, *(bs+2g))

    grad = jnp.zeros((data.shape[0], 1) + tuple(bs))
    for d in range(ndim):
        sl_hi = [slice(None), slice(None)] + [slice(g, g + bs[k]) for k in range(ndim)]
        sl_lo = list(sl_hi)
        sl_hi[2 + d] = slice(g + 1, g + 1 + bs[d])
        sl_lo[2 + d] = slice(g - 1, g - 1 + bs[d])
        diff = gh[tuple(sl_hi)] - gh[tuple(sl_lo)]
        grad = grad + jnp.where(jnp.isnan(diff), 0.0, jnp.abs(diff))

    cell_flag = (grad[:, 0] > threshold).astype(jnp.float32)   # (n_slots, *bs)
    from amr.blocks import cross_block_dilate
    cell_flag = cross_block_dilate(cell_flag, neighbors, cfg, buffer=buffer)
    flagged = refinement_block_mask(cell_flag, ndim)
    active = jnp.arange(data.shape[0]) < n_active
    return flagged & active


def flag_vorticity(data, neighbors, n_active, cfg, threshold, level, buffer=None):
    """Flag a block when the **physical vorticity** |omega| = |dvy/dx - dvx/dy|
    (2D) exceeds ``threshold`` anywhere in it. Unlike the density gradient, the
    /dx makes this a true physical quantity (rotation rate), so it is
    **resolution-independent**: a vortex core has the same |omega| at any base
    resolution, so KH refines even from a well-resolved (GPU-saturating) base and
    keeps refining deeper levels instead of de-refining. Computed on ghost-filled
    velocity; cross-block dilation adds a margin. Dimension-generic: 2D uses the
    scalar out-of-plane curl |dvy/dx - dvx/dy|; 3D uses the full vorticity-vector
    magnitude |curl v| = sqrt(wx^2 + wy^2 + wz^2). Returns (n_slots,) bool.
    """
    ndim = cfg.ndim
    if ndim not in (2, 3):
        raise NotImplementedError("vorticity criterion needs ndim in {2, 3}")
    g = cfg.n_ghost
    bs = cfg.block_shape
    rho = data[:, 0:1]
    # ghost-filled velocity components v[d], each (n_slots, 1, *(bs+2g))
    v = [fill_ghost_same_level(data[:, 1 + d:2 + d] / rho, neighbors, cfg)
         for d in range(ndim)]
    dx = cfg.level_dx(level)

    def cdiff(arr, d):
        hi = [slice(None), slice(None)] + [slice(g, g + bs[k]) for k in range(ndim)]
        lo = list(hi)
        hi[2 + d] = slice(g + 1, g + 1 + bs[d])
        lo[2 + d] = slice(g - 1, g - 1 + bs[d])
        return (arr[tuple(hi)] - arr[tuple(lo)]) / (2.0 * dx[d])

    if ndim == 2:
        omega_sq = (cdiff(v[1], 0) - cdiff(v[0], 1)) ** 2         # (n_slots,1,*bs)
    else:
        # 3D curl: (dvz/dy - dvy/dz, dvx/dz - dvz/dx, dvy/dx - dvx/dy)
        wx = cdiff(v[2], 1) - cdiff(v[1], 2)
        wy = cdiff(v[0], 2) - cdiff(v[2], 0)
        wz = cdiff(v[1], 0) - cdiff(v[0], 1)
        omega_sq = wx ** 2 + wy ** 2 + wz ** 2
    omega = jnp.sqrt(jnp.where(jnp.isnan(omega_sq), 0.0, omega_sq))
    cell_flag = (omega[:, 0] > threshold).astype(jnp.float32)
    from amr.blocks import cross_block_dilate
    cell_flag = cross_block_dilate(cell_flag, neighbors, cfg, buffer=buffer)
    flagged = refinement_block_mask(cell_flag, ndim)
    active = jnp.arange(data.shape[0]) < n_active
    return flagged & active


def flag_pressure_gradient(data, neighbors, n_active, cfg, threshold, buffer=None):
    """Flag on the RELATIVE pressure jump |dP|/P (per cell, summed over axes).

    Uses the internal-energy density e_int = E - 0.5*|mom|^2/rho (= P/(gamma-1));
    the (gamma-1) cancels in the ratio, so this is exactly |dP|/P -- dimensionless
    and scale-invariant. A smooth but high-pressure interior (e.g. a Sedov hot
    bubble) has a small fractional gradient and de-refines; sharp fronts -- the
    shock, and the t=0 injection spike (uniform rho, peaked pressure) -- have an
    O(1) fractional jump and flag. (The earlier absolute |dP| flagged the whole
    bubble because e_int is large there.) ``threshold`` is now a dimensionless
    fraction (~0.2-0.5). Ghost-filled so edge cells see across the block boundary;
    cross-block dilation adds a margin. Returns (n_slots,) bool.
    """
    ndim = cfg.ndim
    g = cfg.n_ghost
    bs = cfg.block_shape
    rho = data[:, 0:1]                                  # (n_slots, 1, *bs)
    mom = data[:, 1:1 + ndim]                           # (n_slots, ndim, *bs)
    E = data[:, -1:]                                    # total energy (last var)
    eint = E - 0.5 * jnp.sum(mom * mom, axis=1, keepdims=True) / rho   # ~ P/(gamma-1)
    gh = fill_ghost_same_level(eint, neighbors, cfg)    # (n_slots, 1, *(bs+2g))

    grad = jnp.zeros((data.shape[0], 1) + tuple(bs))
    for d in range(ndim):
        sl_hi = [slice(None), slice(None)] + [slice(g, g + bs[k]) for k in range(ndim)]
        sl_lo = list(sl_hi)
        sl_hi[2 + d] = slice(g + 1, g + 1 + bs[d])
        sl_lo[2 + d] = slice(g - 1, g - 1 + bs[d])
        hi = gh[tuple(sl_hi)]
        lo = gh[tuple(sl_lo)]
        diff = hi - lo
        # RELATIVE jump |dP|/P (dimensionless, scale-invariant): a smooth but
        # high-pressure interior (e.g. a Sedov hot bubble) has a small fractional
        # gradient and de-refines, while sharp fronts -- the shock and the t=0
        # injection spike -- have an O(1) fractional jump and flag. The absolute
        # |dP| used before flagged the whole bubble because eint is large there.
        rel = jnp.abs(diff) / (jnp.abs(hi) + jnp.abs(lo) + 1e-30)
        grad = grad + jnp.where(jnp.isnan(diff), 0.0, rel)

    cell_flag = (grad[:, 0] > threshold).astype(jnp.float32)
    from amr.blocks import cross_block_dilate
    cell_flag = cross_block_dilate(cell_flag, neighbors, cfg, buffer=buffer)
    flagged = refinement_block_mask(cell_flag, ndim)
    active = jnp.arange(data.shape[0]) < n_active
    return flagged & active


def flag_pressure_density(data, neighbors, n_active, cfg, threshold, buffer=None,
                          rho_ref=None, overdensity=1.2):
    """Combined flag: refine where the relative pressure jump |dP|/P exceeds
    ``threshold``, OR the relative density jump |drho|/rho does AND the gas is
    overdense (rho > overdensity * rho_ref).

    The pressure term locks onto the shock (and the t=0 injection spike, where
    density is still uniform). The density term adds the compressed shell body --
    but gated by overdensity, because relative gradients are evaluated at each
    level's OWN dx: a blast interior spanning decades of density gives O(0.3)
    per-cell jumps on coarse levels even where the profile is smooth, which
    un-gated would cascade refinement across the whole bubble. The low-density
    interior (rho << rho_ref) drops out of the gate at every level, so the
    refinement hugs the shell. ``rho_ref`` defaults to the mean of this level's
    own active cells; pass the level-0 domain mean for a stable reference.
    """
    ndim = cfg.ndim
    g = cfg.n_ghost
    bs = cfg.block_shape
    rho = data[:, 0:1]
    mom = data[:, 1:1 + ndim]
    E = data[:, -1:]
    eint = E - 0.5 * jnp.sum(mom * mom, axis=1, keepdims=True) / rho    # = P/(gamma-1)
    if rho_ref is None:
        rho_ref = jnp.nanmean(rho)                       # sentinel slots are NaN

    def relgrad_flag(field):
        gh = fill_ghost_same_level(field, neighbors, cfg)
        grad = jnp.zeros((data.shape[0], 1) + tuple(bs))
        for d in range(ndim):
            sl_hi = [slice(None), slice(None)] + [slice(g, g + bs[k]) for k in range(ndim)]
            sl_lo = list(sl_hi)
            sl_hi[2 + d] = slice(g + 1, g + 1 + bs[d])
            sl_lo[2 + d] = slice(g - 1, g - 1 + bs[d])
            hi = gh[tuple(sl_hi)]
            lo = gh[tuple(sl_lo)]
            diff = hi - lo
            rel = jnp.abs(diff) / (jnp.abs(hi) + jnp.abs(lo) + 1e-30)
            grad = grad + jnp.where(jnp.isnan(diff), 0.0, rel)
        return grad[:, 0] > threshold

    # density term gated by overdensity (NaN sentinel cells compare False)
    dense = rho[:, 0] > overdensity * rho_ref
    cell_flag = (relgrad_flag(eint) | (relgrad_flag(rho) & dense)).astype(jnp.float32)
    from amr.blocks import cross_block_dilate
    cell_flag = cross_block_dilate(cell_flag, neighbors, cfg, buffer=buffer)
    flagged = refinement_block_mask(cell_flag, ndim)
    active = jnp.arange(data.shape[0]) < n_active
    return flagged & active


def _flag_blocks(parent, cfg, threshold, level, criterion, rho_ref=None):
    """Dispatch to the chosen refinement criterion (used by build_hierarchy/regrid).
    ``rho_ref`` (level-0 domain mean density) feeds the overdensity gate of the
    combined pressure+density criterion."""
    if criterion == "vorticity":
        return flag_vorticity(parent.data, parent.neighbors, parent.n_active, cfg, threshold, level)
    if criterion == "pressure":
        return flag_pressure_gradient(parent.data, parent.neighbors, parent.n_active, cfg, threshold)
    if criterion in ("pressure_density", "prho"):
        return flag_pressure_density(parent.data, parent.neighbors, parent.n_active, cfg, threshold,
                                     rho_ref=rho_ref)
    return flag_density_gradient(parent.data, parent.neighbors, parent.n_active, cfg, threshold)


# jitted flag dispatch (eager: ~250 ms/regrid of tiny-op dispatch); threshold and
# rho_ref stay traced so tuning them never retriggers compilation.
_flag_blocks = jax.jit(_flag_blocks, static_argnames=("cfg", "level", "criterion"))


# ---------------------------------------------------------------------------
# Build a (statically refined) hierarchy
# ---------------------------------------------------------------------------
def _pad_blocks(data, coords, pad_to):
    """Pad the block axis of (data, coords) up to a multiple of ``pad_to`` (the
    device count) so the level can be evenly sharded. Padding slots are inactive
    (coords -1, NaN data). Topology (neighbours / parent slots) is computed
    *after* this on the padded coords, so the NaN sentinel lands at the new last
    slot and every 'missing' gather resolves to it consistently."""
    n = data.shape[0]
    if pad_to <= 1 or n % pad_to == 0:
        return data, coords
    p = ((n + pad_to - 1) // pad_to) * pad_to - n
    data = jnp.concatenate([data, jnp.full((p,) + data.shape[1:], jnp.nan, data.dtype)])
    coords = jnp.concatenate([coords, -jnp.ones((p, coords.shape[1]), coords.dtype)])
    return data, coords


def _sized_parents(flags, cpb, bucket, floor=0):
    """Number of parent slots to allocate for a refined level: the exact flagged
    count (+1 for sentinel headroom). With ``bucket`` set, rounded UP onto the
    geometric ladder ``bucket * 2**k`` and clamped to at least ``floor`` (the
    level's current allocation, in parent units). Geometric growth + never-shrink
    means the jitted ``advance`` recompiles at most ~log2(peak/bucket) times over
    a dynamic run (a linear bucket recompiled on every ``bucket``-crossing --
    dozens of multi-second XLA compiles for a growing Sedov shell) and never
    recompiles on shrink. Padding waste is bounded at 2x the active count.
    ``bucket`` in (None, 0, 1) means exact (no rounding)."""
    n = int(jnp.sum(flags)) + 1
    if bucket and bucket > 1:
        cap = int(bucket)
        while cap < max(n, floor):
            cap *= 2
        return cap
    return max(n, int(floor))


def build_hierarchy(U0, cfg, *, thresholds, capacities=None, periodic=True, bucket=None,
                    pad_to=1, criterion="density"):
    """Construct the level hierarchy from a uniform field.

    thresholds : sequence length ``max_levels`` — density-gradient flag threshold
        to go from level L to L+1.
    capacities : sequence length ``max_levels`` or None — max child blocks per
        refined level (static shapes). None (default) sizes each level to the
        EXACT flagged count (zero NaN padding); set it only to reserve headroom.
    """
    base = build_base_level(U0, cfg)
    # pad coords+data BEFORE topology so the NaN sentinel lands at the padded
    # last slot and find_* resolve every 'missing' to it consistently.
    bdata, bcoords = _pad_blocks(base.data, base.coords, pad_to)
    base_nbrs = _neighbors_j(bcoords, base.n_active, 0, cfg, periodic=periodic)
    levels = [Level(
        data=bdata, coords=bcoords, neighbors=base_nbrs,
        parent_slots=-jnp.ones(bdata.shape[0], jnp.int32), n_active=base.n_active,
    )]

    cpb = cfg.children_per_block
    rho0 = jnp.nanmean(bdata[:, 0])                     # level-0 domain mean density
    for L in range(cfg.max_levels):
        parent = levels[L]
        flags = _flag_blocks(parent, cfg, thresholds[L], L, criterion, rho_ref=rho0)
        if capacities is not None:
            max_parents = capacities[L] // cpb
        else:
            # build runs eagerly (not under jit) -> size to the flagged count
            # (exact, or bucketed for shape stability under dynamic regridding).
            max_parents = _sized_parents(flags, cpb, bucket)
        cc, cd, na = _children_j(parent.coords, parent.data, flags, level=L, cfg=cfg,
                                 max_parents=max_parents)
        cd, cc = _pad_blocks(cd, cc, pad_to)
        nbrs = _neighbors_j(cc, na, L + 1, cfg, periodic=False)
        pslots = _parent_slots_j(cc, parent.coords, parent.n_active, cfg)
        levels.append(Level(data=cd, coords=cc, neighbors=nbrs, parent_slots=pslots, n_active=na))

    return tuple(levels)


def regrid(levels, cfg, *, thresholds, periodic=True, bucket=None, pad_to=1, criterion="density"):
    """Re-evaluate the refinement criterion on the CURRENT solution and rebuild
    every refined level — the adaptive heart of AMR.

    For each level, the same density-gradient flag (`flag_density_gradient`) is
    re-run on the *live* data: features that have sharpened get newly flagged,
    regions that have smoothed drop out. Blocks that persist across the regrid
    carry their evolved data forward (`match_blocks`/`preserve_unaltered`);
    newly-refined blocks get their parent's prolongation (from `build_children`);
    de-refined blocks are dropped — no information lost because `advance` already
    restricted fine results into the coarse level every step.

    Runs eagerly (block counts change, so the jitted `advance` recompiles for the
    new shapes — amortized over the regrid interval). Returns a new hierarchy.
    """
    new = [levels[0]]                                  # level 0 always fully tiles
    rho0 = jnp.nanmean(levels[0].data[:, 0])           # level-0 domain mean density
    for L in range(cfg.max_levels):
        parent = new[L]
        old_child = levels[L + 1]
        flags = _flag_blocks(parent, cfg, thresholds[L], L, criterion, rho_ref=rho0)
        # never-shrink floor: the level's current allocation in parent units
        # (build_children allocates max_parents * children_per_block slots), so
        # regrids only ever grow capacities -> no recompile thrash on shrink.
        prev_parents = old_child.data.shape[0] // cfg.children_per_block
        max_parents = _sized_parents(flags, cfg.children_per_block, bucket,
                                     floor=prev_parents)
        cc, cd, na = _children_j(parent.coords, parent.data, flags, level=L, cfg=cfg,
                                 max_parents=max_parents)
        old_slot = _match_blocks_j(cc, na, old_child.coords, old_child.n_active, cfg.ndim)
        cd = _preserve_j(cd, old_slot, old_child.data)
        cd, cc = _pad_blocks(cd, cc, pad_to)
        nbrs = _neighbors_j(cc, na, L + 1, cfg, periodic=False)
        pslots = _parent_slots_j(cc, parent.coords, parent.n_active, cfg)
        new.append(Level(data=cd, coords=cc, neighbors=nbrs, parent_slots=pslots, n_active=na))
    return tuple(new)


# ---------------------------------------------------------------------------
# Ghost fill + advance (jittable: topology static)
# ---------------------------------------------------------------------------
def _fill_ghosts(levels, cfg, gamma):
    """Per-level ghost-filled PRIMITIVE blocks, coarse->fine. Each level fills its
    same-level face+corner halo, finer levels add the sparse coarse-parent
    underlay, and a physical BC fills any remaining domain-edge NaN ghosts (a
    no-op for periodic runs, whose same-level halo wraps)."""
    ghosted = []
    for L, lvl in enumerate(levels):
        W = cons_to_prim_blocks(lvl.data, gamma)
        gw = fill_ghost_same_level(W, lvl.neighbors, cfg)
        if L > 0:
            gw = fill_ghost_from_coarse_sparse(gw, lvl.coords, lvl.parent_slots, ghosted[L - 1], cfg)
        gw = fill_ghost_physical(gw, W, cfg)
        ghosted.append(gw)
    return ghosted


def _restrict_into_parent(coarse_data, fine, cfg):
    """Overwrite covered coarse cells with restrict(fine), fine authoritative.

    Single parallel scatter: every fine block restricts to ``(n_vars, *h)`` and
    writes a disjoint quadrant of its parent coarse block (children of one parent
    occupy distinct quadrants; different parents are distinct slots), so there are
    no write conflicts. Inactive slots are redirected to the NaN sentinel slot
    (harmless trash). Replaces the old serial per-slot ``scan`` -> O(1) launches.
    """
    ndim = cfg.ndim
    h = tuple(s // 2 for s in cfg.block_shape)
    n_coarse, n_vars = coarse_data.shape[0], coarse_data.shape[1]
    n_fine = fine.data.shape[0]
    sentinel = n_coarse - 1

    restricted = jax.vmap(lambda d: restrict(d, ndim))(fine.data)   # (n_fine, n_vars, *h)
    valid = fine.coords[:, 0] >= 0
    pslot = jnp.where(valid, fine.parent_slots, sentinel).astype(jnp.int32)
    off = (jnp.maximum(fine.coords, 0) % 2) * jnp.asarray(h, jnp.int32)   # (n_fine, ndim)

    # advanced-index arrays broadcasting to (n_fine, n_vars, *h)
    idx = [pslot.reshape((n_fine,) + (1,) * (ndim + 1)),
           jnp.arange(n_vars).reshape((1, n_vars) + (1,) * ndim)]
    for d in range(ndim):
        a = jnp.arange(h[d]).reshape((1, 1) + tuple(h[d] if k == d else 1 for k in range(ndim)))
        idx.append(off[:, d].reshape((n_fine, 1) + (1,) * ndim) + a)
    return coarse_data.at[tuple(idx)].set(restricted)


def advance(levels, dt, cfg, *, gamma, limiter=1, riemann_solver="hllc"):
    """One synchronized same-dt step over the whole hierarchy."""
    ghosted = _fill_ghosts(levels, cfg, gamma)

    # advance each level (conservative same-level seams; C/F faces ghost-based)
    new_data = []
    for L, lvl in enumerate(levels):
        dU = conservative_block_divergence(
            ghosted[L], dt, L, cfg, lvl.neighbors,
            gamma=gamma, limiter=limiter, riemann_solver=riemann_solver,
        )
        new_data.append(lvl.data + dt * dU)

    # restrict fine -> coarse, finest first
    for L in range(len(levels) - 1, 0, -1):
        fine = levels[L]._replace(data=new_data[L])
        new_data[L - 1] = _restrict_into_parent(new_data[L - 1], fine, cfg)

    return tuple(lvl._replace(data=nd) for lvl, nd in zip(levels, new_data))


# ---------------------------------------------------------------------------
# Subcycled (Berger-Oliger) time stepping: the AMR time-win
# ---------------------------------------------------------------------------
def advance_subcycle(levels, dt_base, cfg, *, gamma, limiter=1, riemann_solver="hllc",
                     divergence_fn=None):
    """Berger-Oliger subcycled step (ratio r=2): level L advances by
    ``dt_base / r**L``. The coarse level steps once; each finer level then takes
    ``r`` sub-steps of half the dt, its coarse-fine ghosts gathered from the
    parent level **fully ghosted** (same-level + the parent's own coarse fill) and
    linearly interpolated in time between the parent's pre- and post-step states.
    After the sub-steps the fine level is restricted back into the parent.

    This is the point of AMR in *time*: coarse cells (most of the domain) step
    only once per base step instead of paying the finest timestep everywhere
    (the synchronized :func:`advance`), so total work tracks the refined region.
    Recursion depth is the static ``max_levels`` -> the whole thing jits. No flux
    reflux yet (same coarse-fine conservation level as :func:`advance`).
    """
    r = cfg.refinement_ratio
    lv = list(levels)

    if divergence_fn is None:
        def divergence_fn(gw, neighbors, level, dt_l):
            return conservative_block_divergence(
                gw, dt_l, level, cfg, neighbors,
                gamma=gamma, limiter=limiter, riemann_solver=riemann_solver)

    def rec(L, dt, parent_gprim):
        lvl = lv[L]

        def ghost_prim(data):
            """Level-L fully-ghosted primitives: same-level halo + (L>0) the
            coarse-fine underlay from the fully-ghosted parent (so a grandchild
            never gathers an unfilled parent edge ghost)."""
            gw = fill_ghost_same_level(cons_to_prim_blocks(data, gamma), lvl.neighbors, cfg)
            if L > 0:
                gw = fill_ghost_from_coarse_sparse(gw, lvl.coords, lvl.parent_slots, parent_gprim, cfg)
            return gw

        L_old = lvl.data
        dU = divergence_fn(ghost_prim(L_old), lvl.neighbors, L, dt)
        lv[L] = lvl._replace(data=L_old + dt * dU)

        if L + 1 < len(lv):
            L_new = lv[L].data
            for s in range(r):
                c = (1.0 - s / r) * L_old + (s / r) * L_new      # parent at sub-step start
                rec(L + 1, dt / r, ghost_prim(c))                # pass FULLY-ghosted parent
            lv[L] = lv[L]._replace(data=_restrict_into_parent(lv[L].data, lv[L + 1], cfg))

    rec(0, dt_base, None)                                        # level 0 has no parent
    return tuple(lv)


def compute_dt_subcycle(levels, cfg, *, gamma, CFL):
    """Base (level-0) timestep for subcycling: the largest dt_base such that
    ``dt_base / r**L`` still satisfies the CFL limit on every level L. Level L
    then sub-steps with ``dt_base / r**L``."""
    ndim = cfg.ndim
    r = cfg.refinement_ratio
    dt_base = jnp.inf
    for L, lvl in enumerate(levels):
        W = cons_to_prim_blocks(lvl.data, gamma)
        rho, P = W[:, 0], W[:, -1]
        cs = jnp.sqrt(gamma * P / rho)
        dx = cfg.level_dx(L)
        local = jnp.inf
        for d in range(ndim):
            local = jnp.minimum(local, dx[d] / (jnp.abs(W[:, 1 + d]) + cs))
        cand = (r ** L) * CFL * jnp.nanmin(local)           # scale level-L CFL dt to base
        dt_base = jnp.minimum(dt_base, jnp.where(jnp.isnan(cand), jnp.inf, cand))
    return dt_base


# ---------------------------------------------------------------------------
# CFL timestep over the hierarchy
# ---------------------------------------------------------------------------
def compute_dt(levels, cfg, *, gamma, CFL):
    ndim = cfg.ndim
    dt = jnp.inf
    for L, lvl in enumerate(levels):
        W = cons_to_prim_blocks(lvl.data, gamma)
        rho = W[:, 0]
        P = W[:, -1]
        cs = jnp.sqrt(gamma * P / rho)
        dx = cfg.level_dx(L)
        # per-axis CFL: min_d dx_d / (|v_d| + cs)
        local = jnp.inf
        for d in range(ndim):
            local = jnp.minimum(local, dx[d] / (jnp.abs(W[:, 1 + d]) + cs))
        cand = CFL * jnp.nanmin(local)                  # nan if the level is empty
        dt = jnp.minimum(dt, jnp.where(jnp.isnan(cand), jnp.inf, cand))
    return dt


# ---------------------------------------------------------------------------
# Composite the hierarchy to a dense finest-resolution field (for output)
# ---------------------------------------------------------------------------
def composite_to_dense(levels, cfg):
    """Build a dense field at the finest resolution: each level's blocks are
    prolonged to the finest grid and scattered in, finer levels overwriting."""
    ndim = cfg.ndim
    bs = cfg.block_shape
    n_vars = levels[0].data.shape[1]
    finest = tuple(cfg.base_grid[d] * (cfg.refinement_ratio ** cfg.max_levels) for d in range(ndim))

    img = jnp.full((n_vars, *finest), jnp.nan)
    for L, lvl in enumerate(levels):
        f = cfg.refinement_ratio ** (cfg.max_levels - L)     # block prolong factor
        cell = bs[0] * f                                      # finest cells per block edge

        def body(im, i):
            c = lvl.coords[i]
            valid = c[0] >= 0
            blk = lvl.data[i]                                 # (n_vars, *bs)
            up = blk
            for _ in range(cfg.max_levels - L):
                up = prolong(up, ndim)                        # -> (n_vars, *(bs*f))
            start = [jnp.asarray(0, jnp.int32)] + [
                jnp.asarray(jnp.maximum(c[d], 0) * bs[d] * f, jnp.int32) for d in range(ndim)
            ]
            # Keep the already-composited (coarser) underlay wherever this finer
            # block is NaN: a de-refined block's sentinel / invalid cells must not
            # clobber good coarse data (L0 always tiles the whole domain, so the
            # composite should never have gaps).
            cur = jax.lax.dynamic_slice(im, start, up.shape)
            up = jnp.where(jnp.isnan(up), cur, up)
            placed = jax.lax.dynamic_update_slice(im, up, start)
            return jnp.where(valid, placed, im), None

        img, _ = jax.lax.scan(body, img, jnp.arange(lvl.coords.shape[0]))
    return img
