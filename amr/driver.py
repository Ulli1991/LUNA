"""Multi-level block-AMR hydro driver (dimension-generic 2D/3D).

Level 0 fully tiles the domain; levels >= 1 are sparse refined blocks.
:func:`advance` steps every level with the same dt; :func:`advance_subcycle`
is Berger-Oliger subcycling (r=2). Topology is fixed between regrids, so both
advances jit; :func:`regrid` re-flags on the live solution eagerly. Sharded
driver: :mod:`amr.shard`; gravity/particles plug in as per-level hooks.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from amr.blocks import (
    build_children,
    cross_block_dilate,
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
def _pm_slices(d, g, bs, ndim):
    """(hi, lo, center) index tuples for a central difference along spatial axis
    ``d`` of a ghosted ``(n_slots, n_vars, *(bs+2g))`` block array."""
    c = [slice(None), slice(None)] + [slice(g, g + bs[k]) for k in range(ndim)]
    hi = list(c)
    lo = list(c)
    hi[2 + d] = slice(g + 1, g + 1 + bs[d])
    lo[2 + d] = slice(g - 1, g - 1 + bs[d])
    return tuple(hi), tuple(lo), tuple(c)


def _finish_flags(cell_flag, neighbors, n_active, cfg, buffer):
    """Shared tail of every flag_* criterion: cross-block buffer dilation,
    any-cell-flagged block mask, gate to active slots. Returns (n_slots,) bool."""
    cell_flag = cross_block_dilate(cell_flag, neighbors, cfg, buffer=buffer)
    flagged = refinement_block_mask(cell_flag, cfg.ndim)
    active = jnp.arange(cell_flag.shape[0]) < n_active
    return flagged & active


def _relgrad_flag(field, neighbors, cfg, threshold):
    """Per-cell relative central jump |du| / (|u_hi| + |u_lo|), summed over axes
    on the ghost-filled field; True where the sum exceeds ``threshold``."""
    ndim, g, bs = cfg.ndim, cfg.n_ghost, cfg.block_shape
    gh = fill_ghost_same_level(field, neighbors, cfg)
    grad = jnp.zeros((field.shape[0], 1) + tuple(bs))
    for d in range(ndim):
        sl_hi, sl_lo, _ = _pm_slices(d, g, bs, ndim)
        hi = gh[sl_hi]
        lo = gh[sl_lo]
        diff = hi - lo
        rel = jnp.abs(diff) / (jnp.abs(hi) + jnp.abs(lo) + 1e-30)
        grad = grad + jnp.where(jnp.isnan(diff), 0.0, rel)
    return grad[:, 0] > threshold


def flag_density_gradient(data, neighbors, n_active, cfg, threshold, buffer=None):
    """Flag blocks whose ghost-filled density-gradient magnitude exceeds
    ``threshold`` anywhere; buffer-dilated. Returns (n_slots,) bool."""
    ndim = cfg.ndim
    g = cfg.n_ghost
    bs = cfg.block_shape
    rho = data[:, 0:1]                                 # (n_slots, 1, *bs)
    gh = fill_ghost_same_level(rho, neighbors, cfg)    # (n_slots, 1, *(bs+2g))

    grad = jnp.zeros((data.shape[0], 1) + tuple(bs))
    for d in range(ndim):
        sl_hi, sl_lo, _ = _pm_slices(d, g, bs, ndim)
        diff = gh[sl_hi] - gh[sl_lo]
        grad = grad + jnp.where(jnp.isnan(diff), 0.0, jnp.abs(diff))

    cell_flag = (grad[:, 0] > threshold).astype(jnp.float32)   # (n_slots, *bs)
    return _finish_flags(cell_flag, neighbors, n_active, cfg, buffer)


def flag_vorticity(data, neighbors, n_active, cfg, threshold, level, buffer=None):
    """Flag blocks where the physical vorticity magnitude |curl v| (2D: the
    scalar out-of-plane curl) exceeds ``threshold``. The /dx makes it a true
    rotation rate, hence resolution-INDEPENDENT — KH keeps refining from any
    base resolution. Ghost-filled velocity; returns (n_slots,) bool."""
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
        hi, lo, _ = _pm_slices(d, g, bs, ndim)
        return (arr[hi] - arr[lo]) / (2.0 * dx[d])

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
    return _finish_flags(cell_flag, neighbors, n_active, cfg, buffer)


def flag_pressure_gradient(data, neighbors, n_active, cfg, threshold, buffer=None):
    """Flag on the RELATIVE pressure jump |dP|/P (via e_int; the (gamma-1)
    cancels) -- dimensionless and scale-invariant, so smooth hot interiors
    (e.g. a Sedov bubble) de-refine while sharp fronts have an O(1) jump.
    ``threshold`` is a fraction (~0.2-0.5). Returns (n_slots,) bool."""
    ndim = cfg.ndim
    rho = data[:, 0:1]                                  # (n_slots, 1, *bs)
    mom = data[:, 1:1 + ndim]                           # (n_slots, ndim, *bs)
    E = data[:, -1:]                                    # total energy (last var)
    eint = E - 0.5 * jnp.sum(mom * mom, axis=1, keepdims=True) / rho   # ~ P/(gamma-1)
    cell_flag = _relgrad_flag(eint, neighbors, cfg, threshold).astype(jnp.float32)
    return _finish_flags(cell_flag, neighbors, n_active, cfg, buffer)


def flag_pressure_density(data, neighbors, n_active, cfg, threshold, buffer=None,
                          rho_ref=None, overdensity=1.2):
    """Combined flag: relative pressure jump |dP|/P > ``threshold`` (locks onto
    the shock) OR relative density jump in OVERDENSE gas (rho > overdensity *
    rho_ref; the gate keeps smooth blast interiors from cascading refinement, so
    it hugs the shell). ``rho_ref`` defaults to this level's active-cell mean;
    pass the level-0 domain mean for a stable reference."""
    ndim = cfg.ndim
    rho = data[:, 0:1]
    mom = data[:, 1:1 + ndim]
    E = data[:, -1:]
    eint = E - 0.5 * jnp.sum(mom * mom, axis=1, keepdims=True) / rho    # = P/(gamma-1)
    if rho_ref is None:
        rho_ref = jnp.nanmean(rho)                       # sentinel slots are NaN

    # density term gated by overdensity (NaN sentinel cells compare False)
    dense = rho[:, 0] > overdensity * rho_ref
    cell_flag = (_relgrad_flag(eint, neighbors, cfg, threshold)
                 | (_relgrad_flag(rho, neighbors, cfg, threshold) & dense)
                 ).astype(jnp.float32)
    return _finish_flags(cell_flag, neighbors, n_active, cfg, buffer)


def flag_overdensity(data, neighbors, n_active, cfg, threshold, buffer=None,
                     rho_ref=None):
    """Quasi-Lagrangian (RAMSES constant-mass-per-cell) criterion: flag cells
    with rho > threshold * rho_ref; per-level thresholds ~ 2**(ndim*(L+1)) keep
    roughly constant mass per cell, tracking collapse. ``rho_ref`` is the
    level-0 domain mean (threshold = overdensity 1+delta); NaN never flags."""
    rho = data[:, 0:1]
    if rho_ref is None:
        rho_ref = jnp.nanmean(rho)
    cell_flag = (rho[:, 0] > threshold * rho_ref).astype(jnp.float32)
    return _finish_flags(cell_flag, neighbors, n_active, cfg, buffer)


def flag_jeans(data, neighbors, n_active, cfg, threshold, level, buffer=None):
    """Truelove (1997) refinement: flag cells that under-resolve the local Jeans
    length, ``lambda_J < N_J dx`` (``threshold`` = Jeans number N_J, typ. 4),
    rearranged sqrt-free as ``pi gamma (gamma-1) e_int < G rho^2 (N_J dx)^2``.
    gamma / G come from ``AMRConfig``; NaN sentinel cells never flag."""
    ndim = cfg.ndim
    rho = data[:, 0:1]
    mom = data[:, 1:1 + ndim]
    E = data[:, -1:]
    eint = E - 0.5 * jnp.sum(mom * mom, axis=1, keepdims=True) / rho
    dx = cfg.level_dx(level)[0]
    lhs = jnp.pi * cfg.gamma * (cfg.gamma - 1.0) * eint          # pi gamma P
    rhs = cfg.grav_G * rho * rho * (threshold * dx) ** 2         # G rho^2 (N_J dx)^2
    cell_flag = (lhs[:, 0] < rhs[:, 0]).astype(jnp.float32)      # NaN -> False
    return _finish_flags(cell_flag, neighbors, n_active, cfg, buffer)


def flag_lohner(data, neighbors, n_active, cfg, threshold, buffer=None):
    """Löhner (1987) normalized second-derivative estimator (RAMSES/FLASH/Enzo)
    on ghost-filled variable ``cfg.lohner_var`` (default density): per-axis
    |d2u| over the gradient sum plus an ``eps`` noise filter (``cfg.lohner_eps``,
    default 0.01), combined in quadrature. ~O(1) at sharp features, ->0 in smooth
    flow, so ``threshold`` is a resolution-INDEPENDENT fraction (typ. 0.2-0.8).
    A missing neighbour contributes 0. Returns (n_slots,) bool."""
    ndim = cfg.ndim
    g = cfg.n_ghost
    bs = cfg.block_shape
    var = int(getattr(cfg, "lohner_var", 0))
    eps = float(getattr(cfg, "lohner_eps", 0.01))
    u = data[:, var:var + 1]
    gh = fill_ghost_same_level(u, neighbors, cfg)          # (n_slots, 1, *(bs+2g))

    num2 = jnp.zeros((data.shape[0], 1) + tuple(bs))
    den2 = jnp.zeros((data.shape[0], 1) + tuple(bs))
    for d in range(ndim):
        hi, lo, c = _pm_slices(d, g, bs, ndim)
        uc = gh[c]; uh = gh[hi]; ul = gh[lo]
        num = uh - 2.0 * uc + ul
        den = (jnp.abs(uh - uc) + jnp.abs(uc - ul)
               + eps * (jnp.abs(uh) + 2.0 * jnp.abs(uc) + jnp.abs(ul)))
        num = jnp.where(jnp.isnan(num), 0.0, num)          # missing neighbour -> 0
        den = jnp.where(jnp.isnan(den), 0.0, den)
        num2 = num2 + num * num
        den2 = den2 + den * den
    E = jnp.sqrt(num2 / (den2 + 1e-30))

    cell_flag = (E[:, 0] > threshold).astype(jnp.float32)
    return _finish_flags(cell_flag, neighbors, n_active, cfg, buffer)


def flag_static(parent, cfg, level):
    """Static (geometric) refinement: flag blocks whose CENTER lies inside
    ``cfg.static_region`` = ``((lo, hi), ...)`` in box-fraction coords [0, 1]
    per axis (None -> central half on every axis). Solution-independent."""
    ndim = cfg.ndim
    coords = parent.coords
    region = cfg.static_region
    if region is None:
        region = tuple((0.25, 0.75) for _ in range(ndim))
    nblk = [cfg.base_blocks[d] * (cfg.refinement_ratio ** level) for d in range(ndim)]
    inside = jnp.ones(coords.shape[0], dtype=bool)
    for d in range(ndim):
        c = (jnp.maximum(coords[:, d], 0) + 0.5) / float(nblk[d])
        lo, hi = region[d]
        inside = inside & (c >= lo) & (c < hi)
    active = jnp.arange(coords.shape[0]) < parent.n_active
    return inside & active


def _nested_flags(flags, parent, cfg, level):
    """Enforce PROPER 2:1 nesting: a block on the interior edge of its level's
    patch may not host children — a flush child edge creates a level-skip C/F
    face with no reflux target, silently leaking mass."""
    if level == 0:
        return flags
    sentinel = parent.neighbors.shape[0] - 1
    ok = jnp.ones_like(flags)
    for d in range(cfg.ndim):
        hi_ok = parent.neighbors[:, 2 * d] != sentinel
        lo_ok = parent.neighbors[:, 2 * d + 1] != sentinel
        if cfg.boundary != 0:
            # non-periodic: a physical domain edge is exempt (no reflux exists
            # across it, flush is legal there). In PERIODIC runs coords 0/max
            # WRAP -- they are interior C/F faces, and exempting them re-creates
            # the level-skip face at the wrap (bug: periodic L2 Sod collapse).
            nb_d = ((cfg.base_grid[d] // cfg.block_shape[d])
                    * (cfg.refinement_ratio ** level))
            hi_ok = hi_ok | (parent.coords[:, d] == nb_d - 1)
            lo_ok = lo_ok | (parent.coords[:, d] == 0)
        ok = ok & hi_ok & lo_ok
    return flags & ok


def _flag_blocks(parent, cfg, threshold, level, criterion, rho_ref=None):
    """Dispatch to the chosen refinement criterion; flags are post-filtered by
    :func:`_nested_flags`. ``rho_ref`` (level-0 domain mean density) feeds the
    overdensity-gated criteria."""
    if criterion == "overdensity":
        flags = flag_overdensity(parent.data, parent.neighbors, parent.n_active,
                                 cfg, threshold, rho_ref=rho_ref)
    elif criterion == "vorticity":
        flags = flag_vorticity(parent.data, parent.neighbors, parent.n_active, cfg, threshold, level)
    elif criterion == "pressure":
        flags = flag_pressure_gradient(parent.data, parent.neighbors, parent.n_active, cfg, threshold)
    elif criterion in ("pressure_density", "prho"):
        flags = flag_pressure_density(parent.data, parent.neighbors, parent.n_active, cfg, threshold,
                                      rho_ref=rho_ref)
    elif criterion == "jeans":
        flags = flag_jeans(parent.data, parent.neighbors, parent.n_active, cfg, threshold, level)
    elif criterion == "lohner":
        flags = flag_lohner(parent.data, parent.neighbors, parent.n_active, cfg, threshold)
    elif criterion == "static":
        flags = flag_static(parent, cfg, level)
    else:
        flags = flag_density_gradient(parent.data, parent.neighbors, parent.n_active, cfg, threshold)
    return _nested_flags(flags, parent, cfg, level)


# jitted flag dispatch (eager: ~250 ms/regrid of tiny-op dispatch); threshold and
# rho_ref stay traced so tuning them never retriggers compilation.
_flag_blocks = jax.jit(_flag_blocks, static_argnames=("cfg", "level", "criterion"))


# ---------------------------------------------------------------------------
# Build a (statically refined) hierarchy
# ---------------------------------------------------------------------------
def _pad_blocks(data, coords, pad_to):
    """Pad the block axis up to a multiple of ``pad_to`` (device count) with
    inactive slots (coords -1, NaN data). Topology must be computed AFTER this
    so the NaN sentinel lands at the padded last slot."""
    n = data.shape[0]
    if pad_to <= 1 or n % pad_to == 0:
        return data, coords
    p = ((n + pad_to - 1) // pad_to) * pad_to - n
    data = jnp.concatenate([data, jnp.full((p,) + data.shape[1:], jnp.nan, data.dtype)])
    coords = jnp.concatenate([coords, -jnp.ones((p, coords.shape[1]), coords.dtype)])
    return data, coords


def _sized_parents(flags, cpb, bucket, floor=0, cap=0, shrink=True):
    """Parent-slot allocation for a refined level: exact flagged count (+1
    sentinel), or with ``bucket`` set rounded UP onto a pow2 ladder and clamped
    to ``floor`` (bounds jit recompiles). ``cap`` (parent units, 0 = unlimited)
    is a HARD ceiling: excess parents are simply not refined this regrid.
    ``shrink`` enables a hysteretic one-rung capacity drop when the need falls
    to <= floor/4 (env ``LUNA_AMR_NO_SHRINK=1`` disables)."""
    import os as _os
    n = int(jnp.sum(flags)) + 1
    if cap and cap > 0:
        floor = min(int(floor), int(cap))
    floor = int(floor)
    # pow2 ladder value of the current need (base 8 parents). The old scheme
    # used ``bucket`` as the ladder BASE, silently flooring every level at
    # bucket parents (256 default = 2048 child slots) -- measured 32x padding
    # in memory AND in the batched divergence, which computes over ALL slots.
    need = 8
    while need < n:
        need *= 2
    if shrink and floor > 0 and _os.environ.get("LUNA_AMR_NO_SHRINK", "0") != "1":
        if need * 4 <= floor:
            floor = max(floor // 2, need)
    size = need if (bucket and bucket > 1) else n
    size = max(size, floor)
    if cap and cap > 0:
        size = min(size, int(cap))
    return size


def _ghosted_parent(parent, cfg):
    """1-cell conserved halo of the parent level for ghost-aware child
    prolongation; a missing neighbour clamps to the block's own edge."""
    g, ndim, bs = cfg.n_ghost, cfg.ndim, cfg.block_shape
    gh = fill_ghost_same_level(parent.data, parent.neighbors, cfg)   # (n_slots, nv, *(bs+2g))
    edge = jnp.pad(parent.data, [(0, 0), (0, 0)] + [(g, g)] * ndim, mode="edge")
    gh = jnp.where(jnp.isnan(gh), edge, gh)
    sl = (slice(None), slice(None)) + tuple(slice(g - 1, bs[d] + g + 1) for d in range(ndim))
    return gh[sl]                                                    # (n_slots, nv, *(bs+2))


def _make_child_level(parent, flags, L, cfg, max_parents, pad_to, periodic,
                      old_child=None):
    """Construct level L+1 from the flagged ``parent`` blocks: ghost-aware
    prolongation (``build_children``), optionally carry evolved data of blocks
    that persist across a regrid (``old_child``; match/preserve), pad for even
    sharding, then topology (same-level neighbours + parent slots)."""
    cc, cd, na = _children_j(parent.coords, parent.data, flags, level=L, cfg=cfg,
                             max_parents=max_parents,
                             parent_ghosted=_ghosted_parent(parent, cfg))
    if old_child is not None:
        old_slot = _match_blocks_j(cc, na, old_child.coords, old_child.n_active, cfg.ndim)
        cd = _preserve_j(cd, old_slot, old_child.data)
    cd, cc = _pad_blocks(cd, cc, pad_to)
    nbrs = _neighbors_j(cc, na, L + 1, cfg, periodic=periodic)
    pslots = _parent_slots_j(cc, parent.coords, parent.n_active, cfg)
    return Level(data=cd, coords=cc, neighbors=nbrs, parent_slots=pslots, n_active=na)


def build_hierarchy(U0, cfg, *, thresholds, capacities=None, periodic=True, bucket=None,
                    pad_to=1, criterion="density", max_blocks=0):
    """Construct the level hierarchy from a uniform field. ``thresholds`` (len
    ``max_levels``) is the per-level flag threshold; ``capacities`` (optional,
    len ``max_levels``) reserves a fixed child-block count per refined level
    instead of the default exact (zero-padding) sizing."""
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
            max_parents = _sized_parents(flags, cpb, bucket,
                                         cap=max_blocks // cpb if max_blocks else 0)
        levels.append(_make_child_level(parent, flags, L, cfg, max_parents,
                                        pad_to, periodic))

    return tuple(levels)


def regrid(levels, cfg, *, thresholds, periodic=True, bucket=None, pad_to=1,
           criterion="density", max_blocks=0):
    """Re-flag on the CURRENT solution and rebuild every refined level — the
    adaptive heart of AMR. Persisting blocks carry their evolved data forward,
    new blocks get their parent's prolongation, de-refined blocks drop (already
    restricted into the coarse level every step). Runs eagerly (block counts
    change, so the jitted `advance` recompiles). Returns a new hierarchy."""
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
                                     floor=prev_parents,
                                     cap=max_blocks // cfg.children_per_block
                                     if max_blocks else 0)
        new.append(_make_child_level(parent, flags, L, cfg, max_parents,
                                     pad_to, periodic, old_child=old_child))
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
        # AD-safe conversion: cons_to_prim on NaN sentinel lanes has NaN
        # partials (1/rho, -m/rho^2) that poison reverse-mode even under zero
        # cotangents. Convert on sanitized data, then re-mark sentinel lanes
        # through a where (non-differentiable predicate -> the NaN constant
        # branch carries no gradient). Forward W is bit-identical.
        W = cons_to_prim_blocks(jnp.nan_to_num(lvl.data, nan=1.0), gamma)
        W = jnp.where(jnp.isnan(lvl.data[:, :1]), jnp.nan, W)
        gw = fill_ghost_same_level(W, lvl.neighbors, cfg)
        # physical DOMAIN bands first (zero-gradient of the FINE data, coords-
        # gated): the coarse underlay must never fill wall ghosts -- its
        # interpolation error near steep features injects a fake wall-normal
        # gradient (the levels>=2 conservation leak).
        gw = fill_ghost_physical(gw, W, cfg, coords=lvl.coords, level=L)
        if L > 0:
            gw = fill_ghost_from_coarse_sparse(gw, lvl.coords, lvl.parent_slots, ghosted[L - 1], cfg,
                                               interp="linear")
        gw = fill_ghost_physical(gw, W, cfg)
        # AD barrier: every ghost a real block reads is filled by now; only the
        # NaN-sentinel / padding lanes still hold NaN. Sanitize them (rho=v=P=1,
        # forward values of active blocks unchanged) so reverse-mode never sees
        # a NaN partial — 0-cotangent x NaN would poison the whole gradient
        # (upstream jaxamr does the same nan_to_num before its solver).
        gw = jnp.nan_to_num(gw, nan=1.0)
        ghosted.append(gw)
    return ghosted


def _restrict_into_parent(coarse_data, fine, cfg):
    """Overwrite covered coarse cells with restrict(fine), fine authoritative.
    Single parallel scatter; inactive slots are redirected to the NaN sentinel."""
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


def advance(levels, dt, cfg, *, gamma, limiter=1, riemann_solver="hllc",
            reflux=True):
    """One synchronized same-dt step over the whole hierarchy: per-level update
    (conservative same-level seams), Berger-Colella reflux at every coarse-fine
    interface (the coarse cells outside a fine patch see exactly the
    transverse-averaged fine fluxes -> conserved quantities telescope across
    levels to machine precision), then fine-authoritative restriction."""
    from amr.reflux import apply_reflux_blocks

    ghosted = _fill_ghosts(levels, cfg, gamma)

    new_data = []
    fluxes = []
    for L, lvl in enumerate(levels):
        dU, fl = conservative_block_divergence(
            ghosted[L], dt, L, cfg, lvl.neighbors,
            gamma=gamma, limiter=limiter, riemann_solver=riemann_solver,
            return_face_fluxes=True, coords=lvl.coords,
        )
        new_data.append(lvl.data + dt * dU)
        fluxes.append(fl)

    # reflux finest -> coarsest so corrections propagate down before restriction
    if reflux:
        for L in range(len(levels) - 1, 0, -1):
            new_data[L - 1] = apply_reflux_blocks(
                new_data[L - 1], fluxes[L - 1], fluxes[L],
                levels[L].coords, levels[L].parent_slots, levels[L].neighbors,
                levels[L].n_active, levels[L - 1].neighbors, dt, L - 1, cfg)

    # restrict fine -> coarse, finest first
    for L in range(len(levels) - 1, 0, -1):
        fine = levels[L]._replace(data=new_data[L])
        new_data[L - 1] = _restrict_into_parent(new_data[L - 1], fine, cfg)

    return tuple(lvl._replace(data=nd) for lvl, nd in zip(levels, new_data))


# ---------------------------------------------------------------------------
# Subcycled (Berger-Oliger) time stepping: the AMR time-win
# ---------------------------------------------------------------------------
def advance_subcycle(levels, dt_base, cfg, *, gamma, limiter=1, riemann_solver="hllc",
                     divergence_fn=None, gravity=None, reflux=True, t0=0.0,
                     reflux_fn=None, fill_fn=None):
    """Berger-Oliger subcycled step (r=2): level L advances by ``dt_base/r**L``,
    coarse-fine ghosts linearly time-interpolated between the parent's pre/post
    states, then fine restricted into the parent. Recursion depth is the static
    ``max_levels`` -> jits. ``gravity`` hook ``(L, lvl, parent_phig, dt, t_sub)
    -> (new_data, child_phig)`` runs per level per sub-step after the hydro
    update; ``reflux`` is Berger-Colella, time-accumulated over sub-steps."""
    r = cfg.refinement_ratio
    n = len(levels)
    if reflux and reflux_fn is None:
        # reflux_fn hook: the shard path substitutes a replicated-shard_map
        # wrapper here (GSPMD re-partitions the raw segment-sums under an
        # outer jit, silently dropping shard-remote flux contributions).
        from amr.reflux import apply_reflux_blocks
        reflux_fn = apply_reflux_blocks

    if divergence_fn is None:
        def divergence_fn(gw, neighbors, level, dt_l, want_flux=False):
            return conservative_block_divergence(
                gw, dt_l, level, cfg, neighbors,
                gamma=gamma, limiter=limiter, riemann_solver=riemann_solver,
                return_face_fluxes=want_flux,
                coords=levels[level].coords)

    def ghost_prim(L, data, parent_gprim):
        """Level-L fully-ghosted primitives: same-level halo + (L>0) coarse
        underlay from the fully-ghosted parent. Piecewise-constant C/F ghosts
        on purpose (subcycle C/F fluxes telescope exactly against restriction);
        ``fill_fn`` (shard path) must match this replicated fill bit-for-bit."""
        if fill_fn is not None:
            return fill_fn(L, data, parent_gprim)
        lvl = levels[L]
        # AD-safe conversion + post-fill barrier, mirroring _fill_ghosts:
        # sentinel lanes must never contribute NaN partials to reverse-mode.
        W = cons_to_prim_blocks(jnp.nan_to_num(data, nan=1.0), gamma)
        W = jnp.where(jnp.isnan(data[:, :1]), jnp.nan, W)
        gw = fill_ghost_same_level(W, lvl.neighbors, cfg)
        if L > 0:
            gw = fill_ghost_from_coarse_sparse(gw, lvl.coords, lvl.parent_slots,
                                               parent_gprim, cfg)
        gw = fill_ghost_physical(gw, W, cfg)   # domain-edge BC (no-op if periodic)
        return jnp.nan_to_num(gw, nan=1.0)

    def step_level(L, datas, dt, t_sub, parent_gprim, parent_phig):
        """Advance levels L..n-1 by one level-L step of ``dt``; returns
        (new datas, level-L flux). Pure in its carries so the r sub-steps run
        under ``lax.scan`` (traced graph O(n levels), not O(r**n))."""
        lvl = levels[L]
        L_old = datas[0]
        gp = ghost_prim(L, L_old, parent_gprim)
        if reflux:
            dU, my_flux = divergence_fn(gp, lvl.neighbors, L, dt, want_flux=True)
        else:
            dU = divergence_fn(gp, lvl.neighbors, L, dt)
            my_flux = jnp.zeros(())                              # scan-stable dummy
        data_L = L_old + dt * dU

        # --- per-level gravity: solve (parent-Dirichlet) + kick + floors ---
        if gravity is not None:
            data_L, my_phig = gravity(L, lvl._replace(data=data_L), parent_phig,
                                      dt, t_sub)
        else:
            my_phig = None

        if L + 1 < n:
            L_new = data_L

            def body(child_datas, s):
                w = s / r
                c = (1.0 - w) * L_old + w * L_new                # parent at sub-step start
                pg = ghost_prim(L, c, parent_gprim)              # FULLY-ghosted parent
                nds, cflux = step_level(L + 1, child_datas, dt / r,
                                        t_sub + s * (dt / r), pg, my_phig)
                return nds, cflux

            child_datas, cflux_stack = jax.lax.scan(
                body, datas[1:], jnp.arange(r, dtype=L_old.dtype))
            if reflux:
                child = levels[L + 1]
                acc = tuple(f.sum(axis=0) for f in cflux_stack)  # sum over sub-steps
                data_L = reflux_fn(
                    data_L, my_flux, tuple(a / r for a in acc),
                    child.coords, child.parent_slots, child.neighbors,
                    child.n_active, lvl.neighbors, dt, L, cfg)
            data_L = _restrict_into_parent(
                data_L, levels[L + 1]._replace(data=child_datas[0]), cfg)
            return (data_L,) + tuple(child_datas), my_flux
        return (data_L,), my_flux

    datas, _ = step_level(0, tuple(l.data for l in levels), dt_base, t0, None, None)
    return tuple(l._replace(data=d) for l, d in zip(levels, datas))


def floor_positivity(levels, cfg, *, gamma, rho_floor, p_floor):
    """Opt-in positivity floor: quiescent-reset any active unphysical cell
    (rho <= rho_floor, e_int <= 0, or non-finite) to ``(rho_floor, v=0,
    p_floor)``. Uses ``where`` (not ``max``) so NaN/inf cells are REPLACED, and
    never touches inactive/sentinel blocks. Off by default (rho_floor = 0)."""
    ndim = cfg.ndim
    out = []
    for lvl in levels:
        d = lvl.data
        rho = d[:, 0:1]
        mom = d[:, 1:1 + ndim]
        E = d[:, ndim + 1:ndim + 2]
        eint = E - 0.5 * jnp.sum(mom * mom, axis=1, keepdims=True) / rho
        good = (jnp.isfinite(rho) & jnp.isfinite(E) & (rho > rho_floor)
                & (eint > 0.0))
        active = (lvl.coords[:, 0] >= 0).reshape((-1,) + (1,) * (ndim + 1))
        bad = active & ~good                                # never touch sentinels
        rho_n = jnp.where(bad, rho_floor, rho)
        mom_n = jnp.where(bad, 0.0, mom)
        E_n = jnp.where(bad, p_floor / (gamma - 1.0), E)
        out.append(lvl._replace(data=jnp.concatenate([rho_n, mom_n, E_n], axis=1)))
    return tuple(out)


def floor_from_fraction(levels, cfg, *, gamma, frac):
    """Absolute ``(rho_floor, p_floor)`` = ``frac`` times the base-level mean
    state (``frac`` ~ 1e-8..1e-6). Call once at setup (returns Python floats);
    feed the result to :func:`floor_positivity`."""
    ndim = cfg.ndim
    d = levels[0].data
    rho = d[:, 0]
    mom = d[:, 1:1 + ndim]
    E = d[:, ndim + 1]
    eint = E - 0.5 * jnp.sum(mom * mom, axis=1) / rho
    p = (gamma - 1.0) * eint
    rho_mean = float(jnp.nanmean(rho))
    p_mean = float(jnp.nanmean(p))
    return frac * rho_mean, frac * max(p_mean, 0.0)


# ---------------------------------------------------------------------------
# CFL timestep over the hierarchy
# ---------------------------------------------------------------------------
def _compute_dt(levels, cfg, *, gamma, CFL, subcycle=False):
    """Per-axis CFL dt ``min_d dx_d / (|v_d| + cs)`` over all levels. With
    ``subcycle`` each level's dt is scaled by ``r**L`` back to a base dt (level
    L then sub-steps with ``dt_base / r**L``). Empty (all-NaN) levels are inf."""
    ndim = cfg.ndim
    r = cfg.refinement_ratio
    dt = jnp.inf
    for L, lvl in enumerate(levels):
        W = cons_to_prim_blocks(lvl.data, gamma)
        rho, P = W[:, 0], W[:, -1]
        cs = jnp.sqrt(gamma * P / rho)
        dx = cfg.level_dx(L)
        local = jnp.inf
        for d in range(ndim):
            local = jnp.minimum(local, dx[d] / (jnp.abs(W[:, 1 + d]) + cs))
        if subcycle:
            cand = (r ** L) * CFL * jnp.nanmin(local)   # scale level-L CFL dt to base
        else:
            cand = CFL * jnp.nanmin(local)              # nan if the level is empty
        dt = jnp.minimum(dt, jnp.where(jnp.isnan(cand), jnp.inf, cand))
    return dt


def compute_dt_subcycle(levels, cfg, *, gamma, CFL):
    """Base (level-0) timestep for subcycling: the largest dt_base such that
    ``dt_base / r**L`` still satisfies the CFL limit on every level L."""
    return _compute_dt(levels, cfg, gamma=gamma, CFL=CFL, subcycle=True)


def compute_dt(levels, cfg, *, gamma, CFL):
    """Synchronized CFL timestep: the strictest per-level CFL dt."""
    return _compute_dt(levels, cfg, gamma=gamma, CFL=CFL)


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
