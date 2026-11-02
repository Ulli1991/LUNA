"""Self-gravity on the block-AMR hierarchy: level-by-level Dirichlet multigrid.

Level 0 is solved densely by the validated FMG; each refined level solves on
its block union with Dirichlet ghosts prolonged from the parent phi, via
within-block V-cycles (RBGS smoother; homogeneous-Dirichlet error equation at
coarser depths). The kick mirrors sources/self_gravity, then fine results are
restricted into the parent. All static-shape / fixed-iteration -> jit-safe.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from luna.amr.geometry import prolong, restrict
from luna.amr.halo import (
    fill_ghost_from_coarse_sparse,
    fill_ghost_same_level,
    fill_level_ghosts,
)
from luna.amr.state import assemble_base_level, partition_to_blocks


def _interior(ndim, g):
    """Index tuple selecting the interior of a ghosted (n, 1, *(bs+2g)) array."""
    return (slice(None), slice(None)) + (slice(g, -g),) * ndim


def _shift(gw, d, off, ndim, g, bs):
    """Ghosted (n, 1, *(bs+2g)) -> interior-shaped slice offset by ``off`` along
    spatial axis ``d`` (reads one ghost layer at the faces)."""
    sl = [slice(None), slice(None)] + [slice(g, g + bs[k]) for k in range(ndim)]
    sl[2 + d] = slice(g + off, g + off + bs[d])
    return gw[tuple(sl)]


def _parity(coords, cfg):
    """(n_slots, 1, *bs) global checkerboard parity (0/1) of each cell, from the
    block's integer coords -- so red/black is consistent ACROSS blocks."""
    ndim = cfg.ndim
    bs = cfg.block_shape
    n = coords.shape[0]
    idx = jnp.zeros((n,) + tuple(bs), jnp.int32)
    for d in range(ndim):
        shape = [1] * (1 + ndim)
        shape[1 + d] = bs[d]
        io = jax.lax.iota(jnp.int32, bs[d]).reshape(shape)
        cd = jnp.maximum(coords[:, d], 0).astype(jnp.int32)
        idx = idx + io + cd.reshape((n,) + (1,) * ndim) * bs[d]
    return (idx % 2)[:, None]


def _depth_cfg(cfg, depth):
    """AMRConfig with block_shape halved ``depth`` times (same topology)."""
    if depth == 0:
        return cfg
    bs = tuple(b >> depth for b in cfg.block_shape)
    return dataclasses.replace(cfg, block_shape=bs)


def _ghost_phi(phi_blk, lvl, coarse_ghosted, cfg):
    """Scalar ghost fill: same-level faces/corners, then coarse-Dirichlet
    (parent phi prolonged LINEARLY -- piecewise-constant would put an
    O(dx_coarse) error straight into the Dirichlet values), then physical BC."""
    return fill_level_ghosts(phi_blk, lvl, cfg, parent=coarse_ghosted,
                             interp="linear")


def _ghost_zero(e_blk, neighbors, cfg_k):
    """Ghost fill for the ERROR equation: same-level exchange, every unfilled
    ghost (coarse-fine face, domain edge, missing neighbour) -> 0, the
    homogeneous Dirichlet of the correction scheme."""
    gw = fill_ghost_same_level(e_blk, neighbors, cfg_k)
    return jnp.where(jnp.isnan(gw), 0.0, gw)


def _lap(gw, inv_dx2, ndim, g, bs):
    """7-point Laplacian of a ghosted scalar, on the interior."""
    s = jnp.zeros_like(gw[_interior(ndim, g)])
    c = gw[_interior(ndim, g)]
    for d in range(ndim):
        s = s + (_shift(gw, d, +1, ndim, g, bs)
                 + _shift(gw, d, -1, ndim, g, bs) - 2.0 * c) * inv_dx2[d]
    return s


def _rbgs(phi, rhs, ghost_fn, parity, inv_dx2, ndim, g, bs, n_sweeps):
    """n_sweeps red-black Gauss-Seidel sweeps of lap(phi) = rhs; ``ghost_fn``
    supplies the boundary treatment (real Dirichlet at depth 0, homogeneous
    for the error equation at coarser depths)."""
    diag = 2.0 * sum(inv_dx2)

    def half(phi, color):
        gw = ghost_fn(phi)
        s = jnp.zeros_like(phi)
        for d in range(ndim):
            s = s + (_shift(gw, d, +1, ndim, g, bs)
                     + _shift(gw, d, -1, ndim, g, bs)) * inv_dx2[d]
        new = (s - rhs) / diag
        return jnp.where(parity == color, new, phi)

    def sweep(phi, _):
        return half(half(phi, 0), 1), None

    phi, _ = jax.lax.scan(sweep, phi, None, length=n_sweeps)
    return phi


def _error_vcycle(r, lvl, cfg, level, depth, max_depth, n_pre, n_post, n_coarse):
    """Recursive within-block V-cycle for the error equation lap(e) = r at
    ``depth`` (block_shape halved ``depth`` times), homogeneous Dirichlet."""
    cfg_k = _depth_cfg(cfg, depth)
    ndim, g, bs = cfg_k.ndim, cfg_k.n_ghost, cfg_k.block_shape
    dx = cfg.level_dx(level)
    scale = float(2 ** depth)
    inv_dx2 = [1.0 / (dx[d] * scale) ** 2 for d in range(ndim)]
    parity = _parity(lvl.coords, cfg_k)
    ghost_fn = lambda e: _ghost_zero(e, lvl.neighbors, cfg_k)

    e = jnp.zeros_like(r)
    if depth == max_depth:
        return _rbgs(e, r, ghost_fn, parity, inv_dx2, ndim, g, bs, n_coarse)

    e = _rbgs(e, r, ghost_fn, parity, inv_dx2, ndim, g, bs, n_pre)
    r2 = r - _lap(ghost_fn(e), inv_dx2, ndim, g, bs)
    r_c = jax.vmap(lambda b: restrict(b, ndim))(r2)
    e_c = _error_vcycle(r_c, lvl, cfg, level, depth + 1, max_depth,
                        n_pre, n_post, n_coarse)
    e = e + jax.vmap(lambda b: prolong(b, ndim))(e_c)
    return _rbgs(e, r, ghost_fn, parity, inv_dx2, ndim, g, bs, n_post)


def _vcycle_phi(phi, rhs, lvl, coarse_ghosted, cfg, level, n_pre, n_post, n_coarse):
    """One V-cycle on a refined level's phi with the REAL Dirichlet boundary."""
    ndim, g, bs = cfg.ndim, cfg.n_ghost, cfg.block_shape
    dx = cfg.level_dx(level)
    inv_dx2 = [1.0 / (dx[d] * dx[d]) for d in range(ndim)]
    parity = _parity(lvl.coords, cfg)
    ghost_fn = lambda p: _ghost_phi(p, lvl, coarse_ghosted, cfg)
    # coarsen while the shape stays even and >= 2 cells per dim
    max_depth = 0
    while all((b >> (max_depth + 1)) >= 2 and (b >> max_depth) % 2 == 0
              for b in bs):
        max_depth += 1

    phi = _rbgs(phi, rhs, ghost_fn, parity, inv_dx2, ndim, g, bs, n_pre)
    if max_depth > 0:
        r = rhs - _lap(ghost_fn(phi), inv_dx2, ndim, g, bs)
        r_c = jax.vmap(lambda b: restrict(b, ndim))(r)
        e_c = _error_vcycle(r_c, lvl, cfg, level, 1, max_depth,
                            n_pre, n_post, n_coarse)
        phi = phi + jax.vmap(lambda b: prolong(b, ndim))(e_c)
    return _rbgs(phi, rhs, ghost_fn, parity, inv_dx2, ndim, g, bs, n_post)


def _init_phi_from_parent(lvl, coarse_ghosted, cfg):
    """Initial guess for a refined level's phi: the parent's phi prolonged over
    the WHOLE block -- an all-NaN fine array run through the coarse underlay
    fill (which fills every NaN cell, interior included)."""
    ndim = cfg.ndim
    g = cfg.n_ghost
    bs = cfg.block_shape
    shape = (lvl.data.shape[0], 1) + tuple(s + 2 * g for s in bs)
    nan_blk = jnp.full(shape, jnp.nan, dtype=lvl.data.dtype)
    filled = fill_ghost_from_coarse_sparse(nan_blk, lvl.coords, lvl.parent_slots,
                                           coarse_ghosted, cfg, interp="linear")
    return filled[_interior(ndim, g)]


def solve_phi_hierarchy(levels, cfg, *, fourpiG, boundary, n_vcycles=3,
                        n_pre=4, n_post=4, n_coarse=12, fmg_kwargs=None,
                        rho_blocks=None):
    """Solve Poisson on the whole hierarchy (:func:`solve_phi_level` per level:
    L0 dense FMG, L>0 Dirichlet V-cycles from the parent's phi). Periodic
    domains use the Jeans-swindle source rho - <rho>, <rho> taken once from
    level 0. ``rho_blocks`` (optional, per level) overrides the gravitating
    density (e.g. gas + deposited DM). Returns per-level phi blocks
    ``(n_slots, 1, *block_shape)`` (sentinel slots NaN)."""
    ndim = cfg.ndim
    n_base = 1
    for d in range(ndim):
        n_base *= cfg.base_blocks[d]

    src0 = rho_blocks[0] if rho_blocks is not None else levels[0].data[:, 0:1]
    rho0 = assemble_base_level(src0[:n_base], cfg)[0]
    # Jeans-swindle: the low-level FMG does NOT subtract the mean (the dense
    # self-gravity caller does it) -- an un-subtracted periodic source is
    # insolvable and produces unbounded phi.
    mean_rho = jnp.mean(rho0) if boundary == 0 else 0.0
    phis = [solve_phi_level(levels[0], None, cfg, fourpiG=fourpiG,
                            boundary=boundary, mean_rho=mean_rho, level=0,
                            src=src0, fmg_kwargs=fmg_kwargs)]
    coarse_ghosted = _ghost_phi(phis[0], levels[0], None, cfg)
    for L in range(1, len(levels)):
        lvl = levels[L]
        srcL = rho_blocks[L] if rho_blocks is not None else None
        phis.append(solve_phi_level(lvl, coarse_ghosted, cfg, fourpiG=fourpiG,
                                    boundary=boundary, mean_rho=mean_rho,
                                    level=L, src=srcL, n_vcycles=n_vcycles,
                                    n_pre=n_pre, n_post=n_post,
                                    n_coarse=n_coarse))
        if L + 1 < len(levels):
            coarse_ghosted = _ghost_phi(phis[L], lvl, coarse_ghosted, cfg)
    return phis


def apply_gravity_kick(levels, phis, dt, cfg):
    """Operator-split gravity kick on every level (:func:`kick_level`); fine
    results are then restricted into the parent (fine authoritative), like the
    hydro advance."""
    from luna.amr.driver import _restrict_into_parent

    new_data = []
    coarse_ghosted = None
    for L, (lvl, phi) in enumerate(zip(levels, phis)):
        new_data.append(kick_level(lvl, phi, coarse_ghosted, dt, L, cfg))
        if L + 1 < len(levels):
            coarse_ghosted = _ghost_phi(phi, lvl, coarse_ghosted, cfg)

    # restrict fine -> coarse, finest first (fine authoritative)
    for L in range(len(levels) - 1, 0, -1):
        fine = levels[L]._replace(data=new_data[L])
        new_data[L - 1] = _restrict_into_parent(new_data[L - 1], fine, cfg)
    return tuple(lvl._replace(data=nd) for lvl, nd in zip(levels, new_data))


def solve_phi_level(lvl, coarse_ghosted, cfg, *, fourpiG, boundary, mean_rho,
                    level, src=None, n_vcycles=3, n_pre=4, n_post=4,
                    n_coarse=12, fmg_kwargs=None):
    """Solve Poisson for ONE level -- the subcycling primitive, and the per-level
    body of :func:`solve_phi_hierarchy`. Level 0: dense FMG on the assembled base
    grid; L>0: within-block V-cycles with Dirichlet ghosts from the parent's
    ghosted phi (``coarse_ghosted``). ``mean_rho`` is the Jeans-swindle
    subtraction (base-level mean; 0 for boundary != 0); ``src`` overrides the
    gravitating density (default: the level's own gas density)."""
    from luna.gravity.poisson_multigrid import solve_poisson_fmg

    ndim = cfg.ndim
    srcL = src if src is not None else lvl.data[:, 0:1]
    if level == 0:
        n_base = 1
        for d in range(ndim):
            n_base *= cfg.base_blocks[d]
        rho0 = assemble_base_level(srcL[:n_base], cfg)[0]
        dx0 = cfg.level_dx(0)[0]
        phi0 = solve_poisson_fmg(rho0 - mean_rho, dx0, fourpiG, boundary,
                                 **(fmg_kwargs or {}))
        phi0_blocks = partition_to_blocks(phi0[None], cfg)
        n_slots = lvl.data.shape[0]
        if n_slots > n_base:
            pad = jnp.full((n_slots - n_base,) + phi0_blocks.shape[1:], jnp.nan,
                           dtype=phi0_blocks.dtype)
            phi0_blocks = jnp.concatenate([phi0_blocks, pad], axis=0)
        return phi0_blocks
    phi = _init_phi_from_parent(lvl, coarse_ghosted, cfg)
    rhs = fourpiG * (srcL - mean_rho)
    for _ in range(n_vcycles):
        phi = _vcycle_phi(phi, rhs, lvl, coarse_ghosted, cfg, level,
                          n_pre, n_post, n_coarse)
    return phi


def kick_level(lvl, phi, coarse_ghosted, dt, level, cfg):
    """Operator-split gravity kick on ONE level (no restrict -- the subcycle
    recursion owns fine->coarse restriction): g = -central-diff(phi);
    m += dt*rho*g; E += dt * m_mid . g. Also the per-level body of
    :func:`apply_gravity_kick`; returns the kicked conserved data."""
    ndim = cfg.ndim
    g = cfg.n_ghost
    bs = cfg.block_shape
    gw = _ghost_phi(phi, lvl, coarse_ghosted, cfg)
    dx = cfg.level_dx(level)
    U = lvl.data
    rho = U[:, 0:1]
    mom = U[:, 1:1 + ndim]
    E = U[:, ndim + 1:ndim + 2]
    gvec = []
    for d in range(ndim):
        inv_2dx = 0.5 / dx[d]
        gvec.append(-(_shift(gw, d, +1, ndim, g, bs)
                      - _shift(gw, d, -1, ndim, g, bs)) * inv_2dx)
    gvec = jnp.concatenate(gvec, axis=1)
    new_mom = mom + dt * rho * gvec
    mom_mid = 0.5 * (mom + new_mom)
    new_E = E + dt * jnp.sum(mom_mid * gvec, axis=1, keepdims=True)
    return jnp.concatenate([rho, new_mom, new_E], axis=1)


def make_amr_self_gravity(cfg, acfg, *, is_main_process=True):
    """Runner-facing factory: returns ``apply(levels, dt) -> levels`` or None.

    Reads the same knobs as the dense path (cfg.SELF_GRAVITY, GRAVITATIONAL_G,
    BOUNDARY) plus ``cfg.amr_grav_vcycles`` (per-level V-cycles per solve,
    default 3)."""
    if not cfg.SELF_GRAVITY:
        return None
    fourpiG = 4.0 * jnp.pi * float(cfg.GRAVITATIONAL_G)
    boundary = int(cfg.BOUNDARY)
    n_vcycles = int(cfg.amr_grav_vcycles or 3)
    if is_main_process:
        print(f"  self-gravity (AMR): level-by-level Dirichlet MG, L0 FMG + "
              f"{n_vcycles} within-block V-cycles/level, boundary={boundary}")

    def apply(levels, dt):
        phis = solve_phi_hierarchy(levels, acfg, fourpiG=fourpiG,
                                   boundary=boundary, n_vcycles=n_vcycles)
        return apply_gravity_kick(levels, phis, dt, acfg)

    return apply
