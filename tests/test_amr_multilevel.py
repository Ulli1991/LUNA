"""Multi-level AMR driver: sparse coarse-parent gather, regridding, the driver
step, and a small Kelvin-Helmholtz smoke run."""

import jax
import jax.numpy as jnp
import numpy as np


from luna.amr.config import AMRConfig
from luna.amr.blocks import build_children, find_face_neighbors, find_parent_slots
from luna.amr.halo import (
    fill_ghost_from_coarse,
    fill_ghost_from_coarse_sparse,
    fill_ghost_same_level,
)
from luna.amr.state import build_base_level
from luna.hydro.conversions import prim_to_cons
from tests._helpers import check_conserved

GAMMA = 5.0 / 3.0


def smooth_W(cfg):
    ndim = cfg.ndim
    axes = [np.linspace(0.0, 2 * np.pi, cfg.base_grid[d], endpoint=False) for d in range(ndim)]
    g = np.meshgrid(*axes, indexing="ij")
    rho = 1.5 + 0.3 * np.sin(g[0]) * np.cos(g[1])
    P = 1.0 + 0.2 * np.cos(g[0])
    return jnp.asarray(np.stack([rho, 0.1 * np.sin(g[0]), 0.1 * np.cos(g[1]), P]))


def _face_bands(blk, g, bs):
    """The axis-aligned ghost bands + interior that the per-axis hydro actually
    reads (excludes corners). blk: (n_vars, bs+2g, bs+2g)."""
    return {
        "interior": blk[:, g:g + bs, g:g + bs],
        "-x": blk[:, 0:g, g:g + bs],
        "+x": blk[:, g + bs:g + 2 * bs, g:g + bs][:, :g],
        "-y": blk[:, g:g + bs, 0:g],
        "+y": blk[:, g:g + bs, g + bs:g + 2 * bs][:, :, :g],
    }


def test_sparse_coarse_fill_matches_dense():
    """Sparse gather (from ghosted coarse blocks) reproduces the dense-parent
    fill on the face ghost bands for an interior L0->L1 block."""
    cfg = AMRConfig(ndim=2, box=(1.0, 1.0), base_grid=(24, 24), block_shape=(8, 8),
                    max_levels=2)
    g, bs = cfg.n_ghost, cfg.block_shape[0]
    W = smooth_W(cfg)

    # coarse level as blocks, ghosts filled (same-level, non-periodic interior ok)
    coarse = build_base_level(W, cfg)
    coarse_nbrs = find_face_neighbors(coarse.coords, coarse.n_active, 0, cfg, periodic=False)
    coarse_ghosted = fill_ghost_same_level(coarse.data, coarse_nbrs, cfg)

    # refine the interior block (1,1) [row-major idx 4 in a 3x3]; take one child.
    flags = jnp.zeros(coarse.coords.shape[0], bool).at[4].set(True)
    cc, cd, na = build_children(coarse.coords, coarse.data, flags, level=0, cfg=cfg, max_parents=2)

    # isolate ONE child as a lone fine block (no fine siblings -> all ghosts NaN
    # after same-level fill, so the coarse fill must supply every face).
    one_coords = jnp.array([np.asarray(cc[0]), [-1, -1]], dtype=jnp.int32)
    one_data = jnp.stack([cd[0], jnp.full_like(cd[0], jnp.nan)])
    fine_nbrs = find_face_neighbors(one_coords, 1, 1, cfg, periodic=False)
    fg = fill_ghost_same_level(one_data, fine_nbrs, cfg)

    dense = fill_ghost_from_coarse(fg, one_coords, W, level=1, cfg=cfg, boundary=1)

    pslots = find_parent_slots(one_coords, coarse.coords, coarse.n_active, cfg)
    sparse = fill_ghost_from_coarse_sparse(fg, one_coords, pslots, coarse_ghosted, cfg)

    db, sb = _face_bands(dense[0], g, bs), _face_bands(sparse[0], g, bs)
    for k in db:
        assert jnp.allclose(db[k], sb[k], atol=1e-12, rtol=1e-12), f"mismatch on {k}"


def _kh_ic(cfg, gamma):
    nx, ny = cfg.base_grid
    Lx, Ly = cfg.box
    x = (np.arange(nx) + 0.5) / nx * Lx
    y = (np.arange(ny) + 0.5) / ny * Ly
    X, Y = np.meshgrid(x, y, indexing="ij")
    a = 0.025
    s = np.tanh((Y - Ly / 4) / a) - np.tanh((Y - 3 * Ly / 4) / a) - 1.0  # +1 middle band
    rho = 1.5 + 0.5 * s
    vx = 0.5 * s
    vy = 0.04 * np.sin(4 * np.pi * X / Lx)
    P = 2.5 * np.ones_like(rho)
    return prim_to_cons(jnp.asarray(np.stack([rho, vx, vy, P])), gamma)


def test_driver_kh_runs_and_refines():
    """End-to-end multi-level driver on a small periodic KH: refinement triggers
    on both levels, the step stays finite, and the composite has the finest
    shape. Reflux is on (the default): composite mass, energy, momentum to round-off."""
    from luna.amr import driver

    cfg = AMRConfig(ndim=2, box=(1.0, 1.0), base_grid=(32, 32), block_shape=(8, 8),
                    max_levels=2)
    U0 = _kh_ic(cfg, GAMMA)
    levels = driver.build_hierarchy(U0, cfg, thresholds=[0.05, 0.05], capacities=[128, 256])
    assert int(levels[1].n_active) > 0, "level 1 should refine on the shear layers"
    assert int(levels[2].n_active) > 0, "level 2 should refine"

    dt = float(driver.compute_dt(levels, cfg, gamma=GAMMA, CFL=0.8))
    assert dt > 0 and np.isfinite(dt)

    step = jax.jit(lambda lv, t: driver.advance(lv, t, cfg, gamma=GAMMA))
    C0 = np.asarray(driver.composite_to_dense(levels, cfg))
    for _ in range(5):
        levels = step(levels, dt)
    for lvl in levels:
        assert bool(jnp.all(jnp.isfinite(lvl.data[: int(lvl.n_active)])))

    img = driver.composite_to_dense(levels, cfg)
    assert img.shape == (4, 128, 128)                     # finest = 32 * 2**2
    check_conserved("amr kh 3 levels sync 5 steps composite", C0, img)


def _replace_base(h, U_new, cfg, periodic=True):
    """Swap a hierarchy's level-0 data for a new uniform field (keeps the rest);
    used to simulate the solution evolving between regrids."""
    from luna.amr import driver
    base = build_base_level(U_new, cfg)
    nbrs = find_face_neighbors(base.coords, base.n_active, 0, cfg, periodic=periodic)
    L0 = driver.Level(base.data, base.coords, nbrs,
                      -jnp.ones(base.data.shape[0], jnp.int32), base.n_active)
    return (L0,) + tuple(h[1:])


def test_regrid_adds_and_removes_refinement():
    """The adaptive check: regrid refines where a feature has appeared and
    de-refines where the flow is smooth again, conserving mass."""
    from luna.amr import driver

    cfg = AMRConfig(ndim=2, box=(1.0, 1.0), base_grid=(32, 32), block_shape=(8, 8),
                    max_levels=1)
    feat = _kh_ic(cfg, GAMMA)                                   # sharp shear interfaces
    nx, ny = cfg.base_grid
    y = (np.arange(ny) + 0.5) / ny
    smooth = prim_to_cons(jnp.asarray(np.stack([
        1.5 + 0.05 * np.sin(2 * np.pi * y)[None, :] * np.ones((nx, 1)),
        np.zeros((nx, ny)), np.zeros((nx, ny)), 2.5 * np.ones((nx, ny))])), GAMMA)

    thr = [0.1]
    # smooth start -> nothing to refine
    h = driver.build_hierarchy(smooth, cfg, thresholds=thr)
    assert int(h[1].n_active) == 0

    # a feature appears -> regrid must refine it
    h = _replace_base(h, feat, cfg)
    h = driver.regrid(h, cfg, thresholds=thr)
    assert int(h[1].n_active) > 0

    # mass is conserved across the regrid (new fine blocks are exact prolongs)
    vol_c = cfg.level_dx(0)[0] * cfg.level_dx(0)[1]
    vol_f = cfg.level_dx(cfg.max_levels)[0] * cfg.level_dx(cfg.max_levels)[1]
    m_base = float(jnp.sum(feat[0]) * vol_c)
    m_comp = float(jnp.sum(driver.composite_to_dense(h, cfg)[0]) * vol_f)
    assert abs(m_comp - m_base) / m_base < 1e-12

    # flow smooths out again -> regrid must de-refine
    h = _replace_base(h, smooth, cfg)
    h = driver.regrid(h, cfg, thresholds=thr)
    assert int(h[1].n_active) == 0


def test_subcycle_single_level_matches_sync():
    """With no refinement (max_levels=0) subcycling reduces to the plain step."""
    from luna.amr import driver
    cfg = AMRConfig(ndim=2, box=(1.0, 1.0), base_grid=(16, 16), block_shape=(8, 8), max_levels=0)
    U = _kh_ic(cfg, GAMMA)
    h = driver.build_hierarchy(U, cfg, thresholds=[])
    dt = float(driver.compute_dt(h, cfg, gamma=GAMMA, CFL=0.8))
    na = int(h[0].n_active)
    a = driver.advance(h, dt, cfg, gamma=GAMMA)[0].data[:na]
    b = driver.advance_subcycle(h, dt, cfg, gamma=GAMMA)[0].data[:na]
    assert jnp.allclose(a, b, atol=1e-12)


def test_subcycle_multilevel_runs_and_conserves():
    """Subcycled Berger-Oliger step: coarse takes one dt_base, finer levels
    sub-step. dt_base is r**max_levels larger than the synchronized (finest) dt,
    which is exactly the work saved on the coarse levels. Runs finite; reflux is on
    (the default): composite mass, energy, momentum to round-off."""
    from luna.amr import driver
    cfg = AMRConfig(ndim=2, box=(1.0, 1.0), base_grid=(32, 32), block_shape=(8, 8), max_levels=2)
    U = _kh_ic(cfg, GAMMA)
    h = driver.build_hierarchy(U, cfg, thresholds=[0.05, 0.05])
    assert int(h[2].n_active) > 0

    dt_sync = float(driver.compute_dt(h, cfg, gamma=GAMMA, CFL=0.8))
    dt_base = float(driver.compute_dt_subcycle(h, cfg, gamma=GAMMA, CFL=0.8))
    assert abs(dt_base / dt_sync - cfg.refinement_ratio ** cfg.max_levels) < 0.2

    step = jax.jit(lambda s, d: driver.advance_subcycle(s, d, cfg, gamma=GAMMA))
    C0 = np.asarray(driver.composite_to_dense(h, cfg))
    for _ in range(8):
        h = step(h, dt_base)
    assert all(bool(jnp.all(jnp.isfinite(l.data[: int(l.n_active)]))) for l in h)
    check_conserved("amr kh 3 levels subcycle 8 steps composite", C0, driver.composite_to_dense(h, cfg))


def test_find_parent_slots():
    cfg = AMRConfig(ndim=2, box=(1.0, 1.0), base_grid=(24, 24), block_shape=(8, 8),
                    max_levels=2)
    W = smooth_W(cfg)
    coarse = build_base_level(W, cfg)
    flags = jnp.zeros(coarse.coords.shape[0], bool).at[4].set(True)   # block (1,1)
    cc, _cd, na = build_children(coarse.coords, coarse.data, flags, level=0, cfg=cfg, max_parents=2)
    pslots = find_parent_slots(cc, coarse.coords, coarse.n_active, cfg)
    # every active child (coords // 2 == (1,1)) must resolve to coarse slot of (1,1).
    parent_slot = int(jnp.argmax(jnp.all(coarse.coords == jnp.array([1, 1]), axis=1)))
    assert bool(jnp.all(pslots[:int(na)] == parent_slot))
