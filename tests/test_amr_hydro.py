"""Hydro-on-blocks graft tests.

The central claim: running LUNA's MUSCL-Hancock solver per block, with ghosts
filled by the AMR halo layer, reproduces the *same* solver applied to a globally
periodic-padded grid sliced into block-sized tiles. That validates the whole
plumbing chain end to end — partition -> Morton neighbour finding -> same-level
ghost fill -> per-block solve -> assemble — against a reference that shares none
of that machinery (it uses a global ``jnp.pad(mode='wrap')`` + tile slicing).

Each block is the AMR analogue of a ``shard_map`` shard, so the per-block solver
runs in LUNA's shard-seam mode (``exact_wrap=False`` / ``boundary != 0``): the
block-boundary face flux is taken from the ghost cell. The reference uses the
identical mode, so the match is exact (to fp round-off), not merely close.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from amr.config import AMRConfig
from amr.blocks import build_children, find_face_neighbors
from amr.halo import fill_ghost_from_coarse, fill_ghost_same_level
from amr.hydro import (
    advance_blocks,
    advance_blocks_conservative,
    block_divergence,
    conservative_block_divergence,
    cons_to_prim_blocks,
    prim_to_cons_blocks,
)
from amr.reflux import (
    apply_reflux_dense,
    coarse_fine_face_mask,
    restrict_fine_into_coarse_dense,
)
from amr.geometry import block_cell_centers, prolong
from amr.hydro import block_face_fluxes
from amr.state import assemble_base_level, base_block_coords, build_base_level
from hydro.conversions import cons_to_prim, prim_to_cons
from hydro.muscl_hancock import (
    divergence_from_face_fluxes,
    muscl_hancock_divergence,
    muscl_hancock_face_fluxes,
)

GAMMA = 5.0 / 3.0
LIMITER = 1          # moncen
RIEMANN = "hllc"


# ---------------------------------------------------------------------------
# Smooth, strictly-positive primitive field on a uniform grid.
# ---------------------------------------------------------------------------
def smooth_primitive(cfg, seed=0):
    ndim = cfg.ndim
    axes = [np.linspace(0.0, 2 * np.pi, cfg.base_grid[d], endpoint=False) for d in range(ndim)]
    grids = np.meshgrid(*axes, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(grids[0]) * np.cos(grids[1] if ndim > 1 else 0.0)
    P = 1.0 + 0.2 * np.cos(grids[0])
    comps = [rho]
    for d in range(ndim):
        comps.append(0.1 * np.sin(grids[d] + 0.5 * d))   # velocity per axis
    comps.append(P)
    W = np.stack(comps).astype(np.float64)
    return jnp.asarray(W)


def reference_step(U, cfg, dt):
    """Independent reference: global periodic 1-ghost pad, sliced per tile,
    each tile advanced by the *same* MUSCL-Hancock solver. Shares no AMR code.
    """
    ndim = cfg.ndim
    bs = cfg.block_shape
    nb = cfg.base_blocks
    dx = cfg.level_dx(0)
    inv = [1.0 / dx[d] for d in range(ndim)]
    inv_dx = inv[0]
    inv_dy = inv[1] if ndim > 1 else 1.0
    inv_dz = inv[2] if ndim > 2 else 1.0

    pad_w = [(0, 0)] + [(1, 1)] * ndim
    Upad = jnp.pad(U, pad_w, mode="wrap")
    U_new = U
    for tile in itertools.product(*[range(nb[d]) for d in range(ndim)]):
        ext = [slice(None)] + [slice(tile[d] * bs[d], tile[d] * bs[d] + bs[d] + 2) for d in range(ndim)]
        W_pad = cons_to_prim(Upad[tuple(ext)], GAMMA)
        dU = muscl_hancock_divergence(
            W_pad, dt, gamma=GAMMA, limiter=LIMITER, riemann_solver=RIEMANN,
            spatial_dim=ndim, inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz,
            boundary=1, order=2, exact_wrap=False,
        )
        intr = [slice(None)] + [slice(tile[d] * bs[d], (tile[d] + 1) * bs[d]) for d in range(ndim)]
        U_new = U_new.at[tuple(intr)].set(U[tuple(intr)] + dt * dU)
    return U_new


def block_step(U, cfg, dt):
    """Production path: base-level blocks, periodic same-level halo, per-block
    advance, reassembled to a dense field."""
    ld = build_base_level(U, cfg)
    neighbors = find_face_neighbors(ld.coords, ld.n_active, 0, cfg, periodic=True)
    fill = lambda W: fill_ghost_same_level(W, neighbors, cfg)
    U_new_blocks = advance_blocks(
        ld.data, fill, dt, 0, cfg, gamma=GAMMA, limiter=LIMITER, riemann_solver=RIEMANN
    )
    return assemble_base_level(U_new_blocks[:-1], cfg)   # drop NaN sentinel


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------
def test_cons_prim_blocks_roundtrip():
    cfg = AMRConfig(ndim=2, box=(1.0, 1.0), base_grid=(16, 16), block_shape=(8, 8))
    W = smooth_primitive(cfg)
    U = prim_to_cons(W, GAMMA)
    from amr.state import partition_to_blocks

    U_blocks = partition_to_blocks(U, cfg)
    W_blocks = cons_to_prim_blocks(U_blocks, GAMMA)
    U_round = prim_to_cons_blocks(W_blocks, GAMMA)
    assert jnp.allclose(U_round, U_blocks, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# Face-flux variant reproduces the divergence (basis for reflux)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ndim", [2, 3])
def test_face_fluxes_reproduce_divergence(ndim):
    cfg = AMRConfig(
        ndim=ndim,
        box=(1.0,) * ndim,
        base_grid=(16, 16) if ndim == 2 else (8, 8, 8),
        block_shape=(8, 8) if ndim == 2 else (4, 4, 4),
    )
    W = smooth_primitive(cfg)
    # one padded block (1-ghost) extracted from the global periodic pad
    pad_w = [(0, 0)] + [(1, 1)] * ndim
    Wpad = jnp.pad(W, pad_w, mode="wrap")
    bs = cfg.block_shape
    sl = [slice(None)] + [slice(0, bs[d] + 2) for d in range(ndim)]
    Wp = Wpad[tuple(sl)]

    dx = cfg.level_dx(0)
    inv = dict(inv_dx=1.0 / dx[0], inv_dy=1.0 / dx[1], inv_dz=1.0 / dx[2] if ndim == 3 else 1.0)
    dt = 0.1 * dx[0]

    div_direct = muscl_hancock_divergence(
        Wp, dt, gamma=GAMMA, limiter=LIMITER, riemann_solver=RIEMANN,
        spatial_dim=ndim, boundary=1, order=2, exact_wrap=False, **inv,
    )
    fluxes = muscl_hancock_face_fluxes(
        Wp, dt, gamma=GAMMA, limiter=LIMITER, riemann_solver=RIEMANN,
        spatial_dim=ndim, boundary=1, exact_wrap=False, **inv,
    )
    div_from_flux = divergence_from_face_fluxes(fluxes, spatial_dim=ndim, **inv)
    assert jnp.allclose(div_direct, div_from_flux, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# Single-level equivalence (the integration proof)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ndim,base_grid,block_shape",
    [
        (2, (16, 16), (8, 8)),
        (2, (24, 16), (8, 8)),       # non-square tiling
        (3, (8, 8, 8), (4, 4, 4)),
    ],
)
def test_block_step_matches_reference(ndim, base_grid, block_shape):
    cfg = AMRConfig(ndim=ndim, box=(1.0,) * ndim, base_grid=base_grid, block_shape=block_shape)
    W = smooth_primitive(cfg)
    U = prim_to_cons(W, GAMMA)
    dt = 0.1 * cfg.level_dx(0)[0]

    U_block = block_step(U, cfg, dt)
    U_ref = reference_step(U, cfg, dt)

    assert jnp.all(jnp.isfinite(U_block))
    assert jnp.allclose(U_block, U_ref, atol=1e-10, rtol=1e-10)


def test_block_step_jits():
    """The per-block advance traces under jit with cfg static."""
    cfg = AMRConfig(ndim=2, box=(1.0, 1.0), base_grid=(16, 16), block_shape=(8, 8))
    W = smooth_primitive(cfg)
    U = prim_to_cons(W, GAMMA)
    dt = 0.05 * cfg.level_dx(0)[0]

    ld = build_base_level(U, cfg)
    neighbors = find_face_neighbors(ld.coords, ld.n_active, 0, cfg, periodic=True)

    @jax.jit
    def step(data, nbrs, dt):
        fill = lambda W: fill_ghost_same_level(W, nbrs, cfg)
        return advance_blocks(data, fill, dt, 0, cfg, gamma=GAMMA,
                              limiter=LIMITER, riemann_solver=RIEMANN)

    out = step(ld.data, neighbors, dt)
    assert out.shape == ld.data.shape
    assert jnp.all(jnp.isfinite(out[:-1]))         # active blocks finite
    assert jnp.all(jnp.isnan(out[-1]))             # sentinel stays NaN


def test_sentinel_does_not_contaminate():
    """Active blocks never read the NaN sentinel under periodic tiling."""
    cfg = AMRConfig(ndim=2, box=(1.0, 1.0), base_grid=(16, 16), block_shape=(8, 8))
    W = smooth_primitive(cfg)
    U = prim_to_cons(W, GAMMA)
    U_new = block_step(U, cfg, 0.05 * cfg.level_dx(0)[0])
    assert jnp.all(jnp.isfinite(U_new))


# ---------------------------------------------------------------------------
# Multi-level: refined blocks advance with coarse-parent ghosts
# ---------------------------------------------------------------------------
def test_refined_level_advances_with_coarse_ghosts():
    cfg = AMRConfig(ndim=2, box=(1.0, 1.0), base_grid=(16, 16), block_shape=(8, 8),
                    max_levels=2)
    W = smooth_primitive(cfg)
    U = prim_to_cons(W, GAMMA)

    # base level
    base = build_base_level(U, cfg)
    base_coords = base_block_coords(cfg)             # (n_base, ndim) active coords

    # refine the first base block into 2**ndim children at level 1
    n_base = base_coords.shape[0]
    flags = jnp.zeros(n_base + 1, bool).at[0].set(True)   # include sentinel slot
    child_coords, child_data, n_active = build_children(
        base.coords, base.data, flags, level=0, cfg=cfg, max_parents=2
    )

    # fine-level halo: same-level among the 4 children (external faces -> NaN),
    # then fill the rest from the coarse parent (the dense base field, in prims).
    W_child = cons_to_prim_blocks(child_data, GAMMA)
    fine_nbrs = find_face_neighbors(child_coords, n_active, 1, cfg, periodic=False)
    W_ghosted = fill_ghost_same_level(W_child, fine_nbrs, cfg)
    parent_prim = cons_to_prim(U, GAMMA)             # dense level-0 prim field
    W_ghosted = fill_ghost_from_coarse(
        W_ghosted, child_coords, parent_prim, level=1, cfg=cfg, boundary=0
    )
    assert not jnp.any(jnp.isnan(W_ghosted[: int(n_active)])), "active fine ghosts unfilled"

    dt = 0.1 * cfg.level_dx(1)[0]
    dU = block_divergence(W_ghosted, dt, 1, cfg, gamma=GAMMA,
                          limiter=LIMITER, riemann_solver=RIEMANN)
    U_fine_new = child_data + dt * dU
    assert jnp.all(jnp.isfinite(U_fine_new[: int(n_active)]))


# ---------------------------------------------------------------------------
# Coarse-fine reflux conservation
# ---------------------------------------------------------------------------
def _dense_coarse_step(U_c, cfg, dt):
    """Monolithic periodic coarse advance (exact_wrap -> conserves mass to
    machine precision). Returns (U_c_new, coarse_face_fluxes)."""
    ndim = cfg.ndim
    W_c = cons_to_prim(U_c, GAMMA)
    W_pad = jnp.pad(W_c, [(0, 0)] + [(1, 1)] * ndim, mode="wrap")
    dx = cfg.level_dx(0)
    inv = dict(inv_dx=1.0 / dx[0], inv_dy=1.0 / dx[1], inv_dz=1.0 / dx[2] if ndim == 3 else 1.0)
    fluxes = muscl_hancock_face_fluxes(
        W_pad, dt, gamma=GAMMA, limiter=LIMITER, riemann_solver=RIEMANN,
        spatial_dim=ndim, boundary=0, exact_wrap=True, **inv,
    )
    div = divergence_from_face_fluxes(fluxes, spatial_dim=ndim, **inv)
    return U_c + dt * div, fluxes


def test_reflux_conserves_mass_single_fine_block():
    """Dense (monolithic, seam-free) coarse + ONE fine block: the coarse-fine
    interface is the only non-conservative face, so reflux is the only fix
    needed and total mass must return to machine precision. Without reflux it
    must drift (otherwise the test proves nothing)."""
    cfg = AMRConfig(ndim=2, box=(1.0, 1.0), base_grid=(16, 16), block_shape=(8, 8),
                    max_levels=2)
    W = smooth_primitive(cfg)
    U_c = prim_to_cons(W, GAMMA)
    dx_c = cfg.level_dx(0)
    cell_vol_c = dx_c[0] * dx_c[1]

    # one interior fine block at L1 coords (1,1): covers coarse cells [4:8,4:8].
    cf = jnp.array([[1, 1], [-1, -1]], dtype=jnp.int32)     # block + sentinel
    n_active = 1
    U_f0 = prolong(U_c[:, 4:8, 4:8], 2)[None]               # (1, n_vars, 8, 8)
    nan_blk = jnp.full_like(U_f0[:1], jnp.nan)
    fine_data = jnp.concatenate([U_f0, nan_blk], axis=0)    # (2, n_vars, 8, 8)

    dt = 0.05 * cfg.level_dx(1)[0]

    mass_initial = jnp.sum(U_c[0]) * cell_vol_c             # = total (fine = prolong)

    # --- coarse advance (monolithic, conservative) ---
    U_c_new, Fc = _dense_coarse_step(U_c, cfg, dt)
    assert jnp.allclose(jnp.sum(U_c_new[0]) * cell_vol_c, mass_initial, rtol=1e-12), \
        "monolithic coarse step should conserve mass on its own"

    # --- fine advance (single block, ghosts from coarse parent) ---
    W_f = cons_to_prim_blocks(fine_data, GAMMA)
    fine_nbrs = find_face_neighbors(cf, n_active, 1, cfg, periodic=False)
    W_g = fill_ghost_same_level(W_f, fine_nbrs, cfg)
    W_g = fill_ghost_from_coarse(W_g, cf, cons_to_prim(U_c, GAMMA), level=1, cfg=cfg, boundary=0)
    dU_f = block_divergence(W_g, dt, 1, cfg, gamma=GAMMA, limiter=LIMITER, riemann_solver=RIEMANN)
    U_f_new = fine_data + dt * dU_f
    Ff = block_face_fluxes(W_g, dt, 1, cfg, gamma=GAMMA, limiter=LIMITER, riemann_solver=RIEMANN)

    cf_mask = coarse_fine_face_mask(fine_nbrs, n_active)
    assert bool(jnp.all(cf_mask[0])), "single fine block: all 4 faces are coarse-fine"

    # --- compose: WITHOUT reflux (drifts) ---
    U_c_noflux = restrict_fine_into_coarse_dense(U_c_new, U_f_new, cf, 1, cfg)
    mass_noflux = jnp.sum(U_c_noflux[0]) * cell_vol_c

    # --- compose: WITH reflux (conserves) ---
    U_c_ref = apply_reflux_dense(U_c_new, Fc, Ff, cf, cf_mask, 1, cfg, dt)
    U_c_final = restrict_fine_into_coarse_dense(U_c_ref, U_f_new, cf, 1, cfg)
    mass_final = jnp.sum(U_c_final[0]) * cell_vol_c

    drift_noflux = float(jnp.abs(mass_noflux - mass_initial) / jnp.abs(mass_initial))
    drift_reflux = float(jnp.abs(mass_final - mass_initial) / jnp.abs(mass_initial))

    assert drift_reflux < 1e-11, f"reflux should conserve mass; drift={drift_reflux:.2e}"
    assert drift_noflux > 1e3 * max(drift_reflux, 1e-15), \
        f"without reflux mass should drift (got {drift_noflux:.2e})"


# ---------------------------------------------------------------------------
# Same-level seam consistency: a multi-block single level must conserve
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k,name", [(0, "mass"), (-1, "energy")])
def test_same_level_seam_conserves(k, name):
    """Periodic multi-block (3x3) base level. With consistent seam fluxes every
    block-block face (incl. the periodic wrap) shares one flux, so each conserved
    component telescopes to machine precision. The ghost-based (non-conservative)
    advance must drift."""
    cfg = AMRConfig(ndim=2, box=(1.0, 1.0), base_grid=(24, 24), block_shape=(8, 8))
    W = smooth_primitive(cfg)
    U = prim_to_cons(W, GAMMA)
    dt = 0.05 * cfg.level_dx(0)[0]

    ld = build_base_level(U, cfg)
    nbrs = find_face_neighbors(ld.coords, ld.n_active, 0, cfg, periodic=True)
    fill = lambda Wb: fill_ghost_same_level(Wb, nbrs, cfg)

    total0 = jnp.sum(ld.data[:-1, k])               # drop NaN sentinel; vol cancels in ratio

    U_cons = advance_blocks_conservative(
        ld.data, fill, nbrs, dt, 0, cfg, gamma=GAMMA, limiter=LIMITER, riemann_solver=RIEMANN
    )
    U_nc = advance_blocks(
        ld.data, fill, dt, 0, cfg, gamma=GAMMA, limiter=LIMITER, riemann_solver=RIEMANN
    )
    drift_cons = float(jnp.abs(jnp.sum(U_cons[:-1, k]) - total0) / jnp.abs(total0))
    drift_nc = float(jnp.abs(jnp.sum(U_nc[:-1, k]) - total0) / jnp.abs(total0))

    assert jnp.all(jnp.isfinite(U_cons[:-1]))
    assert drift_cons < 1e-11, f"{name}: seam-consistent advance should conserve (drift={drift_cons:.2e})"
    assert drift_nc > 1e3 * max(drift_cons, 1e-15), \
        f"{name}: ghost-based advance should drift (got {drift_nc:.2e})"


# ---------------------------------------------------------------------------
# Capstone: combined same-level (fine siblings) + coarse-fine reflux conserves
# ---------------------------------------------------------------------------
def test_combined_multilevel_conservation():
    """Dense (seam-free) coarse + a fine patch of 2**ndim sibling blocks. Mass is
    conserved only if BOTH mechanisms work: the fine siblings' shared seam fluxes
    (else the patch leaks internally) AND coarse-fine reflux (else the C/F
    interface leaks). The control advances the fine level non-conservatively and
    must drift even with reflux applied."""
    cfg = AMRConfig(ndim=2, box=(1.0, 1.0), base_grid=(24, 24), block_shape=(8, 8),
                    max_levels=2)
    W = smooth_primitive(cfg)
    U_c = prim_to_cons(W, GAMMA)
    vol_c = cfg.level_dx(0)[0] * cfg.level_dx(0)[1]

    base = build_base_level(U_c, cfg)
    base_coords = base_block_coords(cfg)            # 3x3 row-major
    n_base = base_coords.shape[0]
    # refine the interior block (1,1): row-major index 1*3 + 1 = 4.
    flags = jnp.zeros(n_base + 1, bool).at[4].set(True)
    cc, _cd, n_active = build_children(base.coords, base.data, flags, level=0, cfg=cfg, max_parents=2)
    n_active = int(n_active)
    assert n_active == 4

    # Fine init sampled at fine resolution (NOT a fresh prolong): a fresh prolong
    # has identical first two cells per block, so the boundary slope limits to
    # zero and the seam inconsistency vanishes on step one — which would make the
    # control vacuous. Real sub-structure exposes the sibling-seam leak.
    def _sample_cons(centers):
        th0, th1 = 2 * np.pi * centers[0], 2 * np.pi * centers[1]
        rho = 1.0 + 0.3 * jnp.sin(th0) * jnp.cos(th1)
        P = 1.0 + 0.2 * jnp.cos(th0)
        return prim_to_cons(jnp.stack([rho, 0.1 * jnp.sin(th0), 0.1 * jnp.sin(th1 + 0.5), P]), GAMMA)

    centers = jax.vmap(lambda c: block_cell_centers(c, 1, cfg))(cc)   # (n_slots, ndim, *bs)
    cd = jax.vmap(_sample_cons)(centers).at[-1].set(jnp.nan)          # NaN sentinel

    dt = 0.05 * cfg.level_dx(1)[0]
    # effective dense initial total: coarse outside patch + restrict(fine init).
    mass0 = jnp.sum(restrict_fine_into_coarse_dense(U_c, cd, cc, 1, cfg)[0]) * vol_c

    # coarse advance (monolithic dense, conservative)
    U_c_new, Fc = _dense_coarse_step(U_c, cfg, dt)

    # fine advance with sibling seam fluxes; outer faces stay ghost-based (C/F)
    fine_nbrs = find_face_neighbors(cc, n_active, 1, cfg, periodic=False)
    W_f = cons_to_prim_blocks(cd, GAMMA)
    W_g = fill_ghost_same_level(W_f, fine_nbrs, cfg)
    W_g = fill_ghost_from_coarse(W_g, cc, cons_to_prim(U_c, GAMMA), level=1, cfg=cfg, boundary=0)
    dU_f = conservative_block_divergence(W_g, dt, 1, cfg, fine_nbrs,
                                         gamma=GAMMA, limiter=LIMITER, riemann_solver=RIEMANN)
    U_f_new = cd + dt * dU_f
    Ff = block_face_fluxes(W_g, dt, 1, cfg, gamma=GAMMA, limiter=LIMITER, riemann_solver=RIEMANN)
    cf_mask = coarse_fine_face_mask(fine_nbrs, n_active)

    U_final = apply_reflux_dense(U_c_new, Fc, Ff, cc, cf_mask, 1, cfg, dt)
    U_final = restrict_fine_into_coarse_dense(U_final, U_f_new, cc, 1, cfg)
    drift = float(jnp.abs(jnp.sum(U_final[0]) * vol_c - mass0) / jnp.abs(mass0))
    assert drift < 1e-11, f"combined AMR step should conserve mass; drift={drift:.2e}"

    # control: fine advanced WITHOUT seam consistency -> internal sibling leak
    dU_nc = block_divergence(W_g, dt, 1, cfg, gamma=GAMMA, limiter=LIMITER, riemann_solver=RIEMANN)
    U_f_nc = cd + dt * dU_nc
    U_ctrl = apply_reflux_dense(U_c_new, Fc, Ff, cc, cf_mask, 1, cfg, dt)
    U_ctrl = restrict_fine_into_coarse_dense(U_ctrl, U_f_nc, cc, 1, cfg)
    drift_ctrl = float(jnp.abs(jnp.sum(U_ctrl[0]) * vol_c - mass0) / jnp.abs(mass0))
    assert drift_ctrl > 1e3 * max(drift, 1e-15), \
        f"non-conservative fine siblings should drift (got {drift_ctrl:.2e})"
