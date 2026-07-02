"""Particle -> block map on the AMR hierarchy (amr/particles.py, DM kernel 1)."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from amr.config import AMRConfig
from amr import driver
from amr.particles import locate_particles

N = 32
BOX = 1.0


def _blob_U(n):
    """Gas density blob -> partial refinement around the box centre."""
    x = (jnp.arange(n) + 0.5) * (BOX / n)
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    r2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2
    rho = 1.0 + 4.0 * jnp.exp(-r2 / 0.02)
    U = jnp.zeros((5, n, n, n))
    U = U.at[0].set(rho)
    U = U.at[4].set(1.0)
    return U


def test_locate_particles_containment_and_finest():
    cfg = AMRConfig(ndim=3, box=(BOX,) * 3, base_grid=(N,) * 3,
                    block_shape=(16,) * 3, max_levels=2, boundary=0)
    h = driver.build_hierarchy(_blob_U(N), cfg, thresholds=[1.5, 2.5],
                               criterion="overdensity", periodic=True)
    assert int(h[1].n_active) > 0 and int(h[2].n_active) > 0

    rng = np.random.default_rng(7)
    pos = jnp.asarray(rng.uniform(0, BOX, size=(5000, 3)))
    lev, slot = locate_particles(pos, h, cfg)
    lev, slot = np.asarray(lev), np.asarray(slot)
    p = np.asarray(pos)

    # host-side reference: active block coord sets per level
    active_coords = [set(map(tuple, np.asarray(l.coords[: int(l.n_active)])))
                     for l in h]

    for i in range(p.shape[0]):
        L, s = int(lev[i]), int(slot[i])
        w = BOX / (2 * (2 ** L))                       # base_blocks=2 per dim
        c = np.asarray(h[L].coords[s])
        # 1) containment: the assigned block covers the particle
        assert np.all(c * w <= p[i]) and np.all(p[i] < (c + 1) * w), \
            f"particle {i} not inside its level-{L} block"
        # 2) the assigned slot is active
        assert s < int(h[L].n_active)
        # 3) finest: no ACTIVE block at any finer level contains the particle
        for Lf in range(L + 1, len(h)):
            wf = BOX / (2 * (2 ** Lf))
            cf = tuple((p[i] // wf).astype(int))
            assert cf not in active_coords[Lf], \
                f"particle {i} owned at L{L} but L{Lf} block {cf} is active"


def test_locate_particles_l0_always_owns():
    cfg = AMRConfig(ndim=3, box=(BOX,) * 3, base_grid=(N,) * 3,
                    block_shape=(16,) * 3, max_levels=0, boundary=0)
    h = driver.build_hierarchy(_blob_U(N), cfg, thresholds=[], periodic=True)
    rng = np.random.default_rng(3)
    pos = jnp.asarray(rng.uniform(-0.3, 1.3, size=(2000, 3)))   # incl. wrap
    lev, slot = locate_particles(pos, h, cfg)
    assert np.all(np.asarray(lev) == 0)
    assert np.all(np.asarray(slot) < int(h[0].n_active))


def test_gather_cic_matches_dense_pm():
    """L0-only: block accel + inverse-CIC == the dense ParticleMeshGravity
    sampling of the same phi (same stencil, same trilinear weights)."""
    from dataclasses import replace
    from template_config import Config
    from gravity.particle_mesh import ParticleMeshGravity
    from amr.particles import accel_blocks, gather_cic
    from amr.poisson import _ghost_phi
    from amr.state import partition_to_blocks

    cfg = AMRConfig(ndim=3, box=(BOX,) * 3, base_grid=(N,) * 3,
                    block_shape=(16,) * 3, max_levels=0, boundary=0)
    h = driver.build_hierarchy(_blob_U(N), cfg, thresholds=[], periodic=True)

    x = (jnp.arange(N) + 0.5) * (BOX / N)
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    k = 2 * jnp.pi / BOX
    phi = jnp.sin(k * X) * jnp.cos(2 * k * Y) * jnp.sin(k * Z)

    # dense reference
    dcfg = replace(Config(), SPATIAL_DIMENSION=3, Nx=N, Ny=N, Nz=N,
                   Lx=BOX, Ly=BOX, Lz=BOX)
    pm = ParticleMeshGravity(dcfg)
    rng = np.random.default_rng(11)
    dx = BOX / N
    pos = jnp.asarray(rng.uniform(2 * dx, BOX - 2 * dx, size=(4000, 3)))
    a_ref = pm.accel_at_points(pm.grad_phi(phi, order=2), pos)

    # block path
    phi_b = partition_to_blocks(phi[None], cfg)
    n_slots = h[0].data.shape[0]
    if n_slots > phi_b.shape[0]:
        pad = jnp.full((n_slots - phi_b.shape[0],) + phi_b.shape[1:], jnp.nan)
        phi_b = jnp.concatenate([phi_b, pad], axis=0)
    gw = _ghost_phi(phi_b, h[0], None, cfg)
    acc_b = accel_blocks(gw, 0, cfg)
    lev, slot = locate_particles(pos, h, cfg)
    a_blk = gather_cic([acc_b], h, pos, lev, slot, cfg)

    err = float(jnp.max(jnp.abs(a_blk - a_ref)))
    assert err < 1e-11, f"block CIC gather != dense PM: max err {err:.3e}"


def test_gather_cic_two_level_smoke():
    """2-level: fine-owned particles sample the fine field; values are finite
    and close to the coarse answer for a smooth phi (truncation-level)."""
    from amr.particles import accel_blocks, gather_cic
    from amr.poisson import _ghost_phi, solve_phi_hierarchy

    cfg = AMRConfig(ndim=3, box=(BOX,) * 3, base_grid=(N,) * 3,
                    block_shape=(16,) * 3, max_levels=1, boundary=0)
    h = driver.build_hierarchy(_blob_U(N), cfg, thresholds=[1.5],
                               criterion="overdensity", periodic=True)
    assert int(h[1].n_active) > 0
    phis = solve_phi_hierarchy(h, cfg, fourpiG=4 * np.pi, boundary=0)
    fields, gws = [], []
    coarse_g = None
    for L, (lvl, ph) in enumerate(zip(h, phis)):
        gw = _ghost_phi(ph, lvl, coarse_g, cfg)
        fields.append(accel_blocks(gw, L, cfg))
        if L + 1 < len(h):
            coarse_g = gw

    rng = np.random.default_rng(5)
    pos = jnp.asarray(0.45 + 0.1 * rng.uniform(size=(500, 3)))   # centre blob
    lev, slot = locate_particles(pos, h, cfg)
    assert int((np.asarray(lev) == 1).sum()) > 0                 # fine-owned exist
    a = gather_cic(fields, h, pos, lev, slot, cfg)
    assert np.all(np.isfinite(np.asarray(a)))
    # coarse-only comparison: sample everything from L0
    a0 = gather_cic(fields[:1], h[:1], pos, jnp.zeros_like(lev),
                    locate_particles(pos, h[:1], cfg)[1], cfg)
    rel = float(jnp.max(jnp.abs(a - a0)) / (jnp.max(jnp.abs(a0)) + 1e-30))
    assert rel < 0.2, f"fine vs coarse accel implausibly far: {rel:.3f}"


def test_deposit_cic_conserves_mass():
    """Kernel 2: total composite DM mass == sum of particle masses to machine
    precision, with particles straddling block seams AND coarse-fine faces."""
    from amr.particles import deposit_cic

    cfg = AMRConfig(ndim=3, box=(BOX,) * 3, base_grid=(N,) * 3,
                    block_shape=(16,) * 3, max_levels=2, boundary=0)
    h = driver.build_hierarchy(_blob_U(N), cfg, thresholds=[1.5, 2.5],
                               criterion="overdensity", periodic=True)
    assert int(h[1].n_active) > 0 and int(h[2].n_active) > 0

    rng = np.random.default_rng(23)
    pos = jnp.asarray(rng.uniform(0, BOX, size=(20000, 3)))
    mass = jnp.asarray(rng.uniform(0.5, 1.5, size=(20000,)))
    lev, slot = locate_particles(pos, h, cfg)
    assert int((np.asarray(lev) > 0).sum()) > 0          # refined-owned exist

    rhos = deposit_cic(pos, mass, h, lev, slot, cfg)

    # composite integral: overwrite coarse with fine where covered, so summing
    # the composite at finest resolution counts every gram exactly once.
    comp = driver.composite_to_dense(
        tuple(l._replace(data=r) for l, r in zip(h, rhos)), cfg)
    dv = (BOX / (N * 4)) ** 3                            # finest cell volume
    total = float(jnp.nansum(comp[0])) * dv
    ref = float(jnp.sum(mass))
    rel = abs(total - ref) / ref
    assert rel < 1e-12, f"mass drift {rel:.3e} (total {total} vs {ref})"


def test_deposit_cic_uniform_matches_dense():
    """L0-only deposit == the dense PM cic_deposit on the same particles."""
    from dataclasses import replace
    from template_config import Config
    from gravity.particle_mesh import ParticleMeshGravity
    from amr.particles import deposit_cic
    from amr.state import assemble_base_level

    cfg = AMRConfig(ndim=3, box=(BOX,) * 3, base_grid=(N,) * 3,
                    block_shape=(16,) * 3, max_levels=0, boundary=0)
    h = driver.build_hierarchy(_blob_U(N), cfg, thresholds=[], periodic=True)

    rng = np.random.default_rng(29)
    dx = BOX / N
    # keep away from box edges: the dense cic_deposit CLAMPS instead of wrapping
    pos = jnp.asarray(rng.uniform(2 * dx, BOX - 2 * dx, size=(8000, 3)))
    mass = jnp.asarray(rng.uniform(0.5, 1.5, size=(8000,)))

    lev, slot = locate_particles(pos, h, cfg)
    rho_blk = deposit_cic(pos, mass, h, lev, slot, cfg)[0]
    rho_amr = np.asarray(assemble_base_level(rho_blk[:8], cfg)[0])

    dcfg = replace(Config(), SPATIAL_DIMENSION=3, Nx=N, Ny=N, Nz=N,
                   Lx=BOX, Ly=BOX, Lz=BOX)
    pm = ParticleMeshGravity(dcfg)
    rho_ref = np.asarray(pm.cic_deposit(pos, mass, jnp.ones_like(mass)))
    err = np.max(np.abs(rho_amr - rho_ref))
    assert err < 1e-10, f"block deposit != dense CIC: {err:.3e}"
