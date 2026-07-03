"""Regression test: Berger-Oliger subcycled advance with per-level gravity +
time-accumulated reflux vs the synchronized advance.

Pins the driver-level validation of 2026-07-03 (self-gravitating blob, 2-level
hierarchy): the subcycled composite must track the synchronized one to
truncation level, conserve mass, keep net momentum ~0, and stay NaN-free.
Small enough (32^3, 3 base steps) to run in the CPU test suite; compile is the
dominant cost (~1-2 min).
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from amr.config import AMRConfig
from amr import driver
from amr.poisson import (solve_phi_level, kick_level, _ghost_phi,
                         solve_phi_hierarchy, apply_gravity_kick)
from amr.state import assemble_base_level

GAMMA = 5.0 / 3.0
BOX = 1.0
NB = 32
FOURPIG = 4.0 * np.pi


def _blob_U(n):
    x = (np.arange(n) + 0.5) / n * BOX
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2
    rho = 1.0 + 3.0 * np.exp(-r2 / (2 * 0.08 ** 2))
    U = np.zeros((5, n, n, n))
    U[0] = rho
    U[4] = 1.0 / (GAMMA - 1.0)                       # uniform P, v=0
    return jnp.asarray(U)


def _make_gravity(acfg):
    """Per-level solve+kick hook, the runner's grav_hook pattern."""
    st = {"m": jnp.asarray(0.0)}
    n_base = int(np.prod(acfg.base_blocks))

    def gravity(L, lvl, parent_phig, dt, t_sub):
        if L == 0:
            st["m"] = jnp.mean(assemble_base_level(lvl.data[:n_base, 0:1], acfg)[0])
        phi = solve_phi_level(lvl, parent_phig, acfg, fourpiG=FOURPIG,
                              boundary=0, mean_rho=st["m"], level=L, n_vcycles=3)
        data = kick_level(lvl, phi, parent_phig, dt, L, acfg)
        return data, _ghost_phi(phi, lvl, parent_phig, acfg)

    return gravity


def test_subcycle_gravity_reflux_matches_synchronized():
    levels = 2
    acfg = AMRConfig(ndim=3, box=(BOX,) * 3, base_grid=(NB,) * 3,
                     block_shape=(16,) * 3, max_levels=levels, boundary=0,
                     gamma=GAMMA, grav_G=1.0)
    h0 = driver.build_hierarchy(_blob_U(NB), acfg, thresholds=[1.5, 2.5],
                                criterion="overdensity", periodic=True)
    assert int(h0[1].n_active) > 0 and int(h0[2].n_active) > 0

    r = acfg.refinement_ratio
    gravity = _make_gravity(acfg)

    def sync_step(hh, d):
        hh = driver.advance(hh, d, acfg, gamma=GAMMA, reflux=True)
        phis = solve_phi_hierarchy(hh, acfg, fourpiG=FOURPIG, boundary=0,
                                   n_vcycles=3)
        return apply_gravity_kick(hh, phis, d, acfg)

    sync_jit = jax.jit(sync_step)
    sub_jit = jax.jit(lambda hh, d: driver.advance_subcycle(
        hh, d, acfg, gamma=GAMMA, gravity=gravity, reflux=True, t0=0.0))

    dt_base = float(driver.compute_dt_subcycle(h0, acfg, gamma=GAMMA, CFL=0.4))
    nfine = r ** levels
    dt_fine = dt_base / nfine

    def total_mass(hier):
        comp = driver.composite_to_dense(hier, acfg)
        dv = (BOX / (NB * r ** levels)) ** 3
        return comp, float(jnp.nansum(comp[0]) * dv)

    _, m0 = total_mass(h0)

    hs, hb = h0, h0
    for _ in range(3):                                # 3 base steps
        for _ in range(nfine):
            hs = sync_jit(hs, jnp.asarray(dt_fine))
        hb = sub_jit(hb, jnp.asarray(dt_base))

    comp_s, ms = total_mass(hs)
    comp_b, mb = total_mass(hb)

    # (a) NaN-free
    assert not bool(jnp.any(jnp.isnan(comp_b))), "subcycled composite has NaN"
    # (b) mass conservation: sub within 2x of sync's own (composite-metric) drift
    drift_s = abs(ms - m0) / m0
    drift_b = abs(mb - m0) / m0
    assert drift_b < max(2.0 * drift_s, 1e-10), \
        f"sub mass drift {drift_b:.2e} vs sync {drift_s:.2e}"
    # (c) sub tracks sync to truncation (validated 1.6e-5 at 6 steps; margin)
    l1 = float(jnp.nansum(jnp.abs(comp_b[0] - comp_s[0]))
               / jnp.nansum(jnp.abs(comp_s[0])))
    assert l1 < 1e-3, f"sub vs sync density L1 {l1:.2e} exceeds 1e-3"
    # (d) net momentum stays ~0 (symmetric collapse from rest)
    dv = (BOX / (NB * r ** levels)) ** 3
    p = np.array([float(jnp.nansum(comp_b[i]) * dv) for i in (1, 2, 3)])
    assert np.linalg.norm(p) < 1e-5, f"net momentum {np.linalg.norm(p):.2e}"


if __name__ == "__main__":
    test_subcycle_gravity_reflux_matches_synchronized()
    print("PASS test_subcycle_gravity_reflux_matches_synchronized")
    print("########## SUBCYCLE_TEST_DONE ##########")
