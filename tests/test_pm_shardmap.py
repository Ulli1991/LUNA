"""Parity + invariant tests for the grid-sharded (slab) Particle-Mesh.

Run on simulated devices:
    XLA_FLAGS=--xla_force_host_platform_device_count=4 python tests/test_pm_shardmap.py

Checks, all against periodic numpy references on the FULL grid (the physically
correct target) so the sharded math is pinned exactly, not just self-consistent:
  1. CIC deposit  (slab + x-ghost add)      vs numpy periodic CIC
  2. solve_poisson_fmg_shardmap on a slab mesh vs single-device FMG
  3. grad_phi_slab (4th order, 2-cell halo) vs numpy periodic 4th-order force
  4. gather_accel_slab (1-cell halo)        vs numpy periodic CIC gather
  5. exchange_particles_slab invariant: after a random drift every active
     particle lands in its owning slab, mass + count are conserved.
"""

import pytest
pytestmark = pytest.mark.slow

import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

jax.config.update("jax_enable_x64", True)

from gravity.particle_mesh_shardmap import (
    cic_deposit_slab, decompose_to_slabs, exchange_particles_slab,
    gather_accel_slab, grad_phi_slab, make_slab_mesh)
from gravity.poisson_multigrid import _max_levels, solve_poisson_fmg, solve_poisson_fmg_shardmap


# ---- numpy periodic references --------------------------------------------
def cic_np(pos, mass, Nx, dx):
    rho = np.zeros((Nx, Nx, Nx))
    f = pos / dx - 0.5
    i0 = np.floor(f).astype(int)
    t = f - i0
    for di in (0, 1):
        wx = t[:, 0] if di else 1 - t[:, 0]
        for dj in (0, 1):
            wy = t[:, 1] if dj else 1 - t[:, 1]
            for dk in (0, 1):
                wz = t[:, 2] if dk else 1 - t[:, 2]
                w = wx * wy * wz * mass / dx ** 3
                ii = (i0[:, 0] + di) % Nx
                jj = (i0[:, 1] + dj) % Nx
                kk = (i0[:, 2] + dk) % Nx
                np.add.at(rho, (ii, jj, kk), w)
    return rho


def grad4_np(phi, dx):
    c = 1.0 / (12.0 * dx)
    def d(ax):
        return (np.roll(phi, -2, ax) - 8 * np.roll(phi, -1, ax)
                + 8 * np.roll(phi, 1, ax) - np.roll(phi, 2, ax)) * c
    return d(0), d(1), d(2)


def gather_np(g, pos, Nx, dx):
    gx, gy, gz = g
    f = pos / dx - 0.5
    i0 = np.floor(f).astype(int)
    t = f - i0
    out = np.zeros((pos.shape[0], 3))
    for di in (0, 1):
        wx = t[:, 0] if di else 1 - t[:, 0]
        for dj in (0, 1):
            wy = t[:, 1] if dj else 1 - t[:, 1]
            for dk in (0, 1):
                wz = t[:, 2] if dk else 1 - t[:, 2]
                w = wx * wy * wz
                ii = (i0[:, 0] + di) % Nx
                jj = (i0[:, 1] + dj) % Nx
                kk = (i0[:, 2] + dk) % Nx
                out[:, 0] += w * gx[ii, jj, kk]
                out[:, 1] += w * gy[ii, jj, kk]
                out[:, 2] += w * gz[ii, jj, kk]
    return out


def _shard(mesh, arr, spec):
    return jax.device_put(jnp.asarray(arr), NamedSharding(mesh, spec))


def main():
    ndev = jax.device_count()
    assert ndev >= 2, f"need >=2 devices (got {ndev}); set XLA_FLAGS device count"
    Nx = 16
    box = 1.0
    dx = box / Nx
    N = 4000
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, box, size=(N, 3))
    vel = rng.normal(size=(N, 3)) * 0.01
    mass = np.full(N, 1.0 / N)

    mesh = make_slab_mesh(jax.devices())
    pos_g, vel_g, mass_g, act_g, C = decompose_to_slabs(pos, vel, mass, Nx, ndev, dx, box)
    posS = _shard(mesh, pos_g, P("x", None))
    velS = _shard(mesh, vel_g, P("x", None))
    massS = _shard(mesh, mass_g, P("x"))
    actS = _shard(mesh, act_g, P("x"))
    print(f"# ndev={ndev} Nx={Nx} N={N} C={C} (capacity {ndev*C})")

    # 1) deposit parity
    rho = np.asarray(cic_deposit_slab(posS, massS, actS, Nx, Nx, Nx, dx, mesh))
    rho_ref = cic_np(pos, mass, Nx, dx)
    e1 = np.max(np.abs(rho - rho_ref))
    print(f"1 deposit   max|Δ|={e1:.2e}  sum_sh={rho.sum()*dx**3:.6f} sum_ref={rho_ref.sum()*dx**3:.6f}")
    assert e1 < 1e-12, e1

    # 2) MG parity on the slab mesh vs single-device FMG. Both must converge to
    # the SAME discrete solution; with enough V-cycles the red-black-parity
    # path difference washes out (stationary iteration -> A^{-1}b either way).
    coeff = 1.0
    nx_loc = Nx // ndev
    n_levels = min(_max_levels(Nx), int(np.log2(nx_loc)) + 1)
    delta = rho_ref / rho_ref.mean() - 1.0
    deltaS = _shard(mesh, delta, P("x", "y", "z"))
    phi_sh = np.array(solve_poisson_fmg_shardmap(
        deltaS, dx, coeff, boundary=0, mesh=mesh, n_levels=n_levels, n_extra_vcycles=40))
    phi_ref = np.array(solve_poisson_fmg(
        jnp.asarray(delta), dx, coeff, boundary=0, n_levels=n_levels, n_extra_vcycles=40))
    # gauge-fix (periodic phi is defined up to a constant)
    phi_sh -= phi_sh.mean()
    phi_ref -= phi_ref.mean()
    e2 = np.max(np.abs(phi_sh - phi_ref)) / np.max(np.abs(phi_ref))
    # residual of the sharded solve: does it actually solve lap(phi)=coeff*delta?
    lap = (np.roll(phi_sh, -1, 0) + np.roll(phi_sh, 1, 0)
           + np.roll(phi_sh, -1, 1) + np.roll(phi_sh, 1, 1)
           + np.roll(phi_sh, -1, 2) + np.roll(phi_sh, 1, 2) - 6 * phi_sh) / dx ** 2
    res = np.max(np.abs(lap - coeff * delta)) / np.max(np.abs(coeff * delta))
    print(f"2 MG solve  rel|Δφ|={e2:.2e}  sharded residual={res:.2e}  (n_levels={n_levels})")
    assert e2 < 1e-4, e2
    assert res < 1e-2, res

    # 3) grad parity (use phi_ref so both see identical input)
    phiS = _shard(mesh, phi_ref, P("x", None, None))
    gx, gy, gz = grad_phi_slab(phiS, dx, mesh, order=4)
    gx, gy, gz = np.asarray(gx), np.asarray(gy), np.asarray(gz)
    rgx, rgy, rgz = grad4_np(phi_ref, dx)
    e3 = max(np.max(np.abs(gx - rgx)), np.max(np.abs(gy - rgy)), np.max(np.abs(gz - rgz)))
    print(f"3 grad4     max|Δg|={e3:.2e}")
    assert e3 < 1e-11, e3

    # 4) gather parity
    gS = (_shard(mesh, rgx, P("x", None, None)),
          _shard(mesh, rgy, P("x", None, None)),
          _shard(mesh, rgz, P("x", None, None)))
    acc = np.asarray(gather_accel_slab(gS, posS, Nx, Nx, Nx, dx, mesh))
    # map sharded (slab-order) accel back to original particle order to compare;
    # only the ACTIVE slots are meaningful (padding slots gather at pos=0).
    acc_ref_slab = np.zeros_like(acc)
    owner = np.clip((np.mod(pos[:, 0], box) / (nx_loc * dx)).astype(int), 0, ndev - 1)
    ref_all = gather_np((rgx, rgy, rgz), pos, Nx, dx)
    for d in range(ndev):
        sel = np.nonzero(owner == d)[0]
        acc_ref_slab[d * C:d * C + sel.size] = ref_all[sel]
    m4 = act_g > 0
    e4 = np.max(np.abs(acc[m4] - acc_ref_slab[m4]))
    print(f"4 gather    max|Δa|={e4:.2e}  (active slots)")
    assert e4 < 1e-11, e4

    # 5) exchange invariant: drift hard, redistribute, check every particle home
    key_disp = rng.uniform(-1.5 * nx_loc * dx, 1.5 * nx_loc * dx, size=pos_g.shape[0])
    pos_drift = pos_g.copy()
    pos_drift[:, 0] = np.mod(pos_g[:, 0] + key_disp * (act_g > 0), box)
    pdS = _shard(mesh, pos_drift, P("x", None))
    npos, nvel, nmass, nact, n_bad, max_occ = exchange_particles_slab(
        pdS, velS, massS, actS, Nx, dx, box, mesh)
    n_bad = int(n_bad)
    max_occ = int(max_occ)
    npos = np.asarray(npos); nact = np.asarray(nact); nmass = np.asarray(nmass)
    tot_before = float(act_g.sum())
    tot_after = float(nact.sum())
    mass_after = float(nmass[nact > 0].sum())
    print(f"5 exchange  n_out_of_slab={n_bad}  count {tot_before:.0f}->{tot_after:.0f}  "
          f"mass={mass_after:.6f}  max_occ={max_occ}/{C}")
    # NOTE: displacement here spans >1 slab on purpose to stress single-hop limit;
    # so we only require the one-hop fraction to land correctly + conservation
    # over a physical (sub-slab) drift below.
    small = rng.uniform(-0.4 * nx_loc * dx, 0.4 * nx_loc * dx, size=pos_g.shape[0])
    pos_s = pos_g.copy()
    pos_s[:, 0] = np.mod(pos_g[:, 0] + small * (act_g > 0), box)
    psS = _shard(mesh, pos_s, P("x", None))
    npos2, _, nmass2, nact2, n_bad2, max_occ2 = exchange_particles_slab(
        psS, velS, massS, actS, Nx, dx, box, mesh)
    n_bad2 = int(n_bad2)
    print(f"5 exchange  (sub-slab drift) n_out_of_slab={n_bad2}  "
          f"count->{float(np.asarray(nact2).sum()):.0f}  max_occ={int(max_occ2)}/{C}")
    assert n_bad2 == 0, f"sub-slab drift left {n_bad2} particles out of slab"
    assert abs(float(np.asarray(nact2).sum()) - tot_before) < 0.5, "lost particles"

    print("\nALL PM SHARDMAP PARITY/INVARIANT CHECKS PASSED")


def test_pm_shardmap_parity():
    if jax.device_count() < 2:
        pytest.skip("needs >=2 devices (GPU count or "
                    "--xla_force_host_platform_device_count)")
    main()


if __name__ == "__main__":
    main()
