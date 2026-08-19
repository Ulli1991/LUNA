"""Single-GPU gravity roofline: time the PM pipeline's two hot kernels —
CIC deposit and the multigrid Poisson solve (warm 1-V-cycle and cold FMG) —
in isolation on a MUSIC IC, at one grid size per invocation.

  LUNA_GRAV_IC     MUSIC Arepo-HDF5 IC path
  LUNA_GRAV_NX     PM mesh per side (= particle count per side for unigrid)
  LUNA_GRAV_REPS   timed repetitions per kernel (default 20)

Prints one line per kernel:
  [grav-roofline] nx=512 op=deposit  compile=3.2s  ms=12.34  rate=10883.1 Mpart/s
Rates: deposit/gather in Mpart/s, vcycle/fmg in Mcell/s.  No snapshots, no
integration — pure kernel timing, medians over reps.
"""
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.getcwd())

import jax.numpy as jnp
import numpy as np
from dataclasses import replace

from cosmology.music_ic import music_run_config, read_music_dm_ic
from gravity.particle_mesh import ParticleMeshGravity
from gravity.poisson_multigrid import solve_poisson, solve_poisson_fmg
from setup.cosmo_common import gravity_precision_from_config, poisson_coeff_fac
from template_config import Config


def bench(name, nx, fn, *args, reps, work, unit):
    fn = jax.jit(fn)
    t0 = time.time()
    out = jax.block_until_ready(fn(*args))          # compile + first run
    t_compile = time.time() - t0
    ts = []
    for _ in range(reps):
        t0 = time.time()
        jax.block_until_ready(fn(*args))
        ts.append(time.time() - t0)
    ms = 1e3 * float(np.median(ts))
    print(f"[grav-roofline] nx={nx} op={name:8s} compile={t_compile:6.1f}s "
          f"ms={ms:9.3f} rate={work / ms / 1e3:10.1f} {unit}", flush=True)
    return out


def main():
    ic_path = os.environ["LUNA_GRAV_IC"]
    nx = int(os.environ["LUNA_GRAV_NX"])
    reps = int(os.environ.get("LUNA_GRAV_REPS", "20"))

    ic = read_music_dm_ic(ic_path)
    cfg = replace(Config(), CosmoDM=True, CosmologicalIntegration=True,
                  Nx=nx, music_ic_path=ic_path,
                  production_mode=True, PLOT_AFTER_SIM=False)
    cfg = music_run_config(ic, nx, base=cfg)
    pm = ParticleMeshGravity(cfg)
    prec = gravity_precision_from_config(cfg)
    print(f"[grav-roofline] {ic.n_part} particles  mesh={nx}^3  "
          f"box={float(cfg.Lx):.0f} kpc/h  {prec}", flush=True)

    pos = jnp.asarray(ic.pos, prec.particles)
    mass = jnp.asarray(ic.mass, prec.particles)
    active = jnp.ones(ic.n_part, prec.particles)
    mass_unit = float(jnp.sum(mass)) / float(cfg.Lx) ** 3
    coeff = poisson_coeff_fac(float(cfg.cosmo_Omega_m), float(cfg.cosmo_H0))
    dx = float(pm.dx)
    npart, ncell = ic.n_part, nx ** 3

    def deposit(pos):
        return pm.cic_deposit(pos, mass, active, dtype=prec.deposit,
                              fixed=prec.fixed_int if prec.fixed_deposit else None,
                              fixed_bits=prec.fixed_bits, mass_unit=mass_unit)

    rho = bench("deposit", nx, deposit, pos, reps=reps,
                work=npart, unit="Mpart/s")
    rho = rho.astype(prec.phi)
    rhs = rho - jnp.mean(rho)                        # periodic-solvable RHS

    def fmg(rhs):
        return solve_poisson_fmg(rhs, dx, coeff, 0)

    phi = bench("fmg", nx, fmg, rhs, reps=reps, work=ncell, unit="Mcell/s")

    def vcycle(rhs, phi):                            # warm-start defect V(1,1)
        return solve_poisson(rhs, dx, coeff, 0, n_cycles=1,
                             n_pre=1, n_post=1, phi_init=phi)

    phi = bench("vcycle", nx, vcycle, rhs, phi, reps=reps,
                work=ncell, unit="Mcell/s")

    def gather(phi, pos):
        g = pm.grad_phi(phi, order=4)
        return pm.accel_at_points(g, pos)

    bench("gather", nx, gather, phi, pos, reps=reps, work=npart, unit="Mpart/s")

    # Fused full force step, both production modes: warm (deposit -> 1 V-cycle
    # defect correction on last step's phi -> gather) and cold (deposit -> FMG
    # -> gather). One jit each, so XLA can fuse across the phases.
    def force_warm(pos, phi):
        rho = deposit(pos).astype(prec.phi)
        rhs = rho - jnp.mean(rho)
        phi = solve_poisson(rhs, dx, coeff, 0, n_cycles=1,
                            n_pre=1, n_post=1, phi_init=phi)
        return pm.accel_at_points(pm.grad_phi(phi, order=4), pos), phi

    def force_cold(pos):
        rho = deposit(pos).astype(prec.phi)
        rhs = rho - jnp.mean(rho)
        phi = solve_poisson_fmg(rhs, dx, coeff, 0)
        return pm.accel_at_points(pm.grad_phi(phi, order=4), pos), phi

    bench("f_warm", nx, force_warm, pos, phi, reps=reps,
          work=npart, unit="Mpart/s")
    bench("f_cold", nx, force_cold, pos, reps=reps, work=npart, unit="Mpart/s")

    # Full KDK leapfrog step, device-side only (half-kick -> drift -> warm
    # force -> half-kick), fixed dtau: isolates integrator arithmetic from
    # the runner's host-side dt logic / logging / sync.
    box = float(cfg.Lx)
    dtau = 1e-4
    g0, _ = force_warm(pos, phi)

    def kdk(pos, p, g, phi):
        p_half = p + g * (0.5 * dtau)
        pos = jnp.mod(pos + p_half * dtau, box)
        (g_new, phi), _ = (force_warm(pos, phi), None)
        return pos, p_half + g_new * (0.5 * dtau), g_new, phi

    p0 = jnp.zeros_like(pos)
    bench("kdk", nx, kdk, pos, p0, g0, phi, reps=reps,
          work=npart, unit="Mpart/s")
    print("[grav-roofline] done", flush=True)


if __name__ == "__main__":
    main()
