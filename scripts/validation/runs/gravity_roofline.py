"""Single-GPU gravity roofline: the PM pipeline's hot kernels (CIC deposit,
the multigrid Poisson solve warm and cold, the gather, the fused force step
and one KDK step) timed in isolation on a MUSIC IC, one grid size per call.

    python -m scripts.validation.runs.gravity_roofline ic=music/ic_l9.hdf5 nx=512 reps=20

Prints one line per kernel, e.g.
  [grav-roofline] nx=512 op=deposit  compile=3.2s  ms=12.34  rate=10883.1 Mpart/s
Rates: deposit/gather in Mpart/s, vcycle/fmg in Mcell/s; medians over reps.
Parameters: ic, nx (PM mesh per side), reps.
"""
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp                       # noqa: E402
import numpy as np                            # noqa: E402
from dataclasses import replace               # noqa: E402

from luna.cosmology.music_ic import music_run_config, read_music_dm_ic     # noqa: E402
from luna.gravity.particle_mesh import ParticleMeshGravity                 # noqa: E402
from luna.gravity.poisson_multigrid import solve_poisson, solve_poisson_fmg  # noqa: E402
from luna.setup.cli import args                                            # noqa: E402
from luna.setup.cosmo_common import gravity_precision_from_config, poisson_coeff_fac  # noqa: E402
from luna.template_config import Config                                    # noqa: E402


def bench(name, nx, fn, *fargs, reps, work, unit):
    fn = jax.jit(fn)
    t0 = time.time()
    out = jax.block_until_ready(fn(*fargs))          # compile + first run
    t_compile = time.time() - t0
    ts = []
    for _ in range(reps):
        t0 = time.time()
        jax.block_until_ready(fn(*fargs))
        ts.append(time.time() - t0)
    ms = 1e3 * float(np.median(ts))
    print(f"[grav-roofline] nx={nx} op={name:8s} compile={t_compile:6.1f}s "
          f"ms={ms:9.3f} rate={work / ms / 1e3:10.1f} {unit}", flush=True)
    return out


def main(ic="", nx=512, reps=20):
    if not ic:
        raise SystemExit("ic=<MUSIC Arepo-HDF5 path> is required")
    ic = read_music_dm_ic(ic)
    cfg = replace(Config(), run="dm", CosmologicalIntegration=True, Nx=nx,
                  music_ic_path="(loaded)", PLOT_AFTER_SIM=False)
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

    rho = bench("deposit", nx, deposit, pos, reps=reps, work=npart, unit="Mpart/s")
    rho = rho.astype(prec.phi)
    rhs = rho - jnp.mean(rho)                        # periodic-solvable RHS

    def fmg(rhs):
        return solve_poisson_fmg(rhs, dx, coeff, 0)

    phi = bench("fmg", nx, fmg, rhs, reps=reps, work=ncell, unit="Mcell/s")

    def vcycle(rhs, phi):                            # warm-start defect V(1,1)
        return solve_poisson(rhs, dx, coeff, 0, n_cycles=1, n_pre=1, n_post=1, phi_init=phi)

    phi = bench("vcycle", nx, vcycle, rhs, phi, reps=reps, work=ncell, unit="Mcell/s")

    def gather(phi, pos):
        return pm.accel_at_points(pm.grad_phi(phi, order=4), pos)

    bench("gather", nx, gather, phi, pos, reps=reps, work=npart, unit="Mpart/s")

    # the fused force step in both production modes, one jit each
    def force_warm(pos, phi):
        rho = deposit(pos).astype(prec.phi)
        rhs = rho - jnp.mean(rho)
        phi = solve_poisson(rhs, dx, coeff, 0, n_cycles=1, n_pre=1, n_post=1, phi_init=phi)
        return pm.accel_at_points(pm.grad_phi(phi, order=4), pos), phi

    def force_cold(pos):
        rho = deposit(pos).astype(prec.phi)
        rhs = rho - jnp.mean(rho)
        phi = solve_poisson_fmg(rhs, dx, coeff, 0)
        return pm.accel_at_points(pm.grad_phi(phi, order=4), pos), phi

    bench("f_warm", nx, force_warm, pos, phi, reps=reps, work=npart, unit="Mpart/s")
    bench("f_cold", nx, force_cold, pos, reps=reps, work=npart, unit="Mpart/s")

    # one KDK leapfrog step, device-side only, fixed dtau
    box = float(cfg.Lx)
    dtau = 1e-4
    g0, _ = force_warm(pos, phi)

    def kdk(pos, p, g, phi):
        p_half = p + g * (0.5 * dtau)
        pos = jnp.mod(pos + p_half * dtau, box)
        g_new, phi = force_warm(pos, phi)
        return pos, p_half + g_new * (0.5 * dtau), g_new, phi

    bench("kdk", nx, kdk, pos, jnp.zeros_like(pos), g0, phi, reps=reps, work=npart,
          unit="Mpart/s")
    print("[grav-roofline] done", flush=True)


if __name__ == "__main__":
    main(**args(ic="", nx=512, reps=20))
