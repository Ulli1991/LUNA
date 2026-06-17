"""DM-only cosmological N-body runner (dispatched from main.run for a Cosmo-DM
run). Loads MUSIC dark-matter ICs, configures box + cosmology from the header,
and evolves the particles with a super-comoving KDK leapfrog and the comoving
multigrid particle-mesh gravity:

    CIC deposit  ->  MG Poisson  lap(phi~) = (3/2) H0^2 Omega_m a * delta
                 ->  4th-order force  ->  gather to particles.

Multi-GPU (particle-parallel): the particles are sharded by index over the
device mesh; each shard deposits its own particles, an all-reduce (psum) forms
the global density, the MG solve is replicated, and the force gather is local --
one collective per step, fixed shapes, no particle exchange. (Full grid-sharded
domain decomposition is a later step.)

Variables (Gadget/Arepo units: kpc/h, km/s, 1e10 Msun/h):
  * comoving position x (kpc/h), wrapped periodically in the box;
  * canonical momentum p = a^2 dx/dt (km/s); from the MUSIC Gadget velocity
    u = sqrt(a) dx/dt this is p = u * a^{3/2} at the IC.
  * super-comoving time tau (dtau = dt/a^2): the leapfrog is the plain drift
    x += p dtau and kick p += g~ dtau; a is advanced through the a(tau) table.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

from cosmology import Cosmology
from cosmology.music_ic import music_run_config, read_music_dm_ic
from gravity.particle_mesh import ParticleMeshGravity
from gravity.poisson_multigrid import solve_poisson_fmg


def run_simulation_cosmo_dm(cfg, is_main_process=True):
    ic = read_music_dm_ic(cfg.music_ic_path)
    cfg = music_run_config(ic, cfg.Nx, base=cfg)          # box + cosmology from IC
    cosmo = Cosmology(Omega_m=cfg.cosmo_Omega_m, Omega_L=cfg.cosmo_Omega_L, h=ic.h)
    H0 = float(cfg.cosmo_H0)
    box = float(cfg.Lx)
    Om = float(cfg.cosmo_Omega_m)
    a = float(cfg.a_init)
    a_end = float(getattr(cfg, "a_end", 1.0))
    dloga_max = float(getattr(cfg, "dloga_max", 0.02))
    eta = float(getattr(cfg, "dm_eta", 0.025))            # acceleration-timestep accuracy
    n_extra = int(getattr(cfg, "POISSON_FMG_EXTRA_VCYCLES", 2))
    n_levels = getattr(cfg, "POISSON_N_LEVELS", None)

    pm = ParticleMeshGravity(cfg)
    pos = jnp.asarray(ic.pos)                              # comoving kpc/h
    mass = jnp.asarray(ic.mass)
    active = jnp.ones(ic.n_part)
    p = jnp.asarray(ic.vel) * (a ** 1.5)                  # canonical momentum p=a^2 xdot

    a_tab, tau_tab = cosmo.build_a_of_tau(a, a_end + 0.05, H0)

    def a_of_tau(t):
        return float(np.interp(t, tau_tab, a_tab))

    # --- PM force (returns acceleration + delta_max), particle-parallel sharded ---
    ndev = jax.device_count()
    coeff_fac = 1.5 * Om * H0 * H0

    def force_core(pos, mass, active, a_kick):
        rho = pm.cic_deposit(pos, mass, active)            # this shard's particles
        if ndev > 1:
            rho = jax.lax.psum(rho, "p")                   # global density (Allreduce)
        delta = rho / jnp.mean(rho) - 1.0
        phi = solve_poisson_fmg(delta, pm.dx, coeff_fac * a_kick, boundary=0,
                                n_levels=n_levels, n_extra_vcycles=n_extra)
        g = pm.grad_phi(phi, order=4)
        return pm.accel_at_points(g, pos), jnp.max(delta)

    if ndev > 1:
        from jax.experimental.shard_map import shard_map
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P
        mesh = Mesh(np.asarray(jax.devices()), ("p",))
        _sm = jax.jit(shard_map(
            force_core, mesh=mesh,
            in_specs=(P("p", None), P("p"), P("p"), P()),
            out_specs=(P("p", None), P())))
        pos = jax.device_put(pos, NamedSharding(mesh, P("p", None)))
        p = jax.device_put(p, NamedSharding(mesh, P("p", None)))
        mass = jax.device_put(mass, NamedSharding(mesh, P("p")))
        active = jax.device_put(active, NamedSharding(mesh, P("p")))

        def force(pos, a_kick):
            return _sm(pos, mass, active, a_kick)
        if is_main_process:
            print(f"[cosmo-dm] particle-parallel across {ndev} devices (shard_map)")
    else:
        _jf = jax.jit(force_core)

        def force(pos, a_kick):
            return _jf(pos, mass, active, a_kick)

    # gravitational timestep = acceleration criterion dtau ~ eta sqrt(eps/|a|),
    # softening eps -> PM cell (mesh-scale force resolution); NOT a hydro courant.
    eps_soft = float(getattr(cfg, "dm_softening", 0.0)) or pm.dx

    tau, step, t0 = 0.0, 0, time.time()
    if is_main_process:
        print(f"[cosmo-dm] {ic.n_part} DM particles  box={box:.0f} kpc/h  "
              f"mesh={cfg.Nx}^3  Om={Om} OL={cfg.cosmo_Omega_L} H0={H0:.5f}")
        print(f"[cosmo-dm] a_init={a:.4f} (z={1/a-1:.2f}) -> a_end={a_end:.4f}")
        print(f"[cosmo-dm] {'step':>5} {'a':>8} {'z':>8} {'dloga':>8} {'delta_max':>10}")

    g, dmax = force(pos, a)                                # force + delta_max
    while a < a_end - 1e-9:
        gmax = float(jnp.max(jnp.sqrt(jnp.sum(g ** 2, axis=1))))   # max |acceleration|
        dtau = min(dloga_max / (a * a * cosmo.H(a, H0)),    # cap d ln a
                   eta * np.sqrt(eps_soft / (gmax + 1e-30)))  # gravity accel criterion
        if tau + dtau > tau_tab[-1]:
            dtau = tau_tab[-1] - tau
        a_next = a_of_tau(tau + dtau)
        dloga = np.log(a_next / a)
        # KDK leapfrog (super-comoving): half-kick @a, drift, refresh MG force
        # @a_next, half-kick @a_next; end force reused -> one MG solve/step.
        p = p + g * (0.5 * dtau)
        pos = jnp.mod(pos + p * dtau, box)
        g, dmax = force(pos, a_next)
        p = p + g * (0.5 * dtau)
        tau += dtau
        a = a_next
        step += 1
        if is_main_process and (step % 10 == 0 or a >= a_end - 1e-9):
            print(f"[cosmo-dm] {step:5d} {a:8.4f} {1/a-1:8.4f} {dloga:8.5f} "
                  f"{float(dmax):10.3f}")

    if is_main_process:
        print(f"[cosmo-dm] done: {step} steps, a={a:.4f} (z={1/a-1:.4f}), "
              f"{time.time()-t0:.1f}s")
        _write_snapshot(cfg, np.asarray(pos), np.asarray(p) / a ** 1.5, a, ic)
    return {"a": a, "n_steps": step}


def _write_snapshot(cfg, pos, vel_gadget, a, ic):
    """Gadget/Arepo-style DM particle snapshot + a density-slice figure."""
    import os

    import h5py
    out_dir = getattr(cfg, "output_dir", "/ptmp/uli/jax-hydro/snapshots/cosmo_dm")
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, f"snap_a{a:.4f}.hdf5")
    n = pos.shape[0]
    with h5py.File(fn, "w") as f:
        H = f.create_group("Header")
        H.attrs["BoxSize"] = float(cfg.Lx)
        H.attrs["Time"] = a
        H.attrs["Redshift"] = 1.0 / a - 1.0
        H.attrs["Omega0"] = cfg.cosmo_Omega_m
        H.attrs["OmegaLambda"] = cfg.cosmo_Omega_L
        H.attrs["HubbleParam"] = ic.h
        H.attrs["NumPart_ThisFile"] = np.array([0, n, 0, 0, 0, 0], dtype=np.int64)
        H.attrs["MassTable"] = np.array([0, float(ic.mass[0]), 0, 0, 0, 0])
        g = f.create_group("PartType1")
        g.create_dataset("Coordinates", data=pos.astype(np.float32))
        g.create_dataset("Velocities", data=vel_gadget.astype(np.float32))
    print(f"[cosmo-dm] wrote {fn}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Nx = cfg.Nx
        H2, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=Nx,
                                  range=[[0, cfg.Lx], [0, cfg.Lx]])
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(np.log10(H2.T + 0.5), origin="lower", cmap="inferno")
        ax.set_title(f"DM (z={1/a-1:.2f})"); ax.set_xticks([]); ax.set_yticks([])
        os.makedirs("figures", exist_ok=True)
        fig.tight_layout(); fig.savefig("figures/cosmo_dm.png", dpi=130)
        print("[cosmo-dm] wrote figures/cosmo_dm.png")
    except Exception as e:
        print(f"[cosmo-dm] plot skipped: {e}")
