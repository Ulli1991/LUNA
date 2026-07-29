"""DM-only cosmological N-body runner (cfg.CosmoDM): MUSIC ICs, super-comoving
KDK leapfrog, comoving PM gravity (CIC deposit -> MG Poisson -> 4th-order force
-> gather; one solve/step). Particle-parallel across devices (psum'd density,
replicated solve); grid-sharded slab path via cfg.dm_domain_decomp. Gadget
units: x kpc/h comoving, p = a^2 x_dot km/s (= MUSIC u * a^1.5 at the IC)."""
import time

import jax
import jax.numpy as jnp
import numpy as np

from cosmology import Cosmology
from cosmology.music_ic import music_run_config, read_music_dm_ic
from gravity.particle_mesh import ParticleMeshGravity
from gravity.poisson_multigrid import (laplacian_7pt, solve_poisson,
                                       solve_poisson_fmg)
from file_io.write_gadget import write_gadget_dm_snapshot
from setup.cosmo_common import (CosmoClock, dtau_accel, poisson_coeff_fac,
                                softening_from_config)
from setup.run_report import StepLog, require_finite


def run_simulation_cosmo_dm(cfg, is_main_process=True):
    """DM-only super-comoving KDK N-body from MUSIC ICs (see module docstring).
    Returns ``{"a", "n_steps"}``; `main.run` wraps it in a RunResult."""
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

    # Grid-sharded (slab domain decomposition) path for grids beyond a single
    # device. Shards the mesh + MG solve along x and decomposes the particles
    # spatially; needed for >~512^3. Falls through to the particle-parallel
    # (replicated-grid) path below for the single-device / small-grid case.
    if getattr(cfg, "dm_domain_decomp", False) and jax.device_count() > 1:
        return _run_slab(cfg, ic, cosmo, H0, box, Om, a, a_end,
                         dloga_max, eta, n_extra, is_main_process)

    pm = ParticleMeshGravity(cfg)
    pos = jnp.asarray(ic.pos)                              # comoving kpc/h
    mass = jnp.asarray(ic.mass)
    active = jnp.ones(ic.n_part)
    p = jnp.asarray(ic.vel) * (a ** 1.5)                  # canonical momentum p=a^2 xdot

    clock = CosmoClock(cosmo, a, a_end, H0)

    # --- PM force (returns acceleration + delta_max), particle-parallel sharded ---
    # Warm start (cfg.dm_warm_vcycles = k > 0): carry phi between steps and apply
    # k V-cycles as a defect correction on last step's potential instead of a
    # cold full-FMG solve -- the field barely changes per step (dloga <= 0.02),
    # so 1-2 cycles recover the FMG answer at a fraction of the cost
    # (Horowitz 2026, arXiv:2607.10983). Default 0 = cold FMG, bit-identical
    # to the previous behavior.
    ndev = jax.device_count()
    coeff_fac = poisson_coeff_fac(Om, H0)
    warm = int(getattr(cfg, "dm_warm_vcycles", 0))

    mixed = bool(getattr(cfg, "dm_mixed_precision", False))
    n_pre = int(getattr(cfg, "dm_warm_n_pre", 1))
    n_post = int(getattr(cfg, "dm_warm_n_post", 1))

    extrap = int(getattr(cfg, "dm_warm_extrapolate", 0))

    def force_core(pos, mass, active, a_kick, c1, c2, phi_prev, phi_prev2):
        rho = pm.cic_deposit(pos, mass, active,            # this shard's particles
                             dtype=jnp.float32 if mixed else None)
        if ndev > 1:
            rho = jax.lax.psum(rho, "p")                   # global density (Allreduce)
        delta = rho / jnp.mean(rho) - 1.0
        # Warm-solve seed. extrap=1: rescale the carried phi by the source-
        # amplitude ratio a_next/a_prev (rhs ~ a*delta) — removes the systematic
        # ~dloga staleness for free. extrap=2: linear extrapolation from the
        # last two (rescaled) potentials; host supplies c1/c2.
        if warm > 0 and extrap == 2:
            phi_prev = c1 * phi_prev - c2 * phi_prev2
        elif warm > 0 and extrap == 1:
            phi_prev = c1 * phi_prev
        if warm > 0 and mixed:
            # defect correction: FP64 residual of last step's phi, FP32
            # V-cycle(s) on the correction, FP64 accumulate. Same converged
            # residual as the FP64 cycle at ~2/3 the cost (measured 512^3).
            rhs = jnp.float32(coeff_fac * a_kick) * delta \
                - laplacian_7pt(phi_prev, pm.dx, pm.halo_fn).astype(jnp.float32)
            e = solve_poisson(rhs, pm.dx, 1.0, 0,
                              n_levels=n_levels, n_cycles=warm,
                              n_pre=n_pre, n_post=n_post)
            phi = phi_prev + e
        elif warm > 0:
            phi = solve_poisson(delta, pm.dx, coeff_fac * a_kick, 0,
                                n_levels=n_levels, n_cycles=warm,
                                phi_init=phi_prev, n_pre=n_pre, n_post=n_post)
        else:
            phi = solve_poisson_fmg(delta.astype(phi_prev.dtype), pm.dx,
                                    coeff_fac * a_kick, boundary=0,
                                    n_levels=n_levels, n_extra_vcycles=n_extra)
        g = pm.grad_phi(phi, order=4)
        return pm.accel_at_points(g, pos), jnp.max(delta), phi

    def kdk_core(pos, p, g, mass, active, a_kick, dtau, c1, c2,
                 phi_prev, phi_prev2):
        # KDK leapfrog (super-comoving): half-kick, drift, refresh MG force at
        # a_kick, half-kick. One fused jit per step -- no eager 134M-particle
        # temporaries, and max|g|^2 comes back as a scalar for the next dtau.
        p_half = p + g * (0.5 * dtau)
        pos = jnp.mod(pos + p_half * dtau, box)
        g_new, dmax, phi = force_core(pos, mass, active, a_kick, c1, c2,
                                      phi_prev, phi_prev2)
        p_new = p_half + g_new * (0.5 * dtau)
        g2max = jnp.max(jnp.sum(g_new ** 2, axis=1))
        if ndev > 1:
            g2max = jax.lax.pmax(g2max, "p")
        return pos, p_new, g_new, phi, dmax, g2max

    if ndev > 1:
        from jax.experimental.shard_map import shard_map
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P
        mesh = Mesh(np.asarray(jax.devices()), ("p",))
        _sm = jax.jit(shard_map(
            force_core, mesh=mesh,
            in_specs=(P("p", None), P("p"), P("p"), P(), P(), P(), P(), P()),
            out_specs=(P("p", None), P(), P())))
        _skdk = jax.jit(shard_map(
            kdk_core, mesh=mesh,
            in_specs=(P("p", None), P("p", None), P("p", None), P("p"), P("p"),
                      P(), P(), P(), P(), P(), P()),
            out_specs=(P("p", None), P("p", None), P("p", None), P(), P(), P())))
        pos = jax.device_put(pos, NamedSharding(mesh, P("p", None)))
        p = jax.device_put(p, NamedSharding(mesh, P("p", None)))
        mass = jax.device_put(mass, NamedSharding(mesh, P("p")))
        active = jax.device_put(active, NamedSharding(mesh, P("p")))

        def force(pos, a_kick, c1, c2, phi_prev, phi_prev2):
            return _sm(pos, mass, active, a_kick, c1, c2, phi_prev, phi_prev2)

        def kdk(pos, p, g, phi, phi2, a_kick, dtau, c1, c2):
            return _skdk(pos, p, g, mass, active, a_kick, dtau, c1, c2,
                         phi, phi2)
        if is_main_process:
            print(f"[cosmo-dm] particle-parallel across {ndev} devices (shard_map)")
    else:
        _jf = jax.jit(force_core)
        # donate the phi that is consumed: with linear extrapolation the older
        # field (phi_prev2) dies each step; otherwise phi_prev does.
        _jk = jax.jit(kdk_core,
                      donate_argnums=(0, 1, 2, 10 if extrap == 2 else 9))

        def force(pos, a_kick, c1, c2, phi_prev, phi_prev2):
            return _jf(pos, mass, active, a_kick, c1, c2, phi_prev, phi_prev2)

        def kdk(pos, p, g, phi, phi2, a_kick, dtau, c1, c2):
            return _jk(pos, p, g, mass, active, a_kick, dtau, c1, c2,
                       phi, phi2)

    # gravitational timestep = acceleration criterion dtau ~ eta sqrt(eps/|a|),
    # softening eps -> PM cell (mesh-scale force resolution); NOT a hydro courant.
    eps_soft = softening_from_config(cfg, pm.dx)

    tau, step, t0 = 0.0, 0, time.time()
    if is_main_process:
        print(f"[cosmo-dm] {ic.n_part} DM particles  box={box:.0f} kpc/h  "
              f"mesh={cfg.Nx}^3  Om={Om} OL={cfg.cosmo_Omega_L} H0={H0:.5f}")
        print(f"[cosmo-dm] a_init={a:.4f} (z={1/a-1:.2f}) -> a_end={a_end:.4f}")
        print(f"[cosmo-dm] {'step':>5} {'a':>8} {'z':>8} {'dloga':>8} {'delta_max':>10}")
    rep = StepLog(cfg, is_main_process,
                  history_fields=["step", "a", "dloga", "delta_max"])

    phi = jnp.zeros((cfg.Nx,) * 3)                         # warm-start carry
    phi2 = jnp.zeros((cfg.Nx,) * 3) if extrap == 2 else jnp.float64(0.0)
    g, dmax, phi = force(pos, a, 1.0, 0.0, phi, phi2)      # force + delta_max
    g2max = float(jnp.max(jnp.sum(g ** 2, axis=1)))        # max |accel|^2 (host)
    a_prev, a_prev2 = a, None                              # epochs of phi, phi2
    while a < a_end - 1e-9:
        dtau = clock.cap_expansion(dtau_accel(eta, eps_soft, np.sqrt(g2max)),
                                   a, dloga_max)
        dtau = clock.cap_to_table(dtau, tau)
        a_next = clock.a_of_tau(tau + dtau)
        dloga = np.log(a_next / a)
        if extrap == 2 and a_prev2 is not None:
            c1, c2 = 2.0 * a_next / a_prev, a_next / a_prev2
        else:
            c1, c2 = a_next / a_prev, 0.0                  # rescale-only seed
        pos, p, g, phi_new, dmax, g2 = kdk(pos, p, g, phi, phi2, a_next, dtau,
                                           float(c1), float(c2))
        if extrap == 2:
            phi2, a_prev2 = phi, a_prev
        phi, a_prev = phi_new, a_next
        g2max = float(g2)
        tau += dtau
        a = a_next
        step += 1
        require_finite(float(dmax), what="DM overdensity", step=step, t=tau,
                       hint="PM N-body state went NaN.")
        rep.step(step, f"[cosmo-dm] {step:5d} {a:8.4f} {1/a-1:8.4f} "
                 f"{dloga:8.5f} {float(dmax):10.3f}",
                 row=[step, a, dloga, float(dmax)], force=a >= a_end - 1e-9)

    if is_main_process:
        print(f"[cosmo-dm] done: {step} steps, a={a:.4f} (z={1/a-1:.4f}), "
              f"{time.time()-t0:.1f}s")
        write_gadget_dm_snapshot(cfg, np.asarray(pos), np.asarray(p) / a ** 1.5, a, ic)
    return {"a": a, "n_steps": step}


def _run_slab(cfg, ic, cosmo, H0, box, Om, a, a_end, dloga_max, eta,
              n_extra, is_main_process):
    """Grid-sharded (1-D slab) DM N-body. Mesh + MG shard along x; particles are
    spatially decomposed into per-slab fixed buffers and exchanged on drift.

    Same super-comoving KDK leapfrog and comoving PM gravity as the
    particle-parallel path -- only the data layout differs:
      deposit_slab -> solve_poisson_fmg_shardmap -> grad_phi_slab -> gather_slab,
    one MG solve per step (end-force reuse), exchange after each drift.
    """
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from gravity.particle_mesh_shardmap import (
        cic_deposit_slab, decompose_to_slabs, exchange_particles_slab,
        gather_accel_slab, grad_phi_slab, make_slab_mesh)
    from gravity.poisson_multigrid import _max_levels, solve_poisson_fmg_shardmap

    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    dx = box / Nx
    ndev = jax.device_count()
    if Nx % ndev:
        raise ValueError(f"slab decomposition needs Nx ({Nx}) divisible by ndev ({ndev})")
    mesh = make_slab_mesh(jax.devices())
    nx_loc = Nx // ndev
    n_levels = min(_max_levels(Nx), int(np.log2(nx_loc)) + 1)
    coeff_fac = poisson_coeff_fac(Om, H0)
    cap = float(getattr(cfg, "dm_capacity_factor", 1.5))
    eps_soft = softening_from_config(cfg, dx)

    # host-side spatial decomposition into per-slab fixed-capacity buffers
    pos0 = np.asarray(ic.pos)
    mom0 = np.asarray(ic.vel) * (a ** 1.5)                 # canonical momentum p=a^2 xdot
    mass0 = np.asarray(ic.mass)
    pos_g, mom_g, mass_g, act_g, C = decompose_to_slabs(
        pos0, mom0, mass0, Nx, ndev, dx, box, cap)
    posS = jax.device_put(jnp.asarray(pos_g), NamedSharding(mesh, P("x", None)))
    pS = jax.device_put(jnp.asarray(mom_g), NamedSharding(mesh, P("x", None)))
    massS = jax.device_put(jnp.asarray(mass_g), NamedSharding(mesh, P("x")))
    actS = jax.device_put(jnp.asarray(act_g), NamedSharding(mesh, P("x")))

    clock = CosmoClock(cosmo, a, a_end, H0)

    @jax.jit
    def force(pos, mass, active, a_kick):
        # MUST be jitted: solve_poisson_fmg_shardmap rebuilds a fresh jitted FMG
        # closure each call, so an unjitted force recompiles the multigrid every
        # step. Jitting traces the body once -> FMG compiles once, steps cache-hit.
        rho = cic_deposit_slab(pos, mass, active, Nx, Ny, Nz, dx, mesh)
        delta = rho / jnp.mean(rho) - 1.0
        phi = solve_poisson_fmg_shardmap(
            delta, dx, coeff_fac * a_kick, boundary=0, mesh=mesh,
            n_levels=n_levels, n_extra_vcycles=n_extra)
        g = grad_phi_slab(phi, dx, mesh, order=4)
        return gather_accel_slab(g, pos, Nx, Ny, Nz, dx, mesh), jnp.max(delta)

    if is_main_process:
        print(f"[cosmo-dm] grid-sharded slab decomposition across {ndev} devices "
              f"(nx_loc={nx_loc}, capacity {ndev*C} for {ic.n_part} particles)")
        print(f"[cosmo-dm] {ic.n_part} DM particles  box={box:.0f} kpc/h  "
              f"mesh={Nx}^3  Om={Om} OL={cfg.cosmo_Omega_L} H0={H0:.5f}")
        print(f"[cosmo-dm] a_init={a:.4f} (z={1/a-1:.2f}) -> a_end={a_end:.4f}")
        print(f"[cosmo-dm] {'step':>5} {'a':>8} {'z':>8} {'dloga':>8} "
              f"{'delta_max':>10} {'max_occ':>8} {'n_bad':>6}")

    tau, step, t0 = 0.0, 0, time.time()
    if is_main_process:
        print("[cosmo-dm] compiling first PM force "
              "(deposit -> FMG -> grad -> gather) ...", flush=True)
    tc = time.time()
    g, dmax = force(posS, massS, actS, a)
    jax.block_until_ready(g)
    if is_main_process:
        print(f"[cosmo-dm] first force compiled+ran in {time.time()-tc:.1f}s", flush=True)
    rep = StepLog(cfg, is_main_process,
                  history_fields=["step", "a", "dloga", "delta_max",
                                  "max_occ", "n_bad"])
    n_bad_tot = 0
    while a < a_end - 1e-9:
        gmag = jnp.sqrt(jnp.sum(g ** 2, axis=1))
        gmax = float(jnp.max(jnp.where(actS > 0, gmag, 0.0)))   # active slots only
        dtau = clock.cap_expansion(dtau_accel(eta, eps_soft, gmax), a, dloga_max)
        dtau = clock.cap_to_table(dtau, tau)
        a_next = clock.a_of_tau(tau + dtau)
        dloga = np.log(a_next / a)
        # KDK: half-kick @a, drift, exchange to owning slab, refresh force @a_next, half-kick.
        pS = pS + g * (0.5 * dtau)
        posS = jnp.mod(posS + pS * dtau, box)
        posS, pS, massS, actS, n_bad, max_occ = exchange_particles_slab(
            posS, pS, massS, actS, Nx, dx, box, mesh)
        g, dmax = force(posS, massS, actS, a_next)
        pS = pS + g * (0.5 * dtau)
        tau += dtau
        a = a_next
        step += 1
        n_bad_tot += int(n_bad)
        rep.step(step, f"[cosmo-dm] {step:5d} {a:8.4f} {1/a-1:8.4f} "
                 f"{dloga:8.5f} {float(dmax):10.3f} {int(max_occ):8d} "
                 f"{int(n_bad):6d} {(time.time()-t0)/step:6.2f}s/step",
                 row=[step, a, dloga, float(dmax), int(max_occ), int(n_bad)],
                 force=a >= a_end - 1e-9)

    if is_main_process:
        print(f"[cosmo-dm] done: {step} steps, a={a:.4f} (z={1/a-1:.4f}), "
              f"{time.time()-t0:.1f}s  (cumulative out-of-slab events: {n_bad_tot})")
        # gather active particles to host, restore Gadget velocity u = p / a^{3/2}
        pos_h = np.asarray(posS)
        mom_h = np.asarray(pS)
        act_h = np.asarray(actS) > 0
        sel = np.nonzero(act_h)[0]
        write_gadget_dm_snapshot(cfg, pos_h[sel], mom_h[sel] / a ** 1.5, a, ic)
    return {"a": a, "n_steps": step, "n_out_of_slab": n_bad_tot}
