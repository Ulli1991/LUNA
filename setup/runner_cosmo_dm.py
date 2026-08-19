"""DM-only cosmological N-body runner (cfg.CosmoDM): MUSIC ICs, super-comoving
KDK leapfrog, comoving PM gravity (CIC deposit -> MG Poisson -> 4th-order force
-> gather; one solve/step). Multi-device ALWAYS goes through the grid-sharded
decomposition (cfg.dm_decomp: 3D blocks, or legacy slab) -- the mesh is never
replicated per device. The particle-parallel path is single-device only. Gadget
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
from setup.cosmo_common import (CosmoClock, dtau_accel,
                                gravity_precision_from_config,
                                poisson_coeff_fac, softening_from_config)
from setup.run_report import StepLog, require_finite


def _fixed_ceiling(prec):
    """Largest 1+delta a fixed-point deposit can hold before integer overflow."""
    bits = 63 if prec.fixed_int is jnp.int64 else 31
    return float(2 ** bits - 1) / float(2 ** prec.fixed_bits)


def run_simulation_cosmo_dm(cfg, is_main_process=True):
    """DM-only super-comoving KDK N-body from MUSIC ICs (see module docstring).
    Returns ``{"a", "n_steps"}``; `main.run` wraps it in a RunResult."""
    # Multi-process slab runs read only this rank's contiguous index-chunk
    # (~its x-slab, by the IC's x-major lattice order) — a full read per
    # process is 56 GB at 1024^3 and impossible at 4096^3. The global count
    # keeps the per-slab capacity identical on every rank.
    ic_n_total = None
    if (jax.process_count() > 1
            and getattr(cfg, "dm_decomp", "blocks") == "slab"):
        from cosmology.music_ic import music_part_count
        nproc, pid = jax.process_count(), jax.process_index()
        ic_n_total = music_part_count(cfg.music_ic_path, 1)
        lo = pid * ic_n_total // nproc
        hi = ic_n_total if pid == nproc - 1 else (pid + 1) * ic_n_total // nproc
        ic = read_music_dm_ic(cfg.music_ic_path, 1, lo, hi)
        print(f"[cosmo-dm] rank {pid}/{nproc}: chunked IC read [{lo}:{hi}]",
              flush=True)
    else:
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
    # Dispatch by size (measured, dm_timing_matrix 2026-08-03): the
    # particle-parallel path (replicated mesh, below) wins where the mesh fits
    # comfortably on every device -- 39.2 ms/step at 512^3/2 GPU vs 62.5 for
    # the grid-sharded split. Beyond that (mesh > ~512^3, or multi-process)
    # the grid-sharded decomposition is the only route: the replicated grid
    # caps at single-device memory and its fused graph stops compiling.
    if jax.device_count() > 1 and (cfg.Nx > 512 or jax.process_count() > 1
                                   or getattr(cfg, "dm_force_blocks", False)):
        if getattr(cfg, "dm_decomp", "blocks") == "slab":
            return _run_slab(cfg, ic, cosmo, H0, box, Om, a, a_end,
                             dloga_max, eta, n_extra, is_main_process,
                             n_total=ic_n_total)
        return _run_blocks(cfg, ic, cosmo, H0, box, Om, a, a_end,
                           dloga_max, eta, n_extra, is_main_process)

    pm = ParticleMeshGravity(cfg)
    prec = gravity_precision_from_config(cfg)             # one precision switch
    pos = jnp.asarray(ic.pos, prec.particles)             # comoving kpc/h
    mass = jnp.asarray(ic.mass, prec.particles)
    active = jnp.ones(ic.n_part, prec.particles)
    p = jnp.asarray(ic.vel, prec.particles) * (a ** 1.5)  # canonical momentum p=a^2 xdot
    # mean density: the fixed-point deposit's integer unit (mass is conserved,
    # so this is a constant of the run).
    mass_unit = float(jnp.sum(mass * active)) / (box ** 3)

    clock = CosmoClock(cosmo, a, a_end, H0)

    # --- PM force (returns acceleration + delta_max), particle-parallel sharded ---
    # Warm start (cfg.dm_warm_vcycles = k > 0): carry phi between steps and apply
    # k V-cycles as a defect correction on last step's potential instead of a
    # cold full-FMG solve -- the field barely changes per step (dloga <= 0.02),
    # so 1-2 cycles recover the FMG answer at a fraction of the cost. This is
    # the standard warm start of time-dependent multigrid (RAMSES does the
    # same). Default 0 = cold FMG, bit-identical to the previous behavior.
    ndev = jax.device_count()
    coeff_fac = poisson_coeff_fac(Om, H0)
    warm = int(getattr(cfg, "dm_warm_vcycles", 0))

    mixed = prec.cycle_mixed
    n_pre = int(getattr(cfg, "dm_warm_n_pre", 1))
    n_post = int(getattr(cfg, "dm_warm_n_post", 1))

    extrap = int(getattr(cfg, "dm_warm_extrapolate", 0))
    if is_main_process:
        print(f"[cosmo-dm] {prec}")

    def force_core(pos, mass, active, a_kick, c1, c2, phi_prev, phi_prev2):
        rho = pm.cic_deposit(pos, mass, active,            # this shard's particles
                             dtype=prec.deposit,
                             fixed=prec.fixed_int if prec.fixed_deposit else None,
                             fixed_bits=prec.fixed_bits, mass_unit=mass_unit)
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
        acc = pm.accel_at_points(g, pos).astype(prec.particles)
        return acc, jnp.max(delta), phi

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

    phi = jnp.zeros((cfg.Nx,) * 3, prec.phi)               # warm-start carry
    phi2 = (jnp.zeros((cfg.Nx,) * 3, prec.phi) if extrap == 2
            else jnp.zeros((), prec.phi))
    g, dmax, phi = force(pos, a, 1.0, 0.0, phi, phi2)      # force + delta_max
    g2max = float(jnp.max(jnp.sum(g ** 2, axis=1)))        # max |accel|^2 (host)
    a_prev, a_prev2 = a, None                              # epochs of phi, phi2
    dev_ts = []                            # per-step device wall (see refresh)
    host_every = int(getattr(cfg, "dm_host_every", 1))
    t_ref, step_ref = time.time(), 0
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
        t_dev = time.time()
        pos, p, g, phi_new, dmax, g2 = kdk(pos, p, g, phi, phi2, a_next, dtau,
                                           float(c1), float(c2))
        if extrap == 2:
            phi2, a_prev2 = phi, a_prev
        phi, a_prev = phi_new, a_next
        tau += dtau
        a = a_next
        step += 1
        # Host pulls (float() = device sync) cost ~10-25 ms each on GH200
        # (discrete HBM; near-free on MI300A's unified memory). Amortize:
        # refresh dt criterion + watchdogs every dm_host_every steps, with a
        # safety factor on the reused dtau (dt drifts <1%/step at these z).
        if step % host_every == 0 or a >= a_end - 1e-9:
            g2max = float(g2)                  # host pull = device sync point
            now = time.time()
            dev_ts.append((now - t_ref) / (step - step_ref))
            t_ref, step_ref = now, step
            require_finite(float(dmax), what="DM overdensity", step=step,
                           t=tau, hint="PM N-body state went NaN.")
            if (prec.fixed_deposit
                    and 1.0 + float(dmax) > 0.5 * _fixed_ceiling(prec)):
                raise RuntimeError(
                    f"fixed-point deposit: 1+delta_max = {1 + float(dmax):.0f}"
                    f" is within 2x of the integer ceiling "
                    f"{_fixed_ceiling(prec):.0f} ({prec.fixed_int.__name__} "
                    "at S=2^{prec.fixed_bits}). Lower gravity_deposit_fixed's"
                    " scale or use gravity_precision='double' (int64).")
            rep.step(step, f"[cosmo-dm] {step:5d} {a:8.4f} {1/a-1:8.4f} "
                     f"{dloga:8.5f} {float(dmax):10.3f}",
                     row=[step, a, dloga, float(dmax)],
                     force=a >= a_end - 1e-9)

    if is_main_process:
        wall = time.time() - t0
        med_dev = 1e3 * float(np.median(dev_ts)) if dev_ts else 0.0
        print(f"[cosmo-dm] done: {step} steps, a={a:.4f} (z={1/a-1:.4f}), "
              f"{wall:.1f}s")
        print(f"[cosmo-dm] timing: median device step {med_dev:.1f} ms, "
              f"mean total step {1e3 * wall / max(step, 1):.1f} ms "
              f"(rest = host dt/watchdog/log)")
        if getattr(cfg, "dm_write_snapshot", True):
            write_gadget_dm_snapshot(cfg, np.asarray(pos), np.asarray(p) / a ** 1.5, a, ic)
    return {"a": a, "n_steps": step}


def _balanced_pdims(ndev):
    """Factor ndev into a balanced (px,py,pz), largest factors first."""
    pdims = [1, 1, 1]
    n = ndev
    axis = 0
    while n > 1:
        f = 2
        while n % f:
            f += 1
        pdims[axis % 3] *= f
        n //= f
        axis += 1
    return tuple(sorted(pdims, reverse=True))


def _run_blocks(cfg, ic, cosmo, H0, box, Om, a, a_end, dloga_max, eta,
                n_extra, is_main_process):
    """3D-block DM N-body on the hydro's meshblock decomposition: deposit /
    solve / grad / gather all on one (px,py,pz)-sharded grid, particles
    owned per block with a per-axis ring-exchange cascade
    (docs/SCALING_DESIGN.md). Same fused KDK + warm/mixed gravity as the
    single-device path; conservation watchdog on every step."""
    from jax.experimental.shard_map import shard_map
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from gravity.particle_mesh_blocks import (
        assemble_blocks_sharded, cic_deposit_blocks,
        exchange_particles_blocks, gather_accel_blocks, grad_phi_blocks,
        make_block_mesh, resort_blocks)
    from gravity.poisson_multigrid import (set_mg_mixed,
                                           solve_poisson_fmg_shardmap)

    N = cfg.Nx
    dx = box / N
    ndev = jax.device_count()
    pdims = tuple(getattr(cfg, "dm_pdims", None) or _balanced_pdims(ndev))
    if any(N % p for p in pdims):
        raise ValueError(f"block decomposition {pdims} must divide Nx={N}")
    mesh = make_block_mesh(jax.devices(), pdims)
    coeff_fac = poisson_coeff_fac(Om, H0)
    warm = int(getattr(cfg, "dm_warm_vcycles", 1))
    prec = gravity_precision_from_config(cfg)
    mixed = prec.cycle_mixed
    set_mg_mixed(mixed)          # same precision policy inside the sharded solve
    if prec.fixed_deposit and is_main_process:
        print("[cosmo-dm] NOTE: gravity_deposit_fixed is not wired into the "
              "block-sharded deposit yet; using the float scatter.")
    eps_soft = softening_from_config(cfg, dx)
    cap = float(getattr(cfg, "dm_capacity_factor", 1.5))
    resort_every = int(getattr(cfg, "dm_resort_every", 4))  # in EXCHANGES

    posS, pS, massS, actS, C = assemble_blocks_sharded(
        np.asarray(ic.pos), np.asarray(ic.vel) * (a ** 1.5),
        np.asarray(ic.mass), N, pdims, dx, box, mesh, cap)
    clock = CosmoClock(cosmo, a, a_end, H0)

    from functools import partial

    # --- SPLIT step, all-shard_map: the MG solve is NOT inlined into the
    # step programs. The fused variant asked XLA's SPMD partitioner to
    # swallow the whole multigrid pyramid inside one step graph — compile
    # exploded (12 h at 1024^3, no program) while the SAME solve compiles in
    # minutes standalone. Sequence per step: kick_drift -> [exchange] ->
    # deposit -> field_stats -> solve -> grad+gather -> close_kick. Every
    # program is an explicit shard_map on the hydro's (x,y,z) block mesh
    # (NO auto-sharding, reductions are psum/pmax), built once and reused.
    # We solve the UNSCALED equation lap(psi) = delta and scale the force by
    # coeff_fac*a_kick in close_kick, so the solver's closure constants are
    # step-invariant (one trace/compile ever) and the carried warm seed psi
    # excludes the growth factor (better extrapolation for free).
    S = ("x", "y", "z")
    Pp, Ps, Pg = P(S, None), P(S), P("x", "y", "z")

    @partial(jax.jit, donate_argnums=(0, 1, 2))
    @partial(shard_map, mesh=mesh, in_specs=(Pp, Pp, Pp, P()),
             out_specs=(Pp, Pp))
    def kick_drift(pos_l, p_l, g_l, dtau_):
        p_half = p_l + g_l * (0.5 * dtau_)
        return jnp.mod(pos_l + p_half * dtau_, box), p_half

    @partial(jax.jit, static_argnames=())
    @partial(shard_map, mesh=mesh, in_specs=(Pg, Pp, Ps, Ps),
             out_specs=(Pg, P(), P(), P(), P()))
    def field_stats(rho_l, p_half_l, mass_l, act_l):
        ssum = jax.lax.psum(jnp.sum(rho_l), S)
        cnt = jax.lax.psum(jnp.asarray(rho_l.size, rho_l.dtype), S)
        delta_l = rho_l / (ssum / cnt) - 1.0
        dmax = jax.lax.pmax(jnp.max(delta_l), S)
        p2 = jax.lax.pmax(jnp.max(jnp.where(
            act_l > 0, jnp.sum(p_half_l ** 2, axis=1), 0.0)), S)
        # conservation watchdog inputs: only exchanges can create/drop
        # particles; quiet steps merely rearrange fixed buffers.
        n_act = jax.lax.psum(jnp.sum((act_l > 0).astype(jnp.int32)), S)
        m_tot = jax.lax.psum(jnp.sum(mass_l * (act_l > 0)), S)
        return delta_l.astype(jnp.float64), dmax, p2, n_act, m_tot

    @partial(jax.jit, donate_argnums=(0, 1))
    @partial(shard_map, mesh=mesh, in_specs=(Pp, Pp, Ps, P(), P()),
             out_specs=(Pp, Pp, P()))
    def close_kick(graw_l, p_half_l, act_l, dtau_, scale_):
        g_l = graw_l * scale_
        p_new = p_half_l + g_l * (0.5 * dtau_)
        g2 = jax.lax.pmax(jnp.max(jnp.where(
            act_l > 0, jnp.sum(g_l ** 2, axis=1), 0.0)), S)
        return p_new, g_l, g2

    # Warm-cycle smoothing counts come from the SAME DM-scoped config the
    # particle-parallel path uses (dm_warm_n_pre/n_post, default 1): the
    # solver's own defaults are V(2,2), i.e. twice the per-step smoother work
    # for no measured accuracy gain at dloga <= 0.02.
    n_pre_w = int(getattr(cfg, "dm_warm_n_pre", 1))
    n_post_w = int(getattr(cfg, "dm_warm_n_post", 1))

    def _solve(delta64, psi_prev):
        if psi_prev is None:
            return solve_poisson_fmg_shardmap(
                delta64, dx, 1.0, boundary=0, mesh=mesh,
                n_extra_vcycles=n_extra)
        return solve_poisson_fmg_shardmap(
            delta64, dx, 1.0, boundary=0, mesh=mesh,
            phi_init=psi_prev, warm_cycles=max(warm, 1),
            n_pre=n_pre_w, n_post=n_post_w)

    def _grad_gather(psi, pos):
        return gather_accel_blocks(grad_phi_blocks(psi, dx, mesh, order=4),
                                   pos, N, dx, mesh)
    drift_margin = 0.9 * dx          # trigger before the ghost cell fills

    if is_main_process:
        print(f"[cosmo-dm] 3D-block decomposition {pdims} across {ndev} "
              f"devices (capacity {ndev * C} for {ic.n_part} particles)")
    tau, step, t0 = 0.0, 0, time.time()
    # cold start via the split path: deposit + stats, FMG solve, close_kick
    # with dtau=0 (p unchanged, returns the opening force).
    rho = cic_deposit_blocks(posS, massS, actS, N, dx, mesh,
                             dtype=prec.deposit)
    delta64, dmax, p2, n_act, m_tot = field_stats(rho, pS, massS, actS)
    psi = _solve(delta64, None)
    pS, g, g2 = close_kick(_grad_gather(psi, posS), pS, actS, 0.0,
                           coeff_fac * a)
    jax.block_until_ready(g)
    rep = StepLog(cfg, is_main_process,
                  history_fields=["step", "a", "dloga", "delta_max"])
    g2max = float(g2)
    p2max = float(p2)
    n_act0 = int(n_act)
    m_tot0 = float(m_tot)
    cum_drift, n_exchanges = 0.0, 0
    t_prev = time.time()                       # in-flight per-step wall timing
    while a < a_end - 1e-9:
        dtau = clock.cap_expansion(dtau_accel(eta, eps_soft, np.sqrt(g2max)),
                                   a, dloga_max)
        dtau = clock.cap_to_table(dtau, tau)
        a_next = clock.a_of_tau(tau + dtau)
        dloga = np.log(a_next / a)
        # will this step's worst-case drift overflow the ghost margin?
        # p_half = p + g*dtau/2, so bound |p_half| by last step's maxima.
        step_drift = (np.sqrt(p2max) + 0.5 * np.sqrt(g2max) * dtau) * dtau
        do_ex = (cum_drift + step_drift) > drift_margin
        posS, p_half = kick_drift(posS, pS, g, dtau)
        if do_ex:
            posS, p_half, massS, actS, n_over = exchange_particles_blocks(
                posS, p_half, massS, actS, N, dx, box, mesh)
        else:
            # cadenced exchange: skip while cumulative drift since the last
            # exchange stays inside the one-ghost-cell margin the deposit
            # and gather tolerate (halo-margin ownership, HACC-style).
            n_over = 0
        rho = cic_deposit_blocks(posS, massS, actS, N, dx, mesh,
                                 dtype=prec.deposit)
        delta64, dmax, p2, n_act, m_tot = field_stats(rho, p_half, massS,
                                                      actS)
        # ROCm: an unsynced burst of eager dispatches can wedge the queue
        # permanently (async-pileup, quirks ledger; hit at 64^3 where kernels
        # are us-scale). One mid-step sync bounds the burst; ~free at
        # production sizes. CUDA-retest candidate.
        jax.block_until_ready(delta64)
        psi = _solve(delta64, psi if warm > 0 else None)
        pS, g, g2 = close_kick(_grad_gather(psi, posS), p_half, actS, dtau,
                               coeff_fac * a_next)
        g2max = float(g2)
        p2max = float(p2)
        cum_drift = 0.0 if do_ex else cum_drift + step_drift
        n_exchanges += int(do_ex)
        # cell-order resort cadence: keep deposit/gather memory-local as
        # clustering decays the IC ordering (measured 4-8x effect).
        if resort_every and do_ex and n_exchanges % resort_every == 0:
            posS, pS, massS, actS = resort_blocks(
                posS, pS, massS, actS, N, dx, mesh)
        tau += dtau
        a = a_next
        step += 1
        t_now = time.time()                    # loop is host-synced (float(g2))
        if is_main_process:
            print(f"[cosmo-dm] stepms {step:5d} {(t_now - t_prev) * 1e3:10.1f}"
                  f" ex={int(do_ex)}", flush=True)
        t_prev = t_now
        if do_ex and (int(n_over) != 0 or int(n_act) != n_act0
                      or abs(float(m_tot) - m_tot0) > 1e-8 * m_tot0):
            raise RuntimeError(
                f"[cosmo-dm] block exchange conservation violated at step "
                f"{step}: overflow={int(n_over)}, n_active {int(n_act)} vs "
                f"{n_act0}, mass {float(m_tot):.15e} vs {m_tot0:.15e}")
        rep.step(step, f"[cosmo-dm] {step:5d} {a:8.4f} {1/a-1:8.4f} "
                 f"{dloga:8.5f} {float(dmax):10.3f}",
                 row=[step, a, dloga, float(dmax)],
                 force=a >= a_end - 1e-9)
    if is_main_process:
        print(f"[cosmo-dm] done: {step} steps ({n_exchanges} exchanges), "
              f"a={a:.4f} (z={1/a-1:.4f}), {time.time()-t0:.1f}s")
    if jax.process_count() == 1 and is_main_process:
        sel = np.asarray(actS) > 0
        if getattr(cfg, "dm_write_snapshot", True):
            write_gadget_dm_snapshot(cfg, np.asarray(posS)[sel],
                                 np.asarray(pS)[sel] / a ** 1.5, a, ic)
    return {"a": a, "n_steps": step, "n_exchanges": n_exchanges}


def _run_slab(cfg, ic, cosmo, H0, box, Om, a, a_end, dloga_max, eta,
              n_extra, is_main_process, n_total=None):
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
    send_div = int(getattr(cfg, "dm_exchange_send_div", 64))
    eps_soft = softening_from_config(cfg, dx)
    prec = gravity_precision_from_config(cfg)             # one precision switch

    pos0 = np.asarray(ic.pos)
    mom0 = np.asarray(ic.vel) * (a ** 1.5)                 # canonical momentum p=a^2 xdot
    mass0 = np.asarray(ic.mass)
    if n_total is not None:
        # multi-process chunked IC (this rank holds ~its slab's index-chunk):
        # stitch per-slab buffers from local chunks (global capacity from
        # n_total), then settle boundary crossers with ring exchanges — one
        # hop per round, so loop until no particle is out of its slab.
        from gravity.particle_mesh_shardmap import assemble_slab_from_chunk
        (posS, pS, massS), actS, C = assemble_slab_from_chunk(
            [pos0, mom0, mass0],
            [P("x", None), P("x", None), P("x")], n_total, ndev, mesh,
            capacity_factor=cap)
        # settle moves the chunk/slab-boundary misalignment — MILLIONS of
        # particles in one exchange (measured 6.4M at l10/8), vs ~23k/step in
        # steady state. Always use a generous send buffer here (C//8); the
        # lean cfg send_div applies only to the per-step exchange in kdk.
        for settle in range(8):
            posS, pS, massS, actS, n_bad, occ, _mv = exchange_particles_slab(
                posS, pS, massS, actS, Nx, dx, box, mesh, send_div=8)
            if int(n_bad) == 0:
                break
        n_act = int(jnp.sum(actS > 0))
        if n_act != n_total:
            raise RuntimeError(
                f"[cosmo-dm] IC settle lost particles: {n_act} active vs "
                f"{n_total} in the IC — send buffer (C//{send_div}) or slab "
                "capacity overflowed; lower dm_exchange_send_div or raise "
                "dm_capacity_factor.")
        if is_main_process:
            print(f"[cosmo-dm] IC settle: {settle + 1} rounds, "
                  f"out-of-slab={int(n_bad)} max_occ={int(occ)} cap_C={C}",
                  flush=True)
    else:
        # host-side spatial decomposition into per-slab fixed-capacity buffers
        pos_g, mom_g, mass_g, act_g, C = decompose_to_slabs(
            pos0, mom0, mass0, Nx, ndev, dx, box, cap)
        posS = jax.device_put(jnp.asarray(pos_g), NamedSharding(mesh, P("x", None)))
        pS = jax.device_put(jnp.asarray(mom_g), NamedSharding(mesh, P("x", None)))
        massS = jax.device_put(jnp.asarray(mass_g), NamedSharding(mesh, P("x")))
        actS = jax.device_put(jnp.asarray(act_g), NamedSharding(mesh, P("x")))

    clock = CosmoClock(cosmo, a, a_end, H0)

    warm = int(getattr(cfg, "dm_warm_vcycles", 0))
    # mean density: the fixed-point deposit's integer unit (mass is conserved,
    # so this is a constant of the run).
    mass_unit = float(jnp.sum(massS * (actS > 0))) / (box ** 3)

    def _force(pos, mass, active, a_kick, phi_prev):
        # MUST be jitted by the caller: solve_poisson_fmg_shardmap rebuilds a
        # fresh jitted FMG closure each call; jitting the caller traces once.
        rho = cic_deposit_slab(pos, mass, active, Nx, Ny, Nz, dx, mesh,
                               dtype=prec.deposit,
                               fixed=prec.fixed_int if prec.fixed_deposit
                               else None,
                               fixed_bits=prec.fixed_bits,
                               mass_unit=mass_unit)
        delta = rho / jnp.mean(rho) - 1.0
        # a_kick is traced under the caller's jit, but the solver needs a
        # concrete fourpiG (it calls float() on it): fold the traced factor
        # into the RHS field and hand the solver a literal 1.0.
        rhs = delta * (coeff_fac * a_kick)
        if phi_prev is None:                                # cold FMG (startup)
            phi = solve_poisson_fmg_shardmap(
                rhs, dx, 1.0, boundary=0, mesh=mesh,
                n_levels=n_levels, n_extra_vcycles=n_extra)
        else:                                               # warm carried phi
            phi = solve_poisson_fmg_shardmap(
                rhs, dx, 1.0, boundary=0, mesh=mesh,
                n_levels=n_levels, phi_init=phi_prev,
                warm_cycles=max(warm, 1))
        g = grad_phi_slab(phi, dx, mesh, order=4)
        return (gather_accel_slab(g, pos, Nx, Ny, Nz, dx, mesh),
                jnp.max(delta), phi)

    from functools import partial

    force_cold = jax.jit(lambda pos, m, act, ak: _force(pos, m, act, ak, None))

    @partial(jax.jit, donate_argnums=(0, 1, 2, 4, 5, 7))
    def kdk(pos, p, g, dtau, mass, active, a_kick, phi_prev):
        # fused KDK: half-kick, drift, exchange to owning slab, warm force,
        # half-kick — one jit, no eager 134M temporaries. Also returns the
        # conservation watchdogs (total active count + mass) and max|g|^2.
        p_half = p + g * (0.5 * dtau)
        pos = jnp.mod(pos + p_half * dtau, box)
        (pos, p_half, mass, active, n_bad, max_occ,
         n_move) = exchange_particles_slab(
            pos, p_half, mass, active, Nx, dx, box, mesh, send_div=send_div)
        if warm > 0:
            g_new, dmax, phi = _force(pos, mass, active, a_kick, phi_prev)
        else:
            g_new, dmax, phi = _force(pos, mass, active, a_kick, None)
        p_new = p_half + g_new * (0.5 * dtau)
        g2max = jnp.max(jnp.where(active > 0,
                                  jnp.sum(g_new ** 2, axis=1), 0.0))
        n_act = jnp.sum(active > 0)
        m_tot = jnp.sum(mass * (active > 0))
        return (pos, p_new, g_new, phi, mass, active,
                dmax, g2max, n_bad, max_occ, n_act, m_tot, n_move)

    if is_main_process:
        print(f"[cosmo-dm] grid-sharded slab decomposition across {ndev} devices "
              f"(nx_loc={nx_loc}, capacity {ndev*C} for {ic.n_part} particles)")
        print(f"[cosmo-dm] {prec}  exchange send buffer C//{send_div}")
        print(f"[cosmo-dm] {ic.n_part} DM particles  box={box:.0f} kpc/h  "
              f"mesh={Nx}^3  Om={Om} OL={cfg.cosmo_Omega_L} H0={H0:.5f}")
        print(f"[cosmo-dm] a_init={a:.4f} (z={1/a-1:.2f}) -> a_end={a_end:.4f}")
        print(f"[cosmo-dm] {'step':>5} {'a':>8} {'z':>8} {'dloga':>8} "
              f"{'delta_max':>10} {'max_occ':>8} {'n_bad':>6} {'movers':>8}")

    tau, step, t0 = 0.0, 0, time.time()
    if is_main_process:
        print("[cosmo-dm] compiling first PM force "
              "(deposit -> FMG -> grad -> gather) ...", flush=True)
    tc = time.time()
    g, dmax, phi = force_cold(posS, massS, actS, a)
    jax.block_until_ready(g)
    if is_main_process:
        print(f"[cosmo-dm] first force compiled+ran in {time.time()-tc:.1f}s", flush=True)

    def _grow_capacity(C_new):
        # Reallocate the per-slab particle buffers at C_new slots/slab (JAX
        # cannot resize inside a compiled step — fixed shapes per compile,
        # so regrow host-side between steps; same overflow-handling pattern
        # as JAX-MD's neighbor lists). Each process copies only its own
        # addressable shards; active slots are front-compacted, so a plain
        # prefix copy preserves the layout and new padding lands inactive.
        nonlocal posS, pS, g, massS, actS, C
        from jax.sharding import NamedSharding

        def grow(gx, spec):
            shards = sorted(gx.addressable_shards,
                            key=lambda s: s.index[0].start or 0)
            new_shards = []
            for s in shards:
                old = np.asarray(s.data)
                buf = np.zeros((C_new,) + old.shape[1:], old.dtype)
                buf[:old.shape[0]] = old
                new_shards.append(jax.device_put(jnp.asarray(buf), s.device))
            gshape = (ndev * C_new,) + tuple(gx.shape[1:])
            return jax.make_array_from_single_device_arrays(
                gshape, NamedSharding(mesh, spec), new_shards)

        posS = grow(posS, P("x", None))
        pS = grow(pS, P("x", None))
        g = grow(g, P("x", None))
        massS = grow(massS, P("x"))
        actS = grow(actS, P("x"))
        C = C_new
    rep = StepLog(cfg, is_main_process,
                  history_fields=["step", "a", "dloga", "delta_max",
                                  "max_occ", "n_bad"])
    n_bad_tot = 0
    dev_ts = []                            # per-step device wall (see refresh)
    host_every = int(getattr(cfg, "dm_host_every", 1))
    g2max = float(jnp.max(jnp.where(actS > 0, jnp.sum(g ** 2, axis=1), 0.0)))
    n_act0 = int(jnp.sum(actS > 0))
    m_tot0 = float(jnp.sum(massS * (actS > 0)))
    t_ref, step_ref = time.time(), 0
    while a < a_end - 1e-9:
        dtau = clock.cap_expansion(dtau_accel(eta, eps_soft, np.sqrt(g2max)),
                                   a, dloga_max)
        dtau = clock.cap_to_table(dtau, tau)
        a_next = clock.a_of_tau(tau + dtau)
        dloga = np.log(a_next / a)
        (posS, pS, g, phi, massS, actS, dmax, g2, n_bad, max_occ,
         n_act, m_tot, n_move) = kdk(posS, pS, g, dtau, massS, actS,
                                     a_next, phi)
        tau += dtau
        a = a_next
        step += 1
        # Host pulls (float()/int() = device sync) cost ~10-25 ms each on
        # GH200: refresh dt criterion + watchdogs every dm_host_every steps
        # (same amortization as the particle-parallel loop; dt drifts
        # <1%/step at these z, conservation violations persist until seen).
        if step % host_every == 0 or a >= a_end - 1e-9:
            g2max = float(g2)                  # host pull = device sync point
            now = time.time()
            dev_ts.append((now - t_ref) / (step - step_ref))
            t_ref, step_ref = now, step
            n_bad_tot += int(n_bad)
            # conservation watchdog: the exchange must not create or drop
            # particles/mass (historical bug class; user directive).
            if (int(n_act) != n_act0
                    or abs(float(m_tot) - m_tot0) > 1e-8 * m_tot0):
                raise RuntimeError(
                    f"[cosmo-dm] slab exchange conservation violated at step "
                    f"{step}: n_active {int(n_act)} vs {n_act0}, mass "
                    f"{float(m_tot):.15e} vs {m_tot0:.15e}")
            if (prec.fixed_deposit
                    and 1.0 + float(dmax) > 0.5 * _fixed_ceiling(prec)):
                raise RuntimeError(
                    f"fixed-point deposit: 1+delta_max = {1 + float(dmax):.0f}"
                    f" is within 2x of the integer ceiling "
                    f"{_fixed_ceiling(prec):.0f} ({prec.fixed_int.__name__} "
                    f"at S=2^{prec.fixed_bits}). Lower gravity_deposit_fixed"
                    "'s scale or use gravity_precision='double' (int64).")
            rep.step(step, f"[cosmo-dm] {step:5d} {a:8.4f} {1/a-1:8.4f} "
                     f"{dloga:8.5f} {float(dmax):10.3f} {int(max_occ):8d} "
                     f"{int(n_bad):6d} {int(n_move):8d} "
                     f"{(time.time()-t0)/step:6.2f}s/step",
                     row=[step, a, dloga, float(dmax), int(max_occ),
                          int(n_bad), int(n_move)],
                     force=a >= a_end - 1e-9)
            # Proactive capacity regrow (JAX-MD-style: fixed shapes per
            # compile, occupancy-triggered reallocation between steps on
            # healthy state — the overflow abort below stays as backstop).
            # max_occ is replicated, so every process takes this branch
            # identically; the next kdk call recompiles once (cached).
            if int(max_occ) > 0.9 * C:
                C_new = int(np.ceil(1.3 * C / 4096.0) * 4096)
                if is_main_process:
                    print(f"[cosmo-dm] capacity regrow at step {step}: "
                          f"max_occ {int(max_occ)} > 90% of C={C} -> "
                          f"C={C_new} (one recompile follows)", flush=True)
                _grow_capacity(C_new)

    if is_main_process:
        wall = time.time() - t0
        med_dev = 1e3 * float(np.median(dev_ts)) if dev_ts else 0.0
        print(f"[cosmo-dm] done: {step} steps, a={a:.4f} (z={1/a-1:.4f}), "
              f"{wall:.1f}s  (cumulative out-of-slab events: {n_bad_tot})")
        print(f"[cosmo-dm] timing: median device step {med_dev:.1f} ms, "
              f"mean total step {1e3 * wall / max(step, 1):.1f} ms "
              f"(rest = host dt/watchdog/log)")
        # gather active particles to host, restore Gadget velocity u = p / a^{3/2}
        # (multi-process shards are non-addressable from rank 0 — only touch
        # them when a snapshot is actually requested)
        if getattr(cfg, "dm_write_snapshot", True):
            pos_h = np.asarray(posS)
            mom_h = np.asarray(pS)
            act_h = np.asarray(actS) > 0
            sel = np.nonzero(act_h)[0]
            write_gadget_dm_snapshot(cfg, pos_h[sel], mom_h[sel] / a ** 1.5, a, ic)
    return {"a": a, "n_steps": step, "n_out_of_slab": n_bad_tot}
