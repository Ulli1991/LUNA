"""Block-AMR multi-level hydro runner, wired through ``main.run``.

Mirrors :func:`setup.runner_distributed.run_simulation_distributed` but drives the
block-AMR hierarchy (:mod:`amr.driver`) instead of a single dense grid:

  * build the conservative IC at the base grid via the standard
    ``ic_generation.initial_conditions`` path, then build the refined hierarchy;
  * advance with a synchronized same-dt multi-level step (CFL over all levels);
  * at each snapshot time, **composite the hierarchy to a dense finest-resolution
    field** and write it with the existing HDF5 writer
    (:func:`file_io.write_snapshot_distributed.write_snapshot_distributed`), so
    the standard readers / ``plot_density2d`` work unchanged.

Single device for now (the hierarchy is one host array); ``shard_map`` block
distribution is the next layer. Selected by ``cfg.AMR``.
"""

import copy
import os
import time

import jax
import jax.numpy as jnp
import numpy as np


def run_simulation_amr(cfg, is_main_process=True):
    # AMR bookkeeping (Morton ids, conservative prolong/restrict) wants fp64.
    jax.config.update("jax_enable_x64", True)

    from amr.config import AMRConfig
    from amr import driver
    from ic_generation.ic import initial_conditions
    from file_io.write_snapshot_distributed import write_snapshot_distributed

    ndim = cfg.SPATIAL_DIMENSION
    if ndim not in (2, 3):
        raise NotImplementedError("AMR runner supports SPATIAL_DIMENSION in {2, 3}")

    # ---- coupled gas+DM cosmological box (MUSIC IC) ----------------------
    # Gas (PartType0) is meshed to the base grid; DM (PartType1) stays as
    # particles on the hierarchy (amr/particles.py: locate / ghosted-CIC
    # deposit / inverse-CIC gather). Box + cosmology come from the IC header,
    # so this MUST run before the grid/box reads below.
    dm_state = None
    if getattr(cfg, "CosmoHydro", False):
        import numpy as _np
        from cosmology.music_ic import (music_run_config, read_music_dm_ic,
                                        read_music_gas_ic)
        from gravity.particle_mesh import ParticleMeshGravity
        from setup.runner_cosmo_hydro import mesh_gas_to_grid

        gas = read_music_gas_ic(cfg.music_ic_path)
        dm = read_music_dm_ic(cfg.music_ic_path, part_type=1)
        cfg = music_run_config(dm, cfg.Nx, base=cfg)     # box + cosmology (header)
        a0_music = float(cfg.a_init)
        if float(_np.max(gas.u_int)) <= 0.0:             # MUSIC wrote no u -> seed T
            z = 1.0 / a0_music - 1.0
            T_init = float(getattr(cfg, "cosmo_T_init", 0.0)) or 2.725 * (1.0 + z)
            mu = float(getattr(cfg, "cosmo_mu", 1.22))
            u_init = 8.2543e-3 * T_init / ((float(cfg.gamma) - 1.0) * mu)
            gas.u_int = _np.full(gas.n_part, u_init, dtype=_np.float64)
            if is_main_process:
                print(f"  cosmo-AMR DM: seeding gas T_init={T_init:.1f} K")
        _pm = ParticleMeshGravity(cfg)
        U0_music = jnp.asarray(mesh_gas_to_grid(gas, _pm, a0_music))
        dm_state = dict(pos=jnp.asarray(dm.pos),
                        p=jnp.asarray(dm.vel) * (a0_music ** 1.5),  # canonical p
                        mass=jnp.asarray(dm.mass))
        # bound early: the t=0 snapshot's DM dump reads these via the write()
        # closure before the chunk section (re)assigns them.
        dm_pos, dm_p = dm_state["pos"], dm_state["p"]
        if is_main_process:
            print(f"  cosmo-AMR DM: MUSIC IC {gas.n_part} gas + {dm.n_part} DM, "
                  f"box={cfg.Lx:.1f}, z={1.0 / a0_music - 1.0:.1f}")

    block = int(getattr(cfg, "amr_block_shape", 16))
    levels = int(getattr(cfg, "amr_levels", 2))
    gamma = float(cfg.gamma)
    CFL = float(cfg.CFL)
    riemann = getattr(cfg, "riemann_solver", "hllc")
    limiter = int(getattr(cfg, "LIMITER", 1))

    # ndim-generic domain / grid tuples (Lz/Nz default to the y-values in 2D)
    box_full = (float(cfg.Lx), float(cfg.Ly), float(getattr(cfg, "Lz", cfg.Ly)))
    grid_full = (int(cfg.Nx), int(cfg.Ny), int(getattr(cfg, "Nz", cfg.Ny)))
    acfg = AMRConfig(
        ndim=ndim,
        box=box_full[:ndim],
        base_grid=grid_full[:ndim],
        block_shape=(block,) * ndim,
        max_levels=levels,
        boundary=int(cfg.BOUNDARY),
    )

    # capacities: None -> build_hierarchy sizes each level to the EXACT flagged
    # block count (static refinement), i.e. zero NaN padding. Only set
    # amr_capacities to override (e.g. to leave regrid headroom later).
    caps = getattr(cfg, "amr_capacities", None)
    bucket = int(getattr(cfg, "amr_block_bucket", 0))
    thresholds = getattr(cfg, "amr_thresholds", None) or [0.05] * levels
    criterion = getattr(cfg, "amr_criterion", "density")   # "density" | "vorticity"
    periodic = (cfg.BOUNDARY == 0)

    # --- multi-GPU: shard blocks across devices via shard_map ---
    n_dev = jax.device_count()
    shard_amr = bool(getattr(cfg, "amr_shard", False)) and n_dev > 1
    pad_to = n_dev if shard_amr else 1                  # block axis divisible by #devices
    mesh = None
    if shard_amr:
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        mesh = Mesh(np.asarray(jax.devices()), ("blocks",))

        repl = NamedSharding(mesh, P())                  # replicated across all devices

        def _to_global(a, reshard=False):
            # reshard=True: input is already a GLOBAL multi-process array (regrid
            # output) -> device_put reshards it to replicated ON-DEVICE (cheap,
            # no host round-trip). reshard=False: input is this process's local
            # build array. Single-process (e.g. 1 node, 4 local GPUs): plain
            # device_put replicates correctly -- all mesh devices are addressable.
            # Multi-process: LUNA's create_sharded_array_from_local pattern (put on
            # THE local device, make_array_from_single_device_arrays), because
            # make_array_* wants exactly one shard per *addressable* device and a
            # naive host-array device_put deadlocks across processes.
            if reshard or jax.process_count() == 1:
                return jax.device_put(a, repl)
            local = jax.device_put(np.asarray(a), jax.local_devices()[0])
            return jax.make_array_from_single_device_arrays(local.shape, repl, [local])

        def _global_hierarchy(h, reshard=False):
            g = lambda a: _to_global(a, reshard)
            return tuple(lvl._replace(
                data=g(lvl.data), coords=g(lvl.coords), neighbors=g(lvl.neighbors),
                parent_slots=g(lvl.parent_slots), n_active=g(lvl.n_active)) for lvl in h)

    # --- base-grid conservative IC -> hierarchy ---
    U0 = U0_music if dm_state is not None else jnp.asarray(initial_conditions(cfg))
    hierarchy = driver.build_hierarchy(
        U0, acfg, thresholds=thresholds, capacities=caps, periodic=periodic,
        bucket=bucket, pad_to=pad_to, criterion=criterion,
    )
    if shard_amr:
        hierarchy = _global_hierarchy(hierarchy)
    if is_main_process:
        base_str = "x".join(str(g) for g in grid_full[:ndim])
        print(f"AMR: base {base_str}, block {block}, {levels} levels "
              f"(finest {cfg.Nx * acfg.refinement_ratio**levels}), riemann={riemann}")
        for L, lvl in enumerate(hierarchy):
            print(f"  L{L}: {int(lvl.n_active):6d} active blocks  dx={acfg.level_dx(L)[0]:.3e}")

    subcycle = bool(getattr(cfg, "amr_subcycle", False))
    if shard_amr:
        from amr import shard
        if subcycle:
            step = jax.jit(lambda lv, dt: shard.advance_subcycle_shmap(
                lv, dt, acfg, mesh, gamma=gamma, limiter=limiter, riemann_solver=riemann))
        else:
            step = jax.jit(lambda lv, dt: shard.advance_shmap(
                lv, dt, acfg, mesh, gamma=gamma, limiter=limiter, riemann_solver=riemann))
    elif subcycle:
        step = jax.jit(lambda lv, dt: driver.advance_subcycle(
            lv, dt, acfg, gamma=gamma, limiter=limiter, riemann_solver=riemann))
    else:
        step = jax.jit(lambda lv, dt: driver.advance(
            lv, dt, acfg, gamma=gamma, limiter=limiter, riemann_solver=riemann))
    # jitted: the eager version dispatched ~10 tiny ops per level on the host
    # every single step (a real per-step tax at thousands of blocks).
    dt_fn = jax.jit((lambda lv: driver.compute_dt_subcycle(lv, acfg, gamma=gamma, CFL=CFL)) if subcycle
                    else (lambda lv: driver.compute_dt(lv, acfg, gamma=gamma, CFL=CFL)))
    if is_main_process:
        ti = "subcycled (Berger-Oliger)" if subcycle else "synchronized same-dt"
        print(f"  time integration: {ti}  |  {'shard_map over %d devices' % n_dev if shard_amr else 'single device'}")

    # snapshot output config: write the COMPOSITE at the finest resolution, so
    # the writer's dx / dims metadata describe the finest uniform grid.
    finest = cfg.Nx * acfg.refinement_ratio ** levels
    scfg = copy.copy(cfg)
    scfg.Nx = finest
    scfg.Ny = finest
    if ndim == 3:
        scfg.Nz = finest

    def write(idx, t):
        img = driver.composite_to_dense(hierarchy, acfg)        # (n_vars, finest, finest)
        write_snapshot_distributed(jnp.asarray(img), scfg, t, idx)
        if is_main_process:
            # also dump per-level active block coords for plotting the mesh overlay
            import numpy as _np
            blk = {f"L{L}_coords": _np.asarray(lvl.coords[: int(lvl.n_active)])
                   for L, lvl in enumerate(hierarchy)}
            blk["levels"] = _np.array(len(hierarchy))
            blk["block_shape"] = _np.array(acfg.block_shape)
            blk["t"] = _np.array(float(t))
            _np.savez(os.path.join(cfg.output_dir, f"blocks_{idx:03d}.npz"), **blk)
            if dm_state is not None:                     # DM particle dump
                _np.savez(os.path.join(cfg.output_dir, f"dm_{idx:03d}.npz"),
                          pos=_np.asarray(dm_pos), p=_np.asarray(dm_p))
            print(f"  snapshot #{idx} at t={t:.5f}  (blocks: "
                  + " ".join(f"L{L}={int(l.n_active)}" for L, l in enumerate(hierarchy)) + ")")

    os.makedirs(cfg.output_dir, exist_ok=True)
    snapshot_times = cfg.snapshot_times
    num = len(snapshot_times)
    regrid_interval = int(getattr(cfg, "amr_regrid_interval", 0))

    t, si = 0.0, 0
    if num and t >= snapshot_times[0]:
        write(si, t)
        si += 1

    # cell-updates per base step = sum_L n_active[L] * cells/block * subcycle_factor,
    # where the subcycle advances level L  r**L  times per base step. Recomputed
    # only when the block counts change (after a regrid), so no per-step host sync.
    _cpb, _r = acfg.cells_per_block, acfg.refinement_ratio
    def _cu_per_base_step(hier):
        return sum(int(l.n_active) * _cpb * (_r ** L if subcycle else 1)
                   for L, l in enumerate(hier))

    t0 = time.time()
    nstep = 0
    cu_step = _cu_per_base_step(hierarchy)              # active cell-updates / base step
    cu_total = 0
    cu_last, perf_t = 0, t0                             # for per-interval Mcells/s
    cadence = max(1, int(getattr(cfg, "output_cadence", 10)))
    # ----------------------------------------------------------------------
    # Chunked time loop (single-device AND shard_map): ONE jitted dispatch per
    # regrid interval. A lax.scan advances `chunk_len` steps with dt computed
    # ON DEVICE each step (the old loop's per-step `float(dt_fn(...))` forced a
    # device->host sync + fresh dispatch every step, pinning the GPU at a few %
    # utilization). dt is clamped to the chunk's target time (snapshot / t_end);
    # once the target is reached the remaining scan steps run with dt=0 -- an
    # exact no-op (U + 0*dU) -- so snapshot landings stay exact. The hierarchy
    # buffers are donated, so XLA updates them in place. The sharded `step`
    # (shard_map) traces through the scan like any jittable fn; GSPMD carries
    # the shardings across iterations.
    # ----------------------------------------------------------------------
    chunk_len = regrid_interval if regrid_interval > 0 else 16

    # optional self-gravity: level-by-level Dirichlet solve + operator-split
    # kick (amr/poisson.py), applied after each hydro step INSIDE the scan --
    # the FMG + fixed-sweep RB-GS are fully jittable.
    #
    # Cosmological runs (super-comoving variables): the SAME hydro advance and
    # the SAME gravity machinery, but time is conformal (tt = tau), a(tau) is
    # interpolated on-device from the background table, dt gets the dloga
    # expansion cap, the Poisson coefficient becomes 1.5 Om H0^2 a / <rho>, and
    # the cosmo floor stack runs after each step (amr/cosmo.py).
    from amr.poisson import make_amr_self_gravity
    from ic_generation.ic import is_cosmological
    cosmo_kit = None
    grav_apply = None
    is_cosmo = is_cosmological(cfg.test_problem) or dm_state is not None
    if is_cosmo:
        from amr.cosmo import make_cosmo_amr
        from amr.poisson import _ghost_phi, apply_gravity_kick, solve_phi_hierarchy
        rho_bar0 = float(jnp.nanmean(hierarchy[0].data[:, 0]))
        if dm_state is not None:
            _V = 1.0
            for d in range(ndim):
                _V *= float(acfg.box[d])
            rho_bar0 = rho_bar0 + float(jnp.sum(dm_state["mass"])) / _V
        cosmo_kit = make_cosmo_amr(cfg, acfg, rho_bar0, is_main_process=is_main_process)
        _n_vc = int(getattr(cfg, "amr_grav_vcycles", 3) or 3)
        if dm_state is not None:
            # MUSIC header owns a_init, so tau targets can only be derived here:
            # run to a_end in conformal time, snapshots evenly in tau.
            tau_end = float(np.interp(cosmo_kit["a_end"],
                                      cosmo_kit["a_tab"], cosmo_kit["tau_tab"]))
            cfg.t_end = tau_end
            cfg.time_bet_snaps = tau_end / max(1, int(getattr(cfg, "amr_n_snaps", 4)))
            if is_main_process:
                print(f"  cosmo-AMR DM: a {cosmo_kit['a_start']:.4f} -> "
                      f"{cosmo_kit['a_end']:.4f}, t_end(tau)={tau_end:.4f}")

        def grav_cosmo(levels, dt, aa):
            phis = solve_phi_hierarchy(levels, acfg, fourpiG=cosmo_kit["coeff"](aa),
                                       boundary=0, n_vcycles=_n_vc)
            levels = apply_gravity_kick(levels, phis, dt, acfg)
            return cosmo_kit["floors"](levels, aa)
    else:
        grav_apply = make_amr_self_gravity(cfg, acfg, is_main_process=is_main_process)

    if dm_state is not None:
        # ---- coupled gas+DM step pieces (all jit/scan-safe) ---------------
        from amr.particles import (accel_blocks, deposit_cic, gather_cic,
                                   locate_particles)
        dm_mass = dm_state["mass"]
        _box_arr = jnp.asarray([float(acfg.box[d]) for d in range(ndim)])
        _dm_eta = float(getattr(cfg, "dm_eta", 0.025))
        _eps_soft = (float(getattr(cfg, "dm_softening", 0.0))
                     or float(acfg.level_dx(acfg.max_levels)[0]))

        def _dm_forces(h, pos, aa):
            """One JOINT solve (gas + deposited DM) -> (phis, accel at particles).
            Mirrors the dense runner's forces(): deposit at owning level, total
            source into the level-by-level Dirichlet MG, inverse-CIC gather."""
            lev_p, slot_p = locate_particles(pos, h, acfg)
            rho_dm = deposit_cic(pos, dm_mass, h, lev_p, slot_p, acfg)
            rho_tot = [hh.data[:, 0:1] + rd for hh, rd in zip(h, rho_dm)]
            phis = solve_phi_hierarchy(h, acfg, fourpiG=cosmo_kit["coeff"](aa),
                                       boundary=0, n_vcycles=_n_vc,
                                       rho_blocks=rho_tot)
            fields, cg = [], None
            for L, (hh, ph) in enumerate(zip(h, phis)):
                gw = _ghost_phi(ph, hh, cg, acfg)
                fields.append(accel_blocks(gw, L, acfg))
                if L + 1 < len(h):
                    cg = gw
            gp = gather_cic(fields, h, pos, lev_p, slot_p, acfg)
            return phis, gp

    def _chunk(hier, tt, target):
        def body(carry, _):
            h, tc, taken = carry
            dt = dt_fn(h)
            if cosmo_kit is not None:
                aa = jnp.interp(tc, cosmo_kit["tau_tab"], cosmo_kit["a_tab"])
                dt = jnp.minimum(dt, cosmo_kit["dtau_cap"](aa))
            dt = jnp.minimum(dt, jnp.maximum(target - tc, 0.0))
            h = step(h, dt)
            if cosmo_kit is not None:
                h = grav_cosmo(h, dt, aa)
            elif grav_apply is not None:
                h = grav_apply(h, dt)
            return (h, tc + dt, taken + (dt > 0).astype(jnp.int32)), None
        (h, tf, taken), _ = jax.lax.scan(
            body, (hier, tt, jnp.zeros((), jnp.int32)), None, length=chunk_len)
        return h, tf, taken

    def _chunk_dm(hier, pos, p, g, tt, target):
        """Coupled gas+DM chunk: the dense runner's KDK, per scan step --
        DM half-kick (reusing last force) -> drift -> gas hydro -> ONE joint
        solve at a_next -> gas kick -> DM closing half-kick -> floors."""
        def body(carry, _):
            h, ps, pp, gg, tc, taken = carry
            aa = jnp.interp(tc, cosmo_kit["tau_tab"], cosmo_kit["a_tab"])
            gmax = jnp.max(jnp.sqrt(jnp.sum(gg * gg, axis=1)))
            dt = jnp.minimum(dt_fn(h), cosmo_kit["dtau_cap"](aa))
            dt = jnp.minimum(dt, _dm_eta * jnp.sqrt(_eps_soft / (gmax + 1e-30)))
            dt = jnp.minimum(dt, jnp.maximum(target - tc, 0.0))
            pp = pp + gg * (0.5 * dt)
            ps = jnp.mod(ps + pp * dt, _box_arr)
            h = step(h, dt)
            a_next = jnp.interp(tc + dt, cosmo_kit["tau_tab"], cosmo_kit["a_tab"])
            phis, g2 = _dm_forces(h, ps, a_next)
            h = apply_gravity_kick(h, phis, dt, acfg)
            pp = pp + g2 * (0.5 * dt)
            h = cosmo_kit["floors"](h, a_next)
            return (h, ps, pp, g2, tc + dt,
                    taken + (dt > 0).astype(jnp.int32)), None
        (h, ps, pp, gg, tf, taken), _ = jax.lax.scan(
            body, (hier, pos, p, g, tt, jnp.zeros((), jnp.int32)),
            None, length=chunk_len)
        return h, ps, pp, gg, tf, taken

    if dm_state is not None:
        chunk_step_dm = jax.jit(_chunk_dm, donate_argnums=(0, 1, 2, 3))
        _, dm_g = _dm_forces(hierarchy, dm_pos,
                             jnp.asarray(cosmo_kit["a_start"], jnp.float64))
    else:
        chunk_step = jax.jit(_chunk, donate_argnums=0)

    last_print, t_prev = 0, t
    while t < cfg.t_end - 1e-12:
        target = min(cfg.t_end, snapshot_times[si] if si < num else cfg.t_end)
        if dm_state is not None:
            hierarchy, dm_pos, dm_p, dm_g, t_dev, taken_dev = chunk_step_dm(
                hierarchy, dm_pos, dm_p, dm_g,
                jnp.asarray(t, jnp.float64), jnp.asarray(target, jnp.float64))
        else:
            hierarchy, t_dev, taken_dev = chunk_step(
                hierarchy, jnp.asarray(t, jnp.float64), jnp.asarray(target, jnp.float64))
        taken = int(taken_dev)                           # one host sync per chunk
        t_prev, t = t, float(t_dev)
        if taken == 0:
            break                                        # dt collapsed -- don't spin
        nstep += taken
        cu_total += cu_step * taken
        if si < num and t >= snapshot_times[si] - 1e-12:
            write(si, t)
            si += 1
        # Adaptive re-check on the live solution. Capacities grow on a geometric
        # ladder and never shrink (_sized_parents), so `chunk_step` recompiles
        # only ~log2(peak/bucket) times across the whole run.
        if regrid_interval:
            hierarchy = driver.regrid(hierarchy, acfg, thresholds=thresholds,
                                      periodic=periodic, bucket=bucket, pad_to=pad_to,
                                      criterion=criterion)
            if shard_amr:
                hierarchy = _global_hierarchy(hierarchy, reshard=True)   # on-device, no host round-trip
            cu_step = _cu_per_base_step(hierarchy)       # block counts changed
            if is_main_process:
                counts = " ".join(f"L{L}={int(l.n_active)}" for L, l in enumerate(hierarchy))
                extra = ""
                if cosmo_kit is not None:
                    a_now = float(np.interp(t, cosmo_kit["tau_tab"], cosmo_kit["a_tab"]))
                    extra = f" a={a_now:.4f} z={1.0 / a_now - 1.0:.2f}"
                print(f"  regrid @ step {nstep} (t={t:.4f}{extra}): {counts}")
        if is_main_process and nstep - last_print >= cadence:
            last_print = nstep
            now = time.time()
            # same definition as the dense path (runner_distributed): active
            # cell-updates over the interval / interval wall-time.
            perf = (cu_total - cu_last) / max(now - perf_t, 1e-9) / 1e6
            dt_avg = (t - t_prev) / max(taken, 1)
            print(f"Step {nstep:5d}, t = {t:.5f}, dt = {dt_avg:.5e}, perf = {perf:.2f} Mcells/s "
                  f"(AMR active), {nstep / (now - t0):.1f} steps/s", flush=True)
            cu_last, perf_t = cu_total, now

    # flush any remaining snapshot slots at t_end
    while si < num:
        write(si, cfg.t_end)
        si += 1

    if is_main_process:
        elapsed = time.time() - t0
        avg = cu_total / max(elapsed, 1e-9) / 1e6
        print(f"AMR run done: {nstep} steps to t={t:.5f} in {elapsed:.1f}s")
        print(f"✓ Average performance: {avg:.2f} Mcells/s (AMR active cell-updates)")
    return t, nstep, cfg.output_dir
