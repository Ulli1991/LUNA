"""Block-AMR runner (cfg.AMR): builds the refined hierarchy from the registered
IC, advances it in chunked jitted steps (Berger-Oliger subcycling via
cfg.amr_subcycle), regrids every cfg.amr_regrid_interval base steps, and writes
composited dense snapshots. Multi-device: block DATA sharded via amr/shard
(halo/reflux modes selected by LUNA_AMR_HALO_MODE / LUNA_AMR_REFLUX, see
docs/ENV.md; sparse plans rebuilt after every regrid). Design: amr/DESIGN.md.
"""

import copy
import os
import time
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np

# Parameter bundles threaded through the setup helpers (grouped so no helper
# needs a wall of positional arguments).
_ShardKit = namedtuple("_ShardKit", "shard_amr n_dev mesh repl gather_host")
_HydroOpts = namedtuple("_HydroOpts", "gamma CFL riemann limiter subcycle")
_RegridOpts = namedtuple("_RegridOpts",
                         "thresholds criterion periodic bucket max_blocks pad_to")
_SnapIO = namedtuple("_SnapIO",
                     "cfg scfg acfg dense_snaps gather_host is_main_process")
_DMKit = namedtuple("_DMKit", "forces hook src mass box_arr eta eps_soft")
_Physics = namedtuple("_Physics",
                      "cool cosmo_kit grav_apply grav_cosmo grav_hook dm")
_Steppers = namedtuple("_Steppers", "chunk_step chunk_step_dm")


def _load_music_ic(cfg, is_main_process):
    """Read the MUSIC gas+DM IC: box/cosmology from the header (rebinds cfg),
    gas meshed to the base grid, DM kept as particles with canonical momenta.
    Returns (cfg, dm_state, U0_music)."""
    import numpy as _np
    from cosmology.music_ic import (music_run_config, read_music_dm_ic,
                                    read_music_gas_ic)
    from gravity.particle_mesh import ParticleMeshGravity
    from cosmology.mesh_gas import mesh_gas_to_grid

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
    if is_main_process:
        print(f"  cosmo-AMR DM: MUSIC IC {gas.n_part} gas + {dm.n_part} DM, "
              f"box={cfg.Lx:.1f}, z={1.0 / a0_music - 1.0:.1f}")
    return cfg, dm_state, U0_music


def _to_global(a, sharding, reshard=False):
    """Place an array under `sharding`, multi-process aware. reshard=True:
    input is already a GLOBAL on-device array (regrid output) -> reshard in
    place (no host round-trip)."""
    # Single-process device_put handles both cases; multi-process needs the jit
    # identity for reshard (device_put cannot target non-addressable devices).
    if jax.process_count() == 1:
        return jax.device_put(a, sharding)
    if reshard:
        return jax.jit(lambda x: x, out_shardings=sharding)(a)
    # Multi-process sharded construction: every process has built the
    # IDENTICAL hierarchy on host (deterministic IC + build_hierarchy),
    # so each contributes just its addressable shards of the global
    # array -- the block-AMR analogue of the dense path's
    # create_sharded_array_from_local. At-rest memory = total/n_dev;
    # the full copy each process built is freed after this transfer.
    a_np = np.asarray(a)
    return jax.make_array_from_callback(
        a_np.shape, sharding, lambda idx: a_np[idx])


def _global_hierarchy(h, blk, repl, shard_storage, reshard=False):
    """Place a host-built hierarchy under the global shardings: block DATA on
    the block axis (when shard_storage), topology replicated."""
    # Shard only DATA on the block axis (n_slots x n_vars x block_shape --
    # ~99.9% of the bytes). Topology (coords/neighbors/parent_slots) stays
    # REPLICATED: it is tiny (n_slots x ~6), and regrid_sharded's Morton
    # set-diff / neighbour / parent-slot kernels operate on the GLOBAL
    # block set -- sharding them deadlocks the cross-process gather they'd
    # need (single-process GSPMD hides this; multi-process does not).
    data_arr = blk if shard_storage else repl
    return tuple(lvl._replace(
        data=_to_global(lvl.data, data_arr, reshard),
        coords=_to_global(lvl.coords, repl, reshard),
        neighbors=_to_global(lvl.neighbors, repl, reshard),
        parent_slots=_to_global(lvl.parent_slots, repl, reshard),
        n_active=_to_global(lvl.n_active, repl, reshard)) for lvl in h)


def _build_hierarchy(cfg, U0_music, ndim, is_main_process):
    """Read the AMR knobs, set up (optional) block sharding, build the
    hierarchy from the dense IC, and print the per-level census.
    Returns (hierarchy, acfg, _ShardKit, _HydroOpts, _RegridOpts)."""
    from amr.config import AMRConfig
    from amr import driver
    from ic_generation.ic import initial_conditions

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
        gamma=float(cfg.gamma),
        grav_G=float(getattr(cfg, "GRAVITATIONAL_G", 1.0)),
    )

    # capacities: None -> build_hierarchy sizes each level to the EXACT flagged
    # block count (static refinement), i.e. zero NaN padding. Only set
    # amr_capacities to override (e.g. to leave regrid headroom later).
    caps = getattr(cfg, "amr_capacities", None)
    bucket = int(getattr(cfg, "amr_block_bucket", 0))
    # hard per-level block ceiling (0 = unlimited): the deep-hierarchy memory
    # guard -- blanket refinement x the pow2 ladder otherwise requests
    # device-breaking arrays (41 GiB measured at 5 levels). Excess flags are
    # dropped in slot order; the regrid print shows "AT CAP" when saturated.
    max_blocks = int(getattr(cfg, "amr_max_blocks", 0) or 0)
    thresholds = getattr(cfg, "amr_thresholds", None) or [0.05] * levels
    criterion = getattr(cfg, "amr_criterion", "density")   # "density" | "vorticity"
    periodic = (cfg.BOUNDARY == 0)

    # --- multi-GPU: shard blocks across devices via shard_map ---
    n_dev = jax.device_count()
    shard_amr = bool(getattr(cfg, "amr_shard", False)) and n_dev > 1
    pad_to = n_dev if shard_amr else 1                  # block axis divisible by #devices
    mesh = repl = blk = None
    shard_storage = True
    gather_host = lambda x: x   # host-I/O gather; overridden for multi-process
    if shard_amr:
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        mesh = Mesh(np.asarray(jax.devices()), ("blocks",))

        repl = NamedSharding(mesh, P())                  # replicated across all devices
        blk = NamedSharding(mesh, P("blocks"))           # sharded on the block axis
        # Distributed STORAGE (step 1 of the sharded-memory plan): the per-level
        # block-axis arrays (data/coords/neighbors/parent_slots) live sharded
        # across devices -> between-step memory is total/n_dev. n_active is a
        # scalar (replicated). advance_shmap still all-gathers internally, so the
        # PEAK during a step is unchanged and correctness is identical (step 2
        # removes that gather). Block-sharding needs the block axis divisible by
        # n_dev -- guaranteed by pad_to=n_dev in build_hierarchy/regrid.
        shard_storage = bool(getattr(cfg, "amr_shard_storage", True))

        if jax.process_count() > 1:
            def gather_host(x):
                """Reshard a (possibly block-sharded) global array to replicated
                so host I/O (np.asarray) sees a fully-addressable copy. Transient
                (one array at a time, at snapshot/dump cadence only)."""
                return jax.jit(lambda y: y, out_shardings=repl)(x)

    # --- base-grid conservative IC -> hierarchy ---
    U0 = U0_music if U0_music is not None else jnp.asarray(initial_conditions(cfg))
    hierarchy = driver.build_hierarchy(
        U0, acfg, thresholds=thresholds, capacities=caps, periodic=periodic,
        bucket=bucket, pad_to=pad_to, criterion=criterion, max_blocks=max_blocks,
    )
    if shard_amr:
        hierarchy = _global_hierarchy(hierarchy, blk, repl, shard_storage)
        if os.environ.get("LUNA_AMR_SHARD_DBG"):
            # per-process storage proof: addressable (local) vs global block data
            loc = sum(sum(s.data.nbytes for s in lvl.data.addressable_shards)
                      for lvl in hierarchy)
            tot = sum(lvl.data.nbytes for lvl in hierarchy)
            print(f"  [shard-dbg] proc {jax.process_index()}/{jax.process_count()}: "
                  f"local block-data {loc / 1e6:.1f} MB of {tot / 1e6:.1f} MB global")
    if is_main_process:
        base_str = "x".join(str(g) for g in grid_full[:ndim])
        print(f"AMR: base {base_str}, block {block}, {levels} levels "
              f"(finest {cfg.Nx * acfg.refinement_ratio**levels}), riemann={riemann}")
        for L, lvl in enumerate(hierarchy):
            print(f"  L{L}: {int(lvl.n_active):6d} active blocks  dx={acfg.level_dx(L)[0]:.3e}")

    subcycle = bool(getattr(cfg, "amr_subcycle", False))
    return (hierarchy, acfg,
            _ShardKit(shard_amr, n_dev, mesh, repl, gather_host),
            _HydroOpts(gamma, CFL, riemann, limiter, subcycle),
            _RegridOpts(thresholds, criterion, periodic, bucket, max_blocks,
                        pad_to))


def _build_halo_plans(h, acfg, sk, sparse_halo):
    """Stage-2 sparse halo exchange: build the per-level + global-seam
    all_to_all plans on HOST from the (replicated, static-between-regrids)
    neighbour tables. Rebuilt after every regrid; None unless sparse+sharded."""
    # Threaded through the jitted chunk as ARGUMENTS (a closure-captured plan
    # would be baked as a jit constant and go stale after a same-shape regrid).
    # All three exchanges (same-level, coarse-fine, seam) are bit-exact vs
    # all_gather and the bench shows sparse strictly dominates, so it is the
    # DEFAULT when sharded.
    if not sparse_halo:
        return None
    n_dev, repl = sk.n_dev, sk.repl
    from amr.shard import (_same_level_stencil, build_same_level_plan,
                           build_exchange_plan)
    # per-level fill plans (same-level + coarse-fine)
    level_plans = []
    for L, lvl in enumerate(h):
        if lvl.data.shape[0] % n_dev:
            level_plans.append(None)
            continue
        st = np.asarray(_same_level_stencil(lvl.neighbors, acfg))
        sidx, rpos, _K = build_same_level_plan(st, n_dev)
        # recv_pos passed REPLICATED; the fill's shard_map in_spec slices it
        # per device (validated pattern -- do not reshard here).
        same_plan = (_to_global(jnp.asarray(sidx), repl),
                     _to_global(jnp.asarray(rpos), repl))
        cf_plan = None
        if L > 0 and h[L - 1].data.shape[0] % n_dev == 0:
            ps = np.asarray(lvl.parent_slots)[:, None]           # (n_fine, 1)
            csi, crp, _cK = build_exchange_plan(ps, h[L - 1].data.shape[0], n_dev)
            cf_plan = (_to_global(jnp.asarray(csi), repl),
                       _to_global(jnp.asarray(crp[:, 0]), repl))
        level_plans.append((same_plan, cf_plan))

    # global batched-seam plan: replicate _batch_levels' neighbour
    # concatenation on host (nbr + level-offset), then one exchange plan over
    # the concatenated block axis. Cubic blocks only (stacked Ld/Rd payload) --
    # acfg.block_shape is (block,)*ndim, always cubic.
    # per-level-pair interface-reflux plans (LUNA_AMR_REFLUX=iface)
    reflux_plans = None
    if os.environ.get("LUNA_AMR_REFLUX", "repl") == "iface":
        from amr.shard import build_reflux_plan
        rp = []
        for L in range(1, len(h)):
            if (h[L].data.shape[0] % n_dev) or (h[L - 1].data.shape[0] % n_dev):
                rp.append(None)
                continue
            sidx, meta, _K = build_reflux_plan(
                np.asarray(h[L].coords), np.asarray(h[L].neighbors),
                int(h[L].n_active), np.asarray(h[L].parent_slots),
                np.asarray(h[L - 1].neighbors), h[L - 1].data.shape[0],
                acfg, n_dev)
            rp.append((_to_global(jnp.asarray(sidx), repl),
                       _to_global(jnp.asarray(meta), repl)))
        reflux_plans = tuple(rp)

    seam_plan = None
    all_shardable = all(lvl.data.shape[0] % n_dev == 0 for lvl in h)
    cubic = len(set(acfg.block_shape)) == 1
    if all_shardable and cubic:
        parts, acc = [], 0
        for lvl in h:
            parts.append(np.asarray(lvl.neighbors) + acc)
            acc += lvl.data.shape[0]
        nbr_all = np.concatenate(parts, axis=0)                  # (N, 2ndim)
        ssi, srp, _sK = build_exchange_plan(nbr_all, acc, n_dev)
        seam_plan = (_to_global(jnp.asarray(ssi), repl),
                     _to_global(jnp.asarray(srp), repl))
    return (tuple(level_plans), seam_plan, reflux_plans)


def _snapshot_io(cfg, acfg, sk, is_main_process):
    """Snapshot output config: dense composites are written at the finest
    uniform resolution, so the writer's dx / dims metadata describe the finest
    grid. Returns the _SnapIO bundle that _write_snapshot consumes."""
    # PER-AXIS: composite_to_dense scales each base_grid axis by r**levels --
    # using the x-based value for Ny/Nz silently pads non-cubic domains with
    # garbage (caught by the Sod AMR benchmark, base 64x16x16).
    _rl = acfg.refinement_ratio ** acfg.max_levels
    scfg = copy.copy(cfg)
    scfg.Nx = cfg.Nx * _rl
    scfg.Ny = cfg.Ny * _rl
    if acfg.ndim == 3:
        scfg.Nz = cfg.Nz * _rl

    # Dense-composite snapshots materialize the FULL finest-resolution cube:
    # (Nx * 2**levels)**ndim * n_vars * 8 B -- 40 GiB at 32^3 + 5 levels,
    # regardless of AMR sparsity. Deep hierarchies must write block data only
    # (blocks_*.npz carries the complete AMR state + per-level data below).
    dense_snaps = bool(getattr(cfg, "amr_dense_snapshots", True))
    if not dense_snaps and is_main_process:
        print("  snapshots: block data only (amr_dense_snapshots=False)")
    return _SnapIO(cfg, scfg, acfg, dense_snaps, sk.gather_host, is_main_process)


def _write_snapshot(idx, t, hierarchy, dm_pos, dm_p, io):
    """Write snapshot `idx` at time `t`: dense composite (optional), the
    blocks_*.npz per-level dump, and the DM particle dump when present."""
    import numpy as _np
    from amr import driver
    from file_io.write_snapshot_distributed import write_snapshot_distributed
    if io.dense_snaps:
        img = driver.composite_to_dense(hierarchy, io.acfg)  # (n_vars, finest, finest)
        write_snapshot_distributed(jnp.asarray(io.gather_host(img)), io.scfg, t, idx)
    if io.is_main_process or jax.process_count() > 1:
        # dump per-level active block coords for plotting the mesh overlay.
        # Multi-process: gather_host reshards to replicated (a collective --
        # every process must participate) BEFORE the host slice; only rank 0
        # writes below.
        blk_out = {f"L{L}_coords":
                   _np.asarray(io.gather_host(lvl.coords))[: int(lvl.n_active)]
                   for L, lvl in enumerate(hierarchy)}
        if not io.dense_snaps:                       # block DATA too: the
            for L, lvl in enumerate(hierarchy):      # snapshot IS the blocks
                blk_out[f"L{L}_data"] = (
                    _np.asarray(io.gather_host(lvl.data))[: int(lvl.n_active)])
    if io.is_main_process:
        blk_out["levels"] = _np.array(len(hierarchy))
        blk_out["block_shape"] = _np.array(io.acfg.block_shape)
        blk_out["t"] = _np.array(float(t))
        _np.savez(os.path.join(io.cfg.output_dir, f"blocks_{idx:03d}.npz"),
                  **blk_out)
        if dm_pos is not None:                       # DM particle dump
            _np.savez(os.path.join(io.cfg.output_dir, f"dm_{idx:03d}.npz"),
                      pos=_np.asarray(dm_pos), p=_np.asarray(dm_p))
        print(f"  snapshot #{idx} at t={t:.5f}  (blocks: "
              + " ".join(f"L{L}={int(l.n_active)}" for L, l in enumerate(hierarchy)) + ")")


def _cu_per_base_step(hier, acfg, subcycle):
    """Active cell-updates per base step: sum_L n_active[L] * cells/block *
    subcycle factor (the subcycle advances level L  r**L  times per base step)."""
    cpb, r = acfg.cells_per_block, acfg.refinement_ratio
    return sum(int(l.n_active) * cpb * (r ** L if subcycle else 1)
               for L, l in enumerate(hier))


def _make_physics_hooks(cfg, acfg, hierarchy, dm_state, cooling_solver,
                        subcycle, shard_amr, is_main_process):
    """Build the physics hook callables: pointwise cooling, self-gravity /
    cosmology (incl. the Berger-Oliger per-level gravity hook) and the coupled
    gas+DM pieces. Returns a _Physics bundle; the DM path mutates cfg.t_end."""
    ndim = acfg.ndim

    # --- radiative cooling on the hierarchy (operator-split, like the dense
    # runner): cooling is a POINTWISE energy ODE, so it applies per block with
    # the same dt on every level (synchronized path). vmapped over the block
    # axis; NaN sentinel/padding blocks stay NaN (cooling of NaN is NaN).
    def _cool_hierarchy(lv, dt):
        if cooling_solver is None:
            return lv
        from hydro.cooling import apply_cooling
        return tuple(
            l._replace(data=jax.vmap(lambda u: apply_cooling(u, dt, cfg,
                                                             cooling_solver))(l.data))
            for l in lv)

    # optional self-gravity: level-by-level Dirichlet solve + operator-split
    # kick (amr/poisson.py), applied after each hydro step INSIDE the scan --
    # the FMG + fixed-sweep RB-GS are fully jittable.
    #
    # Cosmological runs (super-comoving variables): the SAME hydro advance and
    # the SAME gravity machinery, but time is conformal (t = tau), a(tau) is
    # interpolated on-device from the background table, dt gets the dloga
    # expansion cap, the Poisson coefficient becomes 1.5 Om H0^2 a / <rho>, and
    # the cosmo floor stack runs after each step (amr/cosmo.py).
    from amr.poisson import make_amr_self_gravity
    from ic_generation.ic import is_cosmological
    cosmo_kit = None
    grav_apply = None
    grav_cosmo = None
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

    # --- Berger-Oliger gravity hook: per-level Poisson solve + kick + floors,
    #     threaded through the subcycle recursion with the parent's phi as the
    #     Dirichlet boundary (a(tau) is re-evaluated at each sub-step's time).
    #     The dense L0 FMG then runs ONCE per BASE step instead of r**Lmax times
    #     -> the subcycling win reaches gravity, not just hydro. DM path (below)
    #     keeps its own base-dt KDK; this covers cosmo-gas and self-gravity.
    grav_hook = None
    if subcycle and dm_state is None and \
            (cosmo_kit is not None or grav_apply is not None):
        from amr.poisson import _ghost_phi, kick_level, solve_phi_level
        from amr.state import assemble_base_level
        _n_base = int(np.prod(acfg.base_blocks))
        _mrho = {"v": jnp.asarray(0.0, jnp.float64)}
        if cosmo_kit is not None:
            _fourpiG_of, _bnd, _nvc_h = (lambda aa: cosmo_kit["coeff"](aa)), 0, _n_vc
        else:
            _G = float(getattr(cfg, "GRAVITATIONAL_G", 1.0))
            _fourpiG_of = lambda aa: 4.0 * jnp.pi * _G
            _bnd = int(cfg.BOUNDARY)
            _nvc_h = int(getattr(cfg, "amr_grav_vcycles", 3) or 3)

        def grav_hook(L, lvl, parent_phig, dt, t_sub):
            aa = (jnp.interp(t_sub, cosmo_kit["tau_tab"], cosmo_kit["a_tab"])
                  if cosmo_kit is not None else jnp.asarray(1.0, jnp.float64))
            if L == 0 and _bnd == 0:            # Jeans-swindle mean, base level, once
                _mrho["v"] = jnp.mean(assemble_base_level(lvl.data[:_n_base, 0:1], acfg)[0])
            mean_rho = _mrho["v"] if _bnd == 0 else jnp.asarray(0.0, jnp.float64)
            phi = solve_phi_level(lvl, parent_phig, acfg, fourpiG=_fourpiG_of(aa),
                                  boundary=_bnd, mean_rho=mean_rho, level=L,
                                  n_vcycles=_nvc_h)
            data = kick_level(lvl, phi, parent_phig, dt, L, acfg)
            if cosmo_kit is not None:
                data = cosmo_kit["floor_level"](data, aa, L)
            return data, _ghost_phi(phi, lvl, parent_phig, acfg)

        if is_main_process:
            print("  Berger-Oliger: per-level subcycled gravity + time-accumulated "
                  f"reflux (L0 FMG once/base step){' | sharded divergence' if shard_amr else ''}")

    dm_kit = None
    if dm_state is not None:
        dm_kit = _make_dm_kit(cfg, acfg, dm_state, cosmo_kit, subcycle,
                              shard_amr, is_main_process)

    return _Physics(_cool_hierarchy, cosmo_kit, grav_apply, grav_cosmo,
                    grav_hook, dm_kit)


def _make_dm_kit(cfg, acfg, dm_state, cosmo_kit, subcycle, shard_amr,
                 is_main_process):
    """Coupled gas+DM step pieces (all jit/scan-safe): the joint-source force
    solve, the frozen-DM Berger-Oliger gravity hook, and the DM step constants.
    Returns a _DMKit."""
    from amr.particles import (accel_blocks, deposit_cic, gather_cic,
                               locate_particles)
    from amr.poisson import _ghost_phi, solve_phi_hierarchy
    ndim = acfg.ndim
    _n_vc = int(getattr(cfg, "amr_grav_vcycles", 3) or 3)
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

    # --- Berger-Oliger for the coupled box: gas hydro+gravity subcycled,
    #     DM particles KDK'd at the BASE dt (v1). The DM density is deposited
    #     once per base step (positions frozen for the gas sub-cycle) and fed
    #     into the per-level gas Poisson source; the DM force is re-solved once
    #     at a_next for the closing half-kick. Gas thus pays the fine dt only
    #     on refined blocks, while the L0 gas FMG + the DM solve run once/base.
    _dm_src = {"rho": None}
    grav_hook_dm = None
    if subcycle and not shard_amr:
        from amr.poisson import kick_level, solve_phi_level
        from amr.state import assemble_base_level
        _n_base_dm = int(np.prod(acfg.base_blocks))
        _mrho_dm = {"v": jnp.asarray(0.0, jnp.float64)}

        def grav_hook_dm(L, lvl, parent_phig, dt, t_sub):
            aa = jnp.interp(t_sub, cosmo_kit["tau_tab"], cosmo_kit["a_tab"])
            src = lvl.data[:, 0:1] + _dm_src["rho"][L]     # gas(now) + frozen DM
            if L == 0:
                _mrho_dm["v"] = jnp.mean(
                    assemble_base_level(src[:_n_base_dm], acfg)[0])
            phi = solve_phi_level(lvl, parent_phig, acfg,
                                  fourpiG=cosmo_kit["coeff"](aa), boundary=0,
                                  mean_rho=_mrho_dm["v"], level=L,
                                  n_vcycles=_n_vc, src=src)
            data = kick_level(lvl, phi, parent_phig, dt, L, acfg)
            data = cosmo_kit["floor_level"](data, aa, L)
            return data, _ghost_phi(phi, lvl, parent_phig, acfg)

        if is_main_process:
            print("  Berger-Oliger (gas+DM): subcycled gas gravity + reflux, "
                  "DM KDK at base dt")
    return _DMKit(_dm_forces, grav_hook_dm, _dm_src, dm_mass,
                  _box_arr, _dm_eta, _eps_soft)


def _floor_params(cfg, hierarchy, acfg, gamma, is_main_process):
    """Resolve the opt-in positivity floor (absolute or fraction-of-base-mean).
    Returns (rho_floor, p_floor); rho_floor == 0 means the floor is off."""
    # opt-in positivity floor (0 = off): quiescent-reset unphysical cells after
    # each step so a capped shock seam degrades gracefully instead of stalling.
    from amr import driver
    rho_floor = float(getattr(cfg, "amr_rho_floor", 0.0) or 0.0)
    p_floor = float(getattr(cfg, "amr_p_floor", 0.0) or 0.0)
    # Relative floor: auto-scale from the base-level mean state when no absolute
    # floor is given. Lets deep runs be robust without a hand-picked density scale.
    frac = float(getattr(cfg, "amr_rho_floor_frac", 0.0) or 0.0)
    if rho_floor == 0.0 and frac > 0.0:
        rho_floor, p_floor = driver.floor_from_fraction(hierarchy, acfg,
                                                        gamma=gamma, frac=frac)
    if rho_floor > 0.0 and is_main_process:
        how = f"frac={frac:.1e} of base mean" if frac > 0.0 else "absolute"
        print(f"  positivity floor ON ({how}): rho>={rho_floor:.3e}, "
              f"p>={p_floor:.3e} (quiescent reset)")
    return rho_floor, p_floor


def _make_chunk_steppers(cfg, acfg, hierarchy, phys, sk, ho, chunk_len,
                         is_main_process):
    """Assemble the jitted chunk advance: hydro step + dt fn, the Berger-Oliger
    sub-advances, the positivity floor and the lax.scan chunk bodies. Returns
    _Steppers with exactly one of (chunk_step, chunk_step_dm) jitted."""
    from amr import driver
    gamma, CFL, riemann, limiter = ho.gamma, ho.CFL, ho.riemann, ho.limiter
    subcycle, shard_amr, mesh = ho.subcycle, sk.shard_amr, sk.mesh
    _cool_hierarchy = phys.cool
    cosmo_kit, grav_apply, grav_cosmo = (phys.cosmo_kit, phys.grav_apply,
                                         phys.grav_cosmo)

    if shard_amr:
        from amr import shard
        if subcycle:
            step = jax.jit(lambda lv, dt, hp=None: _cool_hierarchy(
                shard.advance_subcycle_shmap(
                    lv, dt, acfg, mesh, gamma=gamma, limiter=limiter,
                    riemann_solver=riemann, halo_plans=hp), dt))
        else:
            step = jax.jit(lambda lv, dt, hp=None: _cool_hierarchy(
                shard.advance_shmap(
                    lv, dt, acfg, mesh, gamma=gamma, limiter=limiter,
                    riemann_solver=riemann, halo_plans=hp), dt))
    elif subcycle:
        step = jax.jit(lambda lv, dt, hp=None: _cool_hierarchy(
            driver.advance_subcycle(
                lv, dt, acfg, gamma=gamma, limiter=limiter,
                riemann_solver=riemann), dt))
    else:
        step = jax.jit(lambda lv, dt, hp=None: _cool_hierarchy(
            driver.advance(lv, dt, acfg, gamma=gamma, limiter=limiter,
                           riemann_solver=riemann), dt))
    # jitted: the eager version dispatched ~10 tiny ops per level on the host
    # every single step (a real per-step tax at thousands of blocks).
    dt_fn = jax.jit((lambda lv: driver.compute_dt_subcycle(lv, acfg, gamma=gamma, CFL=CFL)) if subcycle
                    else (lambda lv: driver.compute_dt(lv, acfg, gamma=gamma, CFL=CFL)))

    sub_advance = sub_advance_dm = None
    if phys.grav_hook is not None:
        grav_hook = phys.grav_hook
        if shard_amr:
            # sharded divergence + replicated gravity/reflux (amr/shard.py owns
            # the GSPMD-safe reflux wrapper); same hooks, same semantics.
            def sub_advance(h, dt, t_sub):
                return shard.advance_subcycle_shmap(
                    h, dt, acfg, mesh, gamma=gamma, limiter=limiter,
                    riemann_solver=riemann, gravity=grav_hook, reflux=True,
                    t0=t_sub)
        else:
            def sub_advance(h, dt, t_sub):
                return driver.advance_subcycle(
                    h, dt, acfg, gamma=gamma, limiter=limiter, riemann_solver=riemann,
                    gravity=grav_hook, reflux=True, t0=t_sub)

    dm = phys.dm
    if dm is not None:
        from amr.particles import deposit_cic, locate_particles
        from amr.poisson import apply_gravity_kick
        if dm.hook is not None:
            def sub_advance_dm(h, dt, t_sub):
                return driver.advance_subcycle(
                    h, dt, acfg, gamma=gamma, limiter=limiter,
                    riemann_solver=riemann, gravity=dm.hook, reflux=True,
                    t0=t_sub)

    rho_floor, p_floor = _floor_params(cfg, hierarchy, acfg, gamma,
                                       is_main_process)
    floor_on = rho_floor > 0.0

    def _apply_floor(h):
        if floor_on:
            return driver.floor_positivity(h, acfg, gamma=gamma,
                                           rho_floor=rho_floor, p_floor=p_floor)
        return h

    # ------------------------------------------------------------------
    # Chunked time step (single-device AND shard_map): ONE jitted dispatch per
    # regrid interval. A lax.scan advances `chunk_len` steps with dt computed
    # ON DEVICE each step (the old loop's per-step `float(dt_fn(...))` forced a
    # device->host sync + fresh dispatch every step, pinning the GPU at a few %
    # utilization). dt is clamped to the chunk's target time (snapshot / t_end);
    # once the target is reached the remaining scan steps run with dt=0 -- an
    # exact no-op (U + 0*dU) -- so snapshot landings stay exact. The hierarchy
    # buffers are donated, so XLA updates them in place. The sharded `step`
    # (shard_map) traces through the scan like any jittable fn; GSPMD carries
    # the shardings across iterations.
    # ------------------------------------------------------------------
    def _chunk(hier, hp, t_start, target):
        def body(carry, _):
            h, t, taken = carry
            dt = dt_fn(h)
            if cosmo_kit is not None:
                aa = jnp.interp(t, cosmo_kit["tau_tab"], cosmo_kit["a_tab"])
                dt = jnp.minimum(dt, cosmo_kit["dtau_cap"](aa))
            dt = jnp.minimum(dt, jnp.maximum(target - t, 0.0))
            if sub_advance is not None:                  # Berger-Oliger: fused
                h = sub_advance(h, dt, t)                # hydro+gravity+reflux, subcycled
            else:
                h = step(h, dt, hp)
                if cosmo_kit is not None:
                    h = grav_cosmo(h, dt, aa)
                elif grav_apply is not None:
                    h = grav_apply(h, dt)
            h = _apply_floor(h)
            return (h, t + dt, taken + (dt > 0).astype(jnp.int32)), None
        (h, t_fin, taken), _ = jax.lax.scan(
            body, (hier, t_start, jnp.zeros((), jnp.int32)), None, length=chunk_len)
        return h, t_fin, taken

    def _chunk_dm(hier, pos, p, g, t_start, target):
        """Coupled gas+DM chunk: the dense runner's KDK, per scan step --
        DM half-kick (reusing last force) -> drift -> gas hydro -> ONE joint
        solve at a_next -> gas kick -> DM closing half-kick -> floors."""
        def body(carry, _):
            h, ps, pp, gg, t, taken = carry
            aa = jnp.interp(t, cosmo_kit["tau_tab"], cosmo_kit["a_tab"])
            gmax = jnp.max(jnp.sqrt(jnp.sum(gg * gg, axis=1)))
            dt = jnp.minimum(dt_fn(h), cosmo_kit["dtau_cap"](aa))
            dt = jnp.minimum(dt, dm.eta * jnp.sqrt(dm.eps_soft / (gmax + 1e-30)))
            dt = jnp.minimum(dt, jnp.maximum(target - t, 0.0))
            pp = pp + gg * (0.5 * dt)
            ps = jnp.mod(ps + pp * dt, dm.box_arr)
            a_next = jnp.interp(t + dt, cosmo_kit["tau_tab"], cosmo_kit["a_tab"])
            if sub_advance_dm is not None:               # Berger-Oliger gas + base-dt DM
                lev_p, slot_p = locate_particles(ps, h, acfg)
                dm.src["rho"] = deposit_cic(ps, dm.mass, h, lev_p, slot_p, acfg)
                h = sub_advance_dm(h, dt, t)             # gas fused, DM source frozen
                _, g2 = dm.forces(h, ps, a_next)         # DM force at base dt
            else:
                h = step(h, dt)
                phis, g2 = dm.forces(h, ps, a_next)
                h = apply_gravity_kick(h, phis, dt, acfg)
                h = cosmo_kit["floors"](h, a_next)
            h = _apply_floor(h)
            pp = pp + g2 * (0.5 * dt)
            return (h, ps, pp, g2, t + dt,
                    taken + (dt > 0).astype(jnp.int32)), None
        (h, ps, pp, gg, t_fin, taken), _ = jax.lax.scan(
            body, (hier, pos, p, g, t_start, jnp.zeros((), jnp.int32)),
            None, length=chunk_len)
        return h, ps, pp, gg, t_fin, taken

    if dm is not None:
        return _Steppers(None, jax.jit(_chunk_dm, donate_argnums=(0, 1, 2, 3)))
    return _Steppers(jax.jit(_chunk, donate_argnums=0), None)


def run_simulation_amr(cfg, is_main_process=True):
    """Build the hierarchy, halo plans, physics hooks and chunk steppers, then
    run the chunked time loop: advance -> snapshot -> regrid -> perf report.
    Returns (t, nstep, output_dir); `main.run` wraps it in a RunResult."""
    # AMR bookkeeping (Morton ids, conservative prolong/restrict) wants fp64.
    jax.config.update("jax_enable_x64", True)

    # imported up front (as before the helper split) so module import work
    # keeps its timing; the helpers re-import from the cache.
    from amr.config import AMRConfig                              # noqa: F401
    from amr import driver
    from ic_generation.ic import initial_conditions               # noqa: F401
    from file_io.write_snapshot_distributed import write_snapshot_distributed  # noqa: F401

    ndim = cfg.SPATIAL_DIMENSION
    if ndim not in (2, 3):
        raise NotImplementedError("AMR runner supports SPATIAL_DIMENSION in {2, 3}")

    # ---- coupled gas+DM cosmological box (MUSIC IC) ----------------------
    # Gas (PartType0) is meshed to the base grid; DM (PartType1) stays as
    # particles on the hierarchy (amr/particles.py: locate / ghosted-CIC
    # deposit / inverse-CIC gather). Box + cosmology come from the IC header,
    # so this MUST run before the grid/box reads in _build_hierarchy.
    dm_state = dm_pos = dm_p = U0_music = None
    if getattr(cfg, "CosmoHydro", False):
        cfg, dm_state, U0_music = _load_music_ic(cfg, is_main_process)
        # bound early: the t=0 snapshot's DM dump writes these before the
        # chunk section (re)assigns them.
        dm_pos, dm_p = dm_state["pos"], dm_state["p"]

    hierarchy, acfg, sk, ho, ro = _build_hierarchy(cfg, U0_music, ndim,
                                                   is_main_process)

    # sparse halo exchange (see _build_halo_plans). LUNA_AMR_HALO_MODE =
    # sparse(default)/gather/ring forces the choice; "gather"/"ring" fall back
    # to the _use_ring path.
    _halo_mode = os.environ.get("LUNA_AMR_HALO_MODE", "sparse")
    _sparse_halo = (_halo_mode == "sparse" and sk.shard_amr)

    from sources.cooling import setup_cooling
    cooling_solver = setup_cooling(cfg, is_main_process)

    if sk.shard_amr:
        from amr import shard              # sharded advance / regrid kernels

    halo_plans = _build_halo_plans(hierarchy, acfg, sk, _sparse_halo)
    if _sparse_halo and is_main_process:
        _lp, _sp = halo_plans[0], halo_plans[1]
        _rp = halo_plans[2] if len(halo_plans) > 2 else None
        print("  halo exchange: SPARSE all_to_all "
              f"({sum(p is not None for p in _lp)} level-fill plans, "
              f"seam={'on' if _sp is not None else 'gather'}, "
              f"reflux={'iface' if _rp else 'repl'})", flush=True)
    if is_main_process:
        ti = "subcycled (Berger-Oliger)" if ho.subcycle else "synchronized same-dt"
        print(f"  time integration: {ti}  |  {'shard_map over %d devices' % sk.n_dev if sk.shard_amr else 'single device'}")

    io = _snapshot_io(cfg, acfg, sk, is_main_process)

    os.makedirs(cfg.output_dir, exist_ok=True)
    snapshot_times = cfg.snapshot_times
    num = len(snapshot_times)
    regrid_interval = int(getattr(cfg, "amr_regrid_interval", 0))

    if getattr(cfg, "restart", False):
        raise NotImplementedError(
            "cfg.restart=True: the AMR runner has no checkpoint-restart yet "
            "(only the dense path restores from checkpoints).")
    t = float(getattr(cfg, "t_start", 0.0))
    si = sum(1 for ts in snapshot_times if ts < t)   # file indices track the schedule
    if num and si < num and t >= snapshot_times[si]:
        _write_snapshot(si, t, hierarchy, dm_pos, dm_p, io)
        si += 1

    t0 = time.time()
    nstep = 0
    # active cell-updates / base step: recomputed only when the block counts
    # change (after a regrid), so no per-step host sync.
    cu_step = _cu_per_base_step(hierarchy, acfg, ho.subcycle)
    cu_total = 0
    cu_last, perf_t = 0, t0                             # for per-interval Mcells/s
    cadence = max(1, int(getattr(cfg, "output_cadence", 10)))
    # ONE jitted dispatch per regrid interval (see _make_chunk_steppers).
    chunk_len = regrid_interval if regrid_interval > 0 else 16

    phys = _make_physics_hooks(cfg, acfg, hierarchy, dm_state, cooling_solver,
                               ho.subcycle, sk.shard_amr, is_main_process)
    cosmo_kit = phys.cosmo_kit
    chunk_step, chunk_step_dm = _make_chunk_steppers(
        cfg, acfg, hierarchy, phys, sk, ho, chunk_len, is_main_process)
    if dm_state is not None:
        _, dm_g = phys.dm.forces(hierarchy, dm_pos,
                                 jnp.asarray(cosmo_kit["a_start"], jnp.float64))

    _ldbg = os.environ.get("LUNA_AMR_LOOP_DBG")
    _pi = jax.process_index()
    _lit = 0
    last_print, t_prev = 0, t
    while t < cfg.t_end - 1e-12:
        target = min(cfg.t_end, snapshot_times[si] if si < num else cfg.t_end)
        if _ldbg:
            print(f"[loop] p{_pi} it{_lit}: --> chunk_step (t={t:.6f} tgt={target:.6f})",
                  flush=True)
        if dm_state is not None:
            hierarchy, dm_pos, dm_p, dm_g, t_dev, taken_dev = chunk_step_dm(
                hierarchy, dm_pos, dm_p, dm_g,
                jnp.asarray(t, jnp.float64), jnp.asarray(target, jnp.float64))
        else:
            hierarchy, t_dev, taken_dev = chunk_step(
                hierarchy, halo_plans,
                jnp.asarray(t, jnp.float64), jnp.asarray(target, jnp.float64))
        taken = int(taken_dev)                           # one host sync per chunk
        t_prev, t = t, float(t_dev)
        if sk.shard_amr and jax.process_count() > 1:
            # Multi-process: force the advance collectives to COMPLETE before the
            # next collective op (write / regrid). int(taken_dev) only forces the
            # scalar dt-count, leaving the advance all_gather lazily pending; it
            # then interleaves with regrid's collectives across processes and
            # deadlocks. Materializing the hierarchy here keeps the per-process
            # collective streams in lockstep. (Found via loop instrumentation.)
            hierarchy = jax.block_until_ready(hierarchy)
        if _ldbg:
            print(f"[loop] p{_pi} it{_lit}: chunk+sync OK (taken={taken} t={t:.6f})",
                  flush=True)
        if taken == 0:
            break                                        # dt collapsed -- don't spin
        if (t - t_prev) / taken < 1e-14 * cfg.t_end:
            # dt > 0 but denormal (a vacuum/NaN cell drove the CFL to ~0):
            # taken keeps counting, t is frozen -- without this the loop
            # regrids forever at fixed t until the walltime kills the job
            # (watched happen on the 5-level Sedov). Stop loudly instead.
            if is_main_process:
                print(f"  STALLED: dt_avg={(t - t_prev) / taken:.3e} at t={t:.6f}"
                      f" -- stopping (positivity/vacuum suspected)", flush=True)
            break
        nstep += taken
        cu_total += cu_step * taken
        if si < num and t >= snapshot_times[si] - 1e-12:
            if _ldbg:
                print(f"[loop] p{_pi} it{_lit}: --> write #{si}", flush=True)
            _write_snapshot(si, t, hierarchy, dm_pos, dm_p, io)
            si += 1
            if _ldbg:
                print(f"[loop] p{_pi} it{_lit}: write OK", flush=True)
        # Adaptive re-check on the live solution. Capacities grow on a geometric
        # ladder and never shrink (_sized_parents), so `chunk_step` recompiles
        # only ~log2(peak/bucket) times across the whole run.
        if regrid_interval:
            if _ldbg:
                print(f"[loop] p{_pi} it{_lit}: --> regrid", flush=True)
            if sk.shard_amr:
                # shard-aware regrid: child DATA built sharded (no 70 GiB
                # replicated peak that OOM'd at 6 levels), topology replicated;
                # returns the hierarchy already sharded, no separate reshard.
                hierarchy = shard.regrid_sharded(
                    hierarchy, acfg, sk.mesh, thresholds=ro.thresholds,
                    periodic=ro.periodic, bucket=ro.bucket, pad_to=ro.pad_to,
                    criterion=ro.criterion, max_blocks=ro.max_blocks)
            else:
                hierarchy = driver.regrid(hierarchy, acfg, thresholds=ro.thresholds,
                                          periodic=ro.periodic, bucket=ro.bucket,
                                          pad_to=ro.pad_to, criterion=ro.criterion,
                                          max_blocks=ro.max_blocks)
            if _ldbg:
                jax.block_until_ready(hierarchy[0].data)
                print(f"[loop] p{_pi} it{_lit}: regrid OK", flush=True)
            halo_plans = _build_halo_plans(hierarchy, acfg, sk, _sparse_halo)  # topology changed
            cu_step = _cu_per_base_step(hierarchy, acfg, ho.subcycle)  # block counts changed
            if is_main_process:
                counts = " ".join(
                    f"L{L}={int(l.n_active)}/{l.data.shape[0]}"
                    + (" AT-CAP" if ro.max_blocks and L > 0
                       and int(l.n_active) >= ro.max_blocks - 1 else "")
                    for L, l in enumerate(hierarchy))
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
        _lit += 1

    # flush any remaining snapshot slots at t_end
    while si < num:
        _write_snapshot(si, cfg.t_end, hierarchy, dm_pos, dm_p, io)
        si += 1

    # per-device peak memory: the "memory-lean" claim measured on every run
    peak = max((d.memory_stats() or {}).get("peak_bytes_in_use", 0)
               for d in jax.local_devices()) / 1e9
    if is_main_process:
        elapsed = time.time() - t0
        avg = cu_total / max(elapsed, 1e-9) / 1e6
        print(f"AMR run done: {nstep} steps to t={t:.5f} in {elapsed:.1f}s")
        print(f"✓ Average performance: {avg:.2f} Mcells/s (AMR active cell-updates)")
        print(f"✓ Peak GPU memory: {peak:.2f} GB/device (rank-0 local max)")
    return t, nstep, cfg.output_dir
