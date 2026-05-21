"""
Distributed simulation time loop and execution for multi-node JAX.

This module handles the main time integration loop including:
- Time stepping with sharded arrays
- Distributed snapshot writing (each process writes its shard)
- Distributed checkpoint writing (each process writes its shard)
- Performance tracking across processes
- Validation
"""
import os
import time as pytime
from typing import Tuple, Callable
import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import NamedSharding, PartitionSpec
from jax.experimental import multihost_utils

from file_io.write_snapshot_distributed import write_snapshot_distributed
from file_io.write_checkpoint_distributed import write_checkpoint_distributed
from file_io.checkpoint_utils import manage_checkpoints
from hydro.physics_checks import validate_from_diagnostics
from utils.memory_utils import get_memory_usage
from utils.grid_params import GridParams


@jax.jit
def _isothermal_project_jit(U, cs2_iso, gm1):
    """Reset total energy so p = cs2_iso * rho exactly.

    E_new = p / (gamma - 1) + 0.5 |mom|^2 / rho with p = cs2_iso * rho.
    Leaves rho and momentum components untouched.
    """
    rho = U[0]
    mx, my, mz = U[1], U[2], U[3]
    p_iso = cs2_iso * rho
    KE = 0.5 * (mx * mx + my * my + mz * mz) / rho
    E_new = p_iso / gm1 + KE
    return U.at[4].set(E_new)


def _isothermal_project(U, cs2_iso, gamma):
    return _isothermal_project_jit(U, jnp.float32(cs2_iso), jnp.float32(gamma - 1.0))


def run_simulation_distributed(
    U_sharded: Array,
    t_start: float,
    step_start: int,
    cfg,
    grid: GridParams,
    compute_dt_fn: Callable,
    step_fn: Callable,
    sharding: NamedSharding
) -> Tuple[Array, float, int, float]:
    """Execute the main distributed simulation time loop.

    The runner controls all time-target arithmetic. The integrator's job is
    purely:
        compute_dt_fn(U) -> dt_cfl
        step_fn(U, dt)   -> U_new
    The runner clamps dt to land on the next snapshot time or t_end exactly,
    so no snapshots are ever skipped.
    """
    # Cache process info (avoid repeated function calls)
    is_main_process = (jax.process_index() == 0)

    # Apply sharding constraint to initial array
    U_sharded = jax.lax.with_sharding_constraint(U_sharded, sharding)

    # === Optional OU turbulent driving setup ===
    # Operator-split: after each hydro step_fn, evolve the OU amplitudes by dt
    # then apply F = sum_k a_k exp(i k.x) as a momentum source. Coord arrays
    # are sharded the same way as U's spatial axes so the work stays local.
    forcing_state = None
    if cfg.ou_config is not None:
        from hydro.turbulence import OUDriver
        driver = OUDriver(cfg.ou_config)

        dx, dy, dz = cfg.Lx / cfg.Nx, cfg.Ly / cfg.Ny, cfg.Lz / cfg.Nz
        x_full = (jnp.arange(cfg.Nx, dtype=jnp.float32) + 0.5) * dx
        y_full = (jnp.arange(cfg.Ny, dtype=jnp.float32) + 0.5) * dy
        z_full = (jnp.arange(cfg.Nz, dtype=jnp.float32) + 0.5) * dz
        X, Y, Z = jnp.meshgrid(x_full, y_full, z_full, indexing="ij")
        # U has spec (None, 'x', 'y') for 2D or (None, 'x', 'y', 'z') for 3D;
        # coords are 3D, share axes 1.. of U.
        coord_spec = PartitionSpec(*sharding.spec[1:])
        coord_sharding = NamedSharding(sharding.mesh, coord_spec)
        X = jax.device_put(X, coord_sharding)
        Y = jax.device_put(Y, coord_sharding)
        Z = jax.device_put(Z, coord_sharding)

        key = jax.random.PRNGKey(int(cfg.ou_seed))
        key, sub = jax.random.split(key)
        a_state = driver.init_state(sub)

        @jax.jit
        def _forcing_apply(U, a, key, dt):
            new_key, sub = jax.random.split(key)
            a_new = driver.evolve(a, dt, sub)
            U_new = driver.apply(U, a_new, dt, X, Y, Z)
            return U_new, a_new, new_key

        forcing_state = {"a": a_state, "key": key, "apply": _forcing_apply}
        if is_main_process:
            print(f"✓ OU driver active: {driver.n_modes} modes "
                  f"(k in [{cfg.ou_config.k_min}, {cfg.ou_config.k_max}]), "
                  f"zeta={cfg.ou_config.solenoidal_weight}, "
                  f"T_ac={cfg.ou_config.autocorrelation_time}, "
                  f"rms_a={cfg.ou_config.rms_acceleration}")

    # === Optional CC85 starburst-wind source ===
    # Mdot/Edot deposited uniformly inside a sphere of radius R_*, centered
    # on the box. Operator-split, same pattern as the OU forcing.
    cc85_state = None
    if getattr(cfg, "cc85_enabled", False):
        if not cfg.units.is_enabled:
            raise ValueError("cc85_enabled requires cfg.units (Arepo system).")
        # Convert physical inputs to code-unit rates
        SEC_PER_YR = 3.15576e7
        MSUN = 1.98892e33
        PC = 3.08568e18
        Mdot_cgs = cfg.cc85_eta_M * cfg.cc85_SFR_Msun_per_yr * MSUN / SEC_PER_YR
        Edot_cgs = (cfg.cc85_eta_E * cfg.cc85_SFR_Msun_per_yr / 100.0
                    * 1.0e51 / SEC_PER_YR)
        R_star_cgs = cfg.cc85_R_star_pc * PC

        # Code-unit rates: code-mass per code-time, code-energy per code-time
        Mdot_code = (Mdot_cgs / cfg.units.UnitMass_in_g
                     * cfg.units.UnitTime_in_s)
        Edot_code = (Edot_cgs / cfg.units.UnitEnergy_in_cgs
                     * cfg.units.UnitTime_in_s)
        R_star_code = R_star_cgs / cfg.units.UnitLength_in_cm
        V_sphere_code = (4.0 / 3.0) * jnp.pi * R_star_code ** 3

        # Build the sphere mask. Must be multi-process aware: each rank only
        # builds its local slice (otherwise the full-domain construction +
        # device_put deadlocks in multi-process mode when arrays are large).
        from jax.sharding import PartitionSpec
        import numpy as _np
        from utils.distributed import compute_local_slice
        dx, dy, dz = cfg.Lx / cfg.Nx, cfg.Ly / cfg.Ny, cfg.Lz / cfg.Nz
        cx, cy, cz = 0.5 * cfg.Lx, 0.5 * cfg.Ly, 0.5 * cfg.Lz
        if jax.process_count() > 1:
            xs, ys, zs = compute_local_slice(cfg.Nx, cfg.Ny,
                                             sharding.mesh, jax.process_index(),
                                             cfg.Nz)
            x_loc = (_np.arange(xs[0], xs[1], dtype=_np.float32) + 0.5) * dx - cx
            y_loc = (_np.arange(ys[0], ys[1], dtype=_np.float32) + 0.5) * dy - cy
            z_loc = (_np.arange(zs[0], zs[1], dtype=_np.float32) + 0.5) * dz - cz
            Xl, Yl, Zl = _np.meshgrid(x_loc, y_loc, z_loc, indexing="ij")
            r2_loc = Xl * Xl + Yl * Yl + Zl * Zl
            mask_local = (r2_loc < R_star_code ** 2).astype(_np.float32)
            global_shape = (cfg.Nx, cfg.Ny, cfg.Nz)
            coord_sharding = NamedSharding(sharding.mesh,
                                            PartitionSpec("x", "y", "z"))
            local_dev = jax.local_devices()[0]
            mask_on_dev = jax.device_put(mask_local, local_dev)
            sphere_mask = jax.make_array_from_single_device_arrays(
                shape=global_shape, sharding=coord_sharding,
                arrays=[mask_on_dev]
            )
        else:
            x_full = (jnp.arange(cfg.Nx, dtype=jnp.float32) + 0.5) * dx - cx
            y_full = (jnp.arange(cfg.Ny, dtype=jnp.float32) + 0.5) * dy - cy
            z_full = (jnp.arange(cfg.Nz, dtype=jnp.float32) + 0.5) * dz - cz
            X, Y, Z = jnp.meshgrid(x_full, y_full, z_full, indexing="ij")
            r2 = X * X + Y * Y + Z * Z
            sphere_mask = (r2 < R_star_code ** 2).astype(jnp.float32)
            if cfg.SPATIAL_DIMENSION == 3:
                coord_spec = PartitionSpec(*sharding.spec[1:])
                coord_sharding = NamedSharding(sharding.mesh, coord_spec)
                sphere_mask = jax.device_put(sphere_mask, coord_sharding)

        rho_dot_code = Mdot_code / V_sphere_code
        e_dot_code   = Edot_code / V_sphere_code

        # CRITICAL: pass sphere_mask as a JIT argument (NOT closure),
        # otherwise JAX tries to inline the sharded array as MLIR constants
        # and crashes multi-process. Same lesson as ext_grav above.
        @jax.jit
        def _cc85_apply_inner(U, dt, mask):
            drho = rho_dot_code * dt * mask
            dE   = e_dot_code   * dt * mask
            U_new = U.at[0].add(drho)
            U_new = U_new.at[-1].add(dE)
            return U_new

        def _cc85_apply(U, dt):
            return _cc85_apply_inner(U, dt, sphere_mask)

        cc85_state = {"apply": _cc85_apply}
        if is_main_process:
            # In multi-process mode, sphere_mask is sharded across nodes —
            # jnp.sum and then int() requires a global reduction. Skip the
            # cell-count print for that case to keep setup deadlock-free.
            n_inj = int(jnp.sum(sphere_mask)) if jax.process_count() == 1 else -1
            print(f"✓ CC85 wind source: R*={cfg.cc85_R_star_pc:.1f} pc, "
                  f"SFR={cfg.cc85_SFR_Msun_per_yr} Msun/yr, "
                  f"eta_M={cfg.cc85_eta_M}, eta_E={cfg.cc85_eta_E}")
            print(f"  Mdot = {Mdot_cgs:.3e} g/s = {Mdot_code:.3e} code/code_t")
            print(f"  Edot = {Edot_cgs:.3e} erg/s = {Edot_code:.3e} code")
            print(f"  Injection cells: {n_inj}")

    # === Optional external static-potential gravity (M82 disk setup) ===
    # Apply a position-dependent (g_x, g_y, g_z) to U each step as an
    # operator-split source. The acceleration field is precomputed once
    # from Miyamoto-Nagai (stellar) + NFW (halo) and cached on-device.
    ext_grav_state = None
    if getattr(cfg, "m82_use_external_gravity", False):
        if cfg.SPATIAL_DIMENSION != 3:
            raise ValueError("m82_use_external_gravity requires 3D.")
        from hydro.external_gravity import combined_acceleration
        from jax.sharding import PartitionSpec
        from utils.distributed import compute_local_slice
        import numpy as _np
        dx, dy, dz = cfg.Lx / cfg.Nx, cfg.Ly / cfg.Ny, cfg.Lz / cfg.Nz
        cx, cy, cz = 0.5 * cfg.Lx, 0.5 * cfg.Ly, 0.5 * cfg.Lz
        global_shape = (cfg.Nx, cfg.Ny, cfg.Nz)
        coord_sharding = NamedSharding(sharding.mesh,
                                        PartitionSpec("x", "y", "z"))
        if jax.process_count() > 1:
            xs, ys, zs = compute_local_slice(cfg.Nx, cfg.Ny,
                                             sharding.mesh, jax.process_index(),
                                             cfg.Nz)
            x_loc = (_np.arange(xs[0], xs[1], dtype=_np.float32) + 0.5) * dx - cx
            y_loc = (_np.arange(ys[0], ys[1], dtype=_np.float32) + 0.5) * dy - cy
            z_loc = (_np.arange(zs[0], zs[1], dtype=_np.float32) + 0.5) * dz - cz
            Xl, Yl, Zl = _np.meshgrid(x_loc, y_loc, z_loc, indexing="ij")
            gx_loc, gy_loc, gz_loc = combined_acceleration(
                Xl, Yl, Zl,
                cfg.m82_stellar_MN_M, cfg.m82_stellar_MN_a_kpc,
                cfg.m82_stellar_MN_b_kpc,
                cfg.m82_halo_NFW_M_vir, cfg.m82_halo_NFW_c,
            )
            local_dev = jax.local_devices()[0]
            def _assemble(local_arr):
                d = jax.device_put(_np.asarray(local_arr), local_dev)
                return jax.make_array_from_single_device_arrays(
                    shape=global_shape, sharding=coord_sharding, arrays=[d]
                )
            gx = _assemble(gx_loc); gy = _assemble(gy_loc); gz = _assemble(gz_loc)
        else:
            x_full = (jnp.arange(cfg.Nx, dtype=jnp.float32) + 0.5) * dx - cx
            y_full = (jnp.arange(cfg.Ny, dtype=jnp.float32) + 0.5) * dy - cy
            z_full = (jnp.arange(cfg.Nz, dtype=jnp.float32) + 0.5) * dz - cz
            X, Y, Z = jnp.meshgrid(x_full, y_full, z_full, indexing="ij")
            gx, gy, gz = combined_acceleration(
                X, Y, Z,
                cfg.m82_stellar_MN_M, cfg.m82_stellar_MN_a_kpc, cfg.m82_stellar_MN_b_kpc,
                cfg.m82_halo_NFW_M_vir, cfg.m82_halo_NFW_c,
            )
            coord_spec = PartitionSpec(*sharding.spec[1:])
            coord_sharding = NamedSharding(sharding.mesh, coord_spec)
            gx = jax.device_put(gx, coord_sharding)
            gy = jax.device_put(gy, coord_sharding)
            gz = jax.device_put(gz, coord_sharding)

        # CRITICAL: pass gx, gy, gz as JIT arguments (NOT as closure
        # variables), otherwise JAX tries to inline the sharded global
        # arrays as MLIR constants and crashes in multi-process mode with
        # "Fetching value for jax.Array that spans non-addressable devices".
        @jax.jit
        def _ext_grav_apply_inner(U, dt, gx_in, gy_in, gz_in):
            rho = U[0]
            mx, my, mz = U[1], U[2], U[3]
            new_mx = mx + dt * rho * gx_in
            new_my = my + dt * rho * gy_in
            new_mz = mz + dt * rho * gz_in
            mx_mid = 0.5 * (mx + new_mx)
            my_mid = 0.5 * (my + new_my)
            mz_mid = 0.5 * (mz + new_mz)
            new_E = U[-1] + dt * (mx_mid * gx_in + my_mid * gy_in + mz_mid * gz_in)
            U_new = U.at[1].set(new_mx)
            U_new = U_new.at[2].set(new_my)
            U_new = U_new.at[3].set(new_mz)
            U_new = U_new.at[-1].set(new_E)
            return U_new

        def _ext_grav_apply(U, dt):
            return _ext_grav_apply_inner(U, dt, gx, gy, gz)

        ext_grav_state = {"apply": _ext_grav_apply}
        if is_main_process:
            print(f"✓ External M82 gravity: "
                  f"MN(M={cfg.m82_stellar_MN_M}, a={cfg.m82_stellar_MN_a_kpc}, "
                  f"b={cfg.m82_stellar_MN_b_kpc}) + "
                  f"NFW(M200={cfg.m82_halo_NFW_M_vir}, c={cfg.m82_halo_NFW_c})")

    # === Optional radiative cooling (Townsend 2009 + S&D93 CIE) ===
    cooling_solver = None
    if getattr(cfg, "cooling_enabled", False):
        if not cfg.units.is_enabled:
            raise ValueError("cooling_enabled requires cfg.units (Arepo system).")
        from hydro.cooling import sutherland_dopita_cie, TownsendCoolingSolver
        if cfg.cooling_table == "sd93_solar":
            table = sutherland_dopita_cie(metallicity="solar")
        else:
            raise ValueError(f"Unknown cooling_table: {cfg.cooling_table}")
        mu_cool = getattr(cfg, "cc85_mu", 0.6)
        cooling_solver = TownsendCoolingSolver(table=table, mu=mu_cool)
        if is_main_process:
            print(f"✓ Cooling enabled: table={cfg.cooling_table}, "
                  f"T range [{table.T.min():.2e}, {table.T.max():.2e}] K, "
                  f"mu={mu_cool}")

    # === WARMUP COMPILATION ===
    if is_main_process:
        print("="*60)
        print("COMPILING SHARDED FUNCTIONS")
        print("="*60)
        print("Running one step to trigger JIT compilation...")

    # Warmup: compile both compute_dt and step. Save the IC and restore it
    # afterward so the warmup step does NOT advance the simulation state.
    # (The previous behavior advanced U by one CFL timestep while leaving t
    # at t_start, which introduced a fixed +dt_warmup phase offset visible in
    # short-time convergence tests as a spurious 1st-order error.)
    compile_start = pytime.time()
    U_pre_warmup = U_sharded
    dt_warmup = compute_dt_fn(U_sharded)
    jax.block_until_ready(dt_warmup)
    _ = step_fn(U_sharded, dt_warmup)        # trigger JIT compile, discard result
    jax.block_until_ready(_)
    U_sharded = jax.lax.with_sharding_constraint(U_pre_warmup, sharding)
    compile_time = pytime.time() - compile_start

    if is_main_process:
        print(f"✓ Compilation complete ({compile_time:.1f}s)\n")

    # === SNAPSHOTS AND OUTPUT ===
    # Use pre-computed snapshot times from config (numpy array, not JAX)
    snapshot_times = cfg.snapshot_times
    snapshot_idx = int(np.searchsorted(snapshot_times, t_start)) if cfg.restart else 0
    num_snapshots = len(snapshot_times)

    # Checkpoint timing (wall-clock based)
    last_checkpoint_time = None

    # Output directory
    if is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        if cfg.enable_checkpointing:
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # === TIME LOOP ===
    if is_main_process:
        print("="*60)
        print("SIMULATION START")
        print("="*60)
        print(f"Time range: {t_start:.3f} → {cfg.t_end:.3f}")
        if cfg.SPATIAL_DIMENSION == 3:
            print(f"Global grid: {cfg.Nx} x {cfg.Ny} x {cfg.Nz}")
        elif cfg.SPATIAL_DIMENSION == 2:
            print(f"Global grid: {cfg.Nx} x {cfg.Ny}")
        else:
            print(f"Global grid: {cfg.Nx}")
        print()

    t = t_start
    step = step_start
    start_time = pytime.time()
    last_perf_time = start_time

    # Total cells in global grid
    total_cells = grid.total_cells

    # Main time loop. Runner controls dt:
    #     dt = min(dt_CFL, t_end - t, next_snapshot_time - t)
    # so the step lands exactly on each snapshot time -> no snapshots skipped.
    while t < cfg.t_end:
        t_snapshot_local = (snapshot_times[snapshot_idx]
                            if snapshot_idx < num_snapshots
                            else float('inf'))

        dt_cfl = float(compute_dt_fn(U_sharded))
        target = min(cfg.t_end, t_snapshot_local)
        remaining = target - t
        if dt_cfl >= remaining:
            dt = remaining
            new_t = target          # snap exactly, avoid float roundoff
        else:
            dt = dt_cfl
            new_t = t + dt

        U_sharded = step_fn(U_sharded, dt)

        # Operator-split OU forcing
        if forcing_state is not None:
            U_sharded, forcing_state["a"], forcing_state["key"] = forcing_state["apply"](
                U_sharded, forcing_state["a"], forcing_state["key"], dt)

        # Isothermal-EOS projection: overwrite E so p = cs2_iso * rho.
        # Soft thermostat — leaves rho, momentum untouched.
        if cfg.isothermal_cs2 > 0.0:
            U_sharded = _isothermal_project(U_sharded, cfg.isothermal_cs2, cfg.gamma)

        # Operator-split external static-potential gravity (M82 disk).
        if ext_grav_state is not None:
            U_sharded = ext_grav_state["apply"](U_sharded, dt)

        # Operator-split CC85 central source: Mdot/Edot inside R_*.
        if cc85_state is not None:
            U_sharded = cc85_state["apply"](U_sharded, dt)

        # Operator-split radiative cooling (Townsend exact integration).
        if cooling_solver is not None:
            from hydro.cooling import apply_cooling
            U_sharded = apply_cooling(U_sharded, dt, cfg, cooling_solver)

        t = new_t
        step += 1

        # Validation and progress (only on process 0)
        if is_main_process:
            if step % cfg.output_cadence == 0 or step == 1:
                # Note: Diagnostics removed for performance (was 2nd function call per timestep)
                # if cfg.enable_diagnostics:
                #     validate_from_diagnostics(min_rho, min_p, has_nan, has_inf, step=step, time=t)

                current_time = pytime.time()
                elapsed_since_last = current_time - last_perf_time

                # Get memory usage
                per_device_mem, total_mem = get_memory_usage()
                if total_mem is not None and len(per_device_mem) > 1:
                    per_dev_str = "/".join([f"{m:.2f}" for m in per_device_mem])
                    mem_str = f", mem = {total_mem:.2f} GB ({len(per_device_mem)} GPUs: {per_dev_str})"
                elif total_mem is not None:
                    mem_str = f", mem = {total_mem:.2f} GB"
                else:
                    mem_str = ""

                if step == 1:
                    print(f"Step {step:5d}, t = {t:.5f}, dt = {dt:.5e}{mem_str}")
                else:
                    steps_since_last = cfg.output_cadence if step > cfg.output_cadence else step - 1
                    cell_updates = total_cells * steps_since_last
                    cells_per_sec = cell_updates / elapsed_since_last
                    mcells_per_sec = cells_per_sec / 1e6
                    print(f"Step {step:5d}, t = {t:.5f}, dt = {dt:.5e}, perf = {mcells_per_sec:.2f} Mcells/s{mem_str}")
                last_perf_time = current_time

        # Save snapshot (all processes participate)
        # Only check if we haven't written all snapshots yet
        if snapshot_idx < num_snapshots and t >= snapshot_times[snapshot_idx]:
            if is_main_process:
                print(f"\n{'─'*60}")
                print(f"💾 Saving snapshot #{snapshot_idx} at t = {t:.5f}")
                print(f"{'─'*60}")

            # At very large problem sizes (e.g. 8192^2) the protobuf cache
            # path can produce an array whose sharding metadata is
            # "UnspecifiedValue", which breaks shard.index in the writer.
            # Re-attach the explicit sharding before writing.
            U_sharded = jax.device_put(U_sharded, sharding)

            # Each process writes its own shard to separate file
            write_snapshot_distributed(U_sharded, cfg, t, snapshot_idx)

            # Synchronize all processes after snapshot write
            # This ensures all processes finish I/O before proceeding
            # Overhead is minimal compared to the file I/O itself
            multihost_utils.sync_global_devices("snapshot_write")

            # Update snapshot index
            snapshot_idx += 1

            if is_main_process:
                print(f"✓ All shards saved successfully\n")

        # Checkpoint (distributed - each process writes its own shard)
        if cfg.enable_checkpointing and cfg.checkpoint_wall_time_interval > 0:
            # All processes independently check wall time (they're synchronized by simulation)
            current_wall_time = pytime.time()
            should_checkpoint = False
            if last_checkpoint_time is None:
                should_checkpoint = (step > step_start)
            else:
                elapsed_wall_time = current_wall_time - last_checkpoint_time
                should_checkpoint = (elapsed_wall_time >= cfg.checkpoint_wall_time_interval)

            if should_checkpoint:
                if is_main_process:
                    print(f"\n{'─'*60}")
                    if last_checkpoint_time is not None:
                        elapsed = pytime.time() - last_checkpoint_time
                        print(f"💾 Writing checkpoint (wall-time elapsed: {elapsed:.1f}s)")
                    else:
                        print(f"💾 Writing first checkpoint")
                    print(f"   Simulation: t = {t:.5f}, step = {step}")
                    print(f"{'─'*60}")

                # Each process writes its own checkpoint shard (no gather!)
                write_checkpoint_distributed(U_sharded, cfg, t, step, cfg.checkpoint_dir)

                # Synchronize all processes after checkpoint write
                # This ensures all processes finish I/O before cleanup
                # Overhead is minimal compared to the file I/O itself
                multihost_utils.sync_global_devices("checkpoint_write")

                # Cleanup old checkpoints (only process 0 needs to do this)
                if is_main_process:
                    manage_checkpoints(cfg.checkpoint_dir, cfg.max_checkpoints)
                    print(f"✓ All checkpoint shards saved successfully\n")

                # Update checkpoint time
                last_checkpoint_time = pytime.time()

    return U_sharded, t, step, start_time
