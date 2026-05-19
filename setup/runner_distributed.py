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
