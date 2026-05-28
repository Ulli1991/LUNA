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
from jax.sharding import NamedSharding
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

    # === Operator-split source terms (per-step apply functions) ===
    from setup.source_ou_forcing import setup_ou_forcing
    from setup.source_cc85 import setup_cc85_source
    from setup.source_external_gravity import setup_external_gravity
    from setup.source_cooling import setup_cooling
    forcing_state  = setup_ou_forcing(cfg, sharding, is_main_process)
    cc85_state     = setup_cc85_source(cfg, sharding, is_main_process)
    ext_grav_state = setup_external_gravity(cfg, sharding, is_main_process)
    cooling_solver = setup_cooling(cfg, is_main_process)

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
