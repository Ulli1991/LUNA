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


def run_simulation_distributed(
    U_sharded: Array,
    t_start: float,
    step_start: int,
    cfg,
    grid: GridParams,
    time_step_fn: Callable,  # Now: (U, t, t_snapshot, t_end) -> (U_new, dt)
    sharding: NamedSharding
) -> Tuple[Array, float, int, float]:
    """
    Execute the main distributed simulation time loop.

    Parameters
    ----------
    U_sharded : Array
        Sharded initial state (conserved variables)
    t_start : float
        Start time
    step_start : int
        Start step
    cfg : Config
        Configuration object
    grid : GridParams
        Grid parameters
    time_step_fn : Callable[[Array, float], Array]
        Time stepping function: (U, dt) -> U
    compute_dt_fn : Callable
        Timestep computation function
    sharding : NamedSharding
        Sharding specification

    Returns
    -------
    U_sharded : Array
        Final sharded state
    t : float
        Final time
    step : int
        Final step
    start_time : float
        Wall-clock start time (for performance tracking)
    """
    # Cache process info (avoid repeated function calls)
    is_main_process = (jax.process_index() == 0)

    # Apply sharding constraint to initial array
    U_sharded = jax.lax.with_sharding_constraint(U_sharded, sharding)

    # === WARMUP COMPILATION ===
    if is_main_process:
        print("="*60)
        print("COMPILING SHARDED FUNCTIONS")
        print("="*60)
        print("Running one step to trigger JIT compilation...")

    # Warmup with combined dt+step function
    U_sharded, dt_warmup = time_step_fn(U_sharded, t_start, t_start, cfg.t_end)
    U_sharded = jax.block_until_ready(U_sharded)

    if is_main_process:
        print("✓ Compilation complete\n")

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
        print(f"Global grid: {cfg.Nx} x {cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else 'N/A'}")
        print()

    t = t_start
    step = step_start
    start_time = pytime.time()
    last_perf_time = start_time

    # Total cells in global grid
    total_cells = grid.total_cells

    # Main time loop
    while t < cfg.t_end:
        # MUSCL-Hancock: Combined dt computation + time step in ONE function call
        # This reduces multi-GPU overhead from 2 function calls to 1 per timestep
        t_snapshot_local = snapshot_times[snapshot_idx]

        U_sharded, dt = time_step_fn(U_sharded, t, t_snapshot_local, cfg.t_end)

        t += dt
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
        if t >= snapshot_times[snapshot_idx]:
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
            if snapshot_idx >= num_snapshots:
                snapshot_idx = num_snapshots - 1

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
