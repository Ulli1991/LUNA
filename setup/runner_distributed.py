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
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding

from file_io.checkpoint_utils import manage_checkpoints
from file_io.write_checkpoint_distributed import write_checkpoint_distributed
from file_io.write_snapshot_distributed import write_snapshot_distributed
from utils.grid_params import GridParams
from utils.memory_utils import get_memory_usage


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


@jax.jit
def _barotropic_project_jit(U, unit_density, unit_pressure, T0, rho_AD, mu, gm1):
    """Reset total energy so p follows the barotropic EOS (cgs):
        T(rho) = T0 [1 + (rho/rho_AD)^gm1],  p = k_B T rho / (mu m_H).
    Leaves rho and momentum untouched. dtype follows U (run in fp64)."""
    K_B = 1.380649e-16
    M_H = 1.6726219e-24
    rho = U[0]
    mx, my, mz = U[1], U[2], U[3]
    rho_cgs = rho * unit_density
    T = T0 * (1.0 + (rho_cgs / rho_AD) ** gm1)
    p = (K_B * T * rho_cgs / (mu * M_H)) / unit_pressure
    KE = 0.5 * (mx * mx + my * my + mz * mz) / rho
    E_new = p / gm1 + KE
    return U.at[4].set(E_new)


def _barotropic_project(U, cfg):
    u = cfg.units
    return _barotropic_project_jit(
        U, u.UnitDensity_in_cgs, u.UnitPressure_in_cgs,
        float(cfg.barotropic_T0_K), float(cfg.barotropic_rho_AD_cgs),
        float(cfg.barotropic_mu), cfg.gamma - 1.0)


@jax.jit
def _diag_reductions(U, gamma):
    """Single-pass GPU reductions for periodic-diagnostic prints + history.

    Returns (min_rho, min_P, max_speed, max_mach, sum_rho, sum_E).
    Cost: one elementwise pass + 6 reduce-min/max/sum, dominated by the
    reduction collective in multi-process mode (~5e-7 s/cell/process).
    """
    rho = U[0]
    Nvar = U.shape[0]
    vx = U[1] / rho
    vy = jnp.where(Nvar >= 4, U[2] / rho, 0.0)
    vz = jnp.where(Nvar >= 5, U[3] / rho, 0.0)
    speed2 = vx * vx + vy * vy + vz * vz
    KE = 0.5 * rho * speed2
    P = (gamma - 1.0) * (U[-1] - KE)
    cs2 = gamma * P / rho
    mach2 = speed2 / cs2
    return (
        jnp.min(rho),
        jnp.min(P),
        jnp.sqrt(jnp.max(speed2)),
        jnp.sqrt(jnp.max(mach2)),
        jnp.sum(rho),
        jnp.sum(U[-1]),
    )


def run_simulation_distributed(
    U_sharded: Array,
    t_start: float,
    step_start: int,
    cfg,
    grid: GridParams,
    compute_dt_fn: Callable,
    step_fn: Callable,
    sharding: NamedSharding,
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
    is_main_process = jax.process_index() == 0

    # Apply sharding constraint to initial array
    U_sharded = jax.lax.with_sharding_constraint(U_sharded, sharding)

    # === Operator-split source terms (per-step apply functions) ===
    from sources.cc85 import setup_cc85_source
    from sources.cooling import setup_cooling
    from sources.external_gravity import setup_external_gravity
    from sources.ou_forcing import setup_ou_forcing
    from sources.self_gravity import setup_self_gravity

    forcing_state = setup_ou_forcing(cfg, sharding, is_main_process)
    cc85_state = setup_cc85_source(cfg, sharding, is_main_process)
    ext_grav_state = setup_external_gravity(cfg, sharding, is_main_process)
    self_grav_state = setup_self_gravity(cfg, sharding, is_main_process)
    cooling_solver = setup_cooling(cfg, is_main_process)

    # === WARMUP COMPILATION ===
    if is_main_process:
        print("=" * 60)
        print("COMPILING SHARDED FUNCTIONS")
        print("=" * 60)
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
    _ = step_fn(U_sharded, dt_warmup)  # trigger JIT compile, discard result
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

    # Provenance manifest: a single JSON the runner writes once at startup
    # and appends per-snapshot entries to. Captures git hash, JAX version,
    # config hash, datetime, hostname — enough to reproduce any snapshot.
    import json as _json
    import socket as _socket
    import subprocess as _subprocess
    manifest_path = os.path.join(cfg.output_dir, "manifest.json")
    if is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        try:
            git_hash = _subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__) or ".",
                stderr=_subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            git_hash = "unknown"
        try:
            jax_ver = jax.__version__
        except Exception:
            jax_ver = "unknown"
        manifest = {
            "started_utc": pytime.strftime("%Y-%m-%dT%H:%M:%SZ", pytime.gmtime()),
            "hostname": _socket.gethostname(),
            "git_hash": git_hash,
            "jax_version": jax_ver,
            "test_problem": cfg.test_problem,
            "Nx": cfg.Nx, "Ny": cfg.Ny, "Nz": cfg.Nz,
            "Lx": cfg.Lx, "Ly": cfg.Ly, "Lz": cfg.Lz,
            "gamma": cfg.gamma, "CFL": cfg.CFL,
            "t_start": float(t_start), "t_end": float(cfg.t_end),
            "riemann_solver": getattr(cfg, "riemann_solver", "hllc"),
            "snapshots": [],
        }
        if not (cfg.restart and os.path.exists(manifest_path)):
            with open(manifest_path, "w") as fh:
                _json.dump(manifest, fh, indent=2)

    # Checkpoint timing (wall-clock based)
    last_checkpoint_time = None

    # Output directory
    if is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        if cfg.enable_checkpointing:
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # === TIME LOOP ===
    if is_main_process:
        print("=" * 60)
        print("SIMULATION START")
        print("=" * 60)
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

    # History file: Athena++-style log of conserved totals + extrema per
    # diag_interval. Lets you tell which step a NaN'd run actually went
    # wrong, offline, without rerunning.
    hist_path = os.path.join(cfg.output_dir, "history.csv")
    if is_main_process and cfg.enable_diagnostics:
        os.makedirs(cfg.output_dir, exist_ok=True)
        if not os.path.exists(hist_path) or not cfg.restart:
            with open(hist_path, "w") as fh:
                fh.write("# step,t,dt,min_rho,min_P,max_speed,max_Mach,sum_rho,sum_E\n")

    # Main time loop. Runner controls dt:
    #     dt = min(dt_CFL, t_end - t, next_snapshot_time - t)
    # so the step lands exactly on each snapshot time -> no snapshots skipped.
    # Helper: apply step_fn + all operator-split source terms in order.
    # Returns the new U and the new forcing OU state (a, key). Pure — takes
    # the previous-iteration forcing state, returns the new one. Used in
    # the retry-on-NaN loop below.
    def _attempt_step(U_in, dt, forcing_a, forcing_key):
        U_new = step_fn(U_in, dt)
        new_a, new_key = forcing_a, forcing_key
        if forcing_state is not None:
            U_new, new_a, new_key = forcing_state["apply"](U_new, forcing_a, forcing_key, dt)
        if cfg.isothermal_cs2 > 0.0:
            U_new = _isothermal_project(U_new, cfg.isothermal_cs2, cfg.gamma)
        if getattr(cfg, "barotropic_eos", False):
            U_new = _barotropic_project(U_new, cfg)
        if ext_grav_state is not None:
            U_new = ext_grav_state["apply"](U_new, dt)
        if self_grav_state is not None:
            U_new = self_grav_state["apply"](U_new, dt)
        if cc85_state is not None:
            U_new = cc85_state["apply"](U_new, dt)
        if cooling_solver is not None:
            from hydro.cooling import apply_cooling

            U_new = apply_cooling(U_new, dt, cfg, cooling_solver)
        return U_new, new_a, new_key

    while t < cfg.t_end:
        t_snapshot_local = (
            snapshot_times[snapshot_idx] if snapshot_idx < num_snapshots else float("inf")
        )

        dt_cfl = float(compute_dt_fn(U_sharded))
        target = min(cfg.t_end, t_snapshot_local)
        remaining = target - t
        if dt_cfl >= remaining:
            dt = remaining
            new_t = target  # snap exactly, avoid float roundoff
        else:
            dt = dt_cfl
            new_t = t + dt

        # Single attempt — NaN detection deferred to the next iteration's
        # compute_dt (a NaN in U propagates through cons_to_prim and produces
        # a NaN dt, which triggers a RuntimeError next loop). The previous
        # in-loop retry-on-NaN dt_check called compute_dt twice per step and
        # forced an extra host-device sync — measurable per-step overhead
        # even on the healthy path. The diagnostic prints below still catch
        # NaNs at the diag_interval cadence with negligible cost.
        forcing_a_pre = forcing_state["a"] if forcing_state is not None else None
        forcing_key_pre = forcing_state["key"] if forcing_state is not None else None
        U_sharded, fa_new, fk_new = _attempt_step(
            U_sharded, dt, forcing_a_pre, forcing_key_pre
        )
        if forcing_state is not None:
            forcing_state["a"] = fa_new
            forcing_state["key"] = fk_new

        t = new_t
        step += 1

        # === Periodic diagnostics + history line ===
        # One GPU reduction every diag_interval steps. ~negligible cost.
        if cfg.enable_diagnostics and step % cfg.diag_interval == 0:
            min_rho, min_P, max_speed, max_mach, sum_rho, sum_E = _diag_reductions(
                U_sharded, cfg.gamma
            )
            mr = float(min_rho)
            mp = float(min_P)
            ms = float(max_speed)
            mM = float(max_mach)
            sr = float(sum_rho)
            sE = float(sum_E)
            if is_main_process:
                # ETA from elapsed wall time and simulation-time fraction.
                wall_elapsed = pytime.time() - start_time
                frac = (t - t_start) / max(cfg.t_end - t_start, 1e-30)
                eta_s = wall_elapsed * (1.0 - frac) / max(frac, 1e-30)
                eta_h = eta_s / 3600.0
                eta_str = f"ETA={eta_h:.1f}h" if eta_h > 1.0 else f"ETA={eta_s/60.0:.1f}m"
                print(
                    f"📊 step={step:>7}  t={t:.5e}  min_rho={mr:.3e}  "
                    f"min_P={mp:.3e}  max_|v|={ms:.3e}  max_M={mM:.3e}  {eta_str}"
                )
                with open(hist_path, "a") as fh:
                    fh.write(
                        f"{step},{t:.9e},{dt:.9e},{mr:.9e},{mp:.9e},"
                        f"{ms:.9e},{mM:.9e},{sr:.9e},{sE:.9e}\n"
                    )

        # Validation and progress (only on process 0)
        if is_main_process:
            if step % cfg.output_cadence == 0 or step == 1:

                current_time = pytime.time()
                elapsed_since_last = current_time - last_perf_time

                # Get memory usage
                per_device_mem, total_mem = get_memory_usage()
                if total_mem is not None and len(per_device_mem) > 1:
                    per_dev_str = "/".join([f"{m:.2f}" for m in per_device_mem])
                    mem_str = (
                        f", mem = {total_mem:.2f} GB ({len(per_device_mem)} GPUs: {per_dev_str})"
                    )
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
                    print(
                        f"Step {step:5d}, t = {t:.5f}, dt = {dt:.5e}, perf = {mcells_per_sec:.2f} Mcells/s{mem_str}"
                    )
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

            # Append snapshot record to manifest.
            if is_main_process:
                try:
                    with open(manifest_path, "r") as fh:
                        man = _json.load(fh)
                    man["snapshots"].append({
                        "idx": int(snapshot_idx),
                        "t": float(t),
                        "step": int(step),
                        "wall_time_s": float(pytime.time() - start_time),
                    })
                    with open(manifest_path, "w") as fh:
                        _json.dump(man, fh, indent=2)
                except Exception as e:
                    print(f"(manifest update failed: {e})")

            # Update snapshot index
            snapshot_idx += 1

            if is_main_process:
                print("✓ All shards saved successfully\n")

        # Checkpoint (distributed - each process writes its own shard).
        # Initialize last_checkpoint_time to the start_time on first iteration
        # so the first wall-time interval has to elapse before a checkpoint
        # is taken (avoids writing one on step 1).
        if cfg.enable_checkpointing and cfg.checkpoint_wall_time_interval > 0:
            current_wall_time = pytime.time()
            if last_checkpoint_time is None:
                last_checkpoint_time = start_time
            elapsed_wall_time = current_wall_time - last_checkpoint_time
            should_checkpoint = elapsed_wall_time >= cfg.checkpoint_wall_time_interval

            if should_checkpoint:
                if is_main_process:
                    print(f"\n{'─'*60}")
                    if last_checkpoint_time is not None:
                        elapsed = pytime.time() - last_checkpoint_time
                        print(f"💾 Writing checkpoint (wall-time elapsed: {elapsed:.1f}s)")
                    else:
                        print("💾 Writing first checkpoint")
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
                    print("✓ All checkpoint shards saved successfully\n")

                # Update checkpoint time
                last_checkpoint_time = pytime.time()

    return U_sharded, t, step, start_time
