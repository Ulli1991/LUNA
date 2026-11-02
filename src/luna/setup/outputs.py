"""Everything the loop writes or prints: the snapshot schedule, the progress
line, the history file, the periodic diagnostics, snapshots, checkpoints,
what a restart restores (`restore`) and the end of the run (`finish`).

Snapshots land on exact times: proper-time runs use `cfg.snapshot_times`,
comoving runs the scale factors in `cfg.cosmo_snapshot_a` plus the end.
Which writer runs follows from what the state holds: a gas grid alone, the
gas plus the deposited dark matter, or particles alone.

Checkpoints: a gas-only proper-time run writes the per-shard grid format
(v2, wall-clock cadence, the OU forcing state riding along); every run with
a comoving clock or particle species writes the cosmological set (grid +
particle buffers + a, tau, step, every `cosmo_checkpoint_every` steps).
"""
import json
import os
import socket
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils

from luna.file_io.checkpoint_utils import checkpoint_dir
from luna.setup.run_kind import COUPLED, DM, GAS


class Schedule:
    """The times the loop must land on, in the clock's variable."""

    def __init__(self, cfg, clock, t_start, restart):
        if clock.comoving:
            targets = [clock.target_of(float(a)) for a in (cfg.cosmo_snapshot_a or ())]
            targets.append(clock.t_end)
        else:
            targets = list(cfg.snapshot_times) if cfg.write_snapshots else []
        self.targets = np.unique(np.asarray(targets, dtype=float))
        self.t_end = clock.t_end
        self.idx = int(np.searchsorted(self.targets, t_start)) if restart else 0
        self.n = len(self.targets)
        self.write = bool(cfg.write_snapshots)

    def target(self, t):
        nxt = self.targets[self.idx] if self.idx < self.n else float("inf")
        return min(self.t_end, nxt)

    def due(self, t):
        """The index of the snapshot `t` has reached, advancing past it; None."""
        if self.idx < self.n and t >= self.targets[self.idx] - 1e-12 * max(1.0, abs(t)):
            i = self.idx
            self.idx += 1
            return i
        return None


class Outputs:
    def __init__(self, cfg, physics, mesh, init, is_main=True):
        self.cfg, self.physics, self.mesh, self.init = cfg, physics, mesh, init
        self.is_main = is_main
        self.clock = physics.clock
        self.kind = physics.kind
        self.cosmo_format = self.clock.comoving or self.kind != GAS
        self.t0 = time.time()
        self.last_perf = self.t0
        self.last_step = None
        self.first_perf = None              # (time, step) at the first perf line: compile behind it
        self.last_checkpoint = None
        self.schedule = Schedule(cfg, self.clock, self.clock.t_start
                                 if self.clock.comoving else init.t_start, cfg.restart)
        self.insitu = None
        if getattr(cfg, "insitu", None) is not None and self.kind == GAS:
            from luna.analysis.insitu import InSitu
            self.insitu = InSitu(cfg, mesh, is_main)
        self.hist_path = os.path.join(cfg.output_dir, "history.csv")
        self.manifest_path = os.path.join(cfg.output_dir, "manifest.json")
        if is_main:
            os.makedirs(cfg.output_dir, exist_ok=True)
            if cfg.enable_checkpointing or int(cfg.cosmo_checkpoint_every) > 0:
                os.makedirs(checkpoint_dir(cfg), exist_ok=True)
            self._write_manifest(init)
            if not os.path.exists(self.hist_path) or not cfg.restart:
                with open(self.hist_path, "w") as fh:
                    fh.write("# " + ",".join(self._history_fields()) + "\n")

    # -- provenance ------------------------------------------------------------
    def _write_manifest(self, init):
        cfg = self.cfg
        if cfg.restart and os.path.exists(self.manifest_path):
            return
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__) or ".",
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            git_hash = "unknown"
        man = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "hostname": socket.gethostname(), "git_hash": git_hash,
               "jax_version": jax.__version__, "test_problem": cfg.test_problem,
               "Nx": cfg.Nx, "Ny": cfg.Ny, "Nz": cfg.Nz, "Lx": cfg.Lx, "Ly": cfg.Ly, "Lz": cfg.Lz,
               "gamma": cfg.gamma, "CFL": cfg.CFL, "t_start": float(init.t_start),
               "t_end": float(self.clock.t_end), "riemann_solver": cfg.riemann_solver,
               "snapshots": []}
        with open(self.manifest_path, "w") as fh:
            json.dump(man, fh, indent=2)

    def _history_fields(self):
        f = ["step", "t", "dt"]
        if self.clock.comoving:
            f += ["a"]
        if self.physics.hydro is not None and self.cfg.enable_diagnostics:
            f += ["min_rho", "min_P", "max_speed", "max_Mach", "sum_rho", "sum_E"]
        if self.physics.gravity is not None:
            f += ["delta_max"]
        return f

    # -- restart -------------------------------------------------------------------
    def restore(self, state):
        """Everything a restart restores onto the assembled state, by
        checkpoint format. The dense gas grid, t and step of a grid checkpoint
        were already read by read_ics (luna/setup/ics.py) and the radiation
        field by setup_radiation, each onto its own sharding."""
        if not self.cfg.restart:
            return state
        if self.cosmo_format:
            return self._restore_cosmo_set(state)
        return self._restore_grid_extras(state)

    def _restore_grid_extras(self, state):
        """The grid checkpoint's replicated extras: the OU forcing state."""
        cfg = self.cfg
        if "forcing" in state.sources:
            from luna.file_io.read_checkpoint import find_latest_checkpoint, read_checkpoint_extra
            ck = cfg.restart_file or find_latest_checkpoint(checkpoint_dir(cfg))
            ex = read_checkpoint_extra(ck) if ck else {}
            if "forcing_a" in ex and "forcing_key" in ex:
                state = state.with_source("forcing", (jnp.asarray(ex["forcing_a"]),
                                                      jnp.asarray(ex["forcing_key"])))
                if self.is_main:
                    print("✓ OU forcing state (driver + RNG key) restored from the checkpoint")
            elif self.is_main:
                print("⚠ checkpoint carries no forcing state: OU driver restarts from ou_seed")
        return state

    def _restore_cosmo_set(self, state):
        """The cosmological set: grid, particle buffers, a, tau, step, onto the
        assembled buffers; then the species settle and the schedule resumes."""
        cfg = self.cfg
        from luna.file_io.cosmo_checkpoint import restore_cosmo_state
        arrays = self._checkpoint_arrays(state)
        got, a, tau, step = restore_cosmo_state(cfg, arrays, self.is_main)
        state = state.update(t=tau, a=a, step=step)
        if "U" in got:
            state = state.update(U=got["U"])
        g = self.physics.gravity
        if g is not None:
            species = dict(state.species)
            for sp in g.species:
                species[sp.name] = sp.from_checkpoint(species[sp.name], got)
            state = g.settle_after_restart(state.update(species=species))
        self.schedule = Schedule(cfg, self.clock, state.t, True)
        return state

    def _checkpoint_arrays(self, state):
        arrays = {}
        if state.U is not None:
            arrays["U"] = state.U
        g = self.physics.gravity
        if g is not None:
            for sp in g.species:
                arrays.update(sp.checkpoint_arrays(state.species[sp.name]))
        return arrays

    # -- per step -------------------------------------------------------------------
    def target(self, t):
        return self.schedule.target(t)

    def start(self, state):
        if self.is_main:
            print("=" * 60)
            print("SIMULATION START")
            print("=" * 60)
            print(f"{self.clock.label(state.t)} -> {self.clock.label(self.clock.t_end)}")
            if self.physics.hydro is not None:
                dims = (self.cfg.Nx, self.cfg.Ny, self.cfg.Nz)[:self.cfg.SPATIAL_DIMENSION]
                print(f"Global grid: {' x '.join(str(n) for n in dims)}")
            print(flush=True)
        i = self.schedule.due(state.t)
        if i is not None:
            self.snapshot(state, i)
        if self.insitu is not None:
            self.insitu.after_step(state)

    def after_step(self, state, dt):
        cfg = self.cfg
        if cfg.enable_diagnostics and state.step % cfg.diag_interval == 0 \
                and self.physics.hydro is not None:
            self.diagnostics(state, dt)
        elif self.is_main and (state.step % cfg.output_cadence == 0 or state.step == 1):
            self._history_row(state, dt, None)
        if self.is_main and (state.step % cfg.output_cadence == 0 or state.step == 1):
            self.progress(state, dt)
        i = self.schedule.due(state.t)
        if i is not None:
            self.snapshot(state, i)
        if self.insitu is not None:
            self.insitu.after_step(state)
        self.checkpoint(state)

    def diagnostics(self, state, dt):
        """One reduction pass every diag_interval steps: extrema and totals."""
        from luna.setup.diagnostics import diag_reductions
        vals = diag_reductions(state.U, self.cfg.gamma, mesh=self.mesh)
        mr, mp, ms, mM, sr, sE = (float(v) for v in vals)
        cool = self.physics.cooling
        if cool is not None:
            n_cool = cool.substeps_needed(state.U, dt, self.mesh)
            if self.is_main and n_cool > cool.solver.n_sub:
                print(f"⚠ cooling: {n_cool} sub-steps needed (cooling_dt_frac="
                      f"{cool.solver.dt_frac}) but cooling_n_sub={cool.solver.n_sub}: "
                      "the fastest-cooling cells took the remainder as one floored "
                      "Euler step this step; raise cooling_n_sub", flush=True)
        if not self.is_main:
            return
        wall = time.time() - self.t0
        span = self.clock.t_end - self.clock.t_start
        frac = (state.t - self.clock.t_start) / max(span, 1e-30)
        eta_s = wall * (1.0 - frac) / max(frac, 1e-30)
        eta = f"ETA={eta_s / 3600:.1f}h" if eta_s > 3600 else f"ETA={eta_s / 60:.1f}m"
        print(f"📊 step={state.step:>7}  {self.clock.label(state.t)}  min_rho={mr:.3e}  "
              f"min_P={mp:.3e}  max_|v|={ms:.3e}  max_M={mM:.3e}  {eta}")
        g = self.physics.gravity
        if g is not None and g.report(state):
            print("   " + g.report(state))
        self._history_row(state, dt, [mr, mp, ms, mM, sr, sE])

    def _history_row(self, state, dt, diag):
        row = [state.step, state.t, dt]
        if self.clock.comoving:
            row.append(state.a)
        if self.physics.hydro is not None and self.cfg.enable_diagnostics:
            row += diag if diag is not None else [float("nan")] * 6
        if self.physics.gravity is not None:
            row.append(state.stats.get("delta_max", float("nan")))
        with open(self.hist_path, "a") as fh:
            fh.write(",".join(f"{v:.9e}" if isinstance(v, float) else str(v) for v in row) + "\n")

    def progress(self, state, dt):
        """The perf line: work per wall second over the last cadence interval."""
        from luna.utils.memory_utils import get_memory_usage
        now = time.time()
        per_dev, total = get_memory_usage()
        if total is not None and len(per_dev) > 1:
            mem = f", mem = {total:.2f} GB ({len(per_dev)} GPUs: " \
                  + "/".join(f"{m:.2f}" for m in per_dev) + ")"
        elif total is not None:
            mem = f", mem = {total:.2f} GB"
        else:
            mem = ""
        line = f"Step {state.step:5d}, {self.clock.label(state.t)}, dt = {dt:.5e}"
        if self.last_step is not None and state.step > self.last_step:
            n = state.step - self.last_step
            hydro = self.physics.hydro
            if hydro is not None:
                rate = hydro.grid.total_cells * n / max(now - self.last_perf, 1e-9)
                line += f", perf = {rate / 1e6:.2f} Mcells/s"
                if hydro.flops_per_cell is not None:
                    line += f", ~{rate * hydro.flops_per_cell / 1e12:.2f} TFLOP/s"
            else:
                line += f", {1e3 * (now - self.last_perf) / n:.1f} ms/step"
        g = self.physics.gravity
        if g is not None and g.report(state):
            line += "  " + g.report(state)
        print(line + mem, flush=True)
        self.last_perf, self.last_step = now, state.step
        if self.first_perf is None:
            self.first_perf = (now, state.step)

    # -- snapshots -------------------------------------------------------------------
    def snapshot(self, state, idx):
        """Write snapshot `idx` for what the state holds (all ranks enter)."""
        cfg = self.cfg
        if not self.schedule.write:
            return
        t_label = state.a if self.clock.comoving else state.t
        if self.is_main:
            print(f"\n{'─' * 60}\n💾 Saving snapshot #{idx} at {self.clock.label(state.t)}\n{'─' * 60}")
        g = self.physics.gravity
        if self.kind == GAS:
            from luna.file_io.write_snapshot_distributed import write_snapshot_distributed
            hydro = self.physics.hydro
            U = jax.device_put(state.U, hydro.sharding)   # re-attach the explicit sharding
            write_snapshot_distributed(U, cfg, t_label, idx, mesh=self.mesh)
        elif self.kind == COUPLED:
            from luna.file_io.cosmo_snapshot import write_coupled_snapshot
            rho_dm = g.deposit_of(state, "dm")             # a collective: every rank
            lo, hi = g.backend.block_box()
            rho_gas = g.backend.component(state.U, 0)
            write_coupled_snapshot(cfg, cfg.output_dir, state.a, rho_gas, rho_dm,
                                   lo=lo, hi=hi, is_main_process=self.is_main)
        elif self.kind == DM:
            dm = next(sp for sp in g.species if sp.name == "dm")
            dm.snapshot(cfg, state.species["dm"], state.a, self.init.dm_ic,
                        self.init.n_dm or int(state.stats.get("n_active", 0)))
        if g is not None:
            for sp in g.species:
                if sp.name == "sinks" and self.is_main:
                    sp.snapshot(cfg, state.species[sp.name], state.t, idx)
        multihost_utils.sync_global_devices("snapshot_write")
        if self.is_main:
            try:
                with open(self.manifest_path) as fh:
                    man = json.load(fh)
                man["snapshots"].append({"idx": int(idx), "t": float(t_label),
                                         "step": int(state.step),
                                         "wall_time_s": float(time.time() - self.t0)})
                with open(self.manifest_path, "w") as fh:
                    json.dump(man, fh, indent=2)
            except Exception as e:
                print(f"(manifest update failed: {e})")
            print("✓ snapshot written\n", flush=True)

    # -- checkpoints ---------------------------------------------------------------
    def checkpoint(self, state):
        cfg = self.cfg
        if self.cosmo_format:
            from luna.file_io.cosmo_checkpoint import maybe_write_cosmo_checkpoint
            maybe_write_cosmo_checkpoint(cfg, state.step, state.a, state.t,
                                         self._checkpoint_arrays(state), self.is_main)
            return
        if not (cfg.enable_checkpointing and cfg.checkpoint_wall_time_interval > 0):
            return
        from luna.file_io.checkpoint_sets import due_everywhere, mark_complete, prune
        now = time.time()
        if self.last_checkpoint is None:
            self.last_checkpoint = self.t0     # the first interval must elapse first
        # rank 0's clock decides for everyone: the write is collective
        if not due_everywhere(now - self.last_checkpoint >= cfg.checkpoint_wall_time_interval):
            return
        from luna.file_io.write_checkpoint_distributed import write_checkpoint_distributed
        if self.is_main:
            print(f"\n{'─' * 60}\n💾 Writing checkpoint (wall-time elapsed: "
                  f"{now - self.last_checkpoint:.1f}s)\n   t = {state.t:.5f}, step = {state.step}"
                  f"\n{'─' * 60}")
        extra = None
        if "forcing" in state.sources:
            a, key = state.sources["forcing"]
            extra = {"forcing_a": a, "forcing_key": key}
        write_checkpoint_distributed(state.U, cfg, state.t, state.step, checkpoint_dir(cfg),
                                     extra=extra)
        if "radiation" in state.sources:
            # its own file set: the carry has a different rank from U, and
            # teaching the distributed writer about several shardings is a
            # change to the path every run depends on (radiation_checkpoint).
            from luna.file_io.radiation_checkpoint import write_radiation_checkpoint
            write_radiation_checkpoint(state.sources["radiation"], cfg,
                                       state.t, state.step, checkpoint_dir(cfg))
        multihost_utils.sync_global_devices("checkpoint_write")
        self.last_checkpoint = time.time()
        if self.is_main:
            mark_complete(checkpoint_dir(cfg), state.step)
            prune(checkpoint_dir(cfg), cfg.max_checkpoints)
            print("✓ All checkpoint shards saved successfully\n")

    # -- end of the run ------------------------------------------------------------
    def finish(self, state):
        """The closing line; for a gas run the final report and the plots
        (luna/setup/finalize_distributed.py). Returns the state with its wall time."""
        if self.is_main:
            print(f"[{self.kind}] done: {state.step} steps, {self.clock.label(state.t)}, "
                  f"{time.time() - self.t0:.1f}s", flush=True)
        if self.insitu is not None:
            self.insitu.close()
        now = time.time()
        self.wall = now - self.t0
        stats = {**state.stats, "wall_s": self.wall}
        if self.first_perf is not None and state.step > self.first_perf[1]:
            # the steps after the first perf line, and their wall time: the
            # run's throughput without the compile (output still included)
            stats["timed_steps"] = state.step - self.first_perf[1]
            stats["timed_s"] = now - self.first_perf[0]
        state = state.update(stats=stats)
        if self.kind == GAS:
            from luna.setup.finalize_distributed import finalize_simulation_distributed
            finalize_simulation_distributed(state, self.cfg, self.physics.hydro.grid, self.mesh,
                                            step_start=self.init.step_start,
                                            is_main=self.is_main)
        return state
