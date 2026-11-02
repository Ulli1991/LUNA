"""Shared progress/History reporting for the runners.

StepLog owns the print cadence (cfg.output_cadence), always flushes (buffered
SLURM logs otherwise show nothing until the job ends), and appends an
Athena++-style history.csv so a crashed run can be diagnosed offline.
"""
import os
import time
from dataclasses import dataclass, field


@dataclass
class RunResult:
    """Uniform return of `main.run` across every kind of run.

    t          final time (the scale factor `a` for comoving runs; None for
               autodiff tasks, whose product is `extra`)
    n_steps    steps taken (None where not meaningful)
    output_dir where snapshots / products were written
    extra      kind-specific payload: the final grid, the particles, task results

    Iterates as ``(t, n_steps, output_dir)`` so tuple unpacking keeps working.
    """
    t: float = None
    n_steps: int = None
    output_dir: str = None
    extra: dict = field(default_factory=dict)

    def __iter__(self):
        return iter((self.t, self.n_steps, self.output_dir))


def result_of(state, cfg, physics):
    """The RunResult of an evolved state: the gas grid, the dark matter's
    live rows, and the particle census the harnesses check."""
    from luna.setup.run_kind import DM
    t = state.a if physics.clock.comoving else state.t
    extra = {}
    if state.U is not None:
        extra["U"] = state.U
    if "boundary_mass" in state.sources:
        extra["boundary_mass"] = state.sources["boundary_mass"]
    if "dm" in state.species:
        dm = state.species["dm"]
        extra.update(dm_pos=dm.pos, dm_p=dm.p, dm_active=dm.active, dm_ids=dm.ids,
                     n_particles=int(state.stats.get("n_active", 0)),
                     mass=state.stats.get("mass"))
        g = physics.gravity
        sp = next(s for s in g.species if s.name == "dm")
        extra["n_exchanges"] = sp.n_exchanges
    if physics.kind == DM:
        extra["a"] = state.a
        extra["n_steps"] = state.step
    return RunResult(t=t, n_steps=state.step, output_dir=cfg.output_dir, extra=extra)


class StepLog:
    """Per-run reporter: `log.step(n, line, row=...)` prints `line` every
    `cadence` steps (flushed) and appends `row` to <output_dir>/history.csv."""

    def __init__(self, cfg, is_main, header=None, history_fields=None):
        self.is_main = bool(is_main)
        self.cadence = max(1, int(cfg.output_cadence))
        self.t0 = time.time()
        self.hist = None
        if self.is_main and history_fields:
            out = cfg.output_dir
            if out:
                os.makedirs(out, exist_ok=True)
                self.hist = os.path.join(out, "history.csv")
                with open(self.hist, "w") as fh:
                    fh.write("# " + ",".join(history_fields) + "\n")
        if self.is_main and header:
            print(header, flush=True)

    def due(self, n, force=False):
        return self.is_main and (force or n % self.cadence == 0)

    def step(self, n, line, row=None, force=False):
        if not self.due(n, force):
            return
        print(line, flush=True)
        if self.hist is not None and row is not None:
            with open(self.hist, "a") as fh:
                fh.write(",".join(f"{v:.6e}" if isinstance(v, float) else str(v)
                                  for v in row) + "\n")

    def elapsed(self):
        return time.time() - self.t0

    def done(self, line):
        if self.is_main:
            print(line, flush=True)


def require_finite(value, *, what, step, t, hint=""):
    """Raise RuntimeError if `value` is not finite.

    A NaN state must fail the job LOUDLY: `t + nan` is nan, and every
    `while t < t_end` comparison against nan is False, so a NaN'd run would
    otherwise fall out of its time loop and exit 0 with garbage output.
    """
    import numpy as _np
    if not _np.isfinite(value):
        raise RuntimeError(
            f"non-finite {what} at step {step} (t={t:.6e}): the state went "
            f"NaN/Inf. {hint}".rstrip())
