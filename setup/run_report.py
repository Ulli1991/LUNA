"""Shared progress/History reporting for the runners.

StepLog owns the print cadence (cfg.output_cadence), always flushes (buffered
SLURM logs otherwise show nothing until the job ends), and appends an
Athena++-style history.csv so a crashed run can be diagnosed offline.
"""
import os
import time


class StepLog:
    """Per-run reporter: `log.step(n, line, row=...)` prints `line` every
    `cadence` steps (flushed) and appends `row` to <output_dir>/history.csv."""

    def __init__(self, cfg, is_main, header=None, history_fields=None):
        self.is_main = bool(is_main)
        self.cadence = max(1, int(getattr(cfg, "output_cadence", 10)))
        self.t0 = time.time()
        self.hist = None
        if self.is_main and history_fields:
            out = getattr(cfg, "output_dir", None)
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
