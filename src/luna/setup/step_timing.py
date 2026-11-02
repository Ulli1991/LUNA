"""Per-part wall time of a step and the profiler label of each part (docs/RUNNING.md, Performance).

`part(name)` labels work for the profilers; a timed step (cfg.timing_every) waits after each part."""
import contextlib
import functools
import json
import os
import time

import jax
import numpy as np

JSONL = "step_timing.jsonl"
_ACTIVE = []           # the StepTimer of the step being timed, if any
_ROCTX = None


def _roctx():
    """The roctx library when rocprofv3 is attached (or LUNA_ROCTX is set), else False."""
    global _ROCTX
    if _ROCTX is None:
        _ROCTX = False
        if "rocprofiler" in os.environ.get("LD_PRELOAD", "") or os.environ.get("LUNA_ROCTX"):
            import ctypes
            for lib in ("librocprofiler-sdk-roctx.so", "libroctx64.so"):
                try:
                    _ROCTX = ctypes.CDLL(lib)
                    _ROCTX.roctxRangePushA.argtypes = [ctypes.c_char_p]
                    break
                except OSError:
                    continue
    return _ROCTX


@contextlib.contextmanager
def part(name):
    """Label the enclosed work `name` in traces: named_scope, TraceAnnotation, a roctx range."""
    rx = _roctx()
    if rx:
        rx.roctxRangePushA(name.encode())
    try:
        with jax.named_scope(name), jax.profiler.TraceAnnotation(name):
            yield
    finally:
        if rx:
            rx.roctxRangePop()


def _sync(tree):
    """Wait for every device array in `tree`."""
    for x in jax.tree_util.tree_leaves(tree):
        if isinstance(x, jax.Array):
            x.block_until_ready()


def run_part(name, fn, timer=None):
    """fn() under the label `name`; on a timed step, its device wall ms too."""
    if timer is None:
        with part(name):
            return fn()
    return timer.call(name, fn)


def labelled(name, fn):
    """`fn` labelled `name`, timed as a sub-part of the current part on a timed step."""
    @functools.wraps(fn)
    def call(*args, **kw):
        if not _ACTIVE:
            with part(name):
                return fn(*args, **kw)
        return _ACTIVE[-1].call(name, lambda: fn(*args, **kw), inputs=(args, kw))
    return call


def label_methods(obj, prefix, names):
    """Replace obj's methods `names` (those it has) by `labelled` ones."""
    for n in names:
        if hasattr(obj, n):
            setattr(obj, n, labelled(f"{prefix}.{n}", getattr(obj, n)))
    return obj


class StepTimer:
    """The parts of one step: name -> wall ms, nested names joined by '/'."""

    def __init__(self, step):
        self.step = step
        self.ms = {}
        self.count = {}
        self.stack = []

    def __enter__(self):
        _ACTIVE.append(self)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.total_ms = 1e3 * (time.perf_counter() - self.t0)
        _ACTIVE.remove(self)
        return False

    def call(self, name, fn, inputs=None):
        """fn() timed to the device: its inputs ready first, its outputs waited for."""
        if inputs is not None:
            _sync(inputs)
        key = "/".join(self.stack + [name])
        self.stack.append(name)
        t0 = time.perf_counter()
        try:
            with part(name):
                out = fn()
            _sync(out)
        finally:
            self.stack.pop()
        self.ms[key] = self.ms.get(key, 0.0) + 1e3 * (time.perf_counter() - t0)
        self.count[key] = self.count.get(key, 0) + 1
        return out


def _allgather_json(obj):
    """Every process's `obj` (JSON-able), in process order."""
    if jax.process_count() == 1:
        return [obj]
    from jax.experimental import multihost_utils
    raw = np.frombuffer(json.dumps(obj).encode(), np.uint8)
    n = int(np.max(multihost_utils.process_allgather(np.array([raw.size]))))
    buf = np.zeros(n, np.uint8)
    buf[:raw.size] = raw
    every = np.asarray(multihost_utils.process_allgather(buf)).reshape(-1, n)
    return [json.loads(bytes(r).rstrip(b"\0").decode()) for r in every]


class StepTiming:
    """cfg.timing_every: which steps are timed, the table on the main process, the JSONL record."""

    def __init__(self, every, output_dir, is_main=True, fresh=True):
        self.every = int(every)
        self.is_main = is_main
        self.path = os.path.join(output_dir, JSONL)
        if is_main and fresh and os.path.exists(self.path):
            os.remove(self.path)
        self.last = None
        self._t_end, self._n = None, 0

    def timer(self, step):
        """A StepTimer if step `step` is to be timed, else None."""
        if self._t_end is None:
            self._t_end = time.perf_counter()
        if self.every > 0 and step % self.every == 0:
            return StepTimer(step)
        self._n += 1
        return None

    def record(self, timer, state, dt):
        """Gather the ranks' parts, print the table, append the JSONL record."""
        untimed_ms = 1e3 * (timer.t0 - self._t_end) / self._n if self._n else None
        mine = {"parts": timer.ms, "count": timer.count, "total": timer.total_ms}
        ranks = _allgather_json(mine)
        names = tree_order(list(dict.fromkeys(k for r in ranks for k in r["parts"])))
        parts = {k: [min(r["parts"].get(k, 0.0) for r in ranks),
                     max(r["parts"].get(k, 0.0) for r in ranks)] for k in names}
        total = [min(r["total"] for r in ranks), max(r["total"] for r in ranks)]
        rec = {"step": int(timer.step), "t": float(state.t), "dt": float(dt),
               "ranks": len(ranks), "total_ms": total, "parts_ms": parts,
               "calls": timer.count}
        if untimed_ms is not None:
            rec["untimed_ms_per_step"] = untimed_ms
        if self.is_main:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(format_table(rec), flush=True)
        self.last = rec
        self._t_end, self._n = time.perf_counter(), 0
        return rec


def tree_order(names, prefix=""):
    """`names` with every part followed by its sub-parts (a sub-part is timed before its part ends)."""
    level = [k for k in names if k.startswith(prefix) and k.count("/") == prefix.count("/")]
    out = []
    for k in level:
        out.append(k)
        out.extend(tree_order(names, k + "/"))
    return out


def over_hydro(parts):
    """The top-level parts that cost more than the hydro step (max over ranks)."""
    hydro = parts.get("hydro", [0.0, 0.0])[1]
    return [k for k, v in parts.items() if "/" not in k and k != "hydro" and hydro > 0 and v[1] > hydro]


def format_table(rec):
    """The record as a text table: part, ms (max over ranks), min if it differs, % of the step."""
    parts, total = rec["parts_ms"], rec["total_ms"][1]
    slow = set(over_hydro(parts))
    top = sum(v[1] for k, v in parts.items() if "/" not in k)
    rows = [f"⏱ step {rec['step']} timed part by part ({rec['ranks']} rank(s), ms = max over ranks)",
            f"  {'part':<34}{'ms':>10}{'min ms':>10}{'%':>7}"]
    for k, (lo, hi) in parts.items():
        depth = k.count("/")
        name = "  " * depth + k.rsplit("/", 1)[-1]
        mn = f"{lo:10.2f}" if hi - lo > 0.05 * max(hi, 1e-9) else " " * 10
        flag = "  > hydro" if k in slow else ""
        rows.append(f"  {name:<34}{hi:10.2f}{mn}{100 * hi / max(total, 1e-9):7.1f}{flag}")
    between = rec.get("between_ms", max(total - top, 0.0))
    rows.append(f"  {'(between parts)':<34}{between:10.2f}{'':10}{100 * between / max(total, 1e-9):7.1f}")
    rows.append(f"  {'step total':<34}{total:10.2f}{'':10}{100.0:7.1f}")
    if "untimed_ms_per_step" in rec:
        rows.append(f"  untimed steps since the last timed one: {rec['untimed_ms_per_step']:.2f} ms/step")
    return "\n".join(rows)
