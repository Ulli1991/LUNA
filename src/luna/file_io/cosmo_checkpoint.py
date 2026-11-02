"""Checkpoints for the cosmological runners: gas grid, particle buffers and (a, tau, step).

See docs/RUNNING.md (Checkpoints).
"""
import glob
import os
import re

import h5py
import jax
import numpy as np

from luna.file_io.checkpoint_utils import write_provenance

FORMAT_VERSION = 1
# arrays an older checkpoint of this format may lack
OPTIONAL = ("dm_ids",)
_STEP_RE = re.compile(r"cosmo_(\d+)\.(\d+)\.hdf5$")


def _write_sharded(group, arr):
    group.attrs["global_shape"] = np.asarray(arr.shape, dtype=np.int64)
    sh = group.create_group("shards")
    for k, s in enumerate(arr.addressable_shards):
        start = [int(sl.start or 0) for sl in s.index]
        stop = [int(sl.stop) if sl.stop is not None else n for sl, n in zip(s.index, arr.shape)]
        g = sh.create_group(str(k))
        g.attrs["start"] = np.asarray(start, dtype=np.int64)
        g.attrs["stop"] = np.asarray(stop, dtype=np.int64)
        g.create_dataset("data", data=np.asarray(s.data))


def write_cosmo_checkpoint(checkpoint_dir, step, cfg, arrays, scalars):
    """Write this process's shards of ``arrays`` and the ``scalars`` to cosmo_{step}.{pid}.hdf5; return the path."""
    pid = jax.process_index()
    os.makedirs(checkpoint_dir, exist_ok=True)
    fn = os.path.join(checkpoint_dir, f"cosmo_{int(step):06d}.{pid}.hdf5")
    with h5py.File(fn, "w") as f:
        f.attrs["format_version"] = FORMAT_VERSION
        f.attrs["process_id"] = pid
        f.attrs["process_count"] = jax.process_count()
        write_provenance(f.attrs, cfg)
        for k, v in scalars.items():
            f.attrs[k] = v
        ag = f.create_group("arrays")
        for name, arr in arrays.items():
            _write_sharded(ag.create_group(name), arr)
    return fn


def cosmo_checkpoint_files(fn):
    """Return every process file of the checkpoint set ``fn`` belongs to."""
    d, base = os.path.split(fn)
    m = _STEP_RE.match(base)
    if m is None:
        return [fn]
    files = glob.glob(os.path.join(d, f"cosmo_{int(m.group(1)):06d}.*.hdf5"))
    return sorted(files, key=lambda p: int(_STEP_RE.match(os.path.basename(p)).group(2)))


def find_latest_cosmo_checkpoint(checkpoint_dir):
    files = glob.glob(os.path.join(checkpoint_dir, "cosmo_*.hdf5"))
    if not files:
        return None
    return max(files, key=lambda p: (int(_STEP_RE.match(os.path.basename(p)).group(1)),
                                     os.path.getmtime(p)))


def manage_cosmo_checkpoints(checkpoint_dir, max_checkpoints):
    """Keep the newest ``max_checkpoints`` steps (<= 0 keeps all)."""
    if max_checkpoints <= 0:
        return
    by_step = {}
    for fn in glob.glob(os.path.join(checkpoint_dir, "cosmo_*.hdf5")):
        m = _STEP_RE.match(os.path.basename(fn))
        if m:
            by_step.setdefault(int(m.group(1)), []).append(fn)
    steps = sorted(by_step)
    for st in steps[:max(len(steps) - max_checkpoints, 0)]:
        for fn in by_step[st]:
            os.remove(fn)


def _load_sharded(files, name, sharding, gshape_expected=None):
    stored, dtype, gshape = [], None, None
    for fn in files:
        with h5py.File(fn, "r") as f:
            g = f["arrays"][name]
            gshape = tuple(int(n) for n in g.attrs["global_shape"])
            for k, s in g["shards"].items():
                stored.append((fn, k, tuple(int(v) for v in s.attrs["start"]),
                               tuple(int(v) for v in s.attrs["stop"])))
                if dtype is None:
                    dtype = s["data"].dtype
    if gshape_expected is not None and tuple(gshape_expected) != gshape:
        raise ValueError(
            f"cosmo checkpoint '{name}' has global shape {gshape}, the run expects "
            f"{tuple(gshape_expected)}: particle buffers need the same device count and "
            "capacity (dm_capacity_factor) as the run that wrote the checkpoint")
    bufs = []
    for dev, index in sharding.addressable_devices_indices_map(gshape).items():
        start = [int(sl.start or 0) for sl in index]
        stop = [int(sl.stop) if sl.stop is not None else n for sl, n in zip(index, gshape)]
        block = np.empty([b - a for a, b in zip(start, stop)], dtype=dtype)
        filled = 0
        for fn, k, s0, s1 in stored:
            lo = [max(a, b) for a, b in zip(start, s0)]
            hi = [min(a, b) for a, b in zip(stop, s1)]
            if any(l >= h for l, h in zip(lo, hi)):
                continue
            src = tuple(slice(l - a, h - a) for l, h, a in zip(lo, hi, s0))
            dst = tuple(slice(l - a, h - a) for l, h, a in zip(lo, hi, start))
            with h5py.File(fn, "r") as f:
                block[dst] = f["arrays"][name]["shards"][k]["data"][src]
            filled += int(np.prod([h - l for l, h in zip(lo, hi)]))
        if filled != block.size:
            raise ValueError(f"cosmo checkpoint '{name}': stored shards cover {filled} of "
                             f"{block.size} entries of the block on {dev}")
        bufs.append(jax.device_put(block, dev))
    return jax.make_array_from_single_device_arrays(gshape, sharding, bufs)


def load_cosmo_checkpoint(fn, live_arrays):
    """Restore ``live_arrays`` onto their shardings from the set of ``fn``; return (arrays, scalars).

    Particle buffers must match the live shape (same device count and capacity); the grid may not.
    """
    files = cosmo_checkpoint_files(fn)
    with h5py.File(files[0], "r") as f:
        scalars = {k: f.attrs[k] for k in f.attrs
                   if k not in ("format_version", "process_id", "process_count", "config_json",
                                 "code_revision")}
        names = list(f["arrays"].keys())
    out = {}
    for name, live in live_arrays.items():
        if name not in names and name in OPTIONAL:
            continue
        if name not in names:
            raise KeyError(f"cosmo checkpoint has no array '{name}' (has {names})")
        expected = live.shape if name != "U" else None
        out[name] = _load_sharded(files, name, live.sharding, expected)
    return out, scalars


def restore_cosmo_state(cfg, arrays, is_main_process=True):
    """Restore ``arrays`` from cfg.restart_file or the newest checkpoint; return (arrays, a, tau, step)."""
    from luna.file_io.checkpoint_utils import checkpoint_dir
    fn = cfg.restart_file or find_latest_cosmo_checkpoint(checkpoint_dir(cfg))
    if fn is None:
        raise FileNotFoundError(f"restart=True but no cosmo_*.hdf5 in {checkpoint_dir(cfg)}")
    out, sc = load_cosmo_checkpoint(fn, arrays)
    a, tau, step = float(sc["a"]), float(sc["tau"]), int(sc["step"])
    if is_main_process:
        print(f"[cosmo] restart from {os.path.basename(fn)}: step {step}, a={a:.5f} "
              f"(z={1/a-1:.3f}), tau={tau:.6e}, arrays {list(arrays)}", flush=True)
    return out, a, tau, step


def maybe_write_cosmo_checkpoint(cfg, step, a, tau, arrays, is_main_process=True):
    """Write a set every cfg.cosmo_checkpoint_every steps and prune old ones; every rank must call this."""
    every = int(cfg.cosmo_checkpoint_every)
    if every <= 0 or step % every != 0:
        return None
    from luna.file_io.checkpoint_utils import checkpoint_dir
    fn = write_cosmo_checkpoint(checkpoint_dir(cfg), step, cfg, arrays,
                                {"a": float(a), "tau": float(tau), "step": int(step)})
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices("cosmo_checkpoint")
    if is_main_process:
        manage_cosmo_checkpoints(checkpoint_dir(cfg), int(cfg.max_checkpoints))
        print(f"[cosmo] checkpoint step {step} a={a:.5f} -> {os.path.basename(fn)}", flush=True)
    return fn
