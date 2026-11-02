"""Read HDF5 checkpoints for restart, onto any device count or decomposition.

See docs/RUNNING.md (Checkpoints).
"""
import glob
import os
import re

import h5py
import jax
import jax.numpy as jnp
import numpy as np

_STEP_RE = re.compile(r"checkpoint_(\d+)(?:\.(\d+))?\.hdf5$")


def _read_meta(f):
    grid = {k: (int if k in ("Nx", "Ny", "Nz", "SPATIAL_DIMENSION", "BOUNDARY") else float)(
        f["grid"].attrs[k]) for k in ("Nx", "Ny", "Nz", "Lx", "Ly", "Lz",
                                      "SPATIAL_DIMENSION", "BOUNDARY")}
    physics = {"gamma": float(f["physics"].attrs["gamma"])}
    numerics = {"ORDER": int(f["numerics"].attrs["ORDER"]),
                "LIMITER": int(f["numerics"].attrs["LIMITER"]),
                "CFL": float(f["numerics"].attrs["CFL"])}
    time_control = {k: float(f["time_control"].attrs[k])
                    for k in ("t_start", "t_end", "time_bet_snaps")}
    units = None
    if "units" in f:
        units = {k: float(f["units"].attrs[k]) for k in
                 ("UnitLength_in_cm", "UnitMass_in_g", "UnitVelocity_in_cm_per_s")}
    extra = {}
    if "extra" in f:
        extra = {name: np.asarray(ds) for name, ds in f["extra"].items()}
    return dict(time=float(f["state"].attrs["time"]), step=int(f["state"].attrs["step"]),
                grid=grid, physics=physics, numerics=numerics,
                time_control=time_control, units=units, extra=extra)


def read_checkpoint(checkpoint_filename):
    """Return a checkpoint's metadata dict; 'U' is the global array for version 1, None for version 2."""
    with h5py.File(checkpoint_filename, "r") as f:
        version = int(f.attrs.get("format_version", 1))
        data = _read_meta(f)
        data["format_version"] = version
        if version >= 2:
            data["U"] = None
            data["global_shape"] = tuple(int(n) for n in f.attrs["global_shape"])
        else:
            data["U"] = jnp.array(f["conserved"]["U"])
            data["global_shape"] = tuple(int(n) for n in data["U"].shape)

    print(f"✓ Checkpoint loaded: {checkpoint_filename} (format v{version})")
    print(f"  - Time: {data['time']:.5f}")
    print(f"  - Step: {data['step']}")
    g = data["grid"]
    print(f"  - Grid: {g['Nx']} x {g['Ny']} x {g['Nz']}")
    print(f"  - Spatial dimension: {g['SPATIAL_DIMENSION']}")
    return data


def read_checkpoint_extra(checkpoint_filename):
    """Return the 'extra' datasets (replicated restart state) of a checkpoint."""
    with h5py.File(checkpoint_filename, "r") as f:
        if "extra" not in f:
            return {}
        return {name: np.asarray(ds) for name, ds in f["extra"].items()}


def _step_re(prefix):
    return re.compile(rf"{re.escape(prefix)}_(\d+)(?:\.(\d+))?\.hdf5$")


def checkpoint_step_files(checkpoint_filename, prefix="checkpoint"):
    """Return all files of the set ``checkpoint_filename`` belongs to, sorted by process id.

    ``prefix`` names the file family (e.g. "radiation", written in the same shard format).
    """
    d, base = os.path.split(checkpoint_filename)
    m = _STEP_RE.match(base) or _step_re(prefix).match(base)
    if m is None:
        return [checkpoint_filename]
    step = int(m.group(1))
    files = glob.glob(os.path.join(d, f"{prefix}_{step:06d}.*.hdf5"))
    files += glob.glob(os.path.join(d, f"{prefix}_{step:06d}.hdf5"))

    def _pid(fn):
        mm = _STEP_RE.match(os.path.basename(fn))
        return int(mm.group(2)) if mm and mm.group(2) is not None else -1
    return sorted(set(files), key=_pid)


def _sharding_for(mesh, spatial_dimension):
    from jax.sharding import NamedSharding, PartitionSpec as P
    return NamedSharding(mesh, P(None, *("x", "y", "z")[:spatial_dimension]))


def load_checkpoint_sharded(checkpoint_filename, mesh, spatial_dimension,
                            prefix="checkpoint", sharding=None):
    """Assemble the sharded ``U`` of a checkpoint set on ``mesh``; return (U_sharded, metadata).

    Each device's block is filled from the stored shards that overlap it, so decompositions may differ.
    """
    meta = read_checkpoint(checkpoint_filename)
    if sharding is None:
        sharding = _sharding_for(mesh, spatial_dimension)
    if meta["format_version"] < 2:
        return jax.device_put(meta["U"], sharding), meta

    gshape = meta["global_shape"]
    stored = []                                   # (file, group, start, stop)
    dtype = None
    for fn in checkpoint_step_files(checkpoint_filename, prefix):
        with h5py.File(fn, "r") as f:
            for k, g in f["shards"].items():
                stored.append((fn, k, tuple(int(v) for v in g.attrs["start"]),
                               tuple(int(v) for v in g.attrs["stop"])))
                if dtype is None:
                    dtype = g["U"].dtype
    if not stored:
        raise ValueError(f"{checkpoint_filename}: no shards found in its checkpoint set")

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
            src_sl = tuple(slice(l - a, h - a) for l, h, a in zip(lo, hi, s0))
            dst_sl = tuple(slice(l - a, h - a) for l, h, a in zip(lo, hi, start))
            with h5py.File(fn, "r") as f:
                block[dst_sl] = f["shards"][k]["U"][src_sl]
            filled += int(np.prod([h - l for l, h in zip(lo, hi)]))
        if filled != block.size:
            raise ValueError(f"checkpoint shards cover {filled} of {block.size} cells of "
                             f"the block {start}..{stop} on {dev}")
        bufs.append(jax.device_put(block, dev))
    U = jax.make_array_from_single_device_arrays(gshape, sharding, bufs)
    return U, meta


def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Return one file of the newest complete checkpoint set in checkpoint_dir, or None."""
    from luna.file_io.checkpoint_sets import latest_step
    if not os.path.exists(checkpoint_dir):
        return None
    step = latest_step(checkpoint_dir)
    if step is None:
        return None
    files = glob.glob(os.path.join(checkpoint_dir, f"checkpoint_{step:06d}*.hdf5"))
    if not files:
        return None

    def _key(fn):
        m = _STEP_RE.match(os.path.basename(fn))
        return (int(m.group(1)) if m else -1, os.path.getmtime(fn))
    files.sort(key=_key, reverse=True)
    return files[0]
