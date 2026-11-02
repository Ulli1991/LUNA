"""Radiation-field checkpoint: the carry flattened to U's rank and sharding, own file prefix.

See docs/RADIATION.md (Restart).
"""
import os

import h5py
import jax.numpy as jnp
import numpy as np

PREFIX = "radiation"
FORMAT_VERSION = 1


def _is_ion(carry):
    return isinstance(carry, tuple)


def pack(carry):
    """Return (flat, groups, ionization, moments): the carry as one (n, Nx, Ny, Nz) array."""
    if _is_ion(carry):
        R, x, w = carry
        groups = int(R.shape[0])
        flat = jnp.concatenate([R.reshape(-1, *x.shape), x[None], w[None]])
        return flat, groups, True, int(R.shape[1])
    R = carry
    groups = int(R.shape[0])
    return R.reshape(-1, *R.shape[2:]), groups, False, int(R.shape[1])


def unpack(flat, groups, ionization, moments=4):
    """Inverse of :func:`pack`."""
    m = int(moments)
    grid = flat.shape[1:]
    if ionization:
        R = flat[:groups * m].reshape(groups, m, *grid)
        return R, flat[groups * m], flat[groups * m + 1]
    return flat.reshape(groups, m, *grid)


def write_radiation_checkpoint(carry, cfg, t, step, checkpoint_dir, mesh=None):
    """Write this process's shards of the radiation carry beside the same step's checkpoint; return the path or None."""
    from luna.file_io.checkpoint_utils import _shard_context, _write_meta

    flat, groups, ion, moments = pack(carry)
    process_id, process_count, mesh, local = _shard_context(
        flat, cfg, mesh, per_process=True)
    if len(local) == 0:
        return None

    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir,
                        f"{PREFIX}_{step:06d}.{process_id}.hdf5")
    gshape = tuple(int(n) for n in flat.shape)

    def _write(tmp):
        with h5py.File(tmp, "w") as f:
            f.attrs["format_version"] = 2          # the shard layout the reader wants
            f.attrs["radiation_format"] = FORMAT_VERSION
            f.attrs["global_shape"] = np.asarray(gshape, dtype=np.int64)
            f.attrs["rad_groups"] = groups
            f.attrs["ionization"] = bool(ion)
            f.attrs["rad_moments"] = moments
            d = f.create_group("distributed")
            d.attrs["process_id"] = process_id
            d.attrs["process_count"] = process_count
            d.attrs["n_shards"] = len(local)
            shards = f.create_group("shards")
            for k, s in enumerate(local):
                start = [int(sl.start or 0) for sl in s.index]
                stop = [int(sl.stop) if sl.stop is not None else n
                        for sl, n in zip(s.index, gshape)]
                g = shards.create_group(str(k))
                g.attrs["start"] = np.asarray(start, dtype=np.int64)
                g.attrs["stop"] = np.asarray(stop, dtype=np.int64)
                g.attrs["device"] = str(s.device)
                g.create_dataset("U", data=np.asarray(s.data))   # "U": the reader's name
            _write_meta(f, cfg, t, step)

    from luna.file_io.checkpoint_sets import write_atomically
    write_atomically(path, _write)
    return path


def latest_radiation_step(checkpoint_dir):
    """Return the newest step with a radiation file set, or None."""
    import glob
    import re
    steps = set()
    for fn in glob.glob(os.path.join(checkpoint_dir, f"{PREFIX}_*.hdf5")):
        m = re.search(rf"{PREFIX}_(\d+)", os.path.basename(fn))
        if m:
            steps.add(int(m.group(1)))
    return max(steps) if steps else None


def load_radiation_checkpoint(checkpoint_dir, step, mesh, spatial_dimension,
                              sharding=None):
    """Rebuild the radiation carry written at `step`; None if there is no file set for it."""
    from luna.file_io.read_checkpoint import load_checkpoint_sharded

    import glob
    hits = glob.glob(os.path.join(checkpoint_dir, f"{PREFIX}_{step:06d}.*.hdf5"))
    if not hits:
        return None
    with h5py.File(sorted(hits)[0], "r") as f:
        groups = int(f.attrs["rad_groups"])
        ion = bool(f.attrs["ionization"])
        moments = int(f.attrs.get("rad_moments", 4))    # files before M2: M1
    flat, _ = load_checkpoint_sharded(sorted(hits)[0], mesh, spatial_dimension,
                                      prefix=PREFIX, sharding=sharding)
    return unpack(flat, groups, ion, moments)
