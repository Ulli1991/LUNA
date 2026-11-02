"""Distributed (per-process, per-shard) HDF5 checkpoint writer, format version 2.

See docs/RUNNING.md (Checkpoints).
"""

import os
from typing import Optional
from jax import Array
from jax.sharding import Mesh
import h5py
import numpy as np

from luna.file_io.checkpoint_sets import write_atomically
from luna.file_io.checkpoint_utils import _shard_context, _write_meta, _write_units

FORMAT_VERSION = 2


def write_checkpoint_distributed(
    U_sharded: Array,
    cfg,
    t: float,
    step: int,
    checkpoint_dir: str = "checkpoints",
    mesh: Optional[Mesh] = None,
    extra: Optional[dict] = None,
) -> Optional[str]:
    """Write this process's shards to checkpoint_{step:06d}.{pid}.hdf5; return the path or None.

    ``extra``: small replicated restart state (OU forcing state, RNG key, ``a``) under ``extra/``.
    """
    process_id, process_count, mesh, local_shards = _shard_context(
        U_sharded, cfg, mesh, per_process=True)
    if len(local_shards) == 0:
        return None

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_filename = os.path.join(checkpoint_dir, f"checkpoint_{step:06d}.{process_id}.hdf5")
    gshape = tuple(int(n) for n in U_sharded.shape)

    def _write(path):
        with h5py.File(path, "w") as f:
            f.attrs["format_version"] = FORMAT_VERSION
            f.attrs["global_shape"] = np.asarray(gshape, dtype=np.int64)
            distributed = f.create_group("distributed")
            distributed.attrs["process_id"] = process_id
            distributed.attrs["process_count"] = process_count
            distributed.attrs["n_shards"] = len(local_shards)

            shards = f.create_group("shards")
            for k, s in enumerate(local_shards):
                start = [int(sl.start or 0) for sl in s.index]
                stop = [int(sl.stop) if sl.stop is not None else n for sl, n in zip(s.index, gshape)]
                g = shards.create_group(str(k))
                g.attrs["start"] = np.asarray(start, dtype=np.int64)
                g.attrs["stop"] = np.asarray(stop, dtype=np.int64)
                g.attrs["device"] = str(s.device)
                g.create_dataset("U", data=np.asarray(s.data))

            if extra:
                ex = f.create_group("extra")
                for name, val in extra.items():
                    if val is not None:
                        ex.create_dataset(name, data=np.asarray(val))

            _write_meta(f, cfg, t, step)
            _write_units(f, cfg)

    write_atomically(checkpoint_filename, _write)

    print(f"Process {process_id}: checkpoint shard{'s' if len(local_shards) > 1 else ''} "
          f"saved to {os.path.basename(checkpoint_filename)}")
    return checkpoint_filename
