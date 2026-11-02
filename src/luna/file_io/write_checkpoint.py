"""Serial HDF5 checkpoint writer (full simulation state for exact restart)."""
import os
import h5py

from luna.file_io.checkpoint_utils import _write_meta


def write_checkpoint(U, cfg, t, step, checkpoint_dir="checkpoints"):
    """Write U (conserved) and restart metadata to checkpoint_NNNNNN.hdf5; return the path."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_filename = os.path.join(checkpoint_dir, f"checkpoint_{step:06d}.hdf5")

    with h5py.File(checkpoint_filename, "w") as f:
        conserved = f.create_group("conserved")

        conserved.create_dataset("U", data=U, compression="gzip", compression_opts=4)

        _write_meta(f, cfg, t, step)

    print(f"✓ Checkpoint saved: {checkpoint_filename}")
    return checkpoint_filename
