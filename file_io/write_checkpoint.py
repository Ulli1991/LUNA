"""Serial HDF5 checkpoint writer (full simulation state for exact restart)."""
import os
import h5py

from file_io.checkpoint_utils import _write_meta


def write_checkpoint(U, cfg, t, step, checkpoint_dir="checkpoints"):
    """Write checkpoint_NNNNNN.hdf5 with the conserved variables U (not
    primitives) plus all metadata for exact restart; returns the path."""

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Build checkpoint filename
    checkpoint_filename = os.path.join(checkpoint_dir, f"checkpoint_{step:06d}.hdf5")

    # Write checkpoint file
    with h5py.File(checkpoint_filename, "w") as f:
        # Create main group for conserved variables
        conserved = f.create_group("conserved")

        # Save conserved variables U directly (no conversion to primitives)
        conserved.create_dataset("U", data=U, compression="gzip", compression_opts=4)

        _write_meta(f, cfg, t, step)

    print(f"✓ Checkpoint saved: {checkpoint_filename}")
    return checkpoint_filename
