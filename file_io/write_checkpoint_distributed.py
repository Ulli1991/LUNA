"""
Distributed checkpoint writing for multi-process JAX simulations.

Each process writes its own checkpoint shard to a separate file,
avoiding expensive gather operations.
"""

import os
from typing import Optional
from jax import Array
from jax.sharding import Mesh
import h5py
import numpy as np

from file_io.checkpoint_utils import _shard_context, _write_meta, _write_units


def write_checkpoint_distributed(
    U_sharded: Array,
    cfg,
    t: float,
    step: int,
    checkpoint_dir: str = "checkpoints",
    mesh: Optional[Mesh] = None,
) -> Optional[str]:
    """
    Write distributed checkpoint - each process writes its own shard.

    This avoids the memory-intensive gather operation by having each process
    write its local shard to a separate file. On restart, each process reads
    its own shard back.

    Filename format: checkpoint_{step:06d}.{process_id}.hdf5
    Example: checkpoint_000100.0.hdf5, checkpoint_000100.1.hdf5, etc.

    Parameters
    ----------
    U_sharded : Array
        Distributed conserved variables array
    cfg : Config
        Configuration object
    t : float
        Current simulation time
    step : int
        Current step number
    checkpoint_dir : str, optional
        Directory to save checkpoints (default: "checkpoints")
    mesh : Mesh, optional
        JAX mesh object (will be reconstructed if not provided)

    Returns
    -------
    checkpoint_filename : str or None
        Path to the checkpoint file written by this process,
        or None if this process has no addressable shards
    """
    # Process info + mesh reconstruction (single-process multi-GPU uses
    # jax.device_count(), multi-process uses process_count) + local shards.
    process_id, process_count, mesh, local_shards = _shard_context(
        U_sharded, cfg, mesh, per_process=True)
    if len(local_shards) == 0:
        return None

    # Get the data from the local shard(s) and convert to numpy
    if len(local_shards) == 1:
        U_local = np.array(local_shards[0].data)
    else:
        # Multiple devices per process - concatenate along last axis
        shard_arrays = [np.array(shard.data) for shard in local_shards]
        U_local = np.concatenate(shard_arrays, axis=-1)

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Build checkpoint filename with process ID
    checkpoint_filename = os.path.join(checkpoint_dir, f"checkpoint_{step:06d}.{process_id}.hdf5")

    # Write checkpoint file (each process writes its own)
    with h5py.File(checkpoint_filename, "w") as f:
        # Distributed metadata
        distributed = f.create_group("distributed")
        distributed.attrs["process_id"] = process_id
        distributed.attrs["process_count"] = process_count

        # Create main group for conserved variables
        conserved = f.create_group("conserved")
        # Save LOCAL shard of conserved variables (no compression for speed)
        conserved.create_dataset("U", data=U_local)

        # Metadata (grid is global; local_shape is this process's shard) + units.
        _write_meta(f, cfg, t, step, local_shape=U_local.shape)
        _write_units(f, cfg)

    # Print confirmation (each process prints its own)
    print(
        f"Process {process_id}: Checkpoint shard saved to {os.path.basename(checkpoint_filename)}"
    )

    return checkpoint_filename
