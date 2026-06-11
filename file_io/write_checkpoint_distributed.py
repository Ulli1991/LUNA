"""
Distributed checkpoint writing for multi-process JAX simulations.

Each process writes its own checkpoint shard to a separate file,
avoiding expensive gather operations.
"""

import os
from typing import Optional
import jax
from jax import Array
from jax.sharding import Mesh
import h5py
import numpy as np
from utils.distributed import create_device_mesh


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
    # Get process info
    process_id = jax.process_index()
    process_count = jax.process_count()

    # Reconstruct mesh if not provided. Single-process multi-GPU uses
    # jax.device_count() (e.g., 4), multi-process uses process_count.
    if mesh is None:
        n_devices = jax.device_count() if process_count == 1 else process_count
        mesh, _ = create_device_mesh(n_devices, cfg.SPATIAL_DIMENSION)

    # Get local addressable shard for this process
    local_shards = U_sharded.addressable_shards
    if len(local_shards) == 0:
        print(f"Warning: Process {process_id} has no addressable shards")
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

        # Save simulation state
        state = f.create_group("state")
        state.attrs["time"] = float(t)
        state.attrs["step"] = int(step)

        # Save grid configuration (global, same on all processes)
        grid = f.create_group("grid")
        grid.attrs["Nx"] = cfg.Nx
        grid.attrs["Ny"] = cfg.Ny
        grid.attrs["Nz"] = cfg.Nz
        grid.attrs["Lx"] = cfg.Lx
        grid.attrs["Ly"] = cfg.Ly
        grid.attrs["Lz"] = cfg.Lz
        grid.attrs["SPATIAL_DIMENSION"] = cfg.SPATIAL_DIMENSION
        grid.attrs["BOUNDARY"] = cfg.BOUNDARY

        # Local shard info (varies by process)
        grid.attrs["local_shape"] = U_local.shape

        # Save physics parameters
        physics = f.create_group("physics")
        physics.attrs["gamma"] = cfg.gamma

        # Save numerical scheme parameters
        numerics = f.create_group("numerics")
        numerics.attrs["ORDER"] = cfg.ORDER
        numerics.attrs["LIMITER"] = cfg.LIMITER
        numerics.attrs["CFL"] = cfg.CFL

        # Save time control
        time_control = f.create_group("time_control")
        time_control.attrs["t_start"] = cfg.t_start
        time_control.attrs["t_end"] = cfg.t_end
        time_control.attrs["time_bet_snaps"] = cfg.time_bet_snaps

        # Save unit system if configured
        if cfg.units.is_enabled:
            units_grp = f.create_group("units")
            units_grp.attrs["UnitLength_in_cm"] = cfg.units.UnitLength_in_cm
            units_grp.attrs["UnitMass_in_g"] = cfg.units.UnitMass_in_g
            units_grp.attrs["UnitVelocity_in_cm_per_s"] = cfg.units.UnitVelocity_in_cm_per_s

    # Print confirmation (each process prints its own)
    print(
        f"Process {process_id}: Checkpoint shard saved to {os.path.basename(checkpoint_filename)}"
    )

    return checkpoint_filename
