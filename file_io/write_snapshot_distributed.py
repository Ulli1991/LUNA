import os
import jax
import h5py
import jax.numpy as jnp
import numpy as np
from utils.distributed import compute_local_slice, create_device_mesh

def write_snapshot_distributed(U_sharded, cfg, t, step, mesh=None):
    """
    Save the simulation state from sharded arrays to HDF5 files.
    Each process writes its own shard to a separate file.

    This avoids expensive gather operations that can cause hangs.
    Post-processing can stitch these files together if needed.

    Filename format: {base_name}_{step:03d}.{process_id}.hdf5
    Example: snap_000.0.hdf5, snap_000.1.hdf5, snap_000.2.hdf5, snap_000.3.hdf5

    Parameters
    ----------
    U_sharded : sharded array
        Distributed array across devices
    cfg : Config
        Configuration object
    t : float
        Current simulation time
    step : int
        Snapshot number
    mesh : Mesh, optional
        JAX mesh object (will be reconstructed if not provided)
    """
    # Get process info
    process_id = jax.process_index()
    process_count = jax.process_count()

    # Reconstruct mesh if not provided
    if mesh is None:
        mesh, _ = create_device_mesh(process_count, cfg.SPATIAL_DIMENSION)

    # Get local addressable shard for this process
    # Use addressable_shards to get only the data on this process
    local_shards = U_sharded.addressable_shards
    if len(local_shards) == 0:
        print(f"Warning: Process {process_id} has no addressable shards")
        return

    # Get the data from the local shard(s) and convert to numpy immediately
    # This ensures we're working with local data only
    if len(local_shards) == 1:
        # Single device per process - just get the data
        U_local = np.array(local_shards[0].data)
    else:
        # Multiple devices per process - concatenate along last axis
        shard_arrays = [np.array(shard.data) for shard in local_shards]
        U_local = np.concatenate(shard_arrays, axis=-1)

    # Build output path with process ID in a subdirectory for this snapshot
    # Format: output_dir/snapdir_000/snap_000.0.hdf5 for process 0, etc.
    # This keeps all shards for a given snapshot together in one directory
    snapshot_dir = os.path.join(cfg.output_dir, f"snapdir_{step:03d}")
    os.makedirs(snapshot_dir, exist_ok=True)

    snapshot_filename = os.path.join(
        snapshot_dir,
        f"{cfg.base_name}_{step:03d}.{process_id}.hdf5"
    )

    # --- Extract fields ---
    rho = U_local[0]
    vx = U_local[1] / rho

    if cfg.SPATIAL_DIMENSION >= 2:
        vy = U_local[2] / rho
    else:
        vy = 0.0

    if cfg.SPATIAL_DIMENSION == 3:
        vz = U_local[3] / rho
    else:
        vz = 0.0

    # --- Kinetic energy ---
    vel_sq = vx**2
    if cfg.SPATIAL_DIMENSION >= 2:
        vel_sq = vel_sq + vy**2
    if cfg.SPATIAL_DIMENSION == 3:
        vel_sq = vel_sq + vz**2

    kinetic_energy = 0.5 * rho * vel_sq

    # --- Internal energy and pressure ---
    internal_energy = U_local[-1] - kinetic_energy
    pressure = (cfg.gamma - 1.0) * internal_energy

    # --- Compute local shard position info ---
    # This helps post-processing know where this shard belongs in the global array
    local_shape = U_local.shape

    # Compute shard position in global grid
    if cfg.SPATIAL_DIMENSION == 1:
        (x_start, x_end), _ = compute_local_slice(cfg.Nx, 1, mesh, process_id)
        y_start = 0
        z_start = 0
    elif cfg.SPATIAL_DIMENSION == 2:
        (x_start, x_end), (y_start, y_end) = compute_local_slice(cfg.Nx, cfg.Ny, mesh, process_id)
        z_start = 0
    elif cfg.SPATIAL_DIMENSION == 3:
        # For 3D, we would need a 3D version of compute_local_slice
        # For now, set to 0 as 3D is not yet implemented
        x_start = 0
        y_start = 0
        z_start = 0
    else:
        x_start = 0
        y_start = 0
        z_start = 0

    # --- Write to HDF5 (each process writes its own file) ---
    with h5py.File(snapshot_filename, "w") as f:
        # Create main group
        cells = f.create_group("cells")

        # Write metadata
        cells.attrs["time"] = float(t)
        cells.attrs["step"] = int(step)
        cells.attrs["process_id"] = process_id
        cells.attrs["process_count"] = process_count
        cells.attrs["gamma"] = float(cfg.gamma)
        cells.attrs["spatial_dimension"] = cfg.SPATIAL_DIMENSION
        cells.attrs["order"] = cfg.ORDER
        cells.attrs["limiter"] = cfg.LIMITER

        # Physical domain sizes
        cells.attrs["Lx"] = float(cfg.Lx)
        cells.attrs["Ly"] = float(cfg.Ly) if cfg.SPATIAL_DIMENSION >= 2 else 0.0
        cells.attrs["Lz"] = float(cfg.Lz) if cfg.SPATIAL_DIMENSION == 3 else 0.0

        # Grid spacing
        cells.attrs["dx"] = float(cfg.Lx / cfg.Nx)
        cells.attrs["dy"] = float(cfg.Ly / cfg.Ny) if cfg.SPATIAL_DIMENSION >= 2 else 0.0
        cells.attrs["dz"] = float(cfg.Lz / cfg.Nz) if cfg.SPATIAL_DIMENSION == 3 else 0.0

        # Global grid info (using both naming conventions for compatibility)
        cells.attrs["global_Nx"] = cfg.Nx
        cells.attrs["Nx_global"] = cfg.Nx
        if cfg.SPATIAL_DIMENSION >= 2:
            cells.attrs["global_Ny"] = cfg.Ny
            cells.attrs["Ny_global"] = cfg.Ny
        if cfg.SPATIAL_DIMENSION == 3:
            cells.attrs["global_Nz"] = cfg.Nz
            cells.attrs["Nz_global"] = cfg.Nz

        # Local shard info
        if cfg.SPATIAL_DIMENSION == 1:
            cells.attrs["local_Nx"] = local_shape[1]
        elif cfg.SPATIAL_DIMENSION == 2:
            cells.attrs["local_Nx"] = local_shape[1]
            cells.attrs["local_Ny"] = local_shape[2]
        elif cfg.SPATIAL_DIMENSION == 3:
            cells.attrs["local_Nx"] = local_shape[1]
            cells.attrs["local_Ny"] = local_shape[2]
            cells.attrs["local_Nz"] = local_shape[3]

        # Shard position in global grid
        cells.attrs["x_start"] = x_start
        if cfg.SPATIAL_DIMENSION >= 2:
            cells.attrs["y_start"] = y_start
        if cfg.SPATIAL_DIMENSION == 3:
            cells.attrs["z_start"] = z_start

        # Write fields without compression for maximum IO performance
        cells.create_dataset("density", data=rho)
        cells.create_dataset("velocity_x", data=vx)

        if cfg.SPATIAL_DIMENSION >= 2:
            cells.create_dataset("velocity_y", data=vy)
        if cfg.SPATIAL_DIMENSION == 3:
            cells.create_dataset("velocity_z", data=vz)

        cells.create_dataset("internal_energy", data=internal_energy)
        cells.create_dataset("pressure", data=pressure)

        # Conservative variables (full U array for this shard)
        cells.create_dataset("U", data=U_local)

    # Each process prints its own confirmation
    print(f"Process {process_id}: Snapshot saved to {os.path.basename(snapshot_filename)}")
