"""Distributed HDF5 snapshot writer: one file per shard under
snapdir_NNN/, no gather; stitched back by file_io/read_snapshot.py."""
import os
import jax
import h5py
import jax.numpy as jnp
import numpy as np
from utils.distributed import compute_local_slice, create_device_mesh


def write_snapshot_distributed(U_sharded, cfg, t, step, mesh=None):
    """Write one HDF5 file per addressable shard to
    snapdir_{step:03d}/{base_name}_{step:03d}.{global_shard_id}.hdf5,
    storing primitives + conservatives with x/y/z_start placement attrs."""
    # Get process info
    process_id = jax.process_index()
    process_count = jax.process_count()

    # Reconstruct mesh if not provided. Use total device count, not process
    # count: in single-process multi-GPU mode these differ (e.g. 1 process,
    # 4 GPUs would otherwise build a (1, 1) mesh that doesn't cover devices).
    if mesh is None:
        mesh, _ = create_device_mesh(jax.device_count(), cfg.SPATIAL_DIMENSION)

    # Get local addressable shards. In single-process multi-GPU mode there
    # are multiple shards per process; in true multi-process mode there is
    # typically one. We loop over them and write ONE FILE PER SHARD using
    # shard.index to derive each shard's position in the global grid. This
    # avoids guessing a concatenation axis and matches the reader's stitcher.
    local_shards = U_sharded.addressable_shards
    if len(local_shards) == 0:
        print(f"Warning: Process {process_id} has no addressable shards")
        return

    snapshot_dir = os.path.join(cfg.output_dir, f"snapdir_{step:03d}")
    os.makedirs(snapshot_dir, exist_ok=True)

    for shard_idx, shard in enumerate(local_shards):
        shard_arr = np.array(shard.data)
        # shard.index is a tuple of slices over U's full shape (n_fields, Nx, [Ny, [Nz]]).
        # idx[0] covers the field axis (not sharded); idx[1..] are spatial.
        idx = shard.index
        x_start = idx[1].start or 0
        y_start = (idx[2].start or 0) if cfg.SPATIAL_DIMENSION >= 2 else 0
        z_start = (idx[3].start or 0) if cfg.SPATIAL_DIMENSION == 3 else 0

        # Derive primitives from this shard's conservatives
        rho = shard_arr[0]
        vx = shard_arr[1] / rho
        vy = shard_arr[2] / rho if cfg.SPATIAL_DIMENSION >= 2 else 0.0
        vz = shard_arr[3] / rho if cfg.SPATIAL_DIMENSION == 3 else 0.0
        vel_sq = vx**2
        if cfg.SPATIAL_DIMENSION >= 2:
            vel_sq = vel_sq + vy**2
        if cfg.SPATIAL_DIMENSION == 3:
            vel_sq = vel_sq + vz**2
        kinetic_energy = 0.5 * rho * vel_sq
        internal_energy = shard_arr[-1] - kinetic_energy
        pressure = (cfg.gamma - 1.0) * internal_energy

        local_shape = shard_arr.shape

        # Globally-unique shard ID: works for single-process-multi-GPU AND
        # true multi-process (where each process has one shard).
        global_shard_id = process_id * len(local_shards) + shard_idx
        snapshot_filename = os.path.join(
            snapshot_dir, f"{cfg.base_name}_{step:03d}.{global_shard_id}.hdf5"
        )

        with h5py.File(snapshot_filename, "w") as f:
            cells = f.create_group("cells")

            cells.attrs["time"] = float(t)
            cells.attrs["step"] = int(step)
            cells.attrs["process_id"] = process_id
            cells.attrs["process_count"] = process_count
            cells.attrs["shard_idx"] = int(shard_idx)
            cells.attrs["gamma"] = float(cfg.gamma)
            cells.attrs["spatial_dimension"] = cfg.SPATIAL_DIMENSION
            cells.attrs["order"] = cfg.ORDER
            cells.attrs["limiter"] = cfg.LIMITER

            cells.attrs["Lx"] = float(cfg.Lx)
            cells.attrs["Ly"] = float(cfg.Ly) if cfg.SPATIAL_DIMENSION >= 2 else 0.0
            cells.attrs["Lz"] = float(cfg.Lz) if cfg.SPATIAL_DIMENSION == 3 else 0.0

            cells.attrs["dx"] = float(cfg.Lx / cfg.Nx)
            cells.attrs["dy"] = float(cfg.Ly / cfg.Ny) if cfg.SPATIAL_DIMENSION >= 2 else 0.0
            cells.attrs["dz"] = float(cfg.Lz / cfg.Nz) if cfg.SPATIAL_DIMENSION == 3 else 0.0

            cells.attrs["global_Nx"] = cfg.Nx
            cells.attrs["Nx_global"] = cfg.Nx
            if cfg.SPATIAL_DIMENSION >= 2:
                cells.attrs["global_Ny"] = cfg.Ny
                cells.attrs["Ny_global"] = cfg.Ny
            if cfg.SPATIAL_DIMENSION == 3:
                cells.attrs["global_Nz"] = cfg.Nz
                cells.attrs["Nz_global"] = cfg.Nz

            if cfg.SPATIAL_DIMENSION == 1:
                cells.attrs["local_Nx"] = local_shape[1]
            elif cfg.SPATIAL_DIMENSION == 2:
                cells.attrs["local_Nx"] = local_shape[1]
                cells.attrs["local_Ny"] = local_shape[2]
            elif cfg.SPATIAL_DIMENSION == 3:
                cells.attrs["local_Nx"] = local_shape[1]
                cells.attrs["local_Ny"] = local_shape[2]
                cells.attrs["local_Nz"] = local_shape[3]

            cells.attrs["x_start"] = x_start
            if cfg.SPATIAL_DIMENSION >= 2:
                cells.attrs["y_start"] = y_start
            if cfg.SPATIAL_DIMENSION == 3:
                cells.attrs["z_start"] = z_start

            cells.create_dataset("density", data=rho)
            cells.create_dataset("velocity_x", data=vx)
            if cfg.SPATIAL_DIMENSION >= 2:
                cells.create_dataset("velocity_y", data=vy)
            if cfg.SPATIAL_DIMENSION == 3:
                cells.create_dataset("velocity_z", data=vz)
            cells.create_dataset("internal_energy", data=internal_energy)
            cells.create_dataset("pressure", data=pressure)
            cells.create_dataset("U", data=shard_arr)

            if cfg.units.is_enabled:
                units_grp = f.create_group("units")
                units_grp.attrs["UnitLength_in_cm"] = cfg.units.UnitLength_in_cm
                units_grp.attrs["UnitMass_in_g"] = cfg.units.UnitMass_in_g
                units_grp.attrs["UnitVelocity_in_cm_per_s"] = cfg.units.UnitVelocity_in_cm_per_s
                units_grp.attrs["UnitTime_in_s"] = cfg.units.UnitTime_in_s
                units_grp.attrs["UnitDensity_in_cgs"] = cfg.units.UnitDensity_in_cgs
                units_grp.attrs["UnitPressure_in_cgs"] = cfg.units.UnitPressure_in_cgs
                units_grp.attrs["UnitEnergy_in_cgs"] = cfg.units.UnitEnergy_in_cgs

        print(
            f"Process {process_id} shard {shard_idx}: saved {os.path.basename(snapshot_filename)}"
        )
