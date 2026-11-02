"""Distributed HDF5 snapshot writer: one file per shard under snapdir_NNN/, no gather."""
import os
import h5py
import numpy as np

from luna.file_io.checkpoint_utils import _shard_context, _write_units


def write_snapshot_distributed(U_sharded, cfg, t, step, mesh=None):
    """Write each local shard's primitives and U to snapdir_NNN/{base_name}_NNN.{shard_id}.hdf5."""
    process_id, process_count, mesh, local_shards = _shard_context(U_sharded, cfg, mesh)
    if len(local_shards) == 0:
        return

    snapshot_dir = os.path.join(cfg.output_dir, f"snapdir_{step:03d}")
    os.makedirs(snapshot_dir, exist_ok=True)

    for shard_idx, shard in enumerate(local_shards):
        shard_arr = np.array(shard.data)
        # idx[0] is the (unsharded) field axis; idx[1..] are spatial
        idx = shard.index
        x_start = idx[1].start or 0
        y_start = (idx[2].start or 0) if cfg.SPATIAL_DIMENSION >= 2 else 0
        z_start = (idx[3].start or 0) if cfg.SPATIAL_DIMENSION == 3 else 0

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

        # unique across processes and across shards of one process
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
            from luna.file_io.checkpoint_utils import write_provenance
            write_provenance(cells.attrs, cfg)

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

            _write_units(f, cfg, full=True)

        print(
            f"Process {process_id} shard {shard_idx}: saved {os.path.basename(snapshot_filename)}"
        )
