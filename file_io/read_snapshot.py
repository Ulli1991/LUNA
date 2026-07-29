"""LUNA snapshot reader.

Inverse of file_io/write_snapshot_distributed.py: takes a snapshot directory
(snapdir_NNN/, containing one HDF5 file per process shard) and returns a
dict of stitched, globally-shaped numpy arrays plus metadata.

Single-process snapshots (one .0.hdf5 file) just pass through.
"""

import glob
import os
import h5py
import numpy as np


def _list_shard_files(snapdir):
    files = sorted(glob.glob(os.path.join(snapdir, "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No .hdf5 shards in {snapdir}")
    return files


def _stitch_field(shards, field, ndim):
    """Place each shard's data into the global array using its x/y/z_start.

    Each shard's local array shape is (local_Nx, [local_Ny, [local_Nz]]).
    For 1D, place along x. For 2D, place at (x_start, y_start). 3D similar.
    """
    # Read shape info from first shard
    first = shards[0]
    if ndim == 1:
        Nx = first["cells"].attrs["Nx_global"]
        out = np.empty((Nx,), dtype=np.asarray(first[f"cells/{field}"]).dtype)
    elif ndim == 2:
        Nx = first["cells"].attrs["Nx_global"]
        Ny = first["cells"].attrs["Ny_global"]
        out = np.empty((Nx, Ny), dtype=np.asarray(first[f"cells/{field}"]).dtype)
    else:
        Nx = first["cells"].attrs["Nx_global"]
        Ny = first["cells"].attrs["Ny_global"]
        Nz = first["cells"].attrs["Nz_global"]
        out = np.empty((Nx, Ny, Nz), dtype=np.asarray(first[f"cells/{field}"]).dtype)

    for s in shards:
        cells = s["cells"]
        data = np.asarray(cells[field])
        x_start = int(cells.attrs["x_start"])
        if ndim == 1:
            out[x_start : x_start + data.shape[0]] = data
        elif ndim == 2:
            y_start = int(cells.attrs["y_start"])
            out[x_start : x_start + data.shape[0], y_start : y_start + data.shape[1]] = data
        else:
            y_start = int(cells.attrs["y_start"])
            z_start = int(cells.attrs["z_start"])
            out[
                x_start : x_start + data.shape[0],
                y_start : y_start + data.shape[1],
                z_start : z_start + data.shape[2],
            ] = data
    return out


def read_snapshot(snapdir):
    """Read a snapdir_NNN/ directory into a dict of stitched global fields:
    t, step, gamma, ndim, Lx/dx/Nx (+y/z for ndim>=2/3), cell centers
    x[/y/z], and rho, vx[/vy/vz], P, e_int."""
    files = _list_shard_files(snapdir)
    shards = [h5py.File(f, "r") for f in files]
    try:
        attrs = shards[0]["cells"].attrs
        ndim = int(attrs["spatial_dimension"])
        out = {
            "t": float(attrs["time"]),
            "step": int(attrs["step"]),
            "gamma": float(attrs["gamma"]),
            "ndim": ndim,
            "Lx": float(attrs["Lx"]),
            "dx": float(attrs["dx"]),
            "Nx": int(attrs["Nx_global"]),
        }
        if ndim >= 2:
            out["Ly"] = float(attrs["Ly"])
            out["dy"] = float(attrs["dy"])
            out["Ny"] = int(attrs["Ny_global"])
        if ndim == 3:
            out["Lz"] = float(attrs["Lz"])
            out["dz"] = float(attrs["dz"])
            out["Nz"] = int(attrs["Nz_global"])

        # Cell-center coordinates (uniform grid)
        out["x"] = (np.arange(out["Nx"]) + 0.5) * out["dx"]
        if ndim >= 2:
            out["y"] = (np.arange(out["Ny"]) + 0.5) * out["dy"]
        if ndim == 3:
            out["z"] = (np.arange(out["Nz"]) + 0.5) * out["dz"]

        out["rho"] = _stitch_field(shards, "density", ndim)
        out["vx"] = _stitch_field(shards, "velocity_x", ndim)
        if ndim >= 2:
            out["vy"] = _stitch_field(shards, "velocity_y", ndim)
        if ndim == 3:
            out["vz"] = _stitch_field(shards, "velocity_z", ndim)
        out["P"] = _stitch_field(shards, "pressure", ndim)
        out["e_int"] = _stitch_field(shards, "internal_energy", ndim)
    finally:
        for s in shards:
            s.close()
    return out


def latest_snapshot_dir(output_dir):
    """Return the snapdir_NNN/ with the highest NNN in output_dir."""
    candidates = sorted(glob.glob(os.path.join(output_dir, "snapdir_*")))
    if not candidates:
        raise FileNotFoundError(f"No snapdir_* found under {output_dir}")
    return candidates[-1]


def find_snapshot_at_time(output_dir, t_target, tol=1e-6):
    """Return the snapdir whose snapshot time is closest to t_target
    (closest match is returned even if the difference exceeds tol)."""
    best = None
    best_diff = float("inf")
    for d in sorted(glob.glob(os.path.join(output_dir, "snapdir_*"))):
        files = _list_shard_files(d)
        with h5py.File(files[0], "r") as f:
            t = float(f["cells"].attrs["time"])
        diff = abs(t - t_target)
        if diff < best_diff:
            best_diff = diff
            best = d
    if best is None:
        raise FileNotFoundError(f"No snapshots under {output_dir}")
    if best_diff > tol:
        # Return the closest match but warn the caller might want to interp.
        # Caller can re-check the actual time from the returned dict.
        pass
    return best
