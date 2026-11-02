"""Read a snapdir_NNN/ of per-shard HDF5 files into stitched global numpy arrays."""

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
    """Place each shard's data into the global array at its x/y/z_start."""
    first = shards[0]
    if ndim == 1:
        Nx = first["cells"].attrs["Nx_global"]
        out = np.full((Nx,), np.nan, dtype=np.asarray(first[f"cells/{field}"]).dtype)
    elif ndim == 2:
        Nx = first["cells"].attrs["Nx_global"]
        Ny = first["cells"].attrs["Ny_global"]
        out = np.full((Nx, Ny), np.nan, dtype=np.asarray(first[f"cells/{field}"]).dtype)
    else:
        Nx = first["cells"].attrs["Nx_global"]
        Ny = first["cells"].attrs["Ny_global"]
        Nz = first["cells"].attrs["Nz_global"]
        out = np.full((Nx, Ny, Nz), np.nan, dtype=np.asarray(first[f"cells/{field}"]).dtype)

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
    if np.issubdtype(out.dtype, np.floating) and not np.isfinite(out).all():
        raise ValueError(f"snapshot field '{field}' has cells no shard covers (missing shard file?)")
    return out


def read_snapshot(snapdir):
    """Read a snapdir_NNN/ into a dict of metadata, cell centres and stitched fields.

    Keys: t, step, gamma, ndim, Lx/dx/Nx (+y, z), x[/y/z], rho, vx[/vy/vz], P, e_int.
    """
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
    """Return the snapdir closest in time to t_target (warns if further than tol)."""
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
        import warnings
        warnings.warn(f"closest snapshot to t={t_target} is {best} "
                      f"(|dt|={best_diff:.3g} > tol={tol})", stacklevel=2)
    return best
