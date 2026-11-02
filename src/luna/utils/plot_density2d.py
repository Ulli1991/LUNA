"""Render 2D density maps from single-file or sharded HDF5 snapshots."""
import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_sharded_snapshot(output_dir, base_pattern, snapshot_idx):
    """Return a dict of the stitched global density, positions, time, Lx and Ly of one snapshot."""
    snapshot_subdir = os.path.join(output_dir, f"snapdir_{snapshot_idx:03d}")
    if os.path.isdir(snapshot_subdir):
        search_dir = snapshot_subdir
    else:
        search_dir = output_dir

    shard_files = sorted(
        [
            f
            for f in os.listdir(search_dir)
            if re.match(f"{base_pattern}_{snapshot_idx:03d}\\.\\d+\\.hdf5", f)
        ]
    )

    if not shard_files:
        single_file = f"{base_pattern}_{snapshot_idx:03d}.hdf5"
        if os.path.exists(os.path.join(output_dir, single_file)):
            with h5py.File(os.path.join(output_dir, single_file), "r") as f:
                cells = f["cells"]
                return {
                    "density": np.array(cells["density"]),
                    "x_positions": np.array(cells["x_positions"]),
                    "y_positions": np.array(cells["y_positions"]),
                    "time": cells.attrs["time"],
                    "Lx": cells.attrs["Lx"],
                    "Ly": cells.attrs["Ly"],
                }
        else:
            raise FileNotFoundError(f"No files found for snapshot {snapshot_idx}")

    first_shard = os.path.join(search_dir, shard_files[0])
    with h5py.File(first_shard, "r") as f:
        cells = f["cells"]
        time = cells.attrs["time"]
        Lx = cells.attrs["Lx"]
        Ly = cells.attrs["Ly"]
        Nx_global = cells.attrs.get("Nx_global", None)
        Ny_global = cells.attrs.get("Ny_global", None)

    shards = []
    for shard_file in shard_files:
        fpath = os.path.join(search_dir, shard_file)
        with h5py.File(fpath, "r") as f:
            cells = f["cells"]
            shard_data = {
                "density": np.array(cells["density"]),
                "x_start": cells.attrs.get("x_start", 0),
                "y_start": cells.attrs.get("y_start", 0),
            }
            shards.append(shard_data)

    if Nx_global is None or Ny_global is None:
        max_x_end = max(s["x_start"] + s["density"].shape[0] for s in shards)
        max_y_end = max(s["y_start"] + s["density"].shape[1] for s in shards)
        Nx_global = max_x_end
        Ny_global = max_y_end

    rho_global = np.zeros((Nx_global, Ny_global))
    x_global = np.linspace(0, Lx, Nx_global, endpoint=False)
    y_global = np.linspace(0, Ly, Ny_global, endpoint=False)

    for shard in shards:
        x_start = shard["x_start"]
        y_start = shard["y_start"]
        rho_shard = shard["density"]
        x_end = x_start + rho_shard.shape[0]
        y_end = y_start + rho_shard.shape[1]
        rho_global[x_start:x_end, y_start:y_end] = rho_shard

    return {
        "density": rho_global,
        "x_positions": x_global,
        "y_positions": y_global,
        "time": time,
        "Lx": Lx,
        "Ly": Ly,
    }


def plot_density2d(output_dir, base_name, process_id=0, num_processes=1):
    """Plot a density map per snapshot, colour scale from the last; snapshots
    round-robin over processes. Returns how many this process saved."""
    snapshot_indices = set()

    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith("snapdir_"):
            match = re.match(r"snapdir_(\d+)", item)
            if match:
                snapshot_indices.add(int(match.group(1)))

    all_files = [
        f for f in os.listdir(output_dir) if f.startswith(base_name) and f.endswith(".hdf5")
    ]
    for fname in all_files:
        match = re.match(f"{base_name}_(\\d+)(?:\\.\\d+)?\\.hdf5", fname)
        if match:
            snapshot_indices.add(int(match.group(1)))

    if not snapshot_indices:
        if process_id == 0:
            print(f"No snapshots (snapdir_*) under {output_dir}: nothing plotted")
        return 0

    snapshot_indices = sorted(snapshot_indices)

    my_snapshot_indices = [
        idx for i, idx in enumerate(snapshot_indices) if i % num_processes == process_id
    ]

    if not my_snapshot_indices:
        if process_id == 0:
            print(f"Process {process_id}: No snapshots to plot")
        return 0

    min_density = float("inf")
    max_density = float("-inf")

    png_dir = os.path.join(output_dir, "png")
    os.makedirs(png_dir, exist_ok=True)

    # every process takes the colour scale from the same (last) snapshot
    last_idx = snapshot_indices[-1]
    last_data = load_sharded_snapshot(output_dir, base_name, last_idx)
    min_density = np.min(last_data["density"])
    max_density = np.max(last_data["density"])

    for snap_idx in my_snapshot_indices:
        data = load_sharded_snapshot(output_dir, base_name, snap_idx)

        rho = data["density"]
        t = data["time"]
        Lx = data["Lx"]
        Ly = data["Ly"]

        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        im = ax.imshow(
            rho.T,
            origin="lower",
            extent=[0, Lx, 0, Ly],
            cmap="magma",
            vmin=min_density,
            vmax=max_density,
            interpolation="nearest",
        )

        ax.text(
            0.02,
            0.95,
            f"t={t:.3f}",
            color="white",
            transform=ax.transAxes,
            fontsize=12,
            ha="left",
            va="top",
        )

        ax.tick_params(direction="in", top=True, right=True)

        _cbar = plt.colorbar(im, ax=ax, label="Density")

        png_name = f"{base_name}_{snap_idx:03d}.png"
        plt.savefig(os.path.join(png_dir, png_name), dpi=300)
        plt.close(fig)

        if num_processes > 1:
            print(f"Process {process_id}: Saved density plot for snapshot {snap_idx:03d}")
        else:
            print(f"Saved density plot for snapshot {snap_idx:03d}")
    return len(my_snapshot_indices)
