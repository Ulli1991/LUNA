"""Render 2D density maps from HDF5 snapshots (single-file or sharded
multi-process outputs)."""
import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_sharded_snapshot(output_dir, base_pattern, snapshot_idx):
    """Load a (possibly sharded) snapshot and return a dict with the
    combined global density, positions, time, and domain sizes."""
    # Check if snapshots are in subdirectories (distributed case)
    snapshot_subdir = os.path.join(output_dir, f"snapdir_{snapshot_idx:03d}")
    if os.path.isdir(snapshot_subdir):
        search_dir = snapshot_subdir
    else:
        search_dir = output_dir

    # Find all shards for this snapshot
    pattern = f"{base_pattern}_{snapshot_idx:03d}.*.hdf5"
    shard_files = sorted(
        [
            f
            for f in os.listdir(search_dir)
            if re.match(f"{base_pattern}_{snapshot_idx:03d}\\.\\d+\\.hdf5", f)
        ]
    )

    if not shard_files:
        # Try loading non-sharded file (in output_dir, not in subdirectory)
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

    # Load first shard to get metadata
    first_shard = os.path.join(search_dir, shard_files[0])
    with h5py.File(first_shard, "r") as f:
        cells = f["cells"]
        time = cells.attrs["time"]
        Lx = cells.attrs["Lx"]
        Ly = cells.attrs["Ly"]
        Nx_global = cells.attrs.get("Nx_global", None)
        Ny_global = cells.attrs.get("Ny_global", None)

    # Load all shards and determine global shape
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
            # Position arrays are optional (not needed for distributed snapshots)
            # We compute them from the global domain attributes instead
            shards.append(shard_data)

    # Determine global grid size from shards
    if Nx_global is None or Ny_global is None:
        # Infer from shard positions and sizes
        max_x_end = max(s["x_start"] + s["density"].shape[0] for s in shards)
        max_y_end = max(s["y_start"] + s["density"].shape[1] for s in shards)
        Nx_global = max_x_end
        Ny_global = max_y_end

    # Reconstruct global arrays
    rho_global = np.zeros((Nx_global, Ny_global))
    x_global = np.linspace(0, Lx, Nx_global, endpoint=False)
    y_global = np.linspace(0, Ly, Ny_global, endpoint=False)

    # Place each shard in the global array
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
    """Plot 2D density maps for all snapshots (color scale fixed from the
    last snapshot); snapshots are round-robined over num_processes."""
    # Find all unique snapshot indices
    snapshot_indices = set()

    # Check for distributed snapshots in subdirectories
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith("snapdir_"):
            # Extract snapshot index from directory name
            match = re.match(r"snapdir_(\d+)", item)
            if match:
                snapshot_indices.add(int(match.group(1)))

    # Also check for non-distributed snapshots in main directory
    all_files = [
        f for f in os.listdir(output_dir) if f.startswith(base_name) and f.endswith(".hdf5")
    ]
    for fname in all_files:
        # Match snap_XXX.hdf5 or snap_XXX.Y.hdf5
        match = re.match(f"{base_name}_(\\d+)(?:\\.\\d+)?\\.hdf5", fname)
        if match:
            snapshot_indices.add(int(match.group(1)))

    if not snapshot_indices:
        print("No HDF5 snapshots found!")
        return

    snapshot_indices = sorted(snapshot_indices)

    # Filter snapshots for this process (for parallel plotting)
    my_snapshot_indices = [
        idx for i, idx in enumerate(snapshot_indices) if i % num_processes == process_id
    ]

    if not my_snapshot_indices:
        if process_id == 0:
            print(f"Process {process_id}: No snapshots to plot")
        return

    # Initialize min/max density values
    min_density = float("inf")
    max_density = float("-inf")

    png_dir = os.path.join(output_dir, "png")
    # Only create directory from process 0 to avoid race condition
    if process_id == 0:
        os.makedirs(png_dir, exist_ok=True)

    # First, check the last snapshot for the dynamic colorbar limits
    # All processes use the same last snapshot for consistent coloring
    last_idx = snapshot_indices[-1]
    last_data = load_sharded_snapshot(output_dir, base_name, last_idx)
    min_density = np.min(last_data["density"])
    max_density = np.max(last_data["density"])

    # Now, iterate through snapshots assigned to this process
    for snap_idx in my_snapshot_indices:
        # Load snapshot (handles both sharded and non-sharded)
        data = load_sharded_snapshot(output_dir, base_name, snap_idx)

        rho = data["density"]
        x = data["x_positions"]
        y = data["y_positions"]
        t = data["time"]
        Lx = data["Lx"]
        Ly = data["Ly"]

        # ---- Plot figure ----
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        im = ax.imshow(
            rho.T,  # Transpose to match the x and y axes
            origin="lower",
            extent=[0, Lx, 0, Ly],
            cmap="magma",
            vmin=min_density,  # Dynamically set min/max density from the last snapshot
            vmax=max_density,
            interpolation="nearest",
        )

        # Add time text inside frame
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

        # Ticks inside, all four sides
        ax.tick_params(direction="in", top=True, right=True)

        # Add colorbar with label "Density"
        cbar = plt.colorbar(im, ax=ax, label="Density")

        # Save PNG to png directory
        png_name = f"{base_name}_{snap_idx:03d}.png"
        plt.savefig(os.path.join(png_dir, png_name), dpi=300)
        plt.close(fig)

        if num_processes > 1:
            print(f"Process {process_id}: Saved density plot for snapshot {snap_idx:03d}")
        else:
            print(f"Saved density plot for snapshot {snap_idx:03d}")
