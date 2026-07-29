"""Plot 1D profiles (density, v_x, pressure, internal energy) from HDF5
snapshots; 2D/3D data is sliced through the domain center."""
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Panel order: (field key, y-axis label)
_FIELDS = [
    ("density", "density"),
    ("velocity_x", "v$_{x}$"),
    ("pressure", "pressure"),
    ("internal_energy", "internal energy"),
]


def _read_profile(fpath):
    """Read one snapshot; return (x, fields dict, t, Lx) as 1D center-slice
    profiles (3D/2D data is sliced through the y/z midplane)."""
    with h5py.File(fpath, "r") as f:
        cells = f["cells"]
        x = np.array(cells["x_positions"])
        density = np.array(cells["density"])
        velocity_x = np.array(cells["velocity_x"])
        internal_energy = np.array(cells["internal_energy"])

        # If data is 3D or 2D, take a slice through the center
        if density.ndim == 3:
            j, k = density.shape[1] // 2, density.shape[2] // 2
            density = density[:, j, k]
            velocity_x = velocity_x[:, j, k]
            internal_energy = internal_energy[:, j, k]
        elif density.ndim == 2:
            j = density.shape[1] // 2
            density = density[:, j]
            velocity_x = velocity_x[:, j]
            internal_energy = internal_energy[:, j]
        # else: already 1D, nothing to do

        gamma = cells.attrs["gamma"]
        pressure = (gamma - 1.0) * internal_energy

        t = cells.attrs["time"]
        Lx = cells.attrs["Lx"]
    fields = dict(density=density, velocity_x=velocity_x,
                  pressure=pressure, internal_energy=internal_energy)
    return x, fields, t, Lx


def plot_1d_fluid_quantities(output_dir, base_name):

    # --- find all snapshot files ---
    files = sorted(
        [f for f in os.listdir(output_dir) if f.startswith(base_name) and f.endswith(".hdf5")]
    )

    if not files:
        print("No HDF5 snapshots found!")
        return

    # --- create png directory ---
    png_dir = os.path.join(output_dir, "png")
    os.makedirs(png_dir, exist_ok=True)

    # Dynamic y-axis limits, taken from the last snapshot
    _, last_fields, _, _ = _read_profile(os.path.join(output_dir, files[-1]))
    lims = {name: (np.min(vals), np.max(vals)) for name, vals in last_fields.items()}

    # Now iterate through all snapshot files for plotting
    for fname in files:
        x, fields, t, Lx = _read_profile(os.path.join(output_dir, fname))

        # --- plotting ---
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.ravel()

        for ax, (name, ylabel) in zip(axs, _FIELDS):
            ax.plot(x, fields[name], lw=1.4, color="k")
            ax.set_ylabel(ylabel)
            ax.set_xlim(0, Lx)
            # y-limits from the last snapshot; if min == max, pad off the min
            lo, hi = lims[name]
            ax.set_ylim(lo, hi * 1.1 if hi != lo else lo * 1.1)

        # X-axis labels only on bottom row
        axs[2].set_xlabel("x")
        axs[3].set_xlabel("x")

        # No titles; tick labels inside
        for ax in axs:
            ax.set_title("")
            ax.tick_params(direction="in", which="both", labelsize=10)
            ax.grid(False)

        # Time annotation in top-left panel
        axs[0].text(
            0.80,
            0.92,
            f"t = {t:.4f}",
            transform=axs[0].transAxes,
            fontsize=12,
            color="black",
            ha="left",
            va="top",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )

        plt.tight_layout()

        # Save file inside /png
        png_name = fname.replace(".hdf5", ".png")
        png_path = os.path.join(png_dir, png_name)
        plt.savefig(png_path, dpi=300)
        plt.close()

        print(f"Saved plot: {png_path}")
