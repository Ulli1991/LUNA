"""Plot 1D profiles (density, v_x, pressure, internal energy) from snapdir_NNN/ snapshots."""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from luna.file_io.read_snapshot import read_snapshot

# (field key, y-axis label) per panel
_FIELDS = [
    ("density", "density"),
    ("velocity_x", "v$_{x}$"),
    ("pressure", "pressure"),
    ("internal_energy", "internal energy"),
]


def _read_profile(snapdir):
    """Return (x, fields, t, Lx) of one snapdir_NNN/; 2D/3D data is sliced through the y/z midplane."""
    snap = read_snapshot(snapdir)
    mid = tuple(n // 2 for n in snap["rho"].shape[1:])
    fields = {name: snap[key][(slice(None),) + mid] for name, key in
              (("density", "rho"), ("velocity_x", "vx"), ("pressure", "P"),
               ("internal_energy", "e_int"))}
    return snap["x"], fields, snap["t"], snap["Lx"]


def plot_1d_fluid_quantities(output_dir, base_name):
    """Save a 2x2 profile figure per snapdir_NNN/ to <output_dir>/png/; returns how many."""
    snapdirs = sorted(glob.glob(os.path.join(output_dir, "snapdir_*")),
                      key=lambda d: int(d.rsplit("_", 1)[1]))
    if not snapdirs:
        print(f"No snapshots (snapdir_*) under {output_dir}: nothing plotted")
        return 0

    png_dir = os.path.join(output_dir, "png")
    os.makedirs(png_dir, exist_ok=True)

    # y-limits from the last snapshot
    _, last_fields, _, _ = _read_profile(snapdirs[-1])
    lims = {name: (np.min(vals), np.max(vals)) for name, vals in last_fields.items()}

    for snapdir in snapdirs:
        x, fields, t, Lx = _read_profile(snapdir)

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.ravel()

        for ax, (name, ylabel) in zip(axs, _FIELDS):
            ax.plot(x, fields[name], lw=1.4, color="k")
            ax.set_ylabel(ylabel)
            ax.set_xlim(0, Lx)
            # if min == max, pad off the min
            lo, hi = lims[name]
            ax.set_ylim(lo, hi * 1.1 if hi != lo else lo * 1.1)

        axs[2].set_xlabel("x")
        axs[3].set_xlabel("x")

        for ax in axs:
            ax.set_title("")
            ax.tick_params(direction="in", which="both", labelsize=10)
            ax.grid(False)

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

        png_name = f"{base_name}_{os.path.basename(snapdir).split('_')[-1]}.png"
        png_path = os.path.join(png_dir, png_name)
        plt.savefig(png_path, dpi=300)
        plt.close()

        print(f"Saved plot: {png_path}")
    return len(snapdirs)
