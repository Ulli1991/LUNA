"""Assemble the KH 4×5 density-snapshot grid from per-resolution LUNA snapshots.

Expects snapshots produced by kh_run.py at snapshots/kh_Nx{1024,2048,4096,8192}/.
Each Nx run writes snapshots every 0.5 simulation time; we pick out the
target timestamps {0, 1, 2.5, 4, 5} via find_snapshot_at_time.

Output: figures/kh_grid.{pdf,png}. Same colormap and density range across
all panels so refinement is visually obvious.

Run this on the login node (matplotlib only, no LUNA simulation).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils.figstyle import apply_style, decorate, FIGSIZE

apply_style()

from file_io.read_snapshot import read_snapshot, find_snapshot_at_time


LABEL_FS = 11   # in-panel image annotations only (axis labels inherit rcParams)

NX_SCAN = [1024, 2048, 4096, 8192]
SNAPSHOT_TIMES = [0.0, 1.0, 2.5, 4.0, 5.0]
LX = 2.0
LY = 2.0
VMIN, VMAX = 0.9, 2.1   # shared color range across all panels


def main():
    # Only render rows for resolutions that have snapshots on disk.
    available = [Nx for Nx in NX_SCAN
                 if (Path(f"snapshots/kh_Nx{Nx}").is_dir()
                     and any(Path(f"snapshots/kh_Nx{Nx}").glob("snapdir_*")))]
    if not available:
        print("No KH snapshot dirs found under snapshots/. Nothing to plot.")
        return

    fig, axes = plt.subplots(
        len(available), len(SNAPSHOT_TIMES),
        figsize=(2.2 * len(SNAPSHOT_TIMES), 2.2 * len(available)),
        squeeze=False,
    )

    for r, Nx in enumerate(available):
        outdir = f"snapshots/kh_Nx{Nx}"
        for c, t_target in enumerate(SNAPSHOT_TIMES):
            ax = axes[r, c]
            ax.grid(False)
            try:
                snapdir = find_snapshot_at_time(outdir, t_target, tol=0.05)
                snap = read_snapshot(snapdir)
                rho = snap["rho"]   # shape (Nx, Ny)
                # Plot density with imshow (origin lower-left, extent in physical units)
                ax.imshow(rho.T, origin="lower", extent=(0, LX, 0, LY),
                          vmin=VMIN, vmax=VMAX, cmap="magma",
                          interpolation="nearest", aspect="equal")
            except FileNotFoundError as e:
                ax.text(0.5, 0.5, f"no t≈{t_target}",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=LABEL_FS, color="0.5")
            ax.set_xticks([]); ax.set_yticks([])

    # Row labels (resolutions)
    for r, Nx in enumerate(available):
        axes[r, 0].set_ylabel(f"{Nx}²")
    # Column labels (times)
    for c, t in enumerate(SNAPSHOT_TIMES):
        axes[0, c].text(0.05, 0.92, f"t = {t}", transform=axes[0, c].transAxes,
                        fontsize=LABEL_FS, color="white", fontweight="bold",
                        bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=1.5))

    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/kh_grid.pdf")
    fig.savefig("figures/kh_grid.png", dpi=200)
    print("Saved figures/kh_grid.{pdf,png}")


if __name__ == "__main__":
    main()
