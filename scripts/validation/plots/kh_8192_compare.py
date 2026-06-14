"""KH 8192^2 LUNA vs AthenaK side-by-side at the canonical 5 timestamps.

Top row: LUNA 8192^2 density slices at t = 0, 1, 2.5, 4, 5.
Bottom row: AthenaK 8192^2 (WENOZ+HLLC) at the same times.

LUNA snapshots live under snapshots/kh_Nx8192/ (HDF5 shards).
AthenaK output is in reference/athenak/kh_8192/bin/ (.bin per timestamp).

Saved to figures/kh_8192_luna_vs_athenak.{pdf,png}.
"""
from pathlib import Path
import glob, os
import numpy as np
import matplotlib.pyplot as plt
import h5py

from utils.figstyle import apply_style, decorate, FIGSIZE

apply_style()

from file_io.read_snapshot import read_snapshot, find_snapshot_at_time
from scripts.validation.athenak_io import load_2d as ak_load_2d


LABEL_FS = 11   # in-panel image annotations only (axis labels inherit rcParams)

SNAPSHOT_TIMES = [0.0, 1.0, 2.5, 4.0, 5.0]
LX = LY = 2.0
VMIN, VMAX = 0.9, 2.1
CMAP = "magma"
LUNA_DIR = "snapshots/kh_Nx8192"
AK_DIR = "reference/athenak/kh_8192/bin"


def _find_ak_bin_at(t_target, tol=0.05):
    bins = sorted(glob.glob(os.path.join(AK_DIR, "kh_luna.hydro_w.*.bin")))
    if not bins:
        raise FileNotFoundError(f"No AthenaK bins in {AK_DIR}")
    # Each bin's time is in its header; just walk and pick closest.
    best = None; best_diff = float("inf"); best_t = None
    for b in bins:
        snap = ak_load_2d(b)
        t = snap["t"]
        diff = abs(t - t_target)
        if diff < best_diff:
            best_diff = diff; best = snap; best_t = t
        if diff < tol:
            break
    return best, best_t


def main():
    fig, axes = plt.subplots(2, len(SNAPSHOT_TIMES),
                             figsize=(2.2 * len(SNAPSHOT_TIMES), 2.2 * 2),
                             squeeze=False)

    # LUNA top row
    for c, t in enumerate(SNAPSHOT_TIMES):
        ax = axes[0, c]
        snapdir = find_snapshot_at_time(LUNA_DIR, t, tol=0.05)
        snap = read_snapshot(snapdir)
        ax.imshow(snap["rho"].T, origin="lower", extent=(0, LX, 0, LY),
                  vmin=VMIN, vmax=VMAX, cmap=CMAP,
                  interpolation="nearest", aspect="equal")
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        axes[0, c].text(0.05, 0.92, f"t = {t}", transform=axes[0, c].transAxes,
                        fontsize=LABEL_FS, color="white", fontweight="bold",
                        bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=1.5))

    # AthenaK bottom row
    for c, t in enumerate(SNAPSHOT_TIMES):
        ax = axes[1, c]
        snap, t_actual = _find_ak_bin_at(t)
        ax.imshow(snap["rho"].T, origin="lower",
                  extent=(0, snap["Lx"], 0, snap["Ly"]),
                  vmin=VMIN, vmax=VMAX, cmap=CMAP,
                  interpolation="nearest", aspect="equal")
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])

    axes[0, 0].set_ylabel("LUNA")
    axes[1, 0].set_ylabel("AthenaK")

    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/kh_8192_luna_vs_athenak.pdf")
    fig.savefig("figures/kh_8192_luna_vs_athenak.png", dpi=200)
    print("Saved figures/kh_8192_luna_vs_athenak.{pdf,png}")


if __name__ == "__main__":
    main()
