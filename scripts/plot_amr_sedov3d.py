"""3D Sedov-Taylor blast with block-AMR: central density slice (z=Lz/2) with the
refinement mesh overlaid (block edges colored by level), across snapshots. Shows
the spherical shock expanding and the fine blocks tracking the shell in 3D.

Reads snapshots/amr_sedov3d/{snapdir_NNN/*.hdf5, blocks_NNN.npz}.
"""
import glob
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

OUT = "snapshots/amr_sedov3d"
LX = 1.0
SNAPS = [0, 1, 2]
LEVEL_COLOR = {1: "#00e5ff", 2: "#ffe000", 3: "#7CFC00"}


def load_slice(idx):
    f = sorted(glob.glob(f"{OUT}/snapdir_{idx:03d}/*.hdf5"))[0]
    with h5py.File(f) as h:
        rho = h["cells"]["density"][:]                     # (N, N, N)
    N = rho.shape[2]
    return rho[:, :, N // 2], N                            # central z-slice


def load_blocks(idx):
    z = np.load(f"{OUT}/blocks_{idx:03d}.npz")
    base_blocks = int(z["L0_coords"][:, 0].max()) + 1
    coords = {L: z[f"L{L}_coords"] for L in range(int(z["levels"]))}
    return coords, base_blocks, float(z["t"])


fig, axes = plt.subplots(1, len(SNAPS), figsize=(4.2 * len(SNAPS), 4.2))
for ax, idx in zip(axes, SNAPS):
    sl, N = load_slice(idx)
    coords, base_blocks, t = load_blocks(idx)
    ax.imshow(np.log10(sl.T + 1e-4), origin="lower", extent=[0, LX, 0, LX],
              cmap="inferno", interpolation="nearest")
    z_phys = (N // 2 + 0.5) * LX / N
    for L in range(1, len(coords)):                        # overlay refined levels
        W = LX / (base_blocks * 2 ** L)
        rects = [Rectangle((cx * W, cy * W), W, W)
                 for cx, cy, cz in coords[L] if cz * W <= z_phys < (cz + 1) * W]
        if rects:
            ax.add_collection(PatchCollection(
                rects, facecolor="none", edgecolor=LEVEL_COLOR.get(L, "w"), lw=0.7))
    ax.text(0.035, 0.965, f"t = {t:.3f}", transform=ax.transAxes, ha="left",
            va="top", color="w", fontsize=13,
            bbox=dict(boxstyle="round,pad=0.25", fc="0.1", ec="none", alpha=0.55))
    ax.set_xticks([]); ax.set_yticks([])
axes[0].set_ylabel(r"$z = L_z/2$ slice", fontsize=11)
# level legend
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=c, lw=2, label=f"level {L}") for L, c in LEVEL_COLOR.items() if L < 3]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig("code_paper_figs/amr_sedov3d.png", dpi=200, bbox_inches="tight")
print("wrote code_paper_figs/amr_sedov3d.png")
