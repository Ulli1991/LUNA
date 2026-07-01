"""Physical-units 3D Sedov-Taylor blast: time-evolution comparison.

Top row    : 64^3 fixed grid.
Bottom row : 64^3 -> 512^3 block-AMR, with the refinement mesh overlaid
             (block edges colored by level).
Columns    : time (left -> right), central z = Lz/2 density slice, log scale.

AREPO units (pc, km/s, Msun); SN 1e51 erg into n=1 cm^-3, T=1e4 K ISM, 40 pc box.

Reads /ptmp/uli/jax-hydro/{sedphys_fixed64_evo, sedphys_amr512_evo}/
  snapdir_NNN/*.hdf5   (density; AMR is the composite finest field)
  blocks_NNN.npz       (per-level active block coords; AMR only)
"""
import glob
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

FIXED = os.environ.get("FIXED_OUT", "/ptmp/uli/jax-hydro/sedphys_fixed64_evo")
AMR = os.environ.get("AMR_OUT", "/ptmp/uli/jax-hydro/sedphys_amr512_evo")
LX = 40.0                              # box size (pc)
SNAPS = [1, 2, 3, 4]                   # 2500 / 5000 / 7500 / 10000 yr
YR_PER_CODE = 9.777e5                  # 1 code time in yr (pc / km s^-1)
LEVEL_COLOR = {1: "#00e5ff", 2: "#ffe000", 3: "#7CFC00"}
VMIN, VMAX = -3.0, -0.9                # log10 density (Msun pc^-3)


def zslice(out, idx):
    """Central z-slice of the density (reads only that plane, not the 512^3)."""
    f = sorted(glob.glob(f"{out}/snapdir_{idx:03d}/*.hdf5"))[0]
    with h5py.File(f) as h:
        d = h["cells"]["density"]
        N = d.shape[2]
        sl = d[:, :, N // 2]
    return sl, N


def blocks(out, idx):
    z = np.load(f"{out}/blocks_{idx:03d}.npz")
    base = int(z["L0_coords"][:, 0].max()) + 1
    coords = {L: z[f"L{L}_coords"] for L in range(int(z["levels"]))}
    return coords, base, float(z["t"])


fig, ax = plt.subplots(2, len(SNAPS), figsize=(3.6 * len(SNAPS), 7.5),
                       constrained_layout=True)
im = None
for j, idx in enumerate(SNAPS):
    # top: 64^3 fixed grid
    fsl, _ = zslice(FIXED, idx)
    ax[0, j].imshow(np.log10(fsl.T + 1e-4), origin="lower", extent=[0, LX, 0, LX],
                    cmap="inferno", vmin=VMIN, vmax=VMAX, interpolation="nearest")

    # bottom: AMR composite + refinement mesh
    asl, N = zslice(AMR, idx)
    coords, base, t = blocks(AMR, idx)
    im = ax[1, j].imshow(np.log10(asl.T + 1e-4), origin="lower", extent=[0, LX, 0, LX],
                         cmap="inferno", vmin=VMIN, vmax=VMAX, interpolation="nearest")
    z_phys = (N // 2 + 0.5) * LX / N
    for L in range(1, len(coords)):
        W = LX / (base * 2 ** L)                       # physical block width at level L
        rects = [Rectangle((cx * W, cy * W), W, W)
                 for cx, cy, cz in coords[L] if cz * W <= z_phys < (cz + 1) * W]
        if rects:
            ax[1, j].add_collection(PatchCollection(
                rects, facecolor="none", edgecolor=LEVEL_COLOR.get(L, "w"), lw=0.4))

    ax[0, j].set_title(f"{t * YR_PER_CODE:,.0f} yr", fontsize=13)
    for r in (0, 1):
        ax[r, j].set_xticks([])
        ax[r, j].set_yticks([])

ax[0, 0].set_ylabel("64³ fixed grid", fontsize=14)
ax[1, 0].set_ylabel("64³ → 512³ AMR", fontsize=14)
handles = [Line2D([0], [0], color=c, lw=2, label=f"level {L}")
           for L, c in LEVEL_COLOR.items()]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11,
           bbox_to_anchor=(0.5, -0.02))
cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
cb.set_label(r"$\log_{10}\,\rho$  (M$_\odot$ pc$^{-3}$)", fontsize=11)

os.makedirs("code_paper_figs", exist_ok=True)
fig.savefig("code_paper_figs/amr_sedov3d_evolution.png", dpi=200, bbox_inches="tight")
print("wrote code_paper_figs/amr_sedov3d_evolution.png")
