"""Time-evolution comparison: uniform fixed-grid 1024^2 vs block-AMR
(128^2 base -> 1024^2 effective), with the AMR refinement mesh overlaid
(block edges colored by level). Rows = time, cols = [uniform | AMR + mesh].

Reads:
  snapshots_uniform_1024/snapdir_NNN/*.hdf5   (uniform, x-tiled shards)
  snapshots_amr_evo/snapdir_MMM/*.hdf5        (AMR composite, replicated shards)
  snapshots_amr_evo/blocks_MMM.npz            (per-level active block coords)
"""
import glob
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

LX = 2.0
BASE = 128
BLOCK = 16
UNI_DT, AMR_DT = 0.5, 0.25       # time_bet_snaps for each run
TIMES = [1.0, 2.0, 3.0, 4.0, 5.0]
LEVEL_COLOR = {1: "#00e5ff", 2: "#ffe000", 3: "#7CFC00"}


def read_stitched(snapdir):
    """Assemble full density from sharded HDF5 (handles x/y tiling and replicated)."""
    fs = sorted(glob.glob(snapdir + "/*.hdf5"))
    with h5py.File(fs[0]) as h:
        gNx = int(h["cells"].attrs["global_Nx"])
        gNy = int(h["cells"].attrs.get("global_Ny", gNx))
    full = np.full((gNx, gNy), np.nan)
    for f in fs:
        with h5py.File(f) as h:
            c = h["cells"]; rho = c["density"][:]
            xs = int(c.attrs.get("x_start", 0)); ys = int(c.attrs.get("y_start", 0))
            full[xs:xs + rho.shape[0], ys:ys + rho.shape[1]] = rho
    return full


def block_rects(npz):
    b = np.load(npz)
    out = {}
    for L in (1, 2, 3):
        key = f"L{L}_coords"
        if key not in b.files:
            continue
        bw = BLOCK * LX / (BASE * (2 ** L))      # physical block width at level L
        out[L] = (np.asarray(b[key]), bw)
    return out


nrows = len(TIMES)
fig, ax = plt.subplots(nrows, 2, figsize=(11, 5.2 * nrows), constrained_layout=True)
for i, t in enumerate(TIMES):
    uni = read_stitched(f"snapshots_uniform_1024/snapdir_{round(t / UNI_DT):03d}")
    amr_idx = round(t / AMR_DT)
    amr = read_stitched(f"snapshots_amr_evo/snapdir_{amr_idx:03d}")
    vmin, vmax = 0.9, 2.1
    for a_, img, title in ((ax[i, 0], uni, f"Uniform 1024²  t={t:.1f}"),
                           (ax[i, 1], amr, f"AMR 128²→1024²  t={t:.1f}")):
        a_.imshow(img.T, origin="lower", cmap="magma", extent=[0, LX, 0, LX],
                  aspect="equal", vmin=vmin, vmax=vmax)
        a_.set_xticks([]); a_.set_yticks([]); a_.set_title(title, fontsize=11)
    # overlay refinement mesh on the AMR panel
    for L, (coords, bw) in block_rects(f"snapshots_amr_evo/blocks_{amr_idx:03d}.npz").items():
        rects = [Rectangle((cx * bw, cy * bw), bw, bw) for cx, cy in coords]
        ax[i, 1].add_collection(PatchCollection(
            rects, facecolor="none", edgecolor=LEVEL_COLOR[L], linewidth=0.3, alpha=0.55))

fig.suptitle("Kelvin-Helmholtz: fixed grid vs block-AMR (refinement mesh overlaid)", fontsize=13)
out = "out/amr_vs_uniform_evolution.png"
import os; os.makedirs("out", exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
