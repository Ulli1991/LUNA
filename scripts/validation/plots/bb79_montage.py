"""Boss-Bodenheimer time-evolution montage — the GANDALF Figure 21 analog.

A 2×3 grid of column-density (log₁₀ Σ along the rotation axis) panels at the
six GANDALF Fig 21 times. NB: the panel labels in Hubber+ 2018 Fig 21 read
"0.16…0.36 Myr", but the §6.12 text states t_end = 0.04 Myr — the figure labels
are a factor-of-10 typo; the real times are 0.016–0.036 Myr (≈ 0.9–2.1 t_ff),
the collapse→fragmentation sequence. Those are all within our t_end = 0.04 Myr
run, so no sink particles are needed for this montage.

Reads snapshots/bb79_baro_Nx256 (override with LUNA_BB_SNAP).
"""
import os
import glob

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.figstyle import apply_style, FIGSIZE
from scripts.validation.plots.bb79_iso_luna import luna_column_density

apply_style()

LABEL_FS = 11   # in-panel image annotations only (axis labels inherit rcParams)

SNAP = os.environ.get("LUNA_BB_SNAP", "snapshots/bb79_baro_Nx256_sinks")
TIMES_MYR = [0.020, 0.025, 0.030, 0.0325, 0.035, 0.040]
ZOOM_PC = float(os.environ.get("LUNA_BB_ZOOM_PC", "0.005"))   # 0 -> whole box
PC = 3.0856775814913673e18
MYR = 1.0e6 * 3.15576e7
UNIT_TIME_S = PC / 1.0e5                 # pc / (km/s)
TIMES_CODE = [t * MYR / UNIT_TIME_S for t in TIMES_MYR]


def main():
    dirs = sorted(glob.glob(os.path.join(SNAP, "snapdir_*")))
    if not dirs:
        raise SystemExit(f"no snapshots under {SNAP}")
    times = []
    for d in dirs:
        f = sorted(glob.glob(os.path.join(d, "*.hdf5")))[0]
        with h5py.File(f, "r") as h:
            times.append(float(h["cells"].attrs["time"]))
    times = np.array(times)

    Ss, ts, Ls = [], [], []
    for tc in TIMES_CODE:
        k = int(np.argmin(np.abs(times - tc)))
        S, L, t = luna_column_density(dirs[k])
        Ss.append(np.maximum(S, 1e-30)); ts.append(t); Ls.append(L)
    # Gas-aware scaling: sink point-masses would otherwise pin vmax and wash out
    # the gas; clip to the 99.9th percentile with a 4-dex dynamic range.
    vmax = np.log10(max(np.percentile(s, 99.9) for s in Ss))
    vmin = vmax - 4.0

    fig, axes = plt.subplots(2, 3, figsize=(FIGSIZE[0] * 1.8, FIGSIZE[1] * 1.25))
    im = None
    for ax, S, L, tmyr in zip(axes.ravel(), Ss, Ls, TIMES_MYR):
        im = ax.imshow(np.log10(S).T, origin="lower", extent=(0, L, 0, L),
                       vmin=vmin, vmax=vmax, cmap="magma", interpolation="bilinear")
        ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
        if ZOOM_PC > 0:
            ctr = 0.5 * L
            ax.set_xlim(ctr - ZOOM_PC, ctr + ZOOM_PC)
            ax.set_ylim(ctr - ZOOM_PC, ctr + ZOOM_PC)
        ax.text(0.05, 0.90, f"{tmyr * 1e3:g} kyr", transform=ax.transAxes,
                color="w", fontsize=LABEL_FS, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.92, 0.12, 0.018, 0.76])
    fig.colorbar(im, cax=cax, label=r"$\log_{10}\,\Sigma\ [\mathrm{M_\odot\,pc^{-2}}]$")
    os.makedirs("figures", exist_ok=True)
    tag = "_sinks" if "sink" in SNAP else ""
    out = f"figures/bb79_montage_Nx{Ss[0].shape[0]}{tag}.png"
    fig.savefig(out)
    print(f"  wrote {out}  (times kyr: {[round(t*1e3) for t in TIMES_MYR]})")


if __name__ == "__main__":
    main()
