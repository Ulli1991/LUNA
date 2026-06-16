"""Resolution convergence of the self-gravity collapse tests against AREPO.

For Evrard and the two Bonnor-Ebert spheres we overlay the LUNA radial density
profile at Nx = 64, 128, 256 on the matched AREPO reference, showing convergence
toward the reference solution as resolution increases. Reuses the loaders and
radial binning from arepo_vs_luna.py. Solid lines, no titles, shared style.

256³ snapshots are multi-GB to read — run this on a compute node (SLURM), not
the login node.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.figstyle import apply_style, decorate, FIGSIZE
from scripts.validation.plots.arepo_vs_luna import (
    AREPO_RUN, LUNA_SNAP, load_arepo, load_luna, radial_bins,
    pick_snap_at_time, pick_luna_snap_at_time,
)

apply_style()

# (name, arepo_subdir, luna_subdir_template, target_t)
TESTS = [
    ("Evrard",   "evrard_3d",             "evrard_Nx{N}",   0.80),
    ("BE sub",   "bonnor_ebert_3d_sub",   "be_xi5.0_Nx{N}", 3.00),
    ("BE super", "bonnor_ebert_3d_super", "be_xi10.0_Nx{N}", 3.00),
]
RESOLUTIONS = [64, 128, 256]
RES_COLOR = {64: "C0", 128: "C1", 256: "C2"}


def main():
    os.makedirs("figures", exist_ok=True)
    for name, arepo_sub, luna_tmpl, target_t in TESTS:
        fig, ax = plt.subplots(figsize=FIGSIZE)

        # AREPO reference (black).
        arepo_path = pick_snap_at_time(
            os.path.join(AREPO_RUN, arepo_sub, "output", "snap_*.hdf5"), target_t)
        if arepo_path:
            A = load_arepo(arepo_path)
            r_max = 0.5 * A["L"]
            cA, mA = radial_bins(A["r"], A["rho"], n_bins=40, r_max=r_max, log=True)
            ax.loglog(cA, np.abs(mA), "-", lw=2.4, color="k", label="AREPO")
        else:
            r_max = None
            print(f"  [warn] {name}: no AREPO reference at t={target_t}")

        # LUNA at each resolution.
        for N in RESOLUTIONS:
            luna_dir = os.path.join(LUNA_SNAP, luna_tmpl.format(N=N))
            luna_path = pick_luna_snap_at_time(luna_dir, target_t)
            if not luna_path:
                print(f"  [skip] {name} Nx{N}: no snapshot in {luna_dir}")
                continue
            LU = load_luna(luna_path)
            rm = r_max if r_max is not None else 0.5 * LU["L"]
            cL, mL = radial_bins(LU["r"], LU["rho"], n_bins=40, r_max=rm, log=True)
            ax.loglog(cL, np.abs(mL), "-", lw=1.8, color=RES_COLOR[N],
                      label=fr"LUNA $N={N}^3$")

        ax.set_xlabel(r"$r$"); ax.set_ylabel(r"$\rho$")
        ax.set_ylim(bottom=2e-2)      # crop the low-density numerical floor
        ax.legend(loc="best")
        decorate(ax)
        fig.tight_layout()
        stem = name.lower().replace(" ", "_")
        out = f"figures/gravity_convergence_{stem}.png"
        fig.savefig(out); plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
