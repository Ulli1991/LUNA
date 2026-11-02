"""LUNA BB79 256³ isolated-BC column-density projection.

Projects ρ along z (rotation axis) at the last clean snapshot. The
isolated multipole-Dirichlet Poisson (no periodic image saddles) and
outflow hydro replace the previous periodic setup; this is the
apples-to-apples reference for AREPO's GRAVITY_NOT_PERIODIC run.
"""
from __future__ import annotations
import os
import numpy as np
from scripts.validation.plots._plotkit import mpl, save, load_snap, latest_clean

plt = mpl()
from luna.utils.figstyle import FIGSIZE

SNAP256 = os.environ.get("LUNA_BB_SNAP", "snapshots/bb79_baro_Nx256")


def luna_column_density(snapdir):
    data, a = load_snap(snapdir)
    return data["density"].sum(axis=2) * float(a["dx"]), float(a["Lx"]), float(a["time"])


def main():
    snap256 = latest_clean(SNAP256)
    print(f"  256³ last clean: {os.path.basename(snap256)}")
    S256, L256, t256 = luna_column_density(snap256)
    print(f"  256³: t={t256:.4f}  Σ ∈ [{S256.min():.3e}, {S256.max():.3e}]")

    S256_pos = np.maximum(S256, 1e-30)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(np.log10(S256_pos).T, origin="lower",
                   extent=(0, L256, 0, L256), cmap="magma")
    ax.grid(False)
    ax.set_xlabel("x [pc]"); ax.set_ylabel("y [pc]")
    plt.colorbar(im, ax=ax, shrink=0.85, label=r"$\log_{10}\,\Sigma$")
    fig.tight_layout()
    out = f"code_paper_figs/bb79_256_column_density_Nx{S256.shape[0]}.png"
    save(fig, out, close=True)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
