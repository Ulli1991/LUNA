"""LUNA BB79 256³ isolated-BC column-density projection.

Projects ρ along z (rotation axis) at the last clean snapshot. The
isolated multipole-Dirichlet Poisson (no periodic image saddles) and
outflow hydro replace the previous periodic setup; this is the
apples-to-apples reference for AREPO's GRAVITY_NOT_PERIODIC run.
"""
from __future__ import annotations
import os, glob
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.figstyle import apply_style, decorate, FIGSIZE

apply_style()

SNAP64  = "/raven/u/uli/jax-hydro/snapshots/bb_iso_Nx64"
SNAP256 = "/raven/u/uli/jax-hydro/snapshots/bb_iso_minmod_Nx256"


def luna_column_density(snapdir):
    files = sorted(glob.glob(os.path.join(snapdir, "*.hdf5")))
    with h5py.File(files[0], "r") as h:
        a = h["cells"].attrs
        Nx = int(a["Nx_global"]); Ny = int(a["Ny_global"]); Nz = int(a["Nz_global"])
        L = float(a["Lx"]); dx = float(a["dx"]); t = float(a["time"])
    rho = np.empty((Nx, Ny, Nz), dtype=np.float64)
    for f in files:
        with h5py.File(f, "r") as h:
            c = h["cells"]
            xs = int(c.attrs["x_start"]); ys = int(c.attrs["y_start"]); zs = int(c.attrs["z_start"])
            shp = c["density"].shape
            rho[xs:xs+shp[0], ys:ys+shp[1], zs:zs+shp[2]] = c["density"][:]
    return rho.sum(axis=2) * dx, L, t


def latest_clean(root):
    """Last snapshot before negative ρ appears (if any)."""
    dirs = sorted(glob.glob(os.path.join(root, "snapdir_*")))
    last_good = None
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "*.hdf5")))
        bad = False
        for f in files:
            with h5py.File(f, "r") as h:
                if h["cells"]["density"][:].min() <= 0:
                    bad = True; break
        if bad:
            return last_good
        last_good = d
    return last_good


def main():
    snap64  = latest_clean(SNAP64)
    snap256 = latest_clean(SNAP256)
    print(f"  64³  last clean: {os.path.basename(snap64)}")
    print(f"  256³ last clean: {os.path.basename(snap256)}")

    S64,  L64,  t64  = luna_column_density(snap64)
    S256, L256, t256 = luna_column_density(snap256)
    print(f"  64³:  t={t64:.3f}  Σ ∈ [{S64.min():.3e}, {S64.max():.3e}]")
    print(f"  256³: t={t256:.3f}  Σ ∈ [{S256.min():.3e}, {S256.max():.3e}]")

    S64_pos  = np.maximum(S64,  1e-30)
    S256_pos = np.maximum(S256, 1e-30)
    vmin = np.log10(min(S64_pos.min(), S256_pos.min()))
    vmax = np.log10(max(S64_pos.max(), S256_pos.max()))

    fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE[0]*2, FIGSIZE[1]))
    im0 = axes[0].imshow(np.log10(S64_pos).T,  origin="lower",
                          extent=(0, L64,  0, L64),
                          vmin=vmin, vmax=vmax, cmap="magma")
    axes[0].grid(False)
    axes[0].set_title(f"LUNA 64³  isolated  (t={t64:.2f})")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0], shrink=0.85, label=r"$\log_{10}\Sigma$")

    im1 = axes[1].imshow(np.log10(S256_pos).T, origin="lower",
                          extent=(0, L256, 0, L256),
                          vmin=vmin, vmax=vmax, cmap="magma")
    axes[1].grid(False)
    axes[1].set_title(f"LUNA 256³ isolated (t={t256:.2f})")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1], shrink=0.85, label=r"$\log_{10}\Sigma$")

    fig.suptitle("Boss-Bodenheimer 1979 — LUNA isolated BC, resolution scan, column density along z")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out = "figures/bb79_iso_luna_64_vs_256_minmod.png"
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
