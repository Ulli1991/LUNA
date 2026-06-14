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

SNAP256 = os.environ.get("LUNA_BB_SNAP", "snapshots/bb79_baro_Nx256")


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
    os.makedirs("figures", exist_ok=True)
    out = f"figures/bb79_256_column_density_Nx{S256.shape[0]}.png"
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
