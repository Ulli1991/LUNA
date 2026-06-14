"""BB79 256³ MinMod: isolated BC vs periodic BC diagnostic.

Side-by-side log column density at the last good snap. Same hydro,
same limiter, same resolution — only the BC differs. If the cardinal-
axis lines disappear with periodic BC, they came from our isolated
multipole-Dirichlet implementation. If they stay, it's the HLLC 3D
carbuncle.
"""
import os, glob
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.figstyle import apply_style, decorate, FIGSIZE

apply_style()

ISO_DIR  = "/raven/u/uli/jax-hydro/snapshots/bb_iso_minmod_Nx256"
PER_DIR  = "/raven/u/uli/jax-hydro/snapshots/bb_periodic_minmod_Nx256"


def col_density(snapdir):
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
    dirs = sorted(glob.glob(os.path.join(root, "snapdir_*")))
    last = None
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "*.hdf5")))
        bad = False
        for f in files:
            with h5py.File(f, "r") as h:
                if h["cells"]["density"][:].min() <= 0:
                    bad = True; break
        if bad: return last
        last = d
    return last


def main():
    sd_iso = latest_clean(ISO_DIR)
    sd_per = latest_clean(PER_DIR)
    S_iso, L_iso, t_iso = col_density(sd_iso)
    S_per, L_per, t_per = col_density(sd_per)
    print(f"  iso: t={t_iso:.3f}  Σ∈[{S_iso.min():.2e}, {S_iso.max():.2e}]")
    print(f"  per: t={t_per:.3f}  Σ∈[{S_per.min():.2e}, {S_per.max():.2e}]")

    Sp_iso = np.maximum(S_iso, 1e-30)
    Sp_per = np.maximum(S_per, 1e-30)
    vmin = np.log10(min(Sp_iso.min(), Sp_per.min()))
    vmax = np.log10(max(Sp_iso.max(), Sp_per.max()))

    fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE[0]*2, FIGSIZE[1]))
    im0 = axes[0].imshow(np.log10(Sp_iso).T, origin="lower",
                          extent=(0, L_iso, 0, L_iso),
                          vmin=vmin, vmax=vmax, cmap="magma")
    axes[0].grid(False)
    axes[0].set_title(f"isolated BC (outflow + multipole-Dirichlet) — t={t_iso:.2f}")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0], shrink=0.85, label=r"$\log_{10}\Sigma$")

    im1 = axes[1].imshow(np.log10(Sp_per).T, origin="lower",
                          extent=(0, L_per, 0, L_per),
                          vmin=vmin, vmax=vmax, cmap="magma")
    axes[1].grid(False)
    axes[1].set_title(f"periodic BC (Jeans swindle) — t={t_per:.2f}")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1], shrink=0.85, label=r"$\log_{10}\Sigma$")

    fig.suptitle("BB79 256³ MinMod — BC diagnostic: isolated vs periodic\n"
                 "Same hydro / limiter; cardinal-axis lines test")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    Nx_luna = S_iso.shape[0]   # column density is (Nx, Ny)
    out = f"figures/bb79_iso_vs_periodic_diagnostic_Nx{Nx_luna}.png"
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
