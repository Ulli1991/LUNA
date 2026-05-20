"""CC85 starburst-wind density slice grid plot.

Final-snapshot x-z slice (through y = Ly/2) of log10 number density,
side-by-side for the resolution scan. Empty panels are placeholders for
resolutions that haven't run yet.

Output: figures/cc85_wind_grid.{pdf,png}
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from file_io.read_snapshot import read_snapshot


# Unit conversion (Arepo: kpc / 1e10 Msun / km/s)
_PROTON_MASS = 1.67262e-24
_MSUN = 1.98892e33
_KPC = 3.08568e21
_MU = 0.6


def _density_cgs_from_code(rho_code):
    """Convert rho in (1e10 Msun / kpc^3) to cgs g/cm^3."""
    UnitDensity_cgs = (1e10 * _MSUN) / (_KPC ** 3)
    return rho_code * UnitDensity_cgs


def _latest_snapshot(run_dir):
    if not os.path.isdir(run_dir):
        return None
    snaps = sorted([d for d in os.listdir(run_dir) if d.startswith("snapdir_")])
    if not snaps:
        return None
    return os.path.join(run_dir, snaps[-1])


def _load_run(Nx, snapshots_root):
    run_dir = os.path.join(snapshots_root, f"cc85_Nx{Nx}")
    snapdir = _latest_snapshot(run_dir)
    if snapdir is None:
        return None
    snap = read_snapshot(snapdir)
    rho = snap["rho"]                        # (Nx, Ny, Nz)
    Ny = snap["Ny"]
    rho_slice = rho[:, Ny // 2, :]           # x-z slice through y=Ly/2
    rho_cgs = _density_cgs_from_code(rho_slice)
    n_H = rho_cgs / (_MU * _PROTON_MASS)
    return {
        "logn": np.log10(np.maximum(n_H, 1e-12)),
        "t":    snap["t"],
        "Lx":   snap["Lx"],
        "Lz":   snap["Lz"],
    }


def main(snapshots_root="snapshots", outpath_stem="figures/cc85_wind_grid"):
    resolutions = [128, 256, 512]
    runs = {}
    for Nx in resolutions:
        run = _load_run(Nx, snapshots_root)
        if run is not None:
            runs[Nx] = run
            print(f"  loaded Nx={Nx} (t_code={run['t']:.5f})")
        else:
            print(f"  no data: Nx={Nx}")

    if not runs:
        raise RuntimeError("no CC85 runs found")

    # Box is 10 x 20 kpc (x-z); panel aspect ~ 1:2 so use 9x12 figsize.
    fig, axes = plt.subplots(1, 3, figsize=(11, 8), constrained_layout=True)
    cmap = plt.get_cmap("inferno")
    vmin, vmax = -5.0, -1.0
    im = None
    for j, Nx in enumerate(resolutions):
        ax = axes[j]
        run = runs.get(Nx)
        if run is None:
            ax.text(0.5, 0.5, f"(no data yet)\nNx={Nx}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=14, color="0.4")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        # imshow: rows = x, cols = z; we want z vertical, x horizontal -> transpose
        data = run["logn"].T
        extent = [0, run["Lx"], 0, run["Lz"]]
        im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect="equal",
                       interpolation="nearest")
        # Convert internal time to Myr for the title
        SEC_PER_YR = 3.15576e7; MYR = 1e6 * SEC_PER_YR
        UnitTime_s = _KPC / 1e5         # kpc / (km/s) = ~ 0.978 Gyr
        t_Myr = run["t"] * UnitTime_s / MYR
        ax.set_title(f"Nx={Nx}    t={t_Myr:.1f} Myr", fontsize=11)
        ax.set_xlabel("x (kpc)")
        if j == 0:
            ax.set_ylabel("z (kpc)")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85,
                            label=r"$\log_{10}\, n_{\rm H}\ [{\rm cm}^{-3}]$",
                            pad=0.02)

    for ext in ("pdf", "png"):
        path = f"{outpath_stem}.{ext}"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
