"""CC85 starburst-wind diagnostic grid: density, radial velocity, temperature.

x-z slice through y = Ly/2 of three quantities at a fixed snapshot, plotted
side-by-side across the resolution scan. Empty panels are placeholders for
resolutions that haven't run yet.

Output: figures/cc85_wind_density.{pdf,png}, cc85_wind_vradial.{pdf,png},
        cc85_wind_temperature.{pdf,png}
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from file_io.read_snapshot import read_snapshot
from utils.figstyle import apply_style, decorate, FIGSIZE

apply_style()

LABEL_FS = 11   # in-panel image annotations only (axis labels inherit rcParams)


# Unit conversion (Arepo: kpc / 1e10 Msun / km/s).
_PROTON_MASS = 1.67262e-24   # g
_MSUN = 1.98892e33           # g
_KPC = 3.08568e21            # cm
_KB = 1.38065e-16            # erg/K
_MU = 0.6                    # mean molecular weight
_KM = 1.0e5                  # cm/s


UnitMass_g = 1e10 * _MSUN
UnitLength_cm = _KPC
UnitVel_cms = _KM
UnitDensity_cgs = UnitMass_g / (UnitLength_cm ** 3)
UnitPressure_cgs = UnitDensity_cgs * UnitVel_cms ** 2


# M82 runs: t_end=60 Myr, time_bet_snaps=1 Myr  -> final snap at t=60 is snapdir_060.
_SNAP_IDX = 60
_RUN_PREFIX = "m82_Nx"            # snapshots/m82_NxN/snapdir_NNN


def _snapshot_at_index(run_dir, idx):
    if not os.path.isdir(run_dir):
        return None
    p = os.path.join(run_dir, f"snapdir_{idx:03d}")
    return p if os.path.isdir(p) else None


def _load_run(Nx, snapshots_root):
    run_dir = os.path.join(snapshots_root, f"{_RUN_PREFIX}{Nx}")
    snapdir = _snapshot_at_index(run_dir, _SNAP_IDX)
    if snapdir is None:
        return None
    snap = read_snapshot(snapdir)
    rho = snap["rho"]                            # (Nx, Ny, Nz)
    vx  = snap["vx"]
    vz  = snap["vz"]
    P   = snap["P"]
    Nx_ = snap["Nx"]; Ny_ = snap["Ny"]; Nz_ = snap["Nz"]
    j = Ny_ // 2
    rho_s = rho[:, j, :]
    vx_s  = vx[:, j, :]
    vz_s  = vz[:, j, :]
    P_s   = P[:, j, :]

    # log10 n_H [cm^-3]
    rho_cgs = rho_s * UnitDensity_cgs
    n_H = rho_cgs / (_MU * _PROTON_MASS)
    log_n = np.log10(np.maximum(n_H, 1e-12))

    # Vertical velocity (z-component) — straightforward sign convention
    # (positive = upward, negative = downward, regardless of position).
    Lx = snap["Lx"]; Lz = snap["Lz"]
    v_z = vz_s

    # log10 T [K]
    P_cgs = P_s * UnitPressure_cgs
    T_cgs = (P_cgs / np.maximum(rho_cgs, 1e-40)) * (_MU * _PROTON_MASS / _KB)
    log_T = np.log10(np.maximum(T_cgs, 1e-2))

    return {
        "log_n": log_n,
        "v_z":   v_z,
        "log_T": log_T,
        "t":     snap["t"],
        "Lx":    Lx,
        "Lz":    Lz,
    }


def _plot_grid(field_key, cmap_name, vmin, vmax, label, outpath_stem, runs,
               diverging=False):
    resolutions = [128, 256, 512, 1024]
    fig, axes = plt.subplots(1, 4, figsize=(14, 8), constrained_layout=True)
    cmap = plt.get_cmap(cmap_name)
    SEC_PER_YR = 3.15576e7; MYR = 1e6 * SEC_PER_YR
    UnitTime_s = _KPC / _KM
    im = None
    for j, Nx in enumerate(resolutions):
        ax = axes[j]
        ax.grid(False)
        run = runs.get(Nx)
        if run is None:
            ax.text(0.5, 0.5, f"(no data yet)\nNx={Nx}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=LABEL_FS, color="0.4")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        data = run[field_key].T
        # Center the box so axes go from -L/2 to +L/2 (matches the wind
        # geometry: injection sphere is at the origin, outflow is bi-conical
        # with v_r > 0 above the midplane and v_r < 0 below).
        extent = [-0.5 * run["Lx"], 0.5 * run["Lx"],
                  -0.5 * run["Lz"], 0.5 * run["Lz"]]
        im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect="equal",
                       interpolation="nearest")
        t_Myr = run["t"] * UnitTime_s / MYR
        ax.text(0.05, 0.92, f"Nx={Nx}    t={t_Myr:.1f} Myr",
                transform=ax.transAxes, fontsize=LABEL_FS, color="white", fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=1.5))
        ax.set_xlabel("x (kpc)")
        if j == 0:
            ax.set_ylabel("z (kpc)")

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85,
                     label=label, pad=0.02)

    for ext in ("pdf", "png"):
        path = f"{outpath_stem}.{ext}"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def _radial_profile(rho_code, vx, vy, vz, P_code, Lx, Ly, Lz, n_bins=64):
    """Spherically-average n_H, v_r, T into radial bins. r in kpc."""
    Nx_, Ny_, Nz_ = rho_code.shape
    dx = Lx / Nx_; dy = Ly / Ny_; dz = Lz / Nz_
    x = (np.arange(Nx_) + 0.5) * dx - 0.5 * Lx
    y = (np.arange(Ny_) + 0.5) * dy - 0.5 * Ly
    z = (np.arange(Nz_) + 0.5) * dz - 0.5 * Lz
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2).ravel()

    rho_cgs = (rho_code * UnitDensity_cgs).ravel()
    n_H = rho_cgs / (_MU * _PROTON_MASS)
    R_safe = np.where(R > 0, R, 1.0)
    v_r = ((vx * X + vy * Y + vz * Z).ravel()) / R_safe
    P_cgs = (P_code * UnitPressure_cgs).ravel()
    T_cgs = (P_cgs / np.maximum(rho_cgs, 1e-40)) * (_MU * _PROTON_MASS / _KB)

    r_max = min(0.5 * Lx, 0.5 * Ly, 0.5 * Lz)
    edges = np.linspace(0, r_max, n_bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    counts, _ = np.histogram(R, bins=edges)
    counts = np.maximum(counts, 1)
    n_prof, _ = np.histogram(R, bins=edges, weights=n_H)
    v_prof, _ = np.histogram(R, bins=edges, weights=v_r)
    T_prof, _ = np.histogram(R, bins=edges, weights=T_cgs)
    return centers, n_prof / counts, v_prof / counts, T_prof / counts


def _plot_radial_profiles(runs, snapshots_root="snapshots"):
    fig, axes = plt.subplots(1, 3, figsize=(FIGSIZE[0]*3, FIGSIZE[1]), constrained_layout=True)
    resolutions = sorted(runs.keys())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(resolutions)))
    for i, Nx in enumerate(resolutions):
        run_dir = os.path.join(snapshots_root, f"{_RUN_PREFIX}{Nx}")
        snapdir = _snapshot_at_index(run_dir, _SNAP_IDX)
        snap = read_snapshot(snapdir)
        r, n_H, v_r, T = _radial_profile(
            snap["rho"], snap["vx"], snap["vy"], snap["vz"], snap["P"],
            snap["Lx"], snap["Ly"], snap["Lz"], n_bins=64,
        )
        axes[0].loglog(r, n_H, color=colors[i], lw=1.8, label=f"Nx={Nx}")
        axes[1].plot  (r, v_r, color=colors[i], lw=1.8, label=f"Nx={Nx}")
        axes[2].loglog(r, T,   color=colors[i], lw=1.8, label=f"Nx={Nx}")

    axes[0].set_xlabel("r (kpc)"); axes[0].set_ylabel(r"$n_{\rm H}\ [{\rm cm}^{-3}]$")
    axes[1].set_xlabel("r (kpc)"); axes[1].set_ylabel(r"$v_{\rm rad}\ [{\rm km\,s^{-1}}]$")
    axes[2].set_xlabel("r (kpc)"); axes[2].set_ylabel(r"$T\ [{\rm K}]$")
    for ax in axes:
        ax.legend(loc="best"); decorate(ax)

    for ext in ("pdf", "png"):
        path = f"figures/m82_wind_radial_profiles.{ext}"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def main(snapshots_root="snapshots"):
    resolutions = [128, 256, 512, 1024]
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

    # Auto-percentile color ranges from the union of loaded data
    all_logn = np.concatenate([r["log_n"].ravel() for r in runs.values()])
    all_vr   = np.concatenate([r["v_z"].ravel()   for r in runs.values()])
    all_logT = np.concatenate([r["log_T"].ravel() for r in runs.values()])
    n_lo, n_hi = np.percentile(all_logn, [2, 99])
    vr_mag = np.percentile(np.abs(all_vr), 99)
    T_lo, T_hi = np.percentile(all_logT, [2, 99])

    _plot_grid("log_n", "inferno", n_lo, n_hi,
               r"$\log_{10}\, n_{\rm H}\ [{\rm cm}^{-3}]$",
               "figures/m82_wind_density", runs)
    _plot_grid("v_z", "RdBu_r", -vr_mag, vr_mag,
               r"$v_z\ [{\rm km\,s^{-1}}]$",
               "figures/m82_wind_vz", runs, diverging=True)
    _plot_grid("log_T", "plasma", T_lo, T_hi,
               r"$\log_{10}\, T\ [{\rm K}]$",
               "figures/m82_wind_temperature", runs)
    del runs, all_logn, all_vr, all_logT
    import gc; gc.collect()
    _plot_radial_profiles({Nx: None for Nx in resolutions}, snapshots_root)


if __name__ == "__main__":
    main()
