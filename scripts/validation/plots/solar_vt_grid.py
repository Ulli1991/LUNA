"""V&T solar-convection delta-rho and delta-T grid plots.

Produces two 2x3 figures (HLLC top row, LHLLC bottom row; columns Nx=1024,
2048, 4096). Each panel shows the perturbation field at the latest available
snapshot for that run.

Output: figures/solar_vt_drho_grid.{pdf,png}, figures/solar_vt_dT_grid.{pdf,png}
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from file_io.read_snapshot import read_snapshot


# V&T (2025) 3-layer polytropic atmosphere — must match solar_convection_vt_2d.py
_VT = dict(
    z_bounds=(0.0, 1.0, 3.0),
    Gamma=(2.0, 1.05, 5.0 / 3.0),
    rho_top=(2.7, 0.86, 0.05),
    P_top=(5.94, 3.53, 0.05),
)
_GRAVITY = -2.7   # cfg.gravity_z (downward); g = -cfg.gravity_z = 2.7
_GAMMA = 5.0 / 3.0


def _layer_profile_at_y(y, g, gamma_idx):
    z_lo = _VT["z_bounds"][gamma_idx]
    Gamma = _VT["Gamma"][gamma_idx]
    rho0 = _VT["rho_top"][gamma_idx]
    P0 = _VT["P_top"][gamma_idx]
    eta = 1.0 - ((Gamma - 1.0) / Gamma) * rho0 * g / P0 * (y - z_lo)
    eta = np.maximum(eta, 1e-8)
    rho = rho0 * eta ** (1.0 / (Gamma - 1.0))
    p = P0 * eta ** (Gamma / (Gamma - 1.0))
    return rho, p


def equilibrium_1d(y, g=2.7):
    """1D rho_eq(y), p_eq(y) using V&T's 3-layer atmosphere."""
    z1, z2 = _VT["z_bounds"][1], _VT["z_bounds"][2]
    rho_0, p_0 = _layer_profile_at_y(y, g, 0)
    rho_1, p_1 = _layer_profile_at_y(y, g, 1)
    rho_2, p_2 = _layer_profile_at_y(y, g, 2)
    rho = np.where(y < z1, rho_0, np.where(y < z2, rho_1, rho_2))
    p   = np.where(y < z1, p_0,   np.where(y < z2, p_1,   p_2))
    return rho, p


def _snapshot_at_index(run_dir, idx):
    """Return path to snapdir_{idx:03d} if it exists, else None."""
    if not os.path.isdir(run_dir):
        return None
    p = os.path.join(run_dir, f"snapdir_{idx:03d}")
    return p if os.path.isdir(p) else None


# t_end=1000, time_bet_snaps = t_end/20 = 50  -> t=100 is snapdir_002.
_SNAP_IDX = 2


def _load_run(solver, Nx, snapshots_root):
    run_dir = os.path.join(snapshots_root, f"solar_vt_{solver}_Nx{Nx}")
    snapdir = _snapshot_at_index(run_dir, _SNAP_IDX)
    if snapdir is None:
        return None
    snap = read_snapshot(snapdir)
    rho = snap["rho"]                            # shape (Nx, Ny)
    p   = snap["P"]
    T   = p / rho                                # code units
    y   = snap["y"]
    rho_eq, p_eq = equilibrium_1d(y, g=-_GRAVITY)
    T_eq = p_eq / rho_eq
    drho_rel = (rho - rho_eq[None, :]) / rho_eq[None, :]
    dT_rel   = (T   - T_eq[None, :])   / T_eq[None, :]
    return {
        "drho_rel": drho_rel,
        "dT_rel":   dT_rel,
        "t":        snap["t"],
        "Lx":       snap["Lx"],
        "Ly":       snap["Ly"],
    }


def _plot_grid(field_key, vmin, vmax, label, outpath_stem, runs):
    # Box is 5.5 wide x 11 tall; each panel ~1:2. 3 cols x 2 rows of 1:2
    # boxes -> figure aspect width:height ~ 3 : 4 (with cbar and titles).
    fig, axes = plt.subplots(2, 3, figsize=(9, 12), sharex=True, sharey=True,
                             constrained_layout=True)
    cmap = plt.get_cmap("RdBu_r")
    solvers = ["hllc", "lhllc"]
    resolutions = [1024, 2048, 4096]
    im = None
    for i_row, sol in enumerate(solvers):
        for j_col, Nx in enumerate(resolutions):
            ax = axes[i_row, j_col]
            run = runs.get((sol, Nx))
            if run is None:
                ax.text(0.5, 0.5, f"(no data yet)\n{sol.upper()} Nx={Nx}",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=12, color="0.4")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            data = run[field_key].T          # transpose so y is vertical
            extent = [0, run["Lx"], 0, run["Ly"]]
            im = ax.imshow(data, origin="lower", extent=extent,
                           cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal",
                           interpolation="nearest")
            ax.set_title(f"{sol.upper()}  Nx={Nx}  t={run['t']:.0f}", fontsize=11)
            if i_row == 1:
                ax.set_xlabel("x")
            if j_col == 0:
                ax.set_ylabel("y")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85,
                            label=label, pad=0.02)

    for ext in ("pdf", "png"):
        path = f"{outpath_stem}.{ext}"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def main():
    snapshots_root = "snapshots"
    runs = {}
    for sol in ("hllc", "lhllc"):
        for Nx in (1024, 2048, 4096):
            run = _load_run(sol, Nx, snapshots_root)
            if run is not None:
                runs[(sol, Nx)] = run
                print(f"  loaded {sol} Nx={Nx} (t={run['t']:.1f})")
            else:
                print(f"  no data:  {sol} Nx={Nx}")

    if not runs:
        raise RuntimeError("no runs found under snapshots/solar_vt_*")

    _plot_grid("drho_rel", -0.05, 0.05,
               r"$\delta\rho / \rho_{\rm eq}$",
               "figures/solar_vt_drho_grid", runs)
    _plot_grid("dT_rel", -0.05, 0.05,
               r"$\delta T / T_{\rm eq}$",
               "figures/solar_vt_dT_grid", runs)


if __name__ == "__main__":
    main()
