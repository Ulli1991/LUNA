"""Gresho vortex radial v_phi diagnostic.

Two-panel plot: v_phi(r) at an early time (t ≈ 0.5) and a late time
(t = 3.0) for the Gresho HLLC resolution scan (Nx ∈ {256, 512, 1024, 2048}).
The analytic Gresho profile is piecewise-linear with a peak at r=0.2 and
v_phi=0 outside r=0.4; HLLC dissipates this peak due to its low-Mach
over-dissipation. Resolution sweep shows convergence behaviour.

Output: figures/gresho_vphi_vs_radius.{pdf,png}
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from file_io.read_snapshot import read_snapshot
from utils.figstyle import apply_style, decorate, FIGSIZE

apply_style()


RESOLUTIONS = [256, 512, 1024, 2048]
# t_end=3.0, time_bet_snaps=0.3 -> snapdir_NN at t = 0.3*NN
SNAP_EARLY = 2     # t = 0.6 (closest to t=0.5)
SNAP_LATE = 10     # t = 3.0


def _load_vphi(Nx, snap_idx, root="snapshots"):
    d = os.path.join(root, f"gresho_hllc_Nx{Nx}_M0.01", f"snapdir_{snap_idx:03d}")
    if not os.path.isdir(d):
        return None
    snap = read_snapshot(d)
    vx = snap["vx"]; vy = snap["vy"]
    Lx = snap["Lx"]; Ly = snap["Ly"]
    dx = Lx / Nx; dy = Ly / Nx
    x = (np.arange(Nx) + 0.5) * dx - 0.5 * Lx
    y = (np.arange(Nx) + 0.5) * dy - 0.5 * Ly
    X, Y = np.meshgrid(x, y, indexing="ij")
    R = np.sqrt(X * X + Y * Y)
    R_safe = np.where(R > 0, R, 1.0)
    # v_phi = phi-hat . v = (-Y vx + X vy) / R
    v_phi = (-Y * vx + X * vy) / R_safe

    n_bins = 50
    edges = np.linspace(0, 0.5, n_bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    R_flat = R.ravel(); vphi_flat = v_phi.ravel()
    counts, _ = np.histogram(R_flat, bins=edges)
    counts = np.maximum(counts, 1)
    sums, _ = np.histogram(R_flat, bins=edges, weights=vphi_flat)
    sums2, _ = np.histogram(R_flat, bins=edges, weights=vphi_flat * vphi_flat)
    mean = sums / counts
    var = sums2 / counts - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))
    return {"r": centers, "vphi_mean": mean, "vphi_std": std, "t": snap["t"]}


def _vphi_analytic(r):
    return np.where(r < 0.2, 5.0 * r,
            np.where(r < 0.4, 2.0 - 5.0 * r, 0.0))


def _plot_panel(ax, snap_idx, title):
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(RESOLUTIONS)))
    r_fine = np.linspace(0, 0.5, 200)
    ax.plot(r_fine, _vphi_analytic(r_fine), color="0.5", ls="-", lw=1.0,
            label="analytic")
    for i, Nx in enumerate(RESOLUTIONS):
        d = _load_vphi(Nx, snap_idx)
        if d is None:
            print(f"  no data: Nx={Nx} snap={snap_idx}")
            continue
        ax.plot(d["r"], d["vphi_mean"], color=cmap[i], lw=2, label=f"Nx={Nx}")
        ax.fill_between(d["r"], d["vphi_mean"] - d["vphi_std"],
                        d["vphi_mean"] + d["vphi_std"], color=cmap[i], alpha=0.15)
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("r")
    ax.set_ylabel(r"$v_\phi$")
    # time label as in-axes text (top-left is empty for small r), not a title
    ax.text(0.04, 0.96, title, transform=ax.transAxes, va="top", ha="left",
            fontsize=10)
    ax.legend(loc="upper right", fontsize=9, handlelength=1.4, labelspacing=0.3)
    decorate(ax)


def main(outpath_stem="figures/gresho_vphi_vs_radius"):
    fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE[0]*2, FIGSIZE[1]), constrained_layout=True)
    d_early = _load_vphi(RESOLUTIONS[0], SNAP_EARLY)
    d_late  = _load_vphi(RESOLUTIONS[0], SNAP_LATE)
    t_early = f"{d_early['t']:.2f}" if d_early else "?"
    t_late  = f"{d_late['t']:.2f}"  if d_late  else "?"
    _plot_panel(axes[0], SNAP_EARLY, f"t = {t_early} (early)")
    _plot_panel(axes[1], SNAP_LATE,  f"t = {t_late} (late)")

    for ext in ("pdf", "png"):
        path = f"{outpath_stem}.{ext}"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
