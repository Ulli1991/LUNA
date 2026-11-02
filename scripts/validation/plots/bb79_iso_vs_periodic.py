"""BB79 256³ MinMod: isolated BC vs periodic BC diagnostic.

Side-by-side log column density at the last good snap. Same hydro,
same limiter, same resolution — only the BC differs. If the cardinal-
axis lines disappear with periodic BC, they came from our isolated
multipole-Dirichlet implementation. If they stay, it's the HLLC 3D
carbuncle.
"""
import numpy as np
from scripts.validation.plots._plotkit import mpl, save, load_snap, latest_clean

plt = mpl()
from luna.utils.figstyle import FIGSIZE

LABEL_FS = 11   # in-panel image annotations only (axis labels inherit rcParams)

ISO_DIR  = "snapshots/bb_iso_minmod_Nx256"
PER_DIR  = "snapshots/bb_periodic_minmod_Nx256"


def col_density(snapdir):
    data, a = load_snap(snapdir)
    return data["density"].sum(axis=2) * float(a["dx"]), float(a["Lx"]), float(a["time"])


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
    axes[0].text(0.05, 0.92, f"isolated BC (outflow + multipole-Dirichlet) — t={t_iso:.2f}",
                 transform=axes[0].transAxes, fontsize=LABEL_FS, color="white", fontweight="bold",
                 bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=1.5))
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0], shrink=0.85, label=r"$\log_{10}\Sigma$")

    im1 = axes[1].imshow(np.log10(Sp_per).T, origin="lower",
                          extent=(0, L_per, 0, L_per),
                          vmin=vmin, vmax=vmax, cmap="magma")
    axes[1].grid(False)
    axes[1].text(0.05, 0.92, f"periodic BC (Jeans swindle) — t={t_per:.2f}",
                 transform=axes[1].transAxes, fontsize=LABEL_FS, color="white", fontweight="bold",
                 bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=1.5))
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1], shrink=0.85, label=r"$\log_{10}\Sigma$")

    fig.tight_layout()
    Nx_luna = S_iso.shape[0]   # column density is (Nx, Ny)
    out = f"code_paper_figs/bb79_iso_vs_periodic_diagnostic_Nx{Nx_luna}.png"
    save(fig, out, close=True)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
