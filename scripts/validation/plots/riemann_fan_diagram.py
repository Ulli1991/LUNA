"""Schematic x-t wave fan contrasting the HLL (two-wave) and HLLC (three-wave)
approximate Riemann solvers, for the MUSCL-Hancock section.

(a) HLL collapses the Riemann fan to two acoustic waves S_L, S_R with a single
    constant intermediate ("star") state, so the contact discontinuity is not
    represented and is smeared.
(b) HLLC restores the middle (contact) wave S_M, giving two star states
    U*_L, U*_R either side of the contact -- contacts and shear waves stay sharp.

No data dependency. Writes figures/riemann_fan_hll_hllc.png.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch

UL = "#cfe3f7"     # left state U_L  (blue)
UR = "#f4cccc"     # right state U_R (red)
USL = "#bcd6f2"    # star-left  U*_L
USR = "#f7d9b0"    # star-right U*_R
UHLL = "#d9e8d2"   # HLL single star state U*

# wave endpoints at t = 1 (x positions); origin is the interface at t = 0
XL, XR, XM = -0.62, 0.80, 0.13


def _poly(ax, pts, color):
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="none",
                         zorder=1))


def _wave(ax, x_top, label, lx, color="black", lw=2.0, dashed=False):
    ax.plot([0, x_top], [0, 1], color=color, lw=lw,
            ls="--" if dashed else "-", zorder=3)
    ax.text(lx, 1.04, label, ha="center", va="bottom", fontsize=10, color=color)


def _axes(ax):
    ax.add_patch(FancyArrowPatch((-1, 0), (1.02, 0), arrowstyle="-|>",
                                 mutation_scale=12, color="0.4", lw=1.1, zorder=4))
    ax.add_patch(FancyArrowPatch((0, 0), (0, 1.12), arrowstyle="-|>",
                                 mutation_scale=12, color="0.4", lw=1.1, zorder=4))
    ax.text(1.04, -0.07, "$x$", fontsize=11, color="0.3")
    ax.text(-0.08, 1.12, "$t$", fontsize=11, color="0.3")


def panel_hll(ax):
    ax.set_xlim(-1, 1.12); ax.set_ylim(-0.26, 1.22); ax.axis("off")
    _poly(ax, [(-1, 0), (0, 0), (XL, 1), (-1, 1)], UL)           # U_L
    _poly(ax, [(0, 0), (XL, 1), (XR, 1)], UHLL)                  # U*
    _poly(ax, [(0, 0), (XR, 1), (1, 1), (1, 0)], UR)            # U_R
    _axes(ax)
    _wave(ax, XL, "$S_L$", XL - 0.05)
    _wave(ax, XR, "$S_R$", XR + 0.05)
    ax.text(-0.55, 0.45, "$U_L$", fontsize=12, ha="center")
    ax.text(0.07, 0.62, "$U^{*}$", fontsize=13, ha="center", fontweight="bold")
    ax.text(0.62, 0.45, "$U_R$", fontsize=12, ha="center")
    ax.text(0.06, -0.16, "(a) HLL: two waves, one star state $U^{*}$",
            ha="center", fontsize=10, fontweight="bold")


def panel_hllc(ax):
    ax.set_xlim(-1, 1.12); ax.set_ylim(-0.26, 1.22); ax.axis("off")
    _poly(ax, [(-1, 0), (0, 0), (XL, 1), (-1, 1)], UL)           # U_L
    _poly(ax, [(0, 0), (XL, 1), (XM, 1)], USL)                   # U*_L
    _poly(ax, [(0, 0), (XM, 1), (XR, 1)], USR)                   # U*_R
    _poly(ax, [(0, 0), (XR, 1), (1, 1), (1, 0)], UR)           # U_R
    _axes(ax)
    _wave(ax, XL, "$S_L$", XL - 0.05)
    _wave(ax, XM, "$S_M$ (contact)", XM + 0.02, color="#6a3d9a", dashed=True)
    _wave(ax, XR, "$S_R$", XR + 0.05)
    ax.text(-0.55, 0.45, "$U_L$", fontsize=12, ha="center")
    ax.text(-0.24, 0.66, "$U^{*}_{L}$", fontsize=12, ha="center",
            fontweight="bold")
    ax.text(0.44, 0.62, "$U^{*}_{R}$", fontsize=12, ha="center",
            fontweight="bold")
    ax.text(0.66, 0.45, "$U_R$", fontsize=12, ha="center")
    ax.text(0.06, -0.16, "(b) HLLC: three waves, contact $S_M$ resolved",
            ha="center", fontsize=10, fontweight="bold")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    panel_hll(axes[0])
    panel_hllc(axes[1])
    fig.tight_layout(pad=1.2)
    os.makedirs("figures", exist_ok=True)
    out = "figures/riemann_fan_hll_hllc.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
