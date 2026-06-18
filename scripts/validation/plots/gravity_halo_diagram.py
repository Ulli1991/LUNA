"""Schematic of LUNA's gravity halo exchange, paralleling the hydro halo figure.

A pedagogical 4-panel diagram (not a data plot) showing how the gravity solvers'
communication differs from the single read-halo of one hydro step:

  (a) multigrid     -- one Poisson solve is a V-cycle over a grid hierarchy, and
                       every level halo-exchanges (smoother, residual, prolong):
                       many scalar-field exchanges per solve, not one per step.
  (b) CIC deposit   -- a boundary particle's cloud straddles the edge; the
                       overflow mass is summed (scatter-ADD) onto the neighbour's
                       edge cell -- the reverse of the read halo.
  (c) force gather  -- the 4th-order force stencil reaches two cells across the
                       boundary, so the gather reads a 2-cell halo (vs 1 for gas).
  (d) particle swap -- under the slab decomposition, particles that drift out of
                       a slab are exchanged with the ring neighbour in fixed-size
                       buffers; fluid cells never move between devices.

Writes figures/halo_exchange_gravity.png. No data dependency.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

BLUE = "#1f77b4"      # x-direction ring shift
RED = "#d62728"       # additive deposit / particle motion
INTER = "#cfe9cf"     # interior (never communicated)
XG = "#bcd6f2"        # 1-cell ghost
XG2 = "#7fb0e0"       # 2nd ghost cell (4th-order)
YG = "#f4cccc"        # coarse-level tint
HI = "#ffe9b3"        # highlighted tile / particle


def _box(ax, x, y, w, h, fc, ec="black", lw=1.2, hatch=None, z=2):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                           linewidth=lw, hatch=hatch, zorder=z))


def _arr(ax, p0, p1, color, rad=0.0, lw=1.8, style="-|>", z=4):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle=style, mutation_scale=13,
                                 lw=lw, color=color, zorder=z))


def _grid(ax, x0, y0, cw, cells):
    for c in cells:
        col, row, fc = c[0], c[1], c[2]
        hatch = c[3] if len(c) > 3 else None
        _box(ax, x0 + col * cw, y0 + row * cw, cw, cw, fc, ec="0.55",
             lw=0.7, hatch=hatch)


def _ring_grid(ax, x0, y0, cw, n, ghost=1, hatch=True):
    """n x n interior with a `ghost`-cell ring; returns nothing, draws cells."""
    tot = n + 2 * ghost
    cells = []
    for c in range(tot):
        for r in range(tot):
            edge = min(c, r, tot - 1 - c, tot - 1 - r)
            if edge >= ghost:
                fc, h = INTER, None
            elif edge == 0 and ghost == 2:
                fc, h = XG2, ("////" if hatch else None)
            else:
                fc, h = XG, ("////" if hatch else None)
            cells.append((c, r, fc, h))
    _grid(ax, x0, y0, cw, cells)


def _header(ax, s):
    ax.text(0.02, 0.97, s, transform=ax.transAxes, va="top", ha="left",
            fontsize=10.5, fontweight="bold")


# ----------------------------------------------------------------------------
def panel_a(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    _header(ax, "(a) multigrid: a halo at every level of the V-cycle")
    # three levels, decreasing size, each with a 1-cell ghost ring
    specs = [(0.9, 4, 0.30, "fine"), (4.0, 3, 0.34, "coarse"),
             (6.9, 2, 0.40, "coarsest")]
    centres = []
    yc = 5.2
    for (x0, n, cw, lab) in specs:
        tot = (n + 2) * cw
        y0 = yc - tot / 2
        _ring_grid(ax, x0, y0, cw, n, ghost=1, hatch=False)
        centres.append((x0 + tot / 2, yc, y0, y0 + tot))
        ax.text(x0 + tot / 2, y0 - 0.4, lab, ha="center", fontsize=8, color="0.35")
    # restrict (down/right) along the bottom, prolong (up/left) along the top
    for i in range(2):
        c0, c1 = centres[i], centres[i + 1]
        _arr(ax, (c0[0] + 0.55, c0[2] - 0.05), (c1[0] - 0.55, c1[2] - 0.05),
             "0.4", rad=-0.3, lw=1.5)
        _arr(ax, (c1[0] - 0.55, c1[3] + 0.05), (c0[0] + 0.55, c0[3] + 0.05),
             RED, rad=-0.3, lw=1.5)
    ax.text(2.6, 7.0, "restrict", fontsize=8, color="0.4", ha="center")
    ax.text(5.4, 7.5, "prolong (halo-exchanged)", fontsize=8, color=RED, ha="center")
    ax.text(5, 2.6, "one Poisson solve is a V-cycle over the grid hierarchy; every "
            "level halo-\nexchanges in the smoother, the residual, and the "
            "prolongation.", ha="center", fontsize=8.2, color="0.3")
    ax.text(5, 1.3, "→ many scalar-field exchanges per solve, versus a single "
            "fluid-array\nexchange per hydrodynamic step.", ha="center",
            fontsize=8.2, color=BLUE, style="italic")


def panel_b(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    _header(ax, "(b) CIC deposit: an additive (scatter) halo")
    cw = 0.9
    x0, y0 = 1.2, 3.2
    # device i (cols 0-2) and device i+1 (cols 3-5), boundary between col 2|3
    for c in range(6):
        for r in range(4):
            fc = "#eef4fb" if c < 3 else "#fbeeee"
            _box(ax, x0 + c * cw, y0 + r * cw, cw, cw, fc, ec="0.6", lw=0.8)
    bx = x0 + 3 * cw
    ax.plot([bx, bx], [y0, y0 + 4 * cw], color="0.2", lw=2.0, zorder=3)
    ax.text(x0 + 1.5 * cw, y0 + 4 * cw + 0.3, "device $i$", ha="center", fontsize=9)
    ax.text(x0 + 4.5 * cw, y0 + 4 * cw + 0.3, "device $i{+}1$", ha="center", fontsize=9)
    # particle near the edge; its 2x2 CIC cloud straddles the boundary
    px, py = x0 + 2.7 * cw, y0 + 2.3 * cw
    ax.add_patch(Circle((px, py), 0.13, color=HI, ec="black", lw=1.0, zorder=5))
    for (cc, rr) in [(2, 2), (2, 1), (3, 2), (3, 1)]:
        _box(ax, x0 + cc * cw, y0 + rr * cw, cw, cw, "none", ec=RED, lw=1.8, z=4)
    # overflow contribution arrow crossing the boundary
    _arr(ax, (px, py), (bx + 0.55 * cw, py), RED, lw=2.0)
    ax.text(bx + 0.6 * cw, py + 0.32, r"$+\!\!=$ mass", color=RED, fontsize=9, va="bottom")
    ax.text(5, 1.7, "the cloud of a boundary particle straddles the edge; the "
            "overflow mass\nis summed onto the neighbour's edge cell "
            r"($\mathtt{ppermute}$ then add).", ha="center", fontsize=8.2, color="0.3")
    ax.text(5, 0.6, "the deposit halo flows OUTWARD (scatter-add) — the reverse "
            "of the read halo.", ha="center", fontsize=8.2, color=RED, style="italic")


def panel_c(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    _header(ax, "(c) force gather: a two-cell halo")
    cw = 0.7
    x0, y0 = 1.0, 5.2
    labels = ["$i{-}2$", "$i{-}1$", "$i$", "$i{+}1$", "$i{+}2$"]
    fcs = [INTER, INTER, INTER, XG, XG2]
    for k in range(5):
        _box(ax, x0 + k * cw, y0, cw, cw, fcs[k], ec="0.55", lw=0.9,
             hatch="////" if k >= 3 else None)
        ax.text(x0 + (k + 0.5) * cw, y0 + cw + 0.22, labels[k], ha="center", fontsize=8.5)
    bx = x0 + 3 * cw
    ax.plot([bx, bx], [y0 - 0.5, y0 + cw + 0.5], color="0.2", lw=2.0, zorder=3)
    ax.text(bx, y0 - 0.85, "subdomain boundary", ha="center", fontsize=7.8, color="0.3")
    ax.text(x0 + 1.5 * cw, y0 + cw + 0.95, "interior", ha="center", fontsize=8, color="0.35")
    ax.text(x0 + 4.0 * cw, y0 + cw + 0.95, "2-cell ghost", ha="center", fontsize=8, color=BLUE)
    ax.text(5, 3.4, r"$\partial_x\Phi_i \approx (\Phi_{i-2} - 8\Phi_{i-1} + "
            r"8\Phi_{i+1} - \Phi_{i+2})/12\Delta x$", ha="center", fontsize=9.5)
    ax.text(5, 2.1, "the 4th-order force stencil reaches two cells past the "
            "boundary, so the\ngather reads a 2-cell halo — wider than the gas "
            "stencil's single cell.", ha="center", fontsize=8.2, color="0.3")


def panel_d(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    _header(ax, "(d) particle exchange on drift (slab decomposition)")
    sw, y0, h = 2.0, 3.4, 3.6
    x0 = 1.0
    for i in range(4):
        _box(ax, x0 + i * sw, y0, sw, h, "#eef1f4", ec="0.5", lw=1.0)
        ax.text(x0 + (i + 0.5) * sw, y0 + h + 0.3, f"slab {i}", ha="center", fontsize=8.5)
    ax.text(5.0, y0 - 0.5, r"grid + particles split along $x$", ha="center",
            fontsize=8, color="0.35")
    # a particle drifting from slab 1 into slab 2
    p0 = (x0 + 1.6 * sw, y0 + 2.0)
    p1 = (x0 + 2.4 * sw, y0 + 2.0)
    ax.add_patch(Circle(p0, 0.13, color=HI, ec="black", lw=1.0, zorder=5))
    _arr(ax, (p0[0] + 0.15, p0[1]), (p1[0] - 0.15, p1[1]), RED, lw=2.0)
    ax.add_patch(Circle(p1, 0.13, color=HI, ec="black", lw=1.0, zorder=5))
    ax.text((p0[0] + p1[0]) / 2, p0[1] + 0.3, "drift", ha="center", fontsize=8, color=RED)
    # ppermute send to ring neighbour
    bx = x0 + 2 * sw
    ax.plot([bx, bx], [y0, y0 + h], color="0.2", lw=2.0, zorder=3)
    ax.text(5, 1.9, "a particle leaving its slab is shipped to the ring neighbour "
            r"($\mathtt{ppermute}$)" "\nwithin a fixed-size, mask-padded buffer.",
            ha="center", fontsize=8.2, color="0.3")
    ax.text(5, 0.7, "the fluid never needs this — its cells never move between "
            "devices.", ha="center", fontsize=8.2, color=BLUE, style="italic")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4))
    panel_a(axes[0, 0])
    panel_b(axes[0, 1])
    panel_c(axes[1, 0])
    panel_d(axes[1, 1])
    fig.tight_layout(pad=1.5)
    os.makedirs("figures", exist_ok=True)
    out = "figures/halo_exchange_gravity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
