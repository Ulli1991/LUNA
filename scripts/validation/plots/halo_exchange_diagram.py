"""Schematic of LUNA's shard_map halo exchange (explicit jax.lax.ppermute).

A pedagogical diagram, not a data plot:
  top    -- N device tiles split along one mesh axis, with the two ppermute
            ring shifts (send right edge -> i+1, send left edge -> i-1) and the
            periodic wrap drawn as arrows.
  bottom -- zoom of one shard: interior (never communicated) + the 1-cell ghost
            layers filled by the neighbour edge slabs, plus the MPI equivalence.

Writes figures/halo_exchange_shardmap.png. No data dependency.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

BLUE = "#1f77b4"     # rightward shift  -> fills LEFT ghost
RED = "#d62728"      # leftward shift   -> fills RIGHT ghost
INTER = "#cfe3f7"    # interior
LGHOST = "#bcd6f2"   # left ghost (from left neighbour)
RGHOST = "#f4cccc"   # right ghost (from right neighbour)


def _box(ax, x, y, w, h, fc, ec="black", lw=1.3, hatch=None, z=2):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                           linewidth=lw, hatch=hatch, zorder=z))


def _arrow(ax, p0, p1, color, rad, lw=2.0, z=4):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=16,
                                 lw=lw, color=color, zorder=z))


def main():
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---------------- TOP: ring of N=4 device tiles along axis 'x' ----------
    N = 4
    lefts = [9, 31, 53, 75]
    w, ty, th, gs = 15, 70, 15, 1.5
    for i, x in enumerate(lefts):
        _box(ax, x, ty, w, th, INTER)
        # thin ghost strips on each side to hint at the halo
        _box(ax, x - gs, ty, gs, th, LGHOST, lw=0.8)
        _box(ax, x + w, ty, gs, th, RGHOST, lw=0.8)
        ax.text(x + w / 2, ty + th / 2, f"GPU {i}\n(x-slot {i})",
                ha="center", va="center", fontsize=10, zorder=3)

    ax.text(50, 55, r"global $x$-axis split across $N=4$ GPUs (mesh axis "
            r"'$x$'); edges wrap periodically", ha="center", fontsize=9,
            style="italic", color="0.3")

    # rightward shifts (blue): device i RIGHT edge -> device i+1 LEFT ghost
    for i in range(N - 1):
        _arrow(ax, (lefts[i] + w + gs, ty + th), (lefts[i + 1] - gs, ty + th),
               BLUE, rad=-0.5)
    ax.text(50, 95, r"$\mathtt{perm\_to\_right}=[(i,(i{+}1)\,\%\,N)]$:  "
            "each GPU sends its RIGHT edge slab to neighbour $i{+}1$ "
            r"$\rightarrow$ their LEFT ghost",
            ha="center", fontsize=9.5, color=BLUE)

    # leftward shifts (red): device i LEFT edge -> device i-1 RIGHT ghost
    for i in range(1, N):
        _arrow(ax, (lefts[i] - gs, ty), (lefts[i - 1] + w + gs, ty),
               RED, rad=-0.5)
    ax.text(50, 61.5, r"$\mathtt{perm\_to\_left}=[(i,(i{-}1)\,\%\,N)]$:  "
            "each GPU sends its LEFT edge slab to neighbour $i{-}1$ "
            r"$\rightarrow$ their RIGHT ghost",
            ha="center", fontsize=9.5, color=RED)

    # divider
    ax.plot([3, 97], [50, 50], color="0.85", lw=1.0, zorder=1)

    # ---------------- BOTTOM: zoom of one shard ----------------------------
    zx, zy, zw, zh = 14, 12, 42, 28
    ncol, nrow = 8, 4
    cw = zw / (ncol + 2)        # +2 for the two ghost columns
    cy = zy + zh / 2
    # left ghost column
    _box(ax, zx, zy, cw, zh, LGHOST, hatch="////", lw=1.0)
    # interior grid
    for c in range(ncol):
        for r in range(nrow):
            _box(ax, zx + cw * (c + 1), zy + zh / nrow * r, cw, zh / nrow,
                 INTER, ec="0.7", lw=0.6)
    # right ghost column
    _box(ax, zx + cw * (ncol + 1), zy, cw, zh, RGHOST, hatch="\\\\\\\\", lw=1.0)

    ax.text(zx + zw / 2, zy + zh + 3, "one shard's local array",
            ha="center", fontsize=9.5, color="0.3", style="italic")
    ax.text(zx + cw * (1 + ncol / 2), cy, "interior\n(never moves)",
            ha="center", va="center", fontsize=9)

    # incoming slabs (neighbour edges arriving via ppermute)
    _box(ax, 3, cy - 2.2, cw, 4.4, LGHOST, hatch="////", lw=1.0)
    _arrow(ax, (3 + cw, cy), (zx, cy), BLUE, rad=0.0)
    ax.text(3 + cw / 2, cy + 4, "left nbr's\nright edge",
            ha="center", va="bottom", fontsize=8, color=BLUE)
    _box(ax, zx + zw + 4, cy - 2.2, cw, 4.4, RGHOST, hatch="\\\\\\\\", lw=1.0)
    _arrow(ax, (zx + zw + 4, cy), (zx + zw, cy), RED, rad=0.0)
    ax.text(zx + zw + 4 + cw / 2, cy + 4, "right nbr's\nleft edge",
            ha="center", va="bottom", fontsize=8, color=RED)

    ax.annotate("1-cell ghost layer", (zx + cw / 2, zy), (zx - 6, zy - 7),
                fontsize=8, color=BLUE,
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.0))

    # explanation panel (right)
    tx = 70
    ax.text(tx, zy + zh + 3, r"$\mathtt{ppermute(slab,\ 'x',\ perm)}$",
            fontsize=10, color="black", family="monospace")
    ax.text(tx, zy + zh - 2,
            r"$\equiv$ MPI_Sendrecv on a ring,"
            "\n    but as ONE collective.\n\n"
            "Only the 1-cell slab crosses\n"
            "the wire ($\\sim$5 MB / axis / GPU\n"
            "at $1024^3$ on 8 GPUs); the\n"
            "interior never moves.\n\n"
            "A traced JAX op, so:\n"
            r"  $\bullet$ XLA fuses it w/ the stencil" "\n"
            r"  $\bullet$ reverse-mode AD flows" "\n"
            "    straight through it.",
            fontsize=8.6, va="top", color="0.15")

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out = "figures/halo_exchange_shardmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
