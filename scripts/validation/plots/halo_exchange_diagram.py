"""Schematic of LUNA's shard_map halo exchange (explicit jax.lax.ppermute).

A pedagogical 4-panel diagram (not a data plot), no memory-size numbers:

  (a) decomposition  -- global grid -> device mesh; each GPU owns one tile and
                        needs a halo from every face-neighbour.
  (b) ring ppermute  -- halo fill along one axis = two ring shifts; repeated x,y,z.
  (c) sequential pad -- axes padded x -> y -> z so corner ghosts fill correctly.
  (d) one shard      -- interior never moves; only boundary slabs cross the wire;
                        ppermute is a traced JAX op (fused + differentiable),
                        unlike an opaque MPI_Send.

Writes figures/halo_exchange_shardmap.png. No data dependency.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

BLUE = "#1f77b4"      # x-direction ring shift / x ghosts
RED = "#d62728"       # y-direction ring shift / y ghosts
INTER = "#cfe9cf"     # interior (never communicated)
XG = "#bcd6f2"        # x ghost
YG = "#f4cccc"        # y ghost
CORNER = "#ddd0ec"    # corner ghost
HI = "#ffe9b3"        # highlighted tile


def _box(ax, x, y, w, h, fc, ec="black", lw=1.2, hatch=None, z=2):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                           linewidth=lw, hatch=hatch, zorder=z))


def _arr(ax, p0, p1, color, rad=0.0, lw=1.8, style="-|>", z=4):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle=style, mutation_scale=13,
                                 lw=lw, color=color, zorder=z))


def _grid(ax, x0, y0, cw, cells):
    """cells: list of (col,row,facecolor[,hatch]); origin bottom-left."""
    for c in cells:
        col, row, fc = c[0], c[1], c[2]
        hatch = c[3] if len(c) > 3 else None
        _box(ax, x0 + col * cw, y0 + row * cw, cw, cw, fc, ec="0.55",
             lw=0.7, hatch=hatch)


def _header(ax, s):
    ax.text(0.02, 0.97, s, transform=ax.transAxes, va="top", ha="left",
            fontsize=10.5, fontweight="bold")


# ----------------------------------------------------------------------------
def panel_a(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    _header(ax, "(a) decomposition: global grid → device mesh")
    tw, gap, x0, y0 = 2.3, 0.25, 1.6, 1.4
    for r in range(3):
        for c in range(3):
            x = x0 + c * (tw + gap); y = y0 + r * (tw + gap)
            mid = (c == 1 and r == 1)
            _box(ax, x, y, tw, tw, HI if mid else "#eef1f4")
            if mid:
                ax.text(x + tw / 2, y + tw / 2, "this\nGPU", ha="center",
                        va="center", fontsize=9)
    # face-neighbour halo arrows (double headed) from centre tile
    cx = x0 + 1 * (tw + gap) + tw / 2
    cy = y0 + 1 * (tw + gap) + tw / 2
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        _arr(ax, (cx + dx * 1.0, cy + dy * 1.0),
             (cx + dx * (tw + gap - 0.2), cy + dy * (tw + gap - 0.2)),
             "0.35", style="<|-|>", lw=1.6)
    ax.text(5, 0.7, "each GPU owns one tile; halo needed from every "
            "face-neighbour (6 in 3D).", ha="center", fontsize=8.4,
            color="0.3")
    ax.text(5, 0.05, r"split is a pencil / slab / 3D block by GPU count "
            r"($\mathtt{LUNA\_MESH\_SHAPE}$); an undivided axis $\to$ no-op shift",
            ha="center", fontsize=7.8, color="0.45")


def panel_b(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    _header(ax, "(b) halo fill along one axis = two ring shifts")
    lefts = [0.6, 3.0, 5.4, 7.8]
    w, ty, th, gs = 1.7, 4.2, 2.4, 0.28
    for i, x in enumerate(lefts):
        _box(ax, x, ty, w, th, INTER)
        _box(ax, x - gs, ty, gs, th, XG, lw=0.7)
        _box(ax, x + w, ty, gs, th, XG, lw=0.7)
        ax.text(x + w / 2, ty + th / 2, f"{i}", ha="center", va="center",
                fontsize=10)
    for i in range(len(lefts) - 1):
        _arr(ax, (lefts[i] + w + gs, ty + th), (lefts[i + 1] - gs, ty + th),
             BLUE, rad=-0.55)
        _arr(ax, (lefts[i + 1] - gs, ty), (lefts[i] + w + gs, ty),
             BLUE, rad=-0.55)
    ax.text(5, 8.5, r"$\mathtt{ppermute(right\_edge,'x',i{\to}i{+}1)}$"
            " : fills left ghost", ha="center", fontsize=8.2, color=BLUE)
    ax.text(5, 1.7, r"$\mathtt{ppermute(left\_edge,'x',i{\to}i{-}1)}$"
            " : fills right ghost", ha="center", fontsize=8.2, color=BLUE)
    ax.text(5, 0.5, "edges wrap periodically; one such ring per axis (x, y, z)",
            ha="center", fontsize=8.4, color="0.3", style="italic")


def panel_c(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    _header(ax, "(c) axes padded in sequence (x → y → z)")
    cw = 0.5
    # stage 1: bare interior 3x3
    s1 = [(c, r, INTER) for c in range(3) for r in range(3)]
    _grid(ax, 0.6, 4.0, cw, s1)
    # stage 2: + x ghosts (left/right cols)
    s2 = []
    for c in range(5):
        for r in range(3):
            fc = XG if c in (0, 4) else INTER
            s2.append((c, r, fc, "////" if c in (0, 4) else None))
    _grid(ax, 3.0, 4.0, cw, s2)
    # stage 3: + y ghosts (top/bottom rows); corners now filled
    s3 = []
    for c in range(5):
        for r in range(5):
            if r in (0, 4) and c in (0, 4):
                fc, h = CORNER, "xx"
            elif r in (0, 4):
                fc, h = YG, "\\\\"
            elif c in (0, 4):
                fc, h = XG, "////"
            else:
                fc, h = INTER, None
            s3.append((c, r, fc, h))
    _grid(ax, 6.6, 3.5, cw, s3)
    _arr(ax, (2.3, 4.75), (2.9, 4.75), "0.35", lw=1.6)
    ax.text(2.6, 5.4, "pad x", ha="center", fontsize=8, color="0.3")
    _arr(ax, (5.7, 4.75), (6.5, 4.75), "0.35", lw=1.6)
    ax.text(6.1, 5.4, "pad y", ha="center", fontsize=8, color="0.3")
    ax.text(5, 1.7, "2D slice shown: the y-pad fills the corner ghosts "
            "(purple) for free\nbecause x is already padded. In 3D the same "
            "x→y→z chain fills all\nedge + corner ghosts — no diagonal "
            "messages needed.", ha="center", fontsize=8.0, color="0.3")


def panel_d(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    _header(ax, "(d) one shard: what moves, and why it is special")
    cw, x0, y0 = 0.62, 0.7, 3.0
    cells = []
    for c in range(5):
        for r in range(5):
            if r in (0, 4) and c in (0, 4):
                fc, h = CORNER, "xx"
            elif r in (0, 4):
                fc, h = YG, "\\\\"
            elif c in (0, 4):
                fc, h = XG, "////"
            else:
                fc, h = INTER, None
            cells.append((c, r, fc, h))
    _grid(ax, x0, y0, cw, cells)
    ax.text(x0 + 2.5 * cw, y0 + 2.5 * cw, "interior\n(never\nmoves)",
            ha="center", va="center", fontsize=8)
    ax.text(x0 + 2.5 * cw, y0 + 5 * cw + 0.25, "1-cell ghost ring",
            ha="center", fontsize=8, color="0.3")
    # arrow hinting a slab arriving (representative)
    _arr(ax, (x0 - 0.9, y0 + 2.5 * cw), (x0, y0 + 2.5 * cw), BLUE)

    tx = 4.7
    ax.text(tx, 7.6, r"$\mathtt{ppermute}\ \equiv$ MPI_Sendrecv on a ring,",
            fontsize=9, va="top")
    ax.text(tx, 6.9, "but expressed as one collective.", fontsize=9, va="top")
    ax.text(tx, 5.7, "Only the thin boundary slabs cross\nthe wire; the bulk "
            "interior stays put.", fontsize=9, va="top")
    ax.text(tx, 3.9, "Unlike an opaque MPI_Send, it is a\ntraced JAX op, so:",
            fontsize=9, va="top")
    ax.text(tx, 2.4, r"$\bullet$ XLA fuses it with the stencil math"
            "\n" r"$\bullet$ reverse-mode AD flows straight" "\n"
            "   through it (distributed AND\n   differentiable).",
            fontsize=8.6, va="top", color="0.15")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4))
    panel_a(axes[0, 0])
    panel_b(axes[0, 1])
    panel_c(axes[1, 0])
    panel_d(axes[1, 1])
    fig.tight_layout(pad=1.5)
    os.makedirs("figures", exist_ok=True)
    out = "figures/halo_exchange_shardmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
