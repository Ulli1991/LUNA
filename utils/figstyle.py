"""Shared paper-figure style for all LUNA validation / autodiff figures.

One consistent look across every paper figure:
  - tick LABELS and ticks point INWARD
  - ticks on ALL FOUR sides (top + right too), with minor ticks
  - a coarse (major-only) background grid
  - frameless, large legends
  - one canonical panel size so every figure has the same layout

Usage in a plotting script:

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.figstyle import apply_style, decorate, FIGSIZE

    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(...)
    ax.legend()          # picks up frameless / large style
    decorate(ax)         # 4-side inward ticks + coarse grid (belt & suspenders)
    fig.tight_layout()
    fig.savefig(...)

`decorate(ax)` is idempotent and also fixes axes created before apply_style()
(e.g. twin / secondary axes, or panels in a pre-existing figure).
"""
import matplotlib as mpl

# Canonical single-panel size — every figure uses this (or an (N*W, H) multiple
# for N-column multipanels) so the paper has a uniform layout.
# NOTE: figures carry NO titles — describe them in the caption, never set_title.
FIGSIZE = (6.4, 4.8)

# Grid styling, kept in one place so apply_style() and decorate() agree.
_GRID = dict(color="0.80", linewidth=0.6, linestyle="-", alpha=0.7)


def apply_style():
    """Set global rcParams for the LUNA paper figure style."""
    mpl.rcParams.update({
        # ---- figure / saving ----
        "figure.figsize": FIGSIZE,
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        # ---- fonts ----
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        # ---- ticks: inward, on all four sides, with minor ticks ----
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "xtick.bottom": True,
        "ytick.left": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 6.0,
        "ytick.major.size": 6.0,
        "xtick.minor.size": 3.0,
        "ytick.minor.size": 3.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.7,
        "ytick.minor.width": 0.7,
        # ---- coarse background grid (major lines only) ----
        "axes.grid": True,
        "axes.grid.which": "major",
        "axes.axisbelow": True,            # grid behind the data
        "grid.color": _GRID["color"],
        "grid.linewidth": _GRID["linewidth"],
        "grid.linestyle": _GRID["linestyle"],
        "grid.alpha": _GRID["alpha"],
        # ---- legend: large, no box ----
        "legend.frameon": False,
        "legend.fontsize": 13,
        "legend.handlelength": 1.8,
        "legend.labelspacing": 0.4,
        # ---- axes / lines ----
        "axes.linewidth": 1.1,
        "lines.linewidth": 1.8,
    })


def decorate(ax):
    """Per-axes enforcement of the style (idempotent).

    rcParams already cover axes created after apply_style(); this also fixes
    twin/secondary axes and guarantees the 4-side inward ticks + coarse grid
    regardless of creation order.
    """
    ax.tick_params(which="both", direction="in",
                   top=True, bottom=True, left=True, right=True)
    ax.minorticks_on()
    ax.grid(True, which="major", color=_GRID["color"],
            linewidth=_GRID["linewidth"], linestyle=_GRID["linestyle"],
            alpha=_GRID["alpha"])
    ax.set_axisbelow(True)


def legend(ax, **kwargs):
    """Convenience: a frameless, large legend (overridable via kwargs)."""
    opts = dict(frameon=False, fontsize=13)
    opts.update(kwargs)
    return ax.legend(**opts)
