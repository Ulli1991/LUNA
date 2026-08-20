"""Boss-Bodenheimer collapse — SEREN (Hubber et al. 2011) Figure 13 analog.

Top row: barotropic EOS.  Bottom row: with on-the-fly sink particles.
Columns: t = 25, 33, 55 kyr (the three SEREN Fig 13 times).
Quantity: column density log10 Sigma along the rotation (z) axis, zoomed to the
central +/- 0.005 pc (the SEREN Fig 13 frame). The barotropic and sink rows
should look essentially the same down to where the sinks replace the dense
condensations.

Reads snapshots/bb79_baro_Nx256 (barotropic) and snapshots/bb79_baro_Nx256_sinks
(sinks); override base via LUNA_BB_SNAP_BARO / LUNA_BB_SNAP_SINK.
"""
import os

import numpy as np

from scripts.validation.plots._plotkit import mpl, save, snap_times
from scripts.validation.plots.bb79_iso_luna import luna_column_density

plt = mpl()

PC = 3.0856775814913673e18
MYR = 1.0e6 * 3.15576e7
UNIT_TIME_S = PC / 1.0e5                       # pc / (km/s)
TIMES_MYR = [0.025, 0.033, 0.055]              # SEREN Fig 13 (a),(b),(c)
TIMES_CODE = [t * MYR / UNIT_TIME_S for t in TIMES_MYR]
ZOOM_PC = 0.005                                # half-width of the SEREN Fig 13 frame

ROWS = [
    ("barotropic", os.environ.get("LUNA_BB_SNAP_BARO", "snapshots/bb79_baro_Nx256")),
    ("+ sinks",    os.environ.get("LUNA_BB_SNAP_SINK", "snapshots/bb79_baro_Nx256_sinks")),
]


def main():
    panels = {}                                # (row, col) -> (S, L)
    for r, (_, dp) in enumerate(ROWS):
        dirs, times = snap_times(dp)
        if len(dirs) == 0:
            print(f"  WARN: no snapshots under {dp}")
            continue
        for c, tc in enumerate(TIMES_CODE):
            k = int(np.argmin(np.abs(times - tc)))
            S, L, _ = luna_column_density(dirs[k])
            panels[(r, c)] = (np.maximum(S, 1e-30), L)
            print(f"  row{r} {TIMES_MYR[c]*1e3:g}kyr -> snap t={times[k]*UNIT_TIME_S/MYR*1e3:.1f}kyr")
    if not panels:
        raise SystemExit("no BB79 snapshots found")

    # Scale colors to the GAS column density (barotropic row, which has no sink
    # point-masses) so both rows show the same bar/spiral structure; the sink
    # point-sources simply saturate to white. 4-dex dynamic range for contrast.
    baro = [panels[(0, c)][0] for c in range(len(TIMES_MYR)) if (0, c) in panels]
    ref = baro if baro else [p[0] for p in panels.values()]
    vmax = np.log10(max(np.percentile(p, 99.9) for p in ref))
    vmin = vmax - 4.0

    fig, axes = plt.subplots(2, 3, figsize=(2.2 * 3, 2.2 * 2 + 0.2), squeeze=False)
    im = None
    for r, (rlab, _) in enumerate(ROWS):
        for c, tmyr in enumerate(TIMES_MYR):
            ax = axes[r][c]
            ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
            if (r, c) not in panels:
                ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes,
                        ha="center", va="center", color="0.5", fontsize=10)
            else:
                S, L = panels[(r, c)]
                im = ax.imshow(np.log10(S).T, origin="lower", extent=(0, L, 0, L),
                               vmin=vmin, vmax=vmax, cmap="magma",
                               interpolation="bilinear")
                ctr = 0.5 * L
                ax.set_xlim(ctr - ZOOM_PC, ctr + ZOOM_PC)
                ax.set_ylim(ctr - ZOOM_PC, ctr + ZOOM_PC)
            if r == 0:                          # in-panel time label (no titles)
                ax.text(0.05, 0.90, f"{tmyr * 1e3:g} kyr", transform=ax.transAxes,
                        color="white", fontsize=11, fontweight="bold")
            if c == 0:
                ax.set_ylabel(rlab, fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.92, 0.12, 0.018, 0.76])
    if im is not None:
        fig.colorbar(im, cax=cax, label=r"$\log_{10}\,\Sigma$")
    out = "code_paper_figs/bb79_seren_fig13.png"
    save(fig, out)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
