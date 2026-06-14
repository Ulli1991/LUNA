"""Kelvin-Helmholtz multi-resolution × multi-time density grid plot.

Runs KH at Nx² ∈ {1024², 2048², 4096², 8192²} (multi-GPU recommended for
the larger ones), saves snapshots at t ∈ {0, 1, 2.5, 4, 5}, then assembles
a 4 (resolution) × 5 (time) panel grid of density. Shared colormap and
density range across all panels so refinement is visually obvious.

Saves figures/kh_grid.pdf.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils.figstyle import apply_style, decorate, FIGSIZE

apply_style()

from dataclasses import replace
from template_config import Config
from setup.distributed_setup import initialize_jax_distributed
from main import run


LABEL_FS = 11   # in-panel image annotations only (axis labels inherit rcParams)

NX_SCAN = [1024, 2048, 4096, 8192]
SNAPSHOT_TIMES = [0.0, 1.0, 2.5, 4.0, 5.0]
T_END = max(SNAPSHOT_TIMES)


def main():
    # 1. Run LUNA at each resolution, requesting snapshots at SNAPSHOT_TIMES.
    #    The runner uses cfg.time_bet_snaps for snapshot cadence; here we want
    #    irregular timestamps so we run with a coarse cadence and the runner
    #    will land on them (or we override snapshot_times directly via Config).
    output_dirs = []
    initialize_jax_distributed(enable_x64=False)
    for i, Nx in enumerate(NX_SCAN):
        overrides = dict(
            test_problem="Kelvin-Helmholtz",
            SPATIAL_DIMENSION=2,
            BOUNDARY=0,
            Nx=Nx, Ny=Nx,
            t_end=T_END,
            ORDER=1,
            PLOT_AFTER_SIM=False,
            production_mode=True,
            time_bet_snaps=min(np.diff(SNAPSHOT_TIMES)),
            output_dir=f"snapshots/kh_Nx{Nx}",
            clear_output_dir=True,
        )
        overrides = {k: v for k, v in overrides.items() if k != "enable_x64"}
        _, _, outdir = run(replace(Config(), **overrides))
        output_dirs.append(outdir)

    # 2. Load density at each (resolution, time) and assemble the grid plot.
    fig, axes = plt.subplots(
        len(NX_SCAN), len(SNAPSHOT_TIMES),
        figsize=(2.2 * len(SNAPSHOT_TIMES), 2.2 * len(NX_SCAN)),
        squeeze=False,
    )
    vmin, vmax = None, None
    for r, (Nx, outdir) in enumerate(zip(NX_SCAN, output_dirs)):
        for c, t in enumerate(SNAPSHOT_TIMES):
            ax = axes[r, c]
            ax.grid(False)
            # TODO: load density at time t from outdir.
            raise NotImplementedError(
                "Hook this up to the LUNA snapshot reader once the snapshot "
                "format / per-time file naming is locked in."
            )

    # 3. Decorate axes and save.
    for r, Nx in enumerate(NX_SCAN):
        axes[r, 0].set_ylabel(f"{Nx}²")
    for c, t in enumerate(SNAPSHOT_TIMES):
        axes[0, c].text(0.05, 0.92, f"t = {t}", transform=axes[0, c].transAxes,
                        fontsize=LABEL_FS, color="white", fontweight="bold",
                        bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=1.5))

    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/kh_grid.pdf")
    print("Saved figures/kh_grid.pdf")


if __name__ == "__main__":
    main()
