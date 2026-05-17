"""2D square-advection time-progression plot.

Runs the Square-Advected test at Nx² = 1024² for one full diagonal traversal
(u=v=1, Lx=Ly=1, periodic BCs => t_end = 1). Saves density snapshots at a
handful of times and assembles a horizontal time-progression strip showing
the square exiting one boundary and re-entering through the periodic side.

Saves figures/square_advection.pdf.
"""
from pathlib import Path
import matplotlib.pyplot as plt

from scripts.validation.runner import run_one


NX = 1024
T_END = 1.0
SNAPSHOT_TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]


def main():
    overrides = dict(
        test_problem="Square-Advected",
        SPATIAL_DIMENSION=2,
        BOUNDARY=0,  # periodic
        Nx=NX, Ny=NX,
        t_end=T_END,
        TIME_SOLVER=1,
        ORDER=1,
        PLOT_AFTER_SIM=False,
        production_mode=True,
        time_bet_snaps=0.25,
        output_dir=f"snapshots/square_advection_Nx{NX}",
    )
    _, _, outdir = run_one(overrides)

    fig, axes = plt.subplots(
        1, len(SNAPSHOT_TIMES),
        figsize=(2.5 * len(SNAPSHOT_TIMES), 2.8),
        squeeze=False,
    )
    for c, t in enumerate(SNAPSHOT_TIMES):
        ax = axes[0, c]
        # TODO: load density at time t from outdir, then ax.pcolormesh(...).
        raise NotImplementedError(
            "Hook this up to the LUNA snapshot reader once the snapshot "
            "format / per-time file naming is locked in."
        )
        ax.set_title(f"t = {t}")
        ax.set_aspect("equal")

    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/square_advection.pdf")
    print("Saved figures/square_advection.pdf")


if __name__ == "__main__":
    main()
