"""2D square-advection time-progression plot.

Runs the Square-Advected test at Nx^2 = 1024^2 with u=v=1, Lx=Ly=1, periodic.
One full diagonal traversal => t_end = 1, so t=1 should match t=0 exactly
modulo numerical diffusion. Saves a 1-row strip of density snapshots at
t in {0, 0.25, 0.5, 0.75, 1.0}.

Saves figures/square_advection.{pdf,png}.
"""
from pathlib import Path
import matplotlib.pyplot as plt

from file_io.read_snapshot import read_snapshot, find_snapshot_at_time
from scripts.validation.runner import run_one


NX = 1024
T_END = 1.0
SNAPSHOT_TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]
OUTDIR = f"snapshots/square_advection_Nx{NX}"
VMIN, VMAX = 0.95, 2.05


def main():
    overrides = dict(
        test_problem="Square-Advected",
        SPATIAL_DIMENSION=2,
        BOUNDARY=0,            # periodic
        Nx=NX, Ny=NX,
        Lx=1.0, Ly=1.0,
        t_end=T_END,
        ORDER=1,               # PLM
        LIMITER=1,             # MonCen
        CFL=0.4,
        PLOT_AFTER_SIM=False,
        production_mode=True,
        time_bet_snaps=0.25,
        output_dir=OUTDIR,
        enable_x64=True,
    )
    run_one(overrides)

    fig, axes = plt.subplots(
        1, len(SNAPSHOT_TIMES),
        figsize=(2.5 * len(SNAPSHOT_TIMES), 2.8),
        squeeze=False,
    )

    for c, t in enumerate(SNAPSHOT_TIMES):
        ax = axes[0, c]
        snap = read_snapshot(find_snapshot_at_time(OUTDIR, t, tol=0.02))
        ax.imshow(snap["rho"].T, origin="lower",
                  extent=(0, snap["Lx"], 0, snap["Ly"]),
                  vmin=VMIN, vmax=VMAX, cmap="magma",
                  interpolation="nearest", aspect="equal")
        ax.set_title(f"t = {t}", fontsize=11)
        ax.tick_params(top=True, right=True, direction="in")
        ax.grid(True, linestyle=":", alpha=0.4)
        if c > 0:
            ax.set_yticklabels([])

    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/square_advection.pdf")
    fig.savefig("figures/square_advection.png", dpi=200)
    print("Saved figures/square_advection.{pdf,png}")


if __name__ == "__main__":
    main()
