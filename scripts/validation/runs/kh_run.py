"""Kelvin-Helmholtz at one resolution, snapshots saved for kh_grid.py.

    python -m scripts.validation.runs.kh_run nx=1024 snap=0.5

Lx = Ly = 2 fits the double-tanh IC (shear layers at y = 0.5 and 1.5); a
snapshot every 0.5 over t in [0, 5] gives the 11 frames kh_grid picks from.
Parameters: nx and snap (time between snapshots); output_dir is a field.
"""
from dataclasses import replace

from luna.setup.cli import script_main
from luna.template_config import Config


def config(nx=1024, snap=0.5):
    return replace(
        Config(),
        test_problem="Kelvin-Helmholtz", SPATIAL_DIMENSION=2, BOUNDARY=0,
        Nx=nx, Ny=nx, Lx=2.0, Ly=2.0, gamma=5.0 / 3.0,
        t_end=5.0, time_bet_snaps=snap,
        ORDER=1, LIMITER=1,
        PLOT_AFTER_SIM=False,
        output_dir=f"snapshots/kh_Nx{nx}", clear_output_dir=True,
    )


if __name__ == "__main__":
    script_main(config)
