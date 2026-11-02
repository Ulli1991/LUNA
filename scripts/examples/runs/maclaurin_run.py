"""Maclaurin spheroid equilibrium test.

    python -m scripts.examples.runs.maclaurin_run nx=64 e=0.5 t_end=3

Parameters: nx, e (eccentricity), t_end.
"""
from dataclasses import replace

from luna.setup.cli import script_main
from luna.ic_generation import maclaurin3d
from luna.template_config import Config


def config(nx=64, e=0.5, t_end=3.0):
    a, G = 1.0, 1.0
    L = 4.0 * a
    print(f"Maclaurin spheroid: Nx={nx}, e={e}, a={a}, t_end={t_end:.2f}")
    return replace(
        Config(),
        test_problem="Maclaurin-Spheroid-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=nx, Ny=nx, Nz=nx, Lx=L, Ly=L, Lz=L,
        gamma=5.0 / 3.0, ORDER=1, LIMITER=1,
        t_end=t_end, time_bet_snaps=t_end / 30,
        SELF_GRAVITY=True, GRAVITATIONAL_G=G,
        problem=maclaurin3d.Params(e=e, a=a, rho=1.0),
        PLOT_AFTER_SIM=False,
        output_dir=f"snapshots/maclaurin_e{e:.2f}_Nx{nx}", clear_output_dir=True,
        enable_x64=True, enable_checkpointing=False,
    )


if __name__ == "__main__":
    script_main(config)
