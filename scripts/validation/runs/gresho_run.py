"""Gresho vortex, the 2D low-Mach test.

    python -m scripts.validation.runs.gresho_run nx=256 t_end=3 riemann=hllc mmax=0.01

Parameters: nx, t_end (about three turnovers by default: the core has constant
angular velocity 5), riemann (hllc | hll), mmax (peak Mach number).
"""
from dataclasses import replace

from luna.setup.cli import script_main
from luna.ic_generation import gresho_vortex
from luna.template_config import Config


def config(nx=256, t_end=3.0, riemann="hllc", mmax=0.01):
    riemann = riemann.lower()
    print(f"Gresho 2D: Nx={nx}, t_end={t_end}, riemann={riemann}, M_max={mmax}")
    return replace(
        Config(),
        test_problem="Gresho-Vortex-2D", SPATIAL_DIMENSION=2, BOUNDARY=0,
        Nx=nx, Ny=nx, Lx=1.0, Ly=1.0, gamma=5.0 / 3.0,
        ORDER=1, LIMITER=1,
        t_end=t_end, time_bet_snaps=t_end / 10.0,
        riemann_solver=riemann, problem=gresho_vortex.Params(Mmax=mmax),
        PLOT_AFTER_SIM=False,
        output_dir=f"snapshots/gresho_{riemann}_Nx{nx}_M{mmax}", clear_output_dir=True,
        enable_x64=True,
    )


if __name__ == "__main__":
    script_main(config)
