"""Lane-Emden n=1.5 polytrope with a small radial velocity perturbation: it
oscillates at the fundamental radial mode period.

    python -m scripts.examples.runs.pulsating_polytrope_run nx=64 t_end=5

Parameters: nx, t_end.
"""
from dataclasses import replace

from luna.setup.cli import script_main
from luna.ic_generation import pulsating_polytrope3d
from luna.template_config import Config


def config(nx=64, t_end=5.0):
    R, G = 1.0, 1.0
    L = 4.0 * R
    print(f"Pulsating polytrope: Nx={nx}, n=1.5, R={R}, t_end={t_end:.2f}")
    return replace(
        Config(),
        test_problem="Pulsating-Polytrope-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=nx, Ny=nx, Nz=nx, Lx=L, Ly=L, Lz=L,
        gamma=5.0 / 3.0, ORDER=1, LIMITER=1,
        t_end=t_end, time_bet_snaps=t_end / 40,
        SELF_GRAVITY=True, GRAVITATIONAL_G=G,
        problem=pulsating_polytrope3d.Params(n=1.5, R=R, rho_c=1.0, amp=0.01),
        PLOT_AFTER_SIM=False,
        output_dir=f"snapshots/pulse_Nx{nx}", clear_output_dir=True,
        enable_x64=True, enable_checkpointing=False,
    )


if __name__ == "__main__":
    script_main(config)
