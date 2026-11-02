"""Boss-Bodenheimer 1979 rotating isothermal collapse + m=2 fragmentation.

    python -m scripts.examples.runs.boss_bodenheimer_run nx=64 t_end=2 n_snap=30

Parameters: nx, t_end (code units) and n_snap; output_dir is a Config field.
"""
from dataclasses import replace

from luna.setup.cli import script_main
from luna.ic_generation import boss_bodenheimer3d
from luna.template_config import Config


def config(nx=64, t_end=2.0, n_snap=30):
    R, M, G = 1.0, 1.0, 1.0
    cs2 = 0.25                           # alpha = R cs^2 / (G M) = 0.25
    L = 4.0 * R
    print(f"Boss-Bodenheimer: Nx={nx}, M={M}, R={R}, cs2={cs2}, t_end={t_end:.2f}")
    return replace(
        Config(),
        test_problem="Boss-Bodenheimer-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=nx, Ny=nx, Nz=nx, Lx=L, Ly=L, Lz=L,
        gamma=5.0 / 3.0, ORDER=1, LIMITER=1,
        t_end=t_end, time_bet_snaps=t_end / n_snap,
        SELF_GRAVITY=True, GRAVITATIONAL_G=G,
        problem=boss_bodenheimer3d.Params(beta=0.16, amp=0.1), isothermal_cs2=cs2,
        PLOT_AFTER_SIM=False,
        output_dir=f"snapshots/bb_Nx{nx}", clear_output_dir=True,
        enable_x64=True, enable_checkpointing=False,
    )


if __name__ == "__main__":
    script_main(config)
