"""BB79 with periodic BC + minmod: a diagnostic for cardinal-axis lines.

    python -m scripts.examples.runs.boss_bodenheimer_periodic_minmod_run nx=256

If the orthogonal-axis lines of the isolated-BC 256^3 minmod run show up here
too they are carbuncle; if they disappear they belong to the outflow +
multipole-Dirichlet BC interaction. Parameters: nx, t_end.
"""
from dataclasses import replace

from luna.setup.cli import script_main
from luna.ic_generation import boss_bodenheimer3d
from luna.template_config import Config


def config(nx=256, t_end=2.0):
    R, M, cs2, G = 1.0, 1.0, 0.25, 1.0
    L = 4.0 * R
    print(f"BB79 PERIODIC + MinMod: Nx={nx}, M={M}, R={R}, cs2={cs2}, t_end={t_end:.2f}")
    return replace(
        Config(),
        test_problem="Boss-Bodenheimer-3D", SPATIAL_DIMENSION=3,
        BOUNDARY=0, POISSON_BOUNDARY=0,   # periodic hydro and Poisson (Jeans swindle)
        Nx=nx, Ny=nx, Nz=nx, Lx=L, Ly=L, Lz=L,
        gamma=5.0 / 3.0, ORDER=1, LIMITER=0,
        t_end=t_end, time_bet_snaps=t_end / 30,
        SELF_GRAVITY=True, GRAVITATIONAL_G=G,
        problem=boss_bodenheimer3d.Params(beta=0.16, amp=0.1), isothermal_cs2=cs2,
        PLOT_AFTER_SIM=False,
        output_dir=f"snapshots/bb_periodic_minmod_Nx{nx}", clear_output_dir=True,
        enable_x64=True, enable_checkpointing=False,
    )


if __name__ == "__main__":
    script_main(config)
