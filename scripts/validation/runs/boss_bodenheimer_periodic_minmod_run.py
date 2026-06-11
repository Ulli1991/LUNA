"""BB79 with PERIODIC BC + MinMod at 256³ — diagnostic for cardinal-axis lines.

If the orthogonal-axis lines we see in the isolated-BC 256³ MinMod plot
also show up here, they're carbuncle. If they disappear / move, they're
specific to the outflow + multipole-Dirichlet BC interaction.
"""
import os
import numpy as np
from scripts.validation.runner import run_one


def main():
    Nx = int(os.environ.get("LUNA_BB_NX", "256"))
    t_end = float(os.environ.get("LUNA_BB_TEND", "2.0"))
    R = 1.0
    M = 1.0
    cs2 = 0.25
    L = 4.0 * R
    G = 1.0

    overrides = dict(
        test_problem="Boss-Bodenheimer-3D",
        SPATIAL_DIMENSION=3,
        BOUNDARY=0,                 # periodic hydro
        POISSON_BOUNDARY=0,         # periodic Poisson (Jeans swindle)
        Nx=Nx, Ny=Nx, Nz=Nx,
        Lx=L, Ly=L, Lz=L,
        gamma=5.0 / 3.0,
        ORDER=1, LIMITER=0, CFL=0.3,   # MinMod, same as iso run
        t_end=t_end, time_bet_snaps=t_end / 30,
        SELF_GRAVITY=True, GRAVITATIONAL_G=G,
        POISSON_MAX_CYCLES=15,
        bb_M=M, bb_R=R, bb_cs2=cs2,
        bb_beta=0.16, bb_amp=0.1,
        isothermal_cs2=cs2,
        PLOT_AFTER_SIM=False, production_mode=True,
        output_dir=f"snapshots/bb_periodic_minmod_Nx{Nx}",
        enable_x64=True, enable_checkpointing=False,
    )
    print(f"BB79 PERIODIC + MinMod: Nx={Nx}, M={M}, R={R}, c_s²={cs2}, t_end={t_end:.2f}")
    run_one(overrides)


if __name__ == "__main__":
    main()
