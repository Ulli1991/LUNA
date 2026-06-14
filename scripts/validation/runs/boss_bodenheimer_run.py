"""Boss-Bodenheimer 1979 rotating-isothermal collapse + m=2 fragmentation."""
import os
import numpy as np
from dataclasses import replace
from template_config import Config
from setup.distributed_setup import initialize_jax_distributed
from main import run


def main():
    Nx = int(os.environ.get("LUNA_BB_NX", "64"))
    t_end = float(os.environ.get("LUNA_BB_TEND", "2.0"))
    R = 1.0
    M = 1.0
    cs2 = 0.25   # canonical α = R·cs²/(GM) = 0.25 thermal/grav ratio
    L = 4.0 * R
    G = 1.0

    overrides = dict(
        test_problem="Boss-Bodenheimer-3D",
        SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=Nx, Ny=Nx, Nz=Nx,
        Lx=L, Ly=L, Lz=L,
        gamma=5.0 / 3.0,
        ORDER=1, LIMITER=1, CFL=0.3,
        t_end=t_end, time_bet_snaps=t_end / 30,
        SELF_GRAVITY=True, GRAVITATIONAL_G=G,
        POISSON_MAX_CYCLES=15,
        bb_M=M, bb_R=R, bb_cs2=cs2,
        bb_beta=0.16, bb_amp=0.1,
        # Enforce isothermal during collapse.
        isothermal_cs2=cs2,
        PLOT_AFTER_SIM=False, production_mode=True,
        output_dir=f"snapshots/bb_Nx{Nx}",
        clear_output_dir=True,
        enable_x64=True, enable_checkpointing=False,
    )
    print(f"Boss-Bodenheimer: Nx={Nx}, M={M}, R={R}, c_s²={cs2}, t_end={t_end:.2f}")
    initialize_jax_distributed(enable_x64=bool(overrides.get("enable_x64", False)))
    overrides = {k: v for k, v in overrides.items() if k != "enable_x64"}
    run(replace(Config(), **overrides))


if __name__ == "__main__":
    main()
