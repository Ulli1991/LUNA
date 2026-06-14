"""Lane-Emden n=1.5 polytrope + small radial velocity perturbation.
Expect oscillation at the fundamental radial mode period."""
import os
import numpy as np
from dataclasses import replace
from template_config import Config
from setup.distributed_setup import initialize_jax_distributed
from main import run


def main():
    Nx = int(os.environ.get("LUNA_PULSE_NX", "64"))
    t_end = float(os.environ.get("LUNA_PULSE_TEND", "5.0"))
    R = 1.0
    L = 4.0 * R
    G = 1.0

    overrides = dict(
        test_problem="Pulsating-Polytrope-3D",
        SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=Nx, Ny=Nx, Nz=Nx,
        Lx=L, Ly=L, Lz=L,
        gamma=5.0 / 3.0,
        ORDER=1, LIMITER=1, CFL=0.3,
        t_end=t_end, time_bet_snaps=t_end / 40,
        SELF_GRAVITY=True, GRAVITATIONAL_G=G,
        POISSON_MAX_CYCLES=15,
        lane_n=1.5, lane_R=R, lane_rho_c=1.0,
        pulse_amp=0.01,
        PLOT_AFTER_SIM=False, production_mode=True,
        output_dir=f"snapshots/pulse_Nx{Nx}",
        clear_output_dir=True,
        enable_x64=True, enable_checkpointing=False,
    )
    print(f"Pulsating polytrope: Nx={Nx}, n=1.5, R={R}, t_end={t_end:.2f}")
    initialize_jax_distributed(enable_x64=bool(overrides.get("enable_x64", False)))
    overrides = {k: v for k, v in overrides.items() if k != "enable_x64"}
    run(replace(Config(), **overrides))


if __name__ == "__main__":
    main()
