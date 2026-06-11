"""Maclaurin spheroid equilibrium test."""
import os
import numpy as np
from scripts.validation.runner import run_one


def main():
    Nx = int(os.environ.get("LUNA_MAC_NX", "64"))
    e = float(os.environ.get("LUNA_MAC_E", "0.5"))
    t_end = float(os.environ.get("LUNA_MAC_TEND", "3.0"))
    a = 1.0
    L = 4.0 * a
    G = 1.0

    overrides = dict(
        test_problem="Maclaurin-Spheroid-3D",
        SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=Nx, Ny=Nx, Nz=Nx,
        Lx=L, Ly=L, Lz=L,
        gamma=5.0 / 3.0,
        ORDER=1, LIMITER=1, CFL=0.3,
        t_end=t_end, time_bet_snaps=t_end / 30,
        SELF_GRAVITY=True, GRAVITATIONAL_G=G,
        POISSON_MAX_CYCLES=15,
        mac_e=e, mac_a=a, mac_rho=1.0,
        PLOT_AFTER_SIM=False, production_mode=True,
        output_dir=f"snapshots/maclaurin_e{e:.2f}_Nx{Nx}",
        enable_x64=True, enable_checkpointing=False,
    )
    print(f"Maclaurin spheroid: Nx={Nx}, e={e}, a={a}, t_end={t_end:.2f}")
    run_one(overrides)


if __name__ == "__main__":
    main()
