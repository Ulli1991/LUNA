"""Boss-Bodenheimer 1979 collapse with ISOLATED BCs.

Outflow hydro + multipole-Dirichlet (isolated) Poisson. Matches the
AREPO GRAVITY_NOT_PERIODIC + REFLECTIVE_X/Y/Z=2 setup we resubmitted.
"""
import os
import numpy as np
from scripts.validation.runner import run_one


def main():
    Nx = int(os.environ.get("LUNA_BB_NX", "64"))
    t_end = float(os.environ.get("LUNA_BB_TEND", "2.0"))
    riemann = os.environ.get("LUNA_BB_RIEMANN", "hllc")
    R = 1.0
    M = 1.0
    cs2 = 0.25
    L = 4.0 * R
    G = 1.0

    overrides = dict(
        test_problem="Boss-Bodenheimer-3D",
        SPATIAL_DIMENSION=3,
        BOUNDARY=1,                 # outflow / zero-gradient hydro BC
        POISSON_BOUNDARY=3,         # multipole-Dirichlet isolated BC
        Nx=Nx, Ny=Nx, Nz=Nx,
        Lx=L, Ly=L, Lz=L,
        gamma=5.0 / 3.0,
        ORDER=1, LIMITER=0, CFL=0.3,   # MinMod (more diffusive, matches RAMSES default)
        riemann_solver=riemann,
        t_end=t_end, time_bet_snaps=t_end / 30,
        SELF_GRAVITY=True, GRAVITATIONAL_G=G,
        POISSON_MAX_CYCLES=15,
        bb_M=M, bb_R=R, bb_cs2=cs2,
        bb_beta=0.16, bb_amp=0.1,
        isothermal_cs2=cs2,
        PLOT_AFTER_SIM=False, production_mode=True,
        output_dir=f"snapshots/bb_iso_minmod_{riemann}_Nx{Nx}" if riemann != "hllc"
                   else f"snapshots/bb_iso_minmod_Nx{Nx}",
        enable_x64=True, enable_checkpointing=False,
    )
    print(f"BB79 ISOLATED: Nx={Nx}, M={M}, R={R}, c_s²={cs2}, riemann={riemann}, "
          f"BOUNDARY=1 (outflow), POISSON_BOUNDARY=3 (multipole), t_end={t_end:.2f}")
    run_one(overrides)


if __name__ == "__main__":
    main()
