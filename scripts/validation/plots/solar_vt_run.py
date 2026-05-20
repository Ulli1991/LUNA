"""Velasco-Romero & Teyssier (2025) 2D stellar-convection test.

Three-layer polytropic atmosphere with a small under-dense Gaussian bubble
driving convection through the middle (Gamma=gamma=5/3) zone.

LUNA setup:
    test_problem = "Solar-Convection-VT-2D"
    Box: x in [0, Lx=5.5], y in [0, Ly=11]   (so vertical/Ly = full L)
    g_z = -1 (downward)
    Nx, Ny: env-var driven; defaults Nx=128, Ny=256 (preserves 1:2 aspect)
    Well-balanced + L-HLLC, MUSCL-Hancock + MonCen + CFL=0.4, float64.

Run for t_end=100 (early-phase plumes; V&T show t=100, 1750, 2000 in their
figures — a longer reaching of statistical state needs much more wall time).

Env vars:
    LUNA_SOLAR_VT_NX : x resolution (default 128). Ny = 2*Nx.
    LUNA_SOLAR_VT_TEND : end time (default 100).
"""
import os
from scripts.validation.runner import run_one


L = 11.0
LX = L / 2.0
LY = L


def main():
    Nx = int(os.environ.get("LUNA_SOLAR_VT_NX", "128"))
    Ny = 2 * Nx
    t_end = float(os.environ.get("LUNA_SOLAR_VT_TEND", "100.0"))
    out_dir = os.environ.get("LUNA_SOLAR_VT_OUT", f"snapshots/solar_vt_Nx{Nx}")
    overrides = dict(
        test_problem="Solar-Convection-VT-2D",
        SPATIAL_DIMENSION=2,
        BOUNDARY=1,                 # zero-gradient outflow (standard for stellar convection)
        Nx=Nx, Ny=Ny,
        Lx=LX, Ly=LY,
        gamma=5.0 / 3.0,
        TIME_SOLVER=1,
        ORDER=1,
        LIMITER=1,                  # MonCen
        CFL=0.4,
        t_end=t_end,
        time_bet_snaps=t_end / 20.0,
        gravity_z=-1.0,
        use_well_balanced=True,
        riemann_solver="lhllc",
        PLOT_AFTER_SIM=False,
        production_mode=True,
        output_dir=out_dir,
        enable_x64=True,
    )
    print(f"V&T solar convection: Nx={Nx}, Ny={Ny}, Lx={LX}, Ly={LY}, "
          f"t_end={t_end}")
    run_one(overrides)


if __name__ == "__main__":
    main()
