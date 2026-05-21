"""Gresho vortex 2D low-Mach test runner.

Env vars:
    LUNA_GRESHO_NX      : grid resolution (default 256). Ny = Nx.
    LUNA_GRESHO_TEND    : end time in code units (default 3 — about 3 turnovers
                          since the core has constant angular velocity 5).
    LUNA_GRESHO_RIEMANN : "hllc" or "lhllc" (default lhllc).
    LUNA_GRESHO_MMAX    : peak Mach number (default 0.01).
"""
import os
from scripts.validation.runner import run_one


def main():
    Nx = int(os.environ.get("LUNA_GRESHO_NX", "256"))
    Ny = Nx
    t_end = float(os.environ.get("LUNA_GRESHO_TEND", "3.0"))
    riemann = os.environ.get("LUNA_GRESHO_RIEMANN", "lhllc").lower()
    Mmax = float(os.environ.get("LUNA_GRESHO_MMAX", "0.01"))

    overrides = dict(
        test_problem="Gresho-Vortex-2D",
        SPATIAL_DIMENSION=2,
        BOUNDARY=0,                  # periodic
        Nx=Nx, Ny=Ny,
        Lx=1.0, Ly=1.0,
        gamma=5.0 / 3.0,
        TIME_SOLVER=1,
        ORDER=1,
        LIMITER=1,                   # MonCen
        CFL=0.4,
        t_end=t_end,
        time_bet_snaps=t_end / 10.0,
        riemann_solver=riemann,
        gresho_Mmax=Mmax,
        PLOT_AFTER_SIM=False,
        production_mode=True,
        output_dir=f"snapshots/gresho_{riemann}_Nx{Nx}_M{Mmax}",
        enable_x64=True,
    )
    print(f"Gresho 2D: Nx={Nx}, t_end={t_end}, riemann={riemann}, M_max={Mmax}")
    run_one(overrides)


if __name__ == "__main__":
    main()
