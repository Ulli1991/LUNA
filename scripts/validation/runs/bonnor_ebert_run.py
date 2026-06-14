"""Bonnor-Ebert sphere — sub-critical (stable) or super-critical (collapse).

Env vars:
    LUNA_BE_NX     : grid resolution (default 64)
    LUNA_BE_XI_T   : truncation radius in BE units (5 stable, 10 collapse)
    LUNA_BE_TEND   : end time in code units (default 3.0, several t_dyn)
"""
import os
import numpy as np

from dataclasses import replace
from template_config import Config
from setup.distributed_setup import initialize_jax_distributed
from main import run


def main():
    Nx = int(os.environ.get("LUNA_BE_NX", "64"))
    xi_t = float(os.environ.get("LUNA_BE_XI_T", "5.0"))
    t_end = float(os.environ.get("LUNA_BE_TEND", "3.0"))

    R = 1.0
    cs2 = 1.0
    G = 1.0
    L = 4.0 * R   # periodic box >> cloud

    # Dynamical time t_dyn = 1/√(Gρ̄) where ρ̄ is mean density of the cloud.
    # For BE, ρ̄ depends on ξ_t; rough estimate using ρ_c = cs² ξ_t² / (4πG R²).
    rho_c = cs2 * xi_t * xi_t / (4.0 * np.pi * G * R * R)
    t_dyn = 1.0 / np.sqrt(G * rho_c)

    overrides = dict(
        test_problem="Bonnor-Ebert-3D",
        SPATIAL_DIMENSION=3,
        BOUNDARY=0,
        Nx=Nx, Ny=Nx, Nz=Nx,
        Lx=L, Ly=L, Lz=L,
        gamma=5.0 / 3.0,
        ORDER=1,
        LIMITER=1,
        CFL=0.3,
        t_end=t_end,
        time_bet_snaps=t_end / 30,
        SELF_GRAVITY=True,
        GRAVITATIONAL_G=G,
        POISSON_MAX_CYCLES=15,
        be_xi_t=xi_t,
        be_cs2=cs2,
        be_R=R,
        # Enforce isothermal: P → cs² ρ after each step.
        isothermal_cs2=cs2,
        PLOT_AFTER_SIM=False,
        production_mode=True,
        output_dir=f"snapshots/be_xi{xi_t:.1f}_Nx{Nx}",
        clear_output_dir=True,
        enable_x64=True,
        enable_checkpointing=False,
    )

    regime = "sub-critical (stable)" if xi_t < 6.451 else "super-critical (collapsing)"
    print(
        f"Bonnor-Ebert: Nx={Nx}, ξ_t={xi_t} ({regime}), R={R}, c_s²={cs2}, G={G}, "
        f"ρ_c={rho_c:.3f}, t_dyn={t_dyn:.3f}, t_end={t_end:.3f} (~{t_end/t_dyn:.2f} t_dyn)"
    )
    initialize_jax_distributed(enable_x64=bool(overrides.get("enable_x64", False)))
    overrides = {k: v for k, v in overrides.items() if k != "enable_x64"}
    run(replace(Config(), **overrides))


if __name__ == "__main__":
    main()
