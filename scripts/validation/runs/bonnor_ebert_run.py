"""Bonnor-Ebert sphere: sub-critical (stable) or super-critical (collapse).

    python -m scripts.validation.runs.bonnor_ebert_run nx=64 xi_t=10 t_end=3

Parameters: nx (grid per side), xi_t (truncation radius in BE units: 5 is
stable, 10 collapses), t_end (code units, several t_dyn). Any Config field
can follow as field=value.
"""
from dataclasses import replace

import numpy as np

from luna.setup.cli import script_main
from luna.ic_generation import bonnor_ebert3d
from luna.template_config import Config


def config(nx=64, xi_t=5.0, t_end=3.0):
    R, cs2, G = 1.0, 1.0, 1.0
    L = 4.0 * R                          # periodic box >> cloud
    # rho_c = cs^2 xi_t^2 / (4 pi G R^2); t_dyn = 1/sqrt(G rho_c)
    rho_c = cs2 * xi_t * xi_t / (4.0 * np.pi * G * R * R)
    t_dyn = 1.0 / np.sqrt(G * rho_c)
    regime = "sub-critical (stable)" if xi_t < 6.451 else "super-critical (collapsing)"
    print(f"Bonnor-Ebert: Nx={nx}, xi_t={xi_t} ({regime}), R={R}, cs2={cs2}, G={G}, "
          f"rho_c={rho_c:.3f}, t_dyn={t_dyn:.3f}, t_end={t_end:.3f} (~{t_end / t_dyn:.2f} t_dyn)")
    return replace(
        Config(),
        test_problem="Bonnor-Ebert-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=nx, Ny=nx, Nz=nx, Lx=L, Ly=L, Lz=L,
        gamma=5.0 / 3.0, ORDER=1, LIMITER=1,
        t_end=t_end, time_bet_snaps=t_end / 30,
        SELF_GRAVITY=True, GRAVITATIONAL_G=G,
        problem=bonnor_ebert3d.Params(xi_t=xi_t, cs2=cs2, R=R),
        isothermal_cs2=cs2,                  # P -> cs^2 rho after each step
        PLOT_AFTER_SIM=False,
        output_dir=f"snapshots/be_xi{xi_t:.1f}_Nx{nx}", clear_output_dir=True,
        enable_x64=True, enable_checkpointing=False,
    )


if __name__ == "__main__":
    script_main(config)
