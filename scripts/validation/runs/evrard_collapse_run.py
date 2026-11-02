"""Evrard spherical collapse with self-gravity.

    python -m scripts.validation.runs.evrard_collapse_run nx=64 t_end=1.5
    python -m scripts.validation.runs.evrard_collapse_run nx=64 AMR=1 amr_levels=3

A cold cloud (M=1, R=1, u=0.05, G=1) in a periodic box (L=4R) of low-density
ambient gas free-falls; the accretion shock forms at r/R ~ 0.25 and moves out.
t_ff = pi/2 sqrt(R^3 / 2GM) ~ 1.11.

Parameters: nx, t_end, amr_levels, amr_thr (the pressure_density threshold,
one per level) and amr_regrid. AMR=1 turns the refinement on; it and the
other amr_* fields follow as field=value.
"""
from dataclasses import replace

import numpy as np

from luna.setup.cli import script_main
from luna.ic_generation import evrard3d
from luna.template_config import Config


def config(nx=64, t_end=1.5, amr_levels=3, amr_thr=0.15, amr_regrid=8):
    M, R, u, G = 1.0, 1.0, 0.05, 1.0
    L = 4.0 * R
    t_ff = (np.pi / 2.0) * np.sqrt(R ** 3 / (2.0 * G * M))
    print(f"Evrard collapse: Nx={nx}, M={M}, R={R}, u={u}, G={G}, L={L}, "
          f"t_end={t_end:.3f}, t_ff={t_ff:.3f} (~{t_end / t_ff:.2f} t_ff)")
    return replace(
        Config(),
        test_problem="Evrard-Collapse-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=nx, Ny=nx, Nz=nx, Lx=L, Ly=L, Lz=L,
        gamma=5.0 / 3.0, ORDER=1, LIMITER=1,
        t_end=t_end, time_bet_snaps=t_end / 30,
        SELF_GRAVITY=True, GRAVITATIONAL_G=G,
        POISSON_FMG_VCYCLES_PER_LEVEL=1, POISSON_FMG_EXTRA_VCYCLES=0,
        problem=evrard3d.Params(M=M, R=R, u=u),
        # the cloud's edge is a 160:1 density jump: MC's Hancock faces there go negative (audit C8), as
        # tests/test_gravity_evrard_energy.py FACES_MC counts; first order there, counted in the log
        allow_first_order_faces=True,
        PLOT_AFTER_SIM=False,
        output_dir=f"snapshots/evrard_Nx{nx}", clear_output_dir=True,
        enable_x64=True, enable_checkpointing=False,
        # the AMR path: the gated pressure_density criterion refines the
        # overdense sphere and leaves the ambient coarse
        amr_levels=amr_levels, amr_block_shape=16,
        amr_criterion="pressure_density",
        amr_thresholds=[amr_thr] * max(1, amr_levels),
        amr_regrid_interval=amr_regrid, amr_subcycle=False, amr_shard=False,
    )


if __name__ == "__main__":
    script_main(config)
