"""Evrard spherical collapse with self-gravity.

Cold spherical cloud (M=1, R=1, u=0.05, G=1) in a periodic box (L=4R)
of low-density ambient gas. Cloud free-falls, central pressure rises,
accretion shock forms at r/R ≈ 0.25 and propagates outward.

Run for t_ff ≈ 1.11 (free-fall time = π/2 · √(R³/(2GM))) — by then the
shock has traveled most of the way out.

Env vars:
    LUNA_EVRARD_NX  : grid resolution (default 64)
    LUNA_EVRARD_TEND: end time (default 1.5, ~1.3 t_ff)
"""
import os
import numpy as np

from dataclasses import replace
from template_config import Config
from setup.distributed_setup import initialize_jax_distributed
from main import run


def main():
    Nx = int(os.environ.get("LUNA_EVRARD_NX", "64"))
    t_end = float(os.environ.get("LUNA_EVRARD_TEND", "1.5"))

    M = 1.0
    R = 1.0
    u = 0.05
    G = 1.0
    L = 4.0 * R                       # periodic box >> R
    t_ff = (np.pi / 2.0) * np.sqrt(R ** 3 / (2.0 * G * M))

    overrides = dict(
        test_problem="Evrard-Collapse-3D",
        SPATIAL_DIMENSION=3,
        BOUNDARY=0,                   # periodic — large box approximates isolated
        Nx=Nx, Ny=Nx, Nz=Nx,
        Lx=L, Ly=L, Lz=L,
        gamma=5.0 / 3.0,
        ORDER=1,
        LIMITER=1,
        CFL=0.3,
        t_end=t_end,
        time_bet_snaps=t_end / 30,    # 30 snapshots
        SELF_GRAVITY=True,
        GRAVITATIONAL_G=G,
        POISSON_FMG_VCYCLES_PER_LEVEL=1,
        POISSON_FMG_EXTRA_VCYCLES=0,
        evrard_M=M,
        evrard_R=R,
        evrard_u=u,
        PLOT_AFTER_SIM=False,
        production_mode=True,
        output_dir=os.environ.get("LUNA_EVRARD_OUT", f"snapshots/evrard_Nx{Nx}"),
        clear_output_dir=True,
        enable_x64=True,
        enable_checkpointing=False,
        # Optional AMR path (env-driven): dynamic refinement tracks the
        # collapsing cloud / core through the levels. The gated
        # pressure_density criterion refines the overdense sphere (sharp
        # gradients + rho above the box mean) and leaves the ambient coarse.
        AMR=os.environ.get("LUNA_EVRARD_AMR", "0") == "1",
        amr_levels=int(os.environ.get("LUNA_EVRARD_AMR_LEVELS", "3")),
        amr_block_shape=16,
        amr_criterion="pressure_density",
        amr_thresholds=[float(os.environ.get("LUNA_EVRARD_AMR_THR", "0.15"))]
        * max(1, int(os.environ.get("LUNA_EVRARD_AMR_LEVELS", "3"))),
        amr_regrid_interval=int(os.environ.get("LUNA_EVRARD_AMR_REGRID", "8")),
        amr_subcycle=False,
        amr_shard=os.environ.get("LUNA_EVRARD_AMR_SHARD", "0") == "1",
    )

    print(
        f"Evrard collapse: Nx={Nx}, M={M}, R={R}, u={u}, G={G}, L={L}, "
        f"t_end={t_end:.3f}, t_ff={t_ff:.3f} (~{t_end/t_ff:.2f} t_ff)"
    )
    initialize_jax_distributed(enable_x64=bool(overrides.get("enable_x64", False)))
    overrides = {k: v for k, v in overrides.items() if k != "enable_x64"}
    run(replace(Config(), **overrides))


if __name__ == "__main__":
    main()
