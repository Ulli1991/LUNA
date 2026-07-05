"""3D Sedov-Taylor blast through the block-AMR path (cfg.AMR=True -> runner_amr).

Pressure-gradient refinement: catches the t=0 energy spike AND tracks the
expanding shock shell. Code-unit Sedov IC (rho=1, E=0.5, unit box). Base grid
Nx^3, block 16, amr_levels refinement levels; the composite finest field is
written each snapshot.
"""
import os
from dataclasses import replace
from template_config import Config
from setup.distributed_setup import initialize_jax_distributed
from main import run

N = int(os.environ.get("AMR_SEDOV_NX", "32"))
LEVELS = int(os.environ.get("AMR_LEVELS", "2"))
T_END = float(os.environ.get("AMR_TEND", "0.02"))

cfg = replace(
    Config(),
    test_problem="Sedov-Blast-3D",
    SPATIAL_DIMENSION=3,
    Nx=N, Ny=N, Nz=N,
    Lx=1.0, Ly=1.0, Lz=1.0,
    gamma=5.0 / 3.0,
    CFL=0.4, ORDER=1, LIMITER=0,
    BOUNDARY=1,                     # outflow
    riemann_solver="hllc",
    t_end=T_END,
    time_bet_snaps=T_END / 2.0,     # ~2 snapshots
    AMR=True,
    amr_block_shape=16,
    amr_levels=LEVELS,
    amr_criterion=os.environ.get("AMR_CRITERION", "pressure"),
    amr_subcycle=False,
    amr_regrid_interval=int(os.environ.get("AMR_REGRID", "2")),
    amr_max_blocks=int(os.environ.get("AMR_MAX_BLOCKS", "0")),
    amr_dense_snapshots=os.environ.get("AMR_DENSE_SNAP", "1") == "1",
    amr_rho_floor=float(os.environ.get("AMR_RHO_FLOOR", "0.0")),
    amr_p_floor=float(os.environ.get("AMR_P_FLOOR", "0.0")),
    amr_shard=os.environ.get("AMR_SHARD", "0") == "1",
    enable_checkpointing=False,
    PLOT_AFTER_SIM=False,
    production_mode=True,
    output_dir=os.environ.get("AMR_SEDOV_OUT", "snapshots/amr_sedov3d"),
    clear_output_dir=True,
)

initialize_jax_distributed(enable_x64=True)
run(cfg)
