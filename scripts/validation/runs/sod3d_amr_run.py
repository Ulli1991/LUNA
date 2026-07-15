"""3D planar Sod shock tube through the block-AMR path (benchmark B2).

Density-gradient refinement follows the shock AND the contact (the contact has
no pressure jump, so the pressure criterion would miss it). Classic Sod states,
gamma=1.4, t_end=0.2 (all waves inside the box). Uniform reference = same
script with AMR_LEVELS=0 (identical IC/solver; refinement is the only change).
"""
import os
from dataclasses import replace
from template_config import Config
from setup.distributed_setup import initialize_jax_distributed
from main import run

N = int(os.environ.get("AMR_SOD_NX", "64"))
NT = int(os.environ.get("AMR_SOD_NT", "16"))           # transverse (1 block)
LEVELS = int(os.environ.get("AMR_LEVELS", "2"))
T_END = float(os.environ.get("AMR_TEND", "0.2"))

cfg = replace(
    Config(),
    test_problem="Sod-Shock-3D-AMR",
    SPATIAL_DIMENSION=3,
    Nx=N, Ny=NT, Nz=NT,
    Lx=1.0, Ly=NT / N, Lz=NT / N,                      # cubic cells
    gamma=1.4,
    CFL=0.4, ORDER=1, LIMITER=0,
    BOUNDARY=int(os.environ.get("AMR_BOUNDARY", "1")),  # 1=outflow 0=periodic
    riemann_solver="hllc",
    t_end=T_END,
    time_bet_snaps=T_END / 2.0,
    AMR=True,
    amr_block_shape=16,
    amr_levels=LEVELS,
    amr_criterion=os.environ.get("AMR_CRITERION", "density"),
    amr_subcycle=os.environ.get("AMR_SUBCYCLE", "0") == "1",
    amr_regrid_interval=int(os.environ.get("AMR_REGRID", "2")),
    amr_dense_snapshots=True,
    amr_shard=os.environ.get("AMR_SHARD", "0") == "1",
    enable_checkpointing=False,
    PLOT_AFTER_SIM=False,
    production_mode=True,
    output_dir=os.environ.get("AMR_SOD_OUT", "snapshots/amr_bench_sod"),
    clear_output_dir=True,
)

initialize_jax_distributed(enable_x64=True)
run(cfg)
