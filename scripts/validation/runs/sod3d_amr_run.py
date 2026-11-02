"""3D planar Sod shock tube through the block-AMR path (benchmark B2).

    python -m scripts.validation.runs.sod3d_amr_run nx=64 amr_levels=2
    python -m scripts.validation.runs.sod3d_amr_run nx=64 amr_levels=0 \\
        output_dir=snapshots/sod_uniform

Density-gradient refinement follows the shock and the contact (the contact
has no pressure jump). Classic Sod states, gamma = 1.4, t_end = 0.2. The
uniform reference is the same script with amr_levels=0.

Parameters: nx, nt (transverse cells, one block) and t_end. The rest are
Config fields: amr_levels, amr_criterion, amr_regrid_interval, amr_subcycle,
amr_shard, BOUNDARY (1 outflow, 0 periodic), output_dir.
"""
from dataclasses import replace

from luna.setup.cli import script_main
from luna.template_config import Config


def config(nx=64, nt=16, t_end=0.2):
    return replace(
        Config(),
        test_problem="Sod-Shock-3D-AMR", SPATIAL_DIMENSION=3,
        Nx=nx, Ny=nt, Nz=nt, Lx=1.0, Ly=nt / nx, Lz=nt / nx,      # cubic cells
        gamma=1.4, ORDER=1, LIMITER=0, BOUNDARY=1, riemann_solver="hllc",
        t_end=t_end, time_bet_snaps=t_end / 2.0,
        AMR=True, amr_block_shape=16, amr_levels=2, amr_criterion="density",
        amr_regrid_interval=2, amr_dense_snapshots=True,
        enable_checkpointing=False, PLOT_AFTER_SIM=False,
        output_dir="snapshots/amr_bench_sod", clear_output_dir=True, enable_x64=True,
    )


if __name__ == "__main__":
    script_main(config)
