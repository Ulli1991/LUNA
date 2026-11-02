"""3D Sedov-Taylor blast through the block-AMR path.

    python -m scripts.validation.runs.sedov3d_amr_run nx=32 amr_levels=2 t_end=0.02
    python -m scripts.validation.runs.sedov3d_amr_run nx=64 amr_levels=3 amr_subcycle=1

Pressure-gradient refinement catches the t=0 energy spike and tracks the
expanding shell. Code-unit Sedov IC (rho=1, E=0.5, unit box), base grid nx^3.

Parameters: nx, t_end and n_snap. The AMR knobs are Config fields set as
field=value: amr_levels, amr_block_shape, amr_criterion, amr_regrid_interval,
amr_max_blocks, amr_dense_snapshots, amr_rho_floor, amr_p_floor, amr_subcycle,
amr_shard, output_dir.
"""
from dataclasses import replace

from luna.setup.cli import script_main
from luna.template_config import Config


def config(nx=32, t_end=0.02, n_snap=2):
    return replace(
        Config(),
        test_problem="Sedov-Blast-3D", SPATIAL_DIMENSION=3,
        Nx=nx, Ny=nx, Nz=nx, Lx=1.0, Ly=1.0, Lz=1.0, gamma=5.0 / 3.0,
        ORDER=1, LIMITER=0, BOUNDARY=1, riemann_solver="hllc",
        t_end=t_end, time_bet_snaps=t_end / n_snap,
        AMR=True, amr_block_shape=16, amr_levels=2, amr_criterion="pressure",
        amr_regrid_interval=2, amr_dense_snapshots=True,
        enable_checkpointing=False, PLOT_AFTER_SIM=False,
        output_dir="snapshots/amr_sedov3d", clear_output_dir=True, enable_x64=True,
    )


if __name__ == "__main__":
    script_main(config)
