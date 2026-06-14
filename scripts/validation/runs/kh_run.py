"""Run LUNA Kelvin-Helmholtz at a single resolution and save snapshots.

Driven by env vars so one submit script can dispatch different Nx values:
    LUNA_KH_NX  : integer resolution (default 1024). LUNA runs Nx×Nx grid.
    LUNA_KH_OUT : output dir (default snapshots/kh_Nx{Nx}).

Domain: Lx=Ly=2 to fit LUNA's double-tanh IC (shear layers at y=0.5 and 1.5).
Snapshots written every 0.5 simulation time → 11 snapshots over t∈[0,5].
The kh_grid_plot.py script then picks out t ∈ {0, 1, 2.5, 4, 5} from these.
"""
import os
from dataclasses import replace
from template_config import Config
from setup.distributed_setup import initialize_jax_distributed
from main import run


def main():
    Nx = int(os.environ.get("LUNA_KH_NX", "1024"))
    out_dir = os.environ.get("LUNA_KH_OUT", f"snapshots/kh_Nx{Nx}")
    overrides = dict(
        test_problem="Kelvin-Helmholtz",
        SPATIAL_DIMENSION=2,
        BOUNDARY=0,            # periodic top/bottom and left/right
        Nx=Nx, Ny=Nx,
        Lx=2.0, Ly=2.0,        # LUNA's KH IC uses tanh layers at y=0.5 and 1.5
        gamma=5.0 / 3.0,
        t_end=5.0,
        time_bet_snaps=0.5,
        ORDER=1,
        LIMITER=1,             # MonCen (smooth-flow friendly)
        CFL=0.4,
        PLOT_AFTER_SIM=False,
        production_mode=True,
        output_dir=out_dir,
        clear_output_dir=True,
        DECOMPOSITION="auto",  # let LUNA pick slab/block per GPU count
    )
    initialize_jax_distributed(enable_x64=bool(overrides.get("enable_x64", False)))
    overrides = {k: v for k, v in overrides.items() if k != "enable_x64"}
    run(replace(Config(), **overrides))


if __name__ == "__main__":
    main()
