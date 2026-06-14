"""2D solar convection run (well-balanced + L-HLLC + gravity).

Polytropic stratification with Gamma slightly larger than gamma so the
profile is Schwarzschild-unstable (entropy decreases with height), plus a
small random entropy seed to break the symmetry and trigger convection.

Box: 1x1 periodic-xy, vertical=y. Gravity g_z=-1.

Driven by env vars so the same submit script can dispatch resolutions:
    LUNA_SOLAR_NX    : grid resolution (default 128)
    LUNA_SOLAR_OUT   : output dir (default snapshots/solar_Nx{Nx})
"""
import os
from dataclasses import replace
from template_config import Config
from setup.distributed_setup import initialize_jax_distributed
from main import run


GAMMA = 5.0 / 3.0
SOLAR_GAMMA = 1.70                # slightly > gamma -> superadiabatic, convective
RHO0 = 1.0
P0 = 1.0
GRAVITY = -1.0
LX = LY = 1.0
T_END = 20.0                      # several thermal/turnover times for small superadiabaticity
N_SNAPS = 20
PERT_AMP = 1.0e-3
PERT_SEED = 11
CFL = 0.4


def main():
    Nx = int(os.environ.get("LUNA_SOLAR_NX", "128"))
    out_dir = os.environ.get("LUNA_SOLAR_OUT", f"snapshots/solar_Nx{Nx}")
    overrides = dict(
        test_problem="Solar-Convection-2D",
        SPATIAL_DIMENSION=2,
        BOUNDARY=0,                 # periodic xy (TODO: closed-z BC for a follow-up)
        Nx=Nx, Ny=Nx,
        Lx=LX, Ly=LY,
        gamma=GAMMA,
        ORDER=1,
        LIMITER=1,                  # MonCen (smooth-flow friendly)
        CFL=CFL,
        t_end=T_END,
        time_bet_snaps=T_END / N_SNAPS,
        gravity_z=GRAVITY,
        use_well_balanced=True,
        riemann_solver="lhllc",
        solar_rho0=RHO0,
        solar_p0=P0,
        solar_Gamma=SOLAR_GAMMA,
        solar_pert_amp=PERT_AMP,
        solar_pert_seed=PERT_SEED,
        PLOT_AFTER_SIM=False,
        production_mode=True,
        output_dir=out_dir,
        clear_output_dir=True,
        enable_x64=True,
    )
    print(f"Solar convection 2D: Nx={Nx}, Gamma={SOLAR_GAMMA} (gamma={GAMMA}), "
          f"g_z={GRAVITY}, t_end={T_END}, pert_amp={PERT_AMP}")
    initialize_jax_distributed(enable_x64=bool(overrides.get("enable_x64", False)))
    overrides = {k: v for k, v in overrides.items() if k != "enable_x64"}
    run(replace(Config(), **overrides))


if __name__ == "__main__":
    main()
