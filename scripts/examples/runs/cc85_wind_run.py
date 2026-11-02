"""CC85 starburst wind with M82 nuclear-region parameters, a 3D smoke test.

    python -m scripts.examples.runs.cc85_wind_run nx=128 t_end_myr=10

Box 10 x 10 x 20 kpc on nx x nx x 2nx cells (78 pc per cell at 128).
Parameters: nx, t_end_myr.
"""
from dataclasses import replace

from luna.setup.cli import script_main
from luna.ic_generation import cc85_wind_3d
from luna.template_config import Config
from luna.units import UnitSystem

MYR = 1e6 * 3.15576e7
KPC = 3.08568e21
MSUN = 1.98892e33
LX_KPC, LY_KPC, LZ_KPC = 10.0, 10.0, 20.0


def config(nx=128, t_end_myr=10.0):
    units = UnitSystem(UnitLength_in_cm=KPC, UnitMass_in_g=1.0e10 * MSUN,
                       UnitVelocity_in_cm_per_s=1.0e5)      # Arepo units
    t_end = t_end_myr * MYR / units.UnitTime_in_s
    print(f"CC85 M82 wind: Nx={nx}, Ny={nx}, Nz={2 * nx}; box {LX_KPC}x{LY_KPC}x{LZ_KPC} kpc; "
          f"res={LX_KPC * 1000 / nx:.1f} pc; t_end={t_end_myr} Myr (={t_end:.4e} code)")
    return replace(
        Config(),
        test_problem="CC85-Wind-3D", SPATIAL_DIMENSION=3, BOUNDARY=1,
        Nx=nx, Ny=nx, Nz=2 * nx, Lx=LX_KPC, Ly=LY_KPC, Lz=LZ_KPC,
        gamma=5.0 / 3.0, ORDER=1, LIMITER=0, riemann_solver="hllc",
        t_end=t_end, time_bet_snaps=t_end / 20.0, units=units,
        cc85_enabled=True, cc85_R_star_pc=300.0, cc85_SFR_Msun_per_yr=12.0,
        cc85_eta_M=0.3, cc85_eta_E=0.6, cooling_mu=0.6,
        problem=cc85_wind_3d.Params(n_ambient_cgs=1.0e-4, T_ambient_K=1.0e6),
        cooling_enabled=True, cooling_table="sd93_solar",
        PLOT_AFTER_SIM=False,
        output_dir=f"snapshots/cc85_Nx{nx}", clear_output_dir=True, enable_x64=True,
    )


if __name__ == "__main__":
    script_main(config)
