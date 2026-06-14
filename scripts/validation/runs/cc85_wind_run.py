"""CC85 starburst wind (M82-nuclear-region parameters), 3D smoke test.

Box: 10 x 10 x 20 kpc. Defaults to 128 x 128 x 256 (78 pc per cell).

Env vars:
    LUNA_CC85_NX   : x-resolution (default 128). Ny = Nx, Nz = 2*Nx.
    LUNA_CC85_TEND : simulation end time in Myr (default 10).
"""
import os
from dataclasses import replace
from template_config import Config
from setup.distributed_setup import initialize_jax_distributed
from main import run
from units import UnitSystem


SEC_PER_YR = 3.15576e7
MYR = 1e6 * SEC_PER_YR
KPC = 3.08568e21
MSUN = 1.98892e33

LX_KPC = 10.0
LY_KPC = 10.0
LZ_KPC = 20.0


def main():
    Nx = int(os.environ.get("LUNA_CC85_NX", "128"))
    Ny = Nx
    Nz = 2 * Nx
    t_end_Myr = float(os.environ.get("LUNA_CC85_TEND", "10.0"))

    # Arepo units (kpc, 1e10 Msun, km/s)
    units = UnitSystem(
        UnitLength_in_cm=KPC,
        UnitMass_in_g=1.0e10 * MSUN,
        UnitVelocity_in_cm_per_s=1.0e5,
    )
    t_end_code = t_end_Myr * MYR / units.UnitTime_in_s

    overrides = dict(
        test_problem="CC85-Wind-3D",
        SPATIAL_DIMENSION=3,
        BOUNDARY=1,                          # outflow on all six faces
        Nx=Nx, Ny=Ny, Nz=Nz,
        Lx=LX_KPC * 1e-3 * (KPC / units.UnitLength_in_cm),  # kpc -> code (=1 with our units)
        Ly=LY_KPC * 1e-3 * (KPC / units.UnitLength_in_cm),
        Lz=LZ_KPC * 1e-3 * (KPC / units.UnitLength_in_cm),
        gamma=5.0 / 3.0,
        ORDER=1,
        LIMITER=0,                           # minmod — wind has shocks
        CFL=0.4,
        t_end=t_end_code,
        time_bet_snaps=t_end_code / 20.0,
        units=units,
        riemann_solver="hllc",
        # CC85 source
        cc85_enabled=True,
        cc85_R_star_pc=300.0,
        cc85_SFR_Msun_per_yr=12.0,
        cc85_eta_M=0.3,
        cc85_eta_E=0.6,
        cc85_n_ambient_cgs=1.0e-4,
        cc85_T_ambient_K=1.0e6,
        cc85_mu=0.6,
        # Cooling
        cooling_enabled=True,
        cooling_table="sd93_solar",
        # Output
        PLOT_AFTER_SIM=False,
        production_mode=True,
        output_dir=f"snapshots/cc85_Nx{Nx}",
        clear_output_dir=True,
        enable_x64=True,
    )

    # Sanity: Lx in code units should equal kpc-count since UnitLength = 1 kpc.
    overrides["Lx"] = LX_KPC
    overrides["Ly"] = LY_KPC
    overrides["Lz"] = LZ_KPC

    print(f"CC85 M82 wind: Nx={Nx}, Ny={Ny}, Nz={Nz}; "
          f"box {LX_KPC}x{LY_KPC}x{LZ_KPC} kpc; "
          f"res={LX_KPC*1000/Nx:.1f} pc; t_end={t_end_Myr} Myr "
          f"(={t_end_code:.4e} code)")
    initialize_jax_distributed(enable_x64=bool(overrides.get("enable_x64", False)))
    overrides = {k: v for k, v in overrides.items() if k != "enable_x64"}
    run(replace(Config(), **overrides))


if __name__ == "__main__":
    main()
