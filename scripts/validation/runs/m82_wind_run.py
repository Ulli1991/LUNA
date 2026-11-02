"""M82-like starburst wind (Schneider+ 2018 CC85 setup).

    python -m scripts.validation.runs.m82_wind_run nx=128 t_end_myr=10
    python -m scripts.validation.runs.m82_wind_run nx=64 cool=0 grav=0   # bisection
    python -m scripts.validation.runs.m82_wind_run nx=128 enable_x64=0    # fp32

A rotating exponential gas disk in vertical HSE in a static Miyamoto-Nagai +
NFW potential, CC85 central injection (SFR 20 Msun/yr, eta_M 0.6, eta_E 0.9:
the CGOLS high-mass-loading phase) and Sutherland-Dopita CIE cooling.

Parameters: nx (Ny = nx, Nz = 2 nx), t_end_myr, n_snaps, and the three
feature toggles cc85, grav and cool, which also tag the output directory.
"""
from dataclasses import replace

from luna.setup.cli import script_main
from luna.ic_generation import m82_disk_3d
from luna.template_config import Config
from luna.units import UnitSystem

MYR = 1e6 * 3.15576e7
KPC = 3.08568e21
MSUN = 1.98892e33
LX_KPC, LY_KPC, LZ_KPC = 10.0, 10.0, 20.0


def config(nx=128, t_end_myr=10.0, n_snaps=20, cc85=True, grav=True, cool=True):
    units = UnitSystem(UnitLength_in_cm=KPC, UnitMass_in_g=1.0e10 * MSUN,
                       UnitVelocity_in_cm_per_s=1.0e5)
    t_end = t_end_myr * MYR / units.UnitTime_in_s
    tag = ("" if cc85 else "_no_cc85") + ("" if grav else "_no_grav") + ("" if cool else "_no_cool")
    print(f"M82 disk + CC85 wind: Nx={nx}, Ny={nx}, Nz={2 * nx}; box {LX_KPC}x{LY_KPC}x{LZ_KPC} kpc; "
          f"res={LX_KPC * 1000 / nx:.1f} pc; t_end={t_end_myr} Myr")
    return replace(
        Config(),
        test_problem="M82-Disk-3D", SPATIAL_DIMENSION=3, BOUNDARY=1,
        Nx=nx, Ny=nx, Nz=2 * nx, Lx=LX_KPC, Ly=LY_KPC, Lz=LZ_KPC,
        gamma=5.0 / 3.0, ORDER=1, LIMITER=0, riemann_solver="hllc",
        t_end=t_end, time_bet_snaps=t_end / n_snaps, units=units,
        cc85_enabled=cc85, cc85_R_star_pc=300.0, cc85_SFR_Msun_per_yr=20.0,
        cc85_eta_M=0.6, cc85_eta_E=0.9, cooling_mu=0.6,
        m82_use_external_gravity=grav,
        problem=m82_disk_3d.Params(M_gas=0.25, R_scale_kpc=1.6, R_trunc_kpc=4.5, T_K=1.0e4,
                                   n_ambient_cgs=1.0e-4, T_ambient_K=1.0e6),
        m82_stellar_MN_M=1.0, m82_stellar_MN_a_kpc=0.8, m82_stellar_MN_b_kpc=0.15,
        m82_halo_NFW_M_vir=50.0, m82_halo_NFW_c=20.0,
        cooling_enabled=cool, cooling_table="sd93_solar",
        PLOT_AFTER_SIM=False,
        output_dir=f"snapshots/m82_Nx{nx}{tag}", clear_output_dir=True,
        enable_x64=True, enable_checkpointing=False,
    )


if __name__ == "__main__":
    script_main(config)
