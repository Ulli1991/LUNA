"""M82-like starburst wind (Schneider+ 2018 CC85 setup).

Rotating exponential gas disk in vertical HSE in a static
Miyamoto-Nagai + NFW potential, with CC85 central injection
(SFR = 12 Msun/yr, eta_M = 0.3, eta_E = 0.6) and Sutherland-Dopita CIE
cooling.

Env vars:
    LUNA_M82_NX    : x-resolution (default 128). Ny = Nx, Nz = 2*Nx.
    LUNA_M82_TEND  : end time in Myr (default 10).
"""
import os
from scripts.validation.runner import run_one
from units import UnitSystem


SEC_PER_YR = 3.15576e7
MYR = 1e6 * SEC_PER_YR
KPC = 3.08568e21
MSUN = 1.98892e33

LX_KPC = 10.0
LY_KPC = 10.0
LZ_KPC = 20.0


def main():
    Nx = int(os.environ.get("LUNA_M82_NX", "128"))
    Ny = Nx
    Nz = 2 * Nx
    t_end_Myr = float(os.environ.get("LUNA_M82_TEND", "10.0"))
    n_snaps = int(os.environ.get("LUNA_M82_NSNAPS", "20"))
    use_fp64 = os.environ.get("LUNA_M82_FP32", "0") != "1"
    # Feature toggles for multi-node bisection.
    cc85_on = os.environ.get("LUNA_M82_CC85", "1") == "1"
    grav_on = os.environ.get("LUNA_M82_GRAV", "1") == "1"
    cool_on = os.environ.get("LUNA_M82_COOL", "1") == "1"

    units = UnitSystem(
        UnitLength_in_cm=KPC,
        UnitMass_in_g=1.0e10 * MSUN,
        UnitVelocity_in_cm_per_s=1.0e5,
    )
    t_end_code = t_end_Myr * MYR / units.UnitTime_in_s

    overrides = dict(
        test_problem="M82-Disk-3D",
        SPATIAL_DIMENSION=3,
        BOUNDARY=1,
        Nx=Nx, Ny=Ny, Nz=Nz,
        Lx=LX_KPC, Ly=LY_KPC, Lz=LZ_KPC,
        gamma=5.0 / 3.0,
        TIME_SOLVER=1,
        ORDER=1,
        LIMITER=0,
        CFL=0.3,
        t_end=t_end_code,
        time_bet_snaps=t_end_code / n_snaps,
        units=units,
        riemann_solver="hllc",
        # CC85 central source
        cc85_enabled=cc85_on,
        cc85_R_star_pc=300.0,
        # Schneider+ 2018 CGOLS "high mass-loading" phase (5-40 Myr):
        #   SFR = 20, eta_M = 0.6, eta_E = 0.9
        #   -> Mdot = 12 Msun/yr, Edot = 5.4e42 erg/s
        cc85_SFR_Msun_per_yr=20.0,
        cc85_eta_M=0.6,
        cc85_eta_E=0.9,
        cc85_n_ambient_cgs=1.0e-4,
        cc85_T_ambient_K=1.0e6,
        cc85_mu=0.6,
        # M82 disk + static potential
        m82_use_external_gravity=grav_on,
        m82_disk_M_gas=0.25,
        m82_disk_R_scale_kpc=1.6,
        m82_disk_R_trunc_kpc=4.5,
        m82_disk_T_K=1.0e4,
        m82_stellar_MN_M=1.0,
        m82_stellar_MN_a_kpc=0.8,
        m82_stellar_MN_b_kpc=0.15,
        m82_halo_NFW_M_vir=50.0,
        m82_halo_NFW_c=20.0,
        # Cooling
        cooling_enabled=cool_on,
        cooling_table="sd93_solar",
        # Output
        PLOT_AFTER_SIM=False,
        production_mode=True,
        output_dir=f"snapshots/m82_Nx{Nx}"
                   + ("_no_cc85" if not cc85_on else "")
                   + ("_no_grav" if not grav_on else "")
                   + ("_no_cool" if not cool_on else ""),
        enable_x64=use_fp64,
        enable_checkpointing=False,
    )

    print(f"M82 disk + CC85 wind: Nx={Nx}, Ny={Ny}, Nz={Nz}; "
          f"box {LX_KPC}x{LY_KPC}x{LZ_KPC} kpc; res={LX_KPC*1000/Nx:.1f} pc; "
          f"t_end={t_end_Myr} Myr")
    run_one(overrides)


if __name__ == "__main__":
    main()
