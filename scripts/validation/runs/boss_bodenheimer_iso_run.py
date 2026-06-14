"""Boss-Bodenheimer 1979 collapse — physical (cgs) units + barotropic EOS.

Built verbatim from GANDALF (Hubber, Rosotti & Booth 2018, MNRAS 473, 1603,
§6.12) / Boss & Bodenheimer 1979:
    M ≈ 1 M_sun cloud, R = 0.01 pc, ρ₀ = 1.44e-17 g/cm³,
    ρ = ρ₀[1 + 0.5 sin(2φ)],  Ω = 1.6e-12 rad/s (β ≈ 0.16),
    barotropic EOS  T(ρ) = 10 K [1 + (ρ/1e-14)^(γ-1)], μ = 2.35, γ = 5/3,
    evolved to t_end = 0.04 Myr.

Star-formation UnitSystem (pc, M_sun, km/s). Self-gravity with the isolated
multipole-Dirichlet Poisson BC + a diode (outflow-only) hydro BC — the diode
blocks the axis-aligned ambient inflow jets that otherwise imprint a cardinal
cross, and the barotropic stiffening halts the runaway isothermal collapse.

Env: LUNA_BB_NX, LUNA_BB_TEND_MYR, LUNA_BB_HYDRO_BC (4=diode default, 1=outflow).
"""
import os

from dataclasses import replace

from units import UnitSystem
from template_config import Config
from setup.distributed_setup import initialize_jax_distributed
from main import run

PC = 3.0856775814913673e18
MSUN = 1.98892e33
MYR = 1.0e6 * 3.15576e7


def main():
    Nx = int(os.environ.get("LUNA_BB_NX", "64"))
    t_Myr = float(os.environ.get("LUNA_BB_TEND_MYR", "0.04"))
    hydro_bc = int(os.environ.get("LUNA_BB_HYDRO_BC", "4"))   # 4 = diode (outflow-only)
    riemann = os.environ.get("LUNA_BB_RIEMANN", "hllc")

    # star-formation units: pc, M_sun, km/s
    units = UnitSystem(UnitLength_in_cm=PC, UnitMass_in_g=MSUN,
                       UnitVelocity_in_cm_per_s=1.0e5)
    R_pc = 0.01
    box_fac = float(os.environ.get("LUNA_BB_BOXFAC", "2.0"))
    L = box_fac * R_pc                   # box = box_fac·R (default 2R: Truelove-safe
                                         # at 256³, ~7.6 cells per λ_J at ρ_AD)
    t_end = t_Myr * MYR / units.UnitTime_in_s

    suffix = {1: "_outflow", 0: "_periodic"}.get(hydro_bc, "")
    overrides = dict(
        test_problem="Boss-Bodenheimer-3D",
        SPATIAL_DIMENSION=3,
        BOUNDARY=hydro_bc,
        POISSON_BOUNDARY=3,              # multipole-Dirichlet isolated BC
        Nx=Nx, Ny=Nx, Nz=Nx,
        Lx=L, Ly=L, Lz=L,
        gamma=5.0 / 3.0,
        ORDER=1, LIMITER=0, CFL=0.3,
        riemann_solver=riemann,
        units=units,
        t_end=t_end, time_bet_snaps=t_end / 30,
        SELF_GRAVITY=True, GRAVITATIONAL_G=units.GravityConstant,
        POISSON_MAX_CYCLES=20,
        # ---- paper IC (physical cgs); Ω derived from β=0.16 ----
        bb_rho0_cgs=1.44e-17, bb_R_pc=R_pc, bb_beta=0.16,
        bb_amp=0.5, bb_m=2, bb_rho_ambient_frac=1.0e-3,
        # ---- barotropic EOS ----
        barotropic_eos=True, barotropic_T0_K=10.0,
        barotropic_rho_AD_cgs=1.0e-14, barotropic_mu=2.35,
        PLOT_AFTER_SIM=False, production_mode=True,
        output_dir=f"snapshots/bb79_baro_Nx{Nx}{suffix}",
        clear_output_dir=True,
        enable_x64=True, enable_checkpointing=False,
    )
    print(f"BB79 (physical/barotropic): Nx={Nx}, R={R_pc} pc, rho0=1.44e-17, A=0.5, "
          f"beta=0.16, T0=10K, rho_AD=1e-14, BC={hydro_bc}, riemann={riemann}, "
          f"t_end={t_Myr} Myr ({t_end:.4f} code), G_code={units.GravityConstant:.4e}")
    initialize_jax_distributed(enable_x64=bool(overrides.get("enable_x64", False)))
    overrides = {k: v for k, v in overrides.items() if k != "enable_x64"}
    run(replace(Config(), **overrides))


if __name__ == "__main__":
    main()
