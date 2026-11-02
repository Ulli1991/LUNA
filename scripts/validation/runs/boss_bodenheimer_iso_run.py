"""Boss-Bodenheimer 1979 collapse in physical (cgs) units with a barotropic EOS.

    python -m scripts.validation.runs.boss_bodenheimer_iso_run nx=128 t_end_myr=0.06

Built verbatim from GANDALF (Hubber, Rosotti & Booth 2018, MNRAS 473, 1603,
sec. 6.12) / Boss & Bodenheimer 1979: M ~ 1 Msun, R = 0.01 pc,
rho0 = 1.44e-17 g/cm^3, rho = rho0 [1 + 0.5 sin(2 phi)], Omega = 1.6e-12 rad/s
(beta ~ 0.16), T(rho) = 10 K [1 + (rho/1e-14)^(gamma-1)], mu = 2.35, gamma = 5/3.
Star-formation units (pc, Msun, km/s); isolated multipole-Dirichlet Poisson
BC and a diode hydro BC (blocks the axis-aligned ambient inflow jets).

Parameters: nx, t_end_myr, hydro_bc (4 diode, 1 outflow, 0 periodic; it also
tags the output directory), riemann, sinks (on-the-fly sink particles),
sink_fac (rho_create = sink_fac * rho_AD), box_fac (box = box_fac * R) and
n_snap. output_dir and the other Config fields follow as field=value.
"""
from dataclasses import replace

from luna.setup.cli import script_main
from luna.ic_generation import boss_bodenheimer3d
from luna.template_config import Config
from luna.units import UnitSystem

PC = 3.0856775814913673e18
MSUN = 1.98892e33
MYR = 1.0e6 * 3.15576e7


def config(nx=64, t_end_myr=0.06, hydro_bc=4, riemann="hllc", sinks=False,
           sink_fac=1.0, box_fac=2.0, n_snap=60):
    units = UnitSystem(UnitLength_in_cm=PC, UnitMass_in_g=MSUN,
                       UnitVelocity_in_cm_per_s=1.0e5)
    R_pc = 3.2e16 / PC                   # SEREN R = 3.2e16 cm = 0.010370 pc
    L = box_fac * R_pc                   # default 2R: Truelove-safe at 256^3
    t_end = t_end_myr * MYR / units.UnitTime_in_s
    suffix = {1: "_outflow", 0: "_periodic"}.get(hydro_bc, "")
    over = dict(
        test_problem="Boss-Bodenheimer-3D", SPATIAL_DIMENSION=3, BOUNDARY=hydro_bc,
        POISSON_BOUNDARY=3,              # multipole-Dirichlet isolated BC
        Nx=nx, Ny=nx, Nz=nx, Lx=L, Ly=L, Lz=L,
        gamma=5.0 / 3.0, ORDER=1, LIMITER=0, riemann_solver=riemann,
        units=units, t_end=t_end, time_bet_snaps=t_end / n_snap,
        SELF_GRAVITY=True, GRAVITATIONAL_G=units.GravityConstant,
        problem=boss_bodenheimer3d.Params(rho0_cgs=1.44e-17, R_pc=R_pc, omega_rad_s=1.6e-12,
                                          amp=0.5, m=2, rho_ambient_frac=1.0e-3),
        barotropic_eos=True, barotropic_T0_K=10.0,
        barotropic_rho_AD_cgs=1.0e-14, barotropic_mu=2.35,
        PLOT_AFTER_SIM=False,
        output_dir=f"snapshots/bb79_baro_Nx{nx}{suffix}",
        clear_output_dir=True, enable_x64=True, enable_checkpointing=False,
    )
    if sinks:
        # sinks once a core crosses rho_create = sink_fac * rho_AD (where the
        # barotropic EOS stiffens: the first hydrostatic core; SEREN 2e-12)
        rho_create = sink_fac * 2.0e-12 / units.UnitDensity_in_cgs
        over.update(sinks_enabled=True, sink_max_count=4,
                    sink_creation_density=rho_create, sink_accretion_radius=0.0,
                    sink_softening=0.0, sink_creation_separation=0.0,
                    output_dir=f"snapshots/bb79_baro_Nx{nx}{suffix}_sinks")
        print(f"  SINKS on: rho_create={rho_create:.3e} code ({sink_fac} rho_AD), N_max=4")
    print(f"BB79 (physical/barotropic): Nx={nx}, R={R_pc:.5f} pc, rho0=1.44e-17, A=0.5, "
          f"Omega=1.6e-12, T0=10K, rho_AD=1e-14, BC={hydro_bc}, riemann={riemann}, "
          f"t_end={t_end_myr} Myr ({t_end:.4f} code), G_code={units.GravityConstant:.4e}")
    return replace(Config(), **over)


if __name__ == "__main__":
    script_main(config)
