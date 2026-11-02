"""LUNA config for a Tier-1 coupled run: a turbulent molecular cloud.

Run through Torch via $LUNA_CONFIG, not main.py; see docs/TORCH.md."""
import math
import os
from dataclasses import replace

from luna.ic_generation import turbulent_sphere_3d
from luna.template_config import Config
from luna.units import UnitSystem

PARSEC_CGS = 3.085677581e18
SOLAR_MASS_CGS = 1.98892e33
BOLTZMANN_CGS = 1.380649e-16
HYDROGEN_MASS_CGS = 1.6726219e-24

# Code units: length in pc, mass in Msun, velocity in km/s.
UNITS = UnitSystem(UnitLength_in_cm=PARSEC_CGS,
                   UnitMass_in_g=SOLAR_MASS_CGS,
                   UnitVelocity_in_cm_per_s=1.0e5)

# --- the cloud ------------------------------------------------------------
BOX_PC = 8.0
N = 128
CLOUD_MASS = 2000.0      # Msun
CLOUD_RADIUS = 2.0       # pc
CLOUD_T = 15.0           # K

MEAN_WEIGHT = 2.3        # molecular, in m_H; also passed as sink_star_mu

DX = BOX_PC / N

# Isothermal sound speed of the cold cloud, in km/s.
CS = math.sqrt(BOLTZMANN_CGS * CLOUD_T
               / (MEAN_WEIGHT * HYDROGEN_MASS_CGS)) / 1.0e5

# Sink threshold: the Truelove density for a Jeans length of four cells.
RHO_CREATE = math.pi * CS ** 2 / (UNITS.GravityConstant * (4.0 * DX) ** 2)

# Free-fall time of the cloud as a whole, in code time (pc / (km/s) = 0.978 Myr).
RHO_MEAN = CLOUD_MASS / (4.0 / 3.0 * math.pi * CLOUD_RADIUS ** 3)
T_FF = math.sqrt(3.0 * math.pi / (32.0 * UNITS.GravityConstant * RHO_MEAN))

Config = replace(
    Config(),
    test_problem="Turbulent-Sphere-3D",
    SPATIAL_DIMENSION=3,
    Nx=N, Ny=N, Nz=N,
    Lx=BOX_PC, Ly=BOX_PC, Lz=BOX_PC,
    BOUNDARY=0,                          # periodic box around the sphere
    gamma=5.0 / 3.0,
    ORDER=1, LIMITER=0,                  # minmod
    enable_x64=True,
    units=UNITS,
    GRAVITATIONAL_G=UNITS.GravityConstant,
    problem=turbulent_sphere_3d.Params(mass_msun=CLOUD_MASS, radius_pc=CLOUD_RADIUS,
                                       T_K=CLOUD_T, mach=8.0),
    sink_star_mu=MEAN_WEIGHT,
    seed=1,

    # --- gravity ---------------------------------------------------------
    SELF_GRAVITY=True,
    POISSON_BOUNDARY=0,

    # --- sinks: LUNA forms and accretes them, Torch turns them into stars -
    sinks_enabled=True,
    sink_max_count=256,
    sink_creation_density=RHO_CREATE,
    sink_accretion_radius=0.0,           # 0 -> 2 dx
    sink_softening=0.0,                  # 0 -> 2 dx
    sink_creation_separation=0.0,        # 0 -> 4 dx

    sink_sink_gravity=False,             # PeTar owns star-star; on would double count
    star_max_count=4096,

    # Barotropic, not adiabatic: at mach 8 a total-energy scheme drives the
    # small thermal energy negative; P(rho) cannot go negative.
    barotropic_eos=True,
    barotropic_T0_K=CLOUD_T,
    barotropic_mu=MEAN_WEIGHT,
    barotropic_rho_AD_cgs=1.0e-14,
    cooling_enabled=False,               # the barotropic relation is the thermodynamics
    radiation_enabled=False,

    # --- output ----------------------------------------------------------
    # sinks form after about one free-fall time
    t_end=2.0 * T_FF,
    write_snapshots=True,
    time_bet_snaps=T_FF / 20.0,
    output_dir=os.environ.get("LUNA_OUT", "snapshots/tier1"),
    PLOT_AFTER_SIM=False,
)
