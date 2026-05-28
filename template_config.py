# template_config.py
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional
import numpy as np

from units import UnitSystem
from hydro.turbulence import OUDriverConfig

@dataclass
class Config:

    # -------------------------------------------------------------------------
    # Test problem selection
    # -------------------------------------------------------------------------
    # Available options:
    #   1D:
    #       "Sod-Shock"
    #       "Woodward-Colella"
    #       "Shu-Osher"
    #       "Soundwave"
    #       "Soundwave-Advected"
    #
    #   Einfeldt (1D):
    #       "Einfeldt-Strong"
    #       "Einfeldt-Shear"
    #       "Einfeldt-Transverse"
    #
    #   2D Advection:
    #       "Square-Advected"
    #       "Rotate-Slotted-Disk"
    #       "Soundwave-Advected-2D"
    #
    #   2D Instability / Shock:
    #       "Kelvin-Helmholtz"
    #       "Double-Mach-Reflection"
    #
    #   3D:
    #       "Sod-Shock-3D"
    #
    test_problem: str = "Sod-Shock"

    # ---------------- Device ----------------

    # 0 = CPU
    # 1 = GPU
    # 2 = TPU
    DEVICE: int = 1

    # ---------------- Domain ----------------
    # 1 = 1D
    # 2 = 2D
    # 3 = 3D (not meaningfully implemented yet)
    SPATIAL_DIMENSION: int = 1
    # 0 = Periodic
    # 1 = von Neumann
    # 2 = Reflective
    BOUNDARY: int = 1

    # Domain decomposition strategy (for multi-GPU parallelization)
    # "auto"   = Automatically choose based on problem size and GPU count (recommended)
    # "slab"   = 1D decomposition (pencil/slab) - minimizes communication messages
    # "block"  = 2D/3D decomposition (block) - minimizes surface area
    #
    # Guidelines:
    #   - "slab" is typically faster for P ≤ 32 GPUs (lower latency overhead)
    #   - "block" is better for P > 32 or very large N (> 32768)
    #   - "auto" uses heuristics to choose the best strategy
    DECOMPOSITION: str = "auto"

    Nx: int = 1024
    Ny: int = 1024
    Nz: int = 1024
    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 1.0

    # ---------------- Physics ----------------
    gamma: float = 5.0 / 3.0
    #gamma: float = 1.4

    # ---------------- Time Integration ----------------
    CFL: float = 0.8

    # Only MUSCL-Hancock (predictor-corrector, 2nd-order) is supported.
    # Field retained for backward compatibility of run scripts.
    TIME_SOLVER: int = 1

    # ---------------- Riemann Solver ----------------
    # Reconstruction order:
    # 0 = first order Godunov
    # 1 = PLM (second-order MUSCL)
    # 2 = PPM (future, not implemented yet)
    ORDER: int = 1

    # Slope limiter (used only if ORDER == 1)
    # 0 = Minmod
    # 1 = MonCen
    # 2 = VanLeer
    LIMITER: int = 0

    # Soundwave IC variant. True = single right-going acoustic eigenmode (matches
    # AthenaK's linear_wave). False = u=0 superposition of left+right waves.
    soundwave_single_mode: bool = True

    # ---------------- Time control ----------------
    t_start: float = 0.0
    t_end: float = 0.245

    # Time interval between snapshots (code units)
    time_bet_snaps: float = 0.01

    # ---------------- Output ----------------
    output_dir: str = "snapshots"
    base_name: str = "snap"
    PLOT_AFTER_SIM: bool = True

    # How often to print progress and validate (in steps)
    output_cadence: int = 10

    # Print memory usage during run
    check_memory: bool = True

    # Enable physics diagnostics (min_rho, min_p, NaN/Inf checks)
    # Disable for maximum performance (10-20% speedup)
    enable_diagnostics: bool = True

    # Production mode: disables ALL performance-killing features
    # - Disables validation checks in time loop
    # - Disables verbose progress printing
    # - Disables memory checks
    # - Minimizes CPU-GPU synchronization
    production_mode: bool = False

    # ---------------- Restart / Checkpoint ----------------
    # Enable restart from checkpoint
    restart: bool = False

    # Checkpoint file to restart from (if None, finds latest checkpoint)
    restart_file: str = None

    # Directory for checkpoint files
    checkpoint_dir: str = "checkpoints"

    # Enable automatic checkpointing during simulation
    enable_checkpointing: bool = False

    # Wall-clock time interval between checkpoints (seconds)
    # Set to 0 to disable periodic checkpointing
    # Example: 3600 = checkpoint every hour, 7200 = every 2 hours
    checkpoint_wall_time_interval: float = 7200.0

    # Maximum number of checkpoints to keep (0 = keep all)
    max_checkpoints: int = 5

    # ---------------- Units ----------------
    # Default: dimensionless / reduced units (all base units = 0 -> is_enabled=False).
    # Validation tests (Sod, soundwave, KH, etc.) run dimensionlessly.
    # Override in a derived config (like the user's config.py) to attach a
    # physical unit system for production / ISM physics runs.
    units: UnitSystem = field(default_factory=UnitSystem)

    # ---------------- OU turbulent driving ----------------
    # Optional Ornstein-Uhlenbeck forcing applied as an operator-split momentum
    # source after each hydro step (see hydro/turbulence.py). When None,
    # the runner doesn't touch U after the hydro step.
    ou_config: Optional[OUDriverConfig] = None
    ou_seed: int = 7                                 # PRNG seed for OU evolve()

    # ---------------- Gresho vortex (low-Mach test) ----------------
    gresho_Mmax: float = 0.01

    # ---------------- Solar convection (polytropic atmosphere) ----------------
    solar_rho0: float = 1.0          # density at the bottom of the box
    solar_p0: float = 1.0            # pressure at the bottom of the box
    solar_Gamma: float = 0.0         # polytropic index; 0 -> use cfg.gamma (adiabatic)
    solar_pert_amp: float = 0.0      # random entropy seed amplitude
    solar_pert_seed: int = 11

    # ---------------- CC85 starburst wind (M82-style) ----------------
    # Mdot/Edot injection inside a sphere of radius R_*. Requires cfg.units
    # enabled (we convert SFR + loading factors to code units at setup time).
    cc85_enabled: bool = False
    cc85_R_star_pc: float = 300.0          # injection sphere radius (pc)
    cc85_SFR_Msun_per_yr: float = 12.0     # star formation rate
    cc85_eta_M: float = 1.0                # mass loading: Mdot = eta_M * SFR
    cc85_eta_E: float = 0.6                # energy loading: Edot = eta_E * (SFR/100) * 1e51 erg/yr
    cc85_n_ambient_cgs: float = 1.0e-4     # ambient CGM density
    cc85_T_ambient_K: float = 1.0e6        # ambient CGM temperature
    cc85_mu: float = 0.6                   # mean molecular weight

    # ---------------- M82 disk galaxy (Schneider+ 2018 CC85 setup) ----------------
    # Rotating exponential gas disk in vertical HSE in a static
    # Miyamoto-Nagai (stellar) + NFW (dark halo) potential. When enabled,
    # the runner applies a position-dependent gravity source term every
    # step using hydro/external_gravity.py. The disk IC is selected via
    # test_problem = "M82-Disk-3D".
    m82_use_external_gravity: bool = False
    m82_disk_M_gas: float = 0.25                    # 2.5e9 Msun in code (1e10 Msun)
    m82_disk_R_scale_kpc: float = 1.6
    m82_disk_R_trunc_kpc: float = 4.5
    m82_disk_T_K: float = 1.0e4
    m82_stellar_MN_M: float = 1.0                   # 1e10 Msun
    m82_stellar_MN_a_kpc: float = 0.8
    m82_stellar_MN_b_kpc: float = 0.15
    m82_halo_NFW_M_vir: float = 50.0                # 5e11 Msun in code
    m82_halo_NFW_c: float = 20.0

    # ---------------- Radiative cooling ----------------
    # When True, apply Townsend (2009) cooling step + heating after each
    # hydro step. Requires cfg.units enabled.
    cooling_enabled: bool = False
    cooling_table: str = "sd93_solar"      # "sd93_solar" (Sutherland-Dopita 93)
    cooling_T_floor_K: float = 1.0e4
    cooling_T_max_K:   float = 1.0e9

    # ---------------- Gravity + well-balanced ----------------
    # Constant gravity along z (negative = pointing down). When non-zero and
    # use_well_balanced=True, the integrator subtracts the equilibrium flux
    # F_eq at every face and applies the gravity source term as a perturbation
    # update (Velasco-Romero & Teyssier 2025). Pre-built U_eq is sourced from
    # ic.equilibrium_state(cfg, ...) at setup time.
    gravity_z: float = 0.0
    use_well_balanced: bool = False

    # ---------------- Riemann solver selection ----------------
    # "hllc"  -> hydro.fluxes.hllc_flux_primitive (Davis-style HLLC; default)
    # "lhllc" -> hydro.fluxes_lhllc.lhllc_flux_primitive (Toro 10.71 + V&T 2025
    #           low-Mach correction; intended for stratified low-Mach flows like
    #           solar convection).
    riemann_solver: str = "hllc"

    # ---------------- Isothermal projection ----------------
    # When > 0, after each hydro+forcing step the runner overwrites the total
    # energy so the equation of state effectively becomes p = isothermal_cs2 * rho.
    # This is the "soft thermostat" version of an isothermal EOS: the solver
    # internals keep using gamma (HLLC, CFL) but no thermal energy accumulates.
    isothermal_cs2: float = 0.0

    # -------------------------------------------------------------------------
    # Computed properties (lightweight, timing-related)
    # -------------------------------------------------------------------------

    @cached_property
    def snapshot_times(self) -> np.ndarray:
        """Pre-computed snapshot times (using numpy, not JAX, to avoid GPU allocation)."""
        return np.arange(self.t_start, self.t_end + self.time_bet_snaps/2, self.time_bet_snaps)

    @cached_property
    def use_predictor(self) -> bool:
        """Whether to use predictor step (MUSCL-Hancock only)."""
        return self.TIME_SOLVER == 1
