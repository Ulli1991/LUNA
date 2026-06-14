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

    # ---------------- Domain ----------------
    # 1 = 1D
    # 2 = 2D
    # 3 = 3D (not meaningfully implemented yet)
    SPATIAL_DIMENSION: int = 1
    # 0 = Periodic
    # 1 = von Neumann (zero-gradient; NOTE: permits boundary inflow under an
    #     inward pull — for self-gravitating isolated problems prefer 4)
    # 2 = Reflective
    # 4 = Diode outflow (zero-gradient, ghost normal velocity clamped against
    #     inflow; gas can leave, nothing enters). 3D auto-sharding path only.
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

    # AthenaK-style explicit mesh decomposition. Tuple matching SPATIAL_DIMENSION:
    #   1D: (nx_proc,)            -- product must equal len(jax.devices())
    #   2D: (nx_proc, ny_proc)
    #   3D: (nx_proc, ny_proc, nz_proc)
    # When set, this overrides DECOMPOSITION entirely and a Mesh of that exact
    # shape is built. Use this to escape the auto-heuristic's pencil/2D-slab
    # picks at small device counts when you want balanced 3D blocks (e.g.
    # (2, 2, 2) at 8 GPUs to halve y/z halo cost for 1024^3 fp32 runs).
    MESH_SHAPE: tuple = None

    # shard_map opt-in for the 3D MUSCL-Hancock step. When True the
    # cons_to_prim + halo + flux + update pipeline runs inside a
    # shard_map body — explicit per-shard ppermute for the halo,
    # per-shard intermediates only, sidesteps XLA's auto-sharding
    # working-set blowup at 1024^3 on small device counts. Default
    # False keeps the auto-sharded path (KH 2D, solar, etc.).
    USE_SHARDMAP: bool = False

    # Self-gravity (multigrid V-cycle Poisson solve each step, applied as
    # an operator-split source on momentum + energy). Phase 1: single-
    # process, periodic + Neumann BC only. See docs/multigrid_poisson_plan.md
    SELF_GRAVITY: bool = False
    GRAVITATIONAL_G: float = 1.0          # G in code units (set per problem)
    # Jeans-instability IC knobs (only used by test_problem='Jeans-Instability-3D')
    jeans_rho0: float = 1.0
    jeans_P0: float = 1.0
    jeans_amp: float = 1.0e-4
    jeans_kx: float = 0.0                # set to 2π/Lx in the run script
    jeans_ky: float = 0.0
    jeans_kz: float = 0.0
    # True: seed the pure growing Jeans eigenmode (adiabatic δP = c_s²δρ +
    # matching longitudinal velocity) → clean δρ(t)=A₀e^{σt}. False: legacy
    # static, equal-amplitude perturbation (a cosh+offset, needs a 3-param fit).
    jeans_eigenmode: bool = True
    # Evrard collapse IC knobs (test_problem='Evrard-Collapse-3D')
    evrard_M: float = 1.0
    evrard_R: float = 1.0
    evrard_u: float = 0.05
    evrard_rho_ambient: float = 0.0      # 0 → use default 1e-3 × ρ_cloud_avg
    # Bonnor-Ebert sphere IC (test_problem='Bonnor-Ebert-3D').
    # ξ_t = 5 → sub-critical (stable); ξ_t = 10 → super-critical (collapsing).
    be_xi_t: float = 5.0
    be_rho_c: float = 1.0                # placeholder; the IC unconditionally recomputes
                                         # ρ_c from (ξ_t, R, cs²) so the sphere geometry closes
    be_cs2: float = 1.0
    be_R: float = 1.0
    be_rho_ambient: float = 0.0
    # Lane-Emden polytrope IC (test_problem='Lane-Emden-3D' or 'Pulsating-Polytrope-3D').
    lane_n: float = 1.5                  # polytropic index (1.5 for γ=5/3)
    lane_rho_c: float = 1.0
    lane_R: float = 1.0
    lane_rho_ambient: float = 0.01       # 1% floor — keeps ambient c_s finite (polytropic P→0 at surface)
    pulse_amp: float = 0.01              # radial velocity perturbation amplitude
    # Maclaurin spheroid IC (test_problem='Maclaurin-Spheroid-3D').
    mac_e: float = 0.5                   # eccentricity
    mac_a: float = 1.0                   # equatorial semi-major axis
    mac_rho: float = 1.0
    mac_rho_ambient: float = 0.0
    # Boss-Bodenheimer IC (test_problem='Boss-Bodenheimer-3D').
    bb_M: float = 1.0
    bb_R: float = 1.0
    bb_cs2: float = 0.25                 # isothermal sound speed squared; with M=R=G=1 this equals
                                         # the thermal/grav ratio α = R·cs²/(GM), canonical α = 0.25
    bb_beta: float = 0.16
    bb_amp: float = 0.1
    bb_rho_ambient: float = 0.0
    POISSON_TOL: float = None             # ∞-norm residual stop; None = run all cycles
    POISSON_MAX_CYCLES: int = 10
    POISSON_N_PRE: int = 2                # smoother iters on descent
    POISSON_N_POST: int = 2                # smoother iters on ascent
    POISSON_N_COARSEST: int = 50          # smoother iters at the coarsest level
    POISSON_N_LEVELS: int = None          # auto (until per-side < 4)
    POISSON_FMG_VCYCLES_PER_LEVEL: int = 1
    POISSON_FMG_EXTRA_VCYCLES: int = 0
    POISSON_BOUNDARY: int = None          # None → inherit from BOUNDARY; 0/1/2/3 to override
                                          # 3 = isolated (multipole-Dirichlet)

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

    # Time integration: unsplit MUSCL-Hancock (predictor-corrector, 2nd order)
    # is the only scheme; there is no integrator-selection knob.

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
    # Wipe output_dir before the run. Opt-in (default False so production /
    # checkpoint-restart never deletes data). Validation runs set this True so
    # post-run analysis that globs+stitches every snapdir under output_dir can't
    # pick up stale snapshots from a prior run with different sharding / config.
    clear_output_dir: bool = False
    PLOT_AFTER_SIM: bool = True

    # How often to print progress and validate (in steps)
    output_cadence: int = 10

    # Print memory usage during run
    check_memory: bool = True

    # Periodic in-loop physics diagnostics + NaN abort guard. Both run as
    # single GPU-side reductions every `diag_interval`/`nan_check_interval`
    # steps, negligible cost compared to the step itself. On NaN detected
    # the runner dumps a checkpoint and raises RuntimeError so SLURM exits
    # non-zero — no more silent multi-hour NaN'd runs.
    enable_diagnostics: bool = True
    diag_interval: int = 100
    nan_check_interval: int = 100

    # Production mode: disables ALL performance-killing features
    # - Disables validation checks in time loop
    # - Disables verbose progress printing
    # - Disables memory checks
    # - Minimizes CPU-GPU synchronization
    production_mode: bool = False

    # ---------------- Autodiff run mode ----------------
    # When True, main.run dispatches to the autodiffable scan runner
    # (run_simulation_autodiff): the SAME setup, per-step physics, and sharding
    # as the HPC path, but the time loop is a pure remat'd lax.scan with fixed
    # dt and no IO/checkpoint/NaN-guard, so the whole evolution is
    # differentiable. Works single- and multi-GPU (inherits the sharded
    # step_fn / source applies). The differentiated parameters and observable
    # are problem-specific (e.g. M82: d L_X / d (cc85_eta_M, cc85_eta_E)).
    autodiff_mode: bool = False
    autodiff_n_steps: int = 100
    autodiff_fixed_dt: float = 0.0        # >0 -> fixed dt; 0.0 -> adaptive CFL dt
    # adaptive-dt differentiation mode (used when autodiff_fixed_dt == 0):
    #   "stopgrad" -> dt = stop_gradient(CFL dt): robust, no calibration, but
    #                 the gradient drops the dt-path so AD != FD (FD re-adapts).
    #   "diff"     -> differentiate THROUGH the CFL dt: AD == FD, exact total
    #                 derivative, at the cost of the dt's max-reduction kinks.
    autodiff_dt_mode: str = "stopgrad"
    # gradient mode for the xray_sensitivity task: "reverse" (grad, needs the
    # trajectory tape) or "forward" (jvp, memory-flat in n_steps -> the route
    # to high resolution / many steps for few-parameter sensitivities).
    autodiff_grad_mode: str = "reverse"
    # which differentiable task the autodiff runner performs:
    #   "xray_sensitivity" -> d ln L_X / d (cc85 loadings)   [forward/reverse]
    #   "ic_inversion"     -> recover an IC density perturbation [reverse]
    autodiff_task: str = "xray_sensitivity"
    autodiff_n_iter: int = 100            # optimizer iterations for inversions
    autodiff_ic_field: str = "density"    # ic_inversion: perturb "density" or "energy"

    # ---------------- Restart / Checkpoint ----------------
    # Enable restart from checkpoint
    restart: bool = False

    # Checkpoint file to restart from (if None, finds latest checkpoint)
    restart_file: str = None

    # Directory for checkpoint files
    checkpoint_dir: str = "checkpoints"

    # Automatic checkpointing during simulation. Default ON so that
    # (a) wall-clock checkpoints land for long runs, and (b) the NaN abort
    # guard can dump an emergency checkpoint before raising.
    enable_checkpointing: bool = True

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

