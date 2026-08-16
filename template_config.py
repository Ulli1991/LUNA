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
    # 3 = 3D (production path; use CFL <= 0.5, see validate())
    SPATIAL_DIMENSION: int = 1
    # 0 = Periodic
    # 1 = von Neumann (zero-gradient; NOTE: permits boundary inflow under an
    #     inward pull — for self-gravitating isolated problems prefer 4)
    # 2 = Reflective
    # 4 = Diode outflow (zero-gradient, ghost normal velocity clamped against
    #     inflow; gas can leave, nothing enters).
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

    # Self-gravity (multigrid V-cycle Poisson solve each step, applied as
    # an operator-split source on momentum + energy); periodic + Neumann
    # + isolated (multipole) boundaries — see docs/ARCHITECTURE.md.
    SELF_GRAVITY: bool = False
    GRAVITATIONAL_G: float = 1.0          # G in code units (set per problem)
    # Cosmological (comoving) integration. When False (default) the code
    # integrates the standard Newtonian Euler equations. Cosmological problems
    # (currently only the Zeldovich pancake) require this True and abort
    # otherwise — see is_cosmological() and the guard in main.run.
    CosmologicalIntegration: bool = False
    # Cosmology + Zel'dovich-pancake knobs (only used when CosmologicalIntegration).
    cosmo_Omega_m: float = 1.0      # EdS default -> D(a)=a, clean analytic check
    cosmo_Omega_L: float = 0.0
    cosmo_H0: float = 1.0           # H0 in code units
    a_init: float = 0.3             # initial scale factor (start redshift)
    a_end: float = 1.0              # final scale factor (z=0)
    dloga_max: float = 0.01         # cap on d(ln a) per step
    cosmo_kick_scheme: str = "strang"  # gravity-kick centering: strang (2nd order, 2 FMG/step,
                                       # default) | half (1 solve, midpoint coeff) | end (legacy)
    zel_a_caustic: float = 1.5      # pancake caustic scale factor (sets amplitude)
    zel_cs_frac: float = 1.0e-2     # initial sound speed / peak velocity (cold gas)
    cosmo_p_floor: float = 0.0      # pressure/temperature floor (0 = off)
    # DM-only cosmological N-body (Cosmo-DM): collisionless particles from a
    # MUSIC IC, evolved by super-comoving PM gravity (no gas grid).
    CosmoDM: bool = False
    music_ic_path: str = ""         # MUSIC/Gadget Arepo-HDF5 DM IC path
    dm_softening: float = 0.0       # gravitational softening (0 -> PM cell size)
    dm_eta: float = 0.025           # acceleration-timestep accuracy parameter
    dm_domain_decomp: bool = False  # grid-sharded slab decomposition (>512^3, multi-GPU)
    dm_capacity_factor: float = 1.5  # per-slab particle buffer headroom for clustering
    dm_exchange_send_div: int = 64  # slab exchange send-buffer slots = C // this (overflow is counted, never silent)
    # DM PM-gravity fast path (measured 512^3/MI300A: 46 ms/step, 2x tighter than
    # cold-start V(2,2); set dm_warm_vcycles=0 for the bit-conservative cold FMG).
    dm_warm_vcycles: int = 1        # >0: warm-start PM gravity — carry phi, k V-cycles/step (0 = cold FMG)
    dm_write_snapshot: bool = True  # write the end-of-run gadget snapshot (off for pure timing runs)
    dm_host_every: int = 1          # refresh dt criterion + watchdogs every k steps (host syncs are ~10-25 ms on GH200)
    # ONE switch for the whole gravity path (docs/PRECISION_SURVEY.md R6).
    # 'double': FP64 everywhere.  'mixed' (default): FP64 particles + FP64 phi
    # carry/residual, FP32 deposit + FP32 defect-correction V-cycle (textbook
    # 3-precision iterative refinement).  'single': FP32 throughout.
    gravity_precision: str = "mixed"  # 'double' | 'mixed' | 'single'
    # Fixed-point CIC accumulation (QuanTaichi 2021 / OpenMM 2012; survey R3):
    # deposit round(w*S) in units of the mean density into an integer grid.
    # Bitwise reproducible under particle reordering (integer add is
    # associative) and exact-mass by construction; int32@2^17 in mixed/single,
    # int64@2^40 in double.
    gravity_deposit_fixed: bool = False  # fixed-point CIC: bitwise reproducible + exact mass (int32@2^17, int64@2^40 in 'double')
    dm_mixed_precision: bool = True   # DEPRECATED alias, use gravity_precision
    dm_warm_extrapolate: int = 2    # warm-solve seed: 1 = rescale phi by a_next/a_prev; 2 = linear extrap from last two phi
    dm_warm_n_pre: int = 1          # smoother sweeps in the warm V-cycle (DM path only;
    dm_warm_n_post: int = 1         # global POISSON_N_PRE/POST are untouched)
    dm_decomp: str = "blocks"       # multi-GPU DM layout: "blocks" (3D, production) | "slab" (legacy)
    dm_resort_every: int = 4        # cell-order resort every N particle exchanges (0 = off)
    dm_fp32: bool = False            # DEPRECATED alias for gravity_precision='single' (fp32 particle state)
    CosmoHydro: bool = False        # coupled gas+DM box (gas meshed from MUSIC baryon IC)
    cosmo_jeans_ncell: float = 0.0  # Truelove Jeans-floor cells (0 = off; ~4 for halos)
    cosmo_T_floor: float = 0.0      # temperature floor [K] (10 K is standard for cosmo runs)
    cosmo_rho_floor: float = 0.0    # density floor as fraction of mean (0 = off; 1e-3 default in run)
    cosmo_vmax: float = 0.0         # |v~| velocity ceiling, code units (0 = off); tames v=p/rho runaways
    cosmo_Tmax: float = 0.0         # temperature ceiling [K] (0 = off); kills spurious ~1e11 K diffuse cells
    cosmo_floor_buffer: float = 1.0  # momentum-zero cells within this factor of rho_floor (>=1)
    DualEnergy: bool = False        # Bryan+1995 dual-energy (track e_int, switch when KE>>e_int)
    cosmo_dual_eta: float = 1.0e-3  # dual-energy switch: use e_int when (E-KE)/E < eta
    cosmo_debug: bool = False       # per-step gas/DM extrema diagnostics (blow-up debugging)
    cosmo_T_init: float = 0.0       # initial gas temperature [K]; 0 -> CMB 2.725(1+z)
    cosmo_mu: float = 1.22          # mean molecular weight (neutral primordial H+He)
    cosmo_snap_every: int = 0       # periodic gas+DM density snapshot every N steps (0 = end only)
    cosmo_profile: bool = False     # per-phase timestep breakdown -> <out_dir>/profile_steps.csv (every 10 steps)
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
    # Physical (cgs) parameters straight from GANDALF (Hubber+ 2018 §6.12) /
    # Boss & Bodenheimer 1979; run with a star-formation UnitSystem (pc, Msun,
    # km/s) and the barotropic EOS below.
    bb_rho0_cgs: float = 1.44e-17        # mean cloud density (g/cm^3)
    bb_R_pc: float = 0.01                # cloud radius (pc)
    bb_omega_rad_s: float = 1.6e-12      # rigid-body angular velocity (rad/s)
    bb_amp: float = 0.5                  # m-mode perturbation amplitude (paper: A=0.5)
    bb_m: int = 2                        # azimuthal mode number
    bb_rho_ambient_frac: float = 1.0e-3  # ambient density as a fraction of rho0
    bb_beta: float = 0.16
    bb_rho_ambient: float = 0.0
    POISSON_N_PRE: int = 2                # smoother iters on descent
    POISSON_N_POST: int = 2                # smoother iters on ascent
    POISSON_N_COARSEST: int = 50          # smoother iters at the coarsest level
    POISSON_N_LEVELS: int = None          # auto (until per-side < 4)
    POISSON_FMG_VCYCLES_PER_LEVEL: int = 1
    POISSON_FMG_EXTRA_VCYCLES: int = 0
    POISSON_FMG_N_COARSEST: int = 50      # coarsest-grid smooths; 10 == 50 to rel-L2 1e-11, ~1.5x faster
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
    # Unsplit scheme: the stable limit scales ~1/ndim. Measured on a smooth
    # 3D soundwave (2026-07-29): 1D stable to ~0.95, 2D to ~0.6, 3D to ~0.45.
    # This default is the 1D value — USE 0.4 FOR 3D (all 3D run scripts do).
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

    # Benchmark step cap: if > 0, the time loop stops after this many steps
    # regardless of t_end (equal-work, bounded scaling runs). 0 = disabled.
    max_steps: int = 0

    # Disable ALL snapshot I/O (scaling/benchmark runs). Default True keeps
    # production behaviour; set False so a high-resolution timing run doesn't
    # emit a multi-TB dump that dwarfs the compute being measured.
    write_snapshots: bool = True

    # Enable float64 (double precision) for the whole solver. Default False
    # (float32) for production throughput; True matches the FP64 scaling study.
    # main() reads this to set jax_enable_x64 before any array is created.
    enable_x64: bool = False

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

    # Periodic in-loop physics diagnostics run as a single GPU-side reduction
    # every `diag_interval` steps (negligible cost); the NaN abort guard checks
    # every step, piggybacked on compute_dt. On NaN detected the runner dumps a
    # checkpoint and raises RuntimeError so SLURM exits non-zero — no more silent
    # multi-hour NaN'd runs.
    enable_diagnostics: bool = True
    diag_interval: int = 100

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
    # ic_inversion: spectral band-limit of the TRUE injected IC perturbation.
    # 0 = broadband (power to Nyquist) -> the r(k) collapse is a genuine
    # measurement of the solver's scale-resolved recovery limit, not an artifact
    # of the input spectrum. >0 restricts injected power to k <= this (legacy).
    autodiff_ic_kmax: int = 0
    autodiff_ic_slope: float = 0.0        # power-law slope of broadband IC (|F|~k^slope); 0 = white
    autodiff_ic_lr: float = 1.0e-2        # Adam lr for the IC-inversion recovery loop
    # luna_emergence task: optimize the IC field so the late-time field forms a
    # target image (the word "LUNA") purely from the dynamics (paper fig19).
    autodiff_emergence_driver: str = "blast"   # "blast" | "turbulence"
    autodiff_emergence_target: str = "LUNA"
    autodiff_emergence_image: str = ""         # path to a [0,1] target field (.npy/png); overrides text target
    autodiff_emergence_init_seeds: int = 0     # blast: seed N scattered hot Sedov sites in e(x,0) (blast-wave-noise start)
    autodiff_emergence_emax: float = 50.0      # blast: max internal energy (x floor)
    autodiff_emergence_contrast: float = 4.0   # target hi/lo ratio
    autodiff_emergence_match: str = "e"        # observe "e" (internal energy) | "rho"
    autodiff_emergence_cfl: float = 0.25       # fixed-dt CFL for the long rollout
    autodiff_emergence_lr: float = 0.2         # Adam learning rate
    autodiff_emergence_nc: int = 32            # turbulence: control-grid side (band-limit)
    autodiff_emergence_mach: float = 3.0       # turbulence: stirring Mach number

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

    # ---------------- Barotropic EOS (star formation) ----------------
    # When True, after each hydro step the runner resets the pressure to the
    # barotropic relation  P = k_B T(ρ) ρ / (μ m_H)  with
    # T(ρ) = T0 [1 + (ρ/ρ_AD)^(γ-1)]: an isothermal floor at T0 that stiffens
    # adiabatically above ρ_AD (GANDALF/Boss-Bodenheimer). Requires cfg.units.
    # The stiffening halts the runaway isothermal collapse. Mutually exclusive
    # with isothermal_cs2.
    barotropic_eos: bool = False
    barotropic_T0_K: float = 10.0
    barotropic_rho_AD_cgs: float = 1.0e-14
    barotropic_mu: float = 2.35

    # ---------------- Particle (point-mass) module ----------------
    # A small set of point masses (central star + planet; later sinks) that
    # exert Plummer-softened gravity on the gas each step (sources/particles.py).
    # Used for the disc-planet interaction test (de Val-Borro 2006 / GANDALF
    # §6.11). 2D. The planet is held on a prescribed circular Keplerian orbit.
    particles_enabled: bool = False
    particle_softening: float = 0.0       # Plummer ε (code length); ~ a few dx
    particle_star_mass: float = 1.0       # central star mass (code units)
    particle_planet_q: float = 1.0e-3     # planet/star mass ratio (Jupiter/Sun)
    particle_planet_a: float = 1.0        # planet orbital radius (code units)
    # Sink (accretion): drain gas within particle_sink_radius of a sink particle
    # each step (fraction removed = particle_sink_rate * dt). The planet is a
    # sink when enabled; the central star too if particle_sink_on_star.
    particle_sink_enabled: bool = False
    particle_sink_radius: float = 0.0     # planet accretion radius (code length); ~0.5 R_Hill
    particle_sink_rate: float = 0.0       # accretion efficiency (1/code time)
    particle_sink_on_star: bool = False   # also drain gas at the central star (inner cavity)
    particle_star_sink_radius: float = 0.0  # star cavity radius; defaults to particle_sink_radius
    particle_sink_vdamp: float = 0.0      # velocity damping rate in sink regions (de Val-Borro zones)

    # ---------------- Disc-planet gas disc (Disc-Planet-2D IC) ----------------
    # 2D globally-isothermal gas disc in rotational balance with the softened
    # central-star potential (de Val-Borro 2006 / GANDALF §6.11). Requires
    # isothermal_cs2 > 0. Σ ∝ R^-disc_sigma_index, tapered between R_in and R_out.
    disc_sigma0: float = 1.0              # surface-density normalization at R=1
    disc_sigma_index: float = 0.5         # Σ ∝ R^-index
    disc_R_in: float = 0.4                # inner taper radius (code length)
    disc_R_out: float = 2.5               # outer taper radius (code length)
    disc_floor: float = 1.0e-4            # ambient surface-density floor
    # de Val-Borro (2006) boundary damping zones: near the inner/outer radial
    # boundaries relax the state toward the INITIAL axisymmetric disc (the IC,
    # planet-free) with a rate ramping smoothly to the edge. This is what keeps
    # the cold disc stable — a vacuum cavity drain does not.
    disc_damping_enabled: bool = False
    disc_damp_R_in: float = 0.5           # inner zone: damp where R < this
    disc_damp_R_out: float = 2.2          # outer zone: damp where R > this
    disc_damp_rate: float = 5.0           # peak relaxation rate (1/code time)

    # ---------------- Sink particles (on-the-fly, 3D; sources/sinks.py) -------
    # Form sinks dynamically when a density peak exceeds sink_creation_density
    # (and is far enough from existing sinks); accrete gas inside the accretion
    # radius; integrate with gas + self-gravity by the hybrid scheme — mesh
    # Poisson for gas self-gravity + gas→sink, direct N-body for sink→gas and
    # sink→sink. Pre-allocated sink_max_count slots (static shapes; no recompile
    # when a sink spawns). Requires SELF_GRAVITY-style config + SPATIAL_DIMENSION=3.
    sinks_enabled: bool = False
    sink_max_count: int = 4               # pre-allocated sink slots (BB79 needs ~2)
    sink_creation_density: float = 0.0    # spawn a sink above this density (0 → never)
    sink_accretion_radius: float = 0.0    # gas inside this radius accretes (0 → 2·dx)
    sink_softening: float = 0.0           # ε for sink→gas / sink→sink (0 → 2·dx)
    sink_creation_separation: float = 0.0  # min distance to an existing sink (0 → 4·dx)

    # ---------------- Block-AMR (multi-level hydro) ----------------
    # Opt-in. When True, main.run dispatches to setup.runner_amr instead of the
    # dense gas path: the base grid (Nx,Ny) is refined by `amr_levels` factor-2
    # levels around steep density gradients, evolved with a synchronized same-dt
    # multi-level step, and snapshots are written as the composited finest-
    # resolution field through the standard HDF5 writer. 2D only for now.
    AMR: bool = False
    amr_levels: int = 2                  # refinement levels above the base grid
    amr_block_shape: int = 16            # cells per block per dim (must divide Nx,Ny)
    amr_thresholds: list = None          # per-level flag threshold; None -> [0.05]*levels
    # Refinement criterion (all dimension-generic, 2D & 3D):
    #   "density"   |grad rho|  — resolution-dependent shock/contact tracker
    #   "pressure"  |grad P|    — catches the Sedov energy spike at t=0 AND the
    #                             shock front; the right trigger for blasts
    #   "vorticity" |curl v|    — resolution-INDEPENDENT; refines vortex cores even
    #                             from a well-resolved base; the right trigger for KH
    amr_criterion: str = "density"
    amr_capacities: list = None          # per-level max child blocks; None -> exact (no padding)
    # Hard per-level block ceiling (0 = unlimited): the deep-hierarchy memory
    # guard. Blanket refinement x the pow2 bucket ladder otherwise requests
    # device-breaking arrays (41 GiB measured at 5 levels); excess flagged
    # blocks are simply not refined that regrid (dropped in slot order; the
    # regrid log prints AT-CAP when a level saturates).
    amr_max_blocks: int = 0
    # Opt-in positivity floor for the AMR advance (0 = off). When a capped level
    # seams a strong shock, the coarse-fine flux can drive a cell negative and
    # collapse dt; the floor quiescent-resets such cells to (rho_floor, v=0,
    # p_floor) instead of spiraling. A validated shallow run never triggers it.
    amr_rho_floor: float = 0.0
    amr_p_floor: float = 0.0
    # Relative (auto-scaling) alternative to the absolute floors above: a fraction
    # of the base-level MEAN (rho, p). Lets deep runs be made robust without
    # knowing the absolute density scale (frac ~ 1e-8..1e-6 sits far below any
    # resolved structure). Used only when amr_rho_floor == 0; 0 = off.
    amr_rho_floor_frac: float = 0.0
    # Dense-composite snapshots materialize the FULL finest-resolution cube
    # ((Nx*2^levels)^ndim * n_vars * 8 B = 40 GiB at 32^3+5 levels) regardless
    # of AMR sparsity. False -> write per-level block data into blocks_*.npz
    # instead (the complete AMR state, sized to what's actually refined).
    amr_dense_snapshots: bool = True
    # Adaptive re-check cadence: re-run the refinement criterion on the live
    # solution and rebuild the grid every N steps (0 -> static, flag once at t=0).
    # The flagging dilation margin (n_ghost cells) must cover how far a feature
    # moves in N steps, so keep N small (a few) unless you widen the margin.
    amr_regrid_interval: int = 4
    # Round each level's block capacity up to this many parents so array shapes
    # stay stable across regrids (the jitted step recompiles only when a bucket
    # boundary is crossed, not on every regrid). 0/1 -> exact (no padding) but
    # recompiles whenever the block count changes. Bounded padding, big speedup.
    amr_block_bucket: int = 256
    # Subcycle in time (Berger-Oliger): each refinement level steps with its own
    # dt = dt_base / 2**level, so coarse cells don't pay the finest timestep. This
    # is what makes AMR a *time* win, not just a memory win. False -> synchronized
    # same-dt (every level on the finest dt).
    amr_subcycle: bool = True
    # Distribute blocks across devices with jax.shard_map (multi-GPU). Each device
    # owns a shard of the block axis; the per-block flux/divergence is sharded and
    # the irregular block-neighbour seam coupling is an all_gather. Auto no-op on a
    # single device. The block axis is padded to a multiple of the device count.
    amr_shard: bool = False

    # -------------------------------------------------------------------------
    # Computed properties (lightweight, timing-related)
    # -------------------------------------------------------------------------

    @cached_property
    def snapshot_times(self) -> np.ndarray:
        """Pre-computed snapshot times (using numpy, not JAX, to avoid GPU allocation)."""
        return np.arange(self.t_start, self.t_end + self.time_bet_snaps/2, self.time_bet_snaps)


    def validate(self):
        """Fail fast on the known config traps (called by main.run)."""
        errs = []
        if self.SPATIAL_DIMENSION >= 2 and self.CFL > 0.5:
            errs.append(f"CFL={self.CFL} with SPATIAL_DIMENSION="
                        f"{self.SPATIAL_DIMENSION}: unsplit MUSCL-Hancock "
                        "needs CFL <= 0.5 in 2D/3D (0.4 recommended; the "
                        "template default 0.8 is the 1D value)")
        if self.BOUNDARY not in (0, 1, 2, 3, 4):
            errs.append(f"BOUNDARY={self.BOUNDARY}: valid codes are "
                        "0 periodic, 1 zero-gradient, 2 reflective, "
                        "3 isolated, 4 diode")
        if getattr(self, "dm_decomp", "blocks") not in ("blocks", "slab"):
            errs.append(f"dm_decomp={self.dm_decomp!r}: 'blocks' or 'slab'")
        if getattr(self, "gravity_precision", "mixed") not in (
                "double", "mixed", "single"):
            errs.append(f"gravity_precision={self.gravity_precision!r}: "
                        "'double', 'mixed' or 'single'")
        if getattr(self, "cosmo_kick_scheme", "strang") not in (
                "strang", "half", "end"):
            errs.append(f"cosmo_kick_scheme={self.cosmo_kick_scheme!r}: "
                        "'strang', 'half' or 'end'")
        if (getattr(self, "CosmoDM", False)
                or getattr(self, "CosmoHydro", False)) \
                and not getattr(self, "music_ic_path", ""):
            errs.append("CosmoDM/CosmoHydro requires music_ic_path")
        if errs:
            raise ValueError("Config validation failed:\n  - "
                             + "\n  - ".join(errs))
