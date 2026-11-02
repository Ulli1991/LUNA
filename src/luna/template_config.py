"""Every setting of a LUNA run, one frozen dataclass; docs/CONFIG.md explains each section.
A problem's own parameters live in its IC module's Params, reached as cfg.problem."""
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Optional
import numpy as np

from luna.units import UnitSystem
from luna.hydro.turbulence import OUDriverConfig
from luna.analysis.config import InsituConfig


@dataclass(frozen=True)
class Config:
    # ---------------- Problem and grid --------------------------------------------
    test_problem: str = "Sod-Shock"       # a key of _IC_REGISTRY in luna/ic_generation/ic.py
    problem: Any = None
    seed: int = 0
    SPATIAL_DIMENSION: int = 1
    Nx: int = 1024
    Ny: int = 1024
    Nz: int = 1024
    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 1.0
    BOUNDARY: int = 1                     # 0 periodic | 1 zero-gradient | 2 reflective | 4 diode outflow
    gamma: float = 5.0 / 3.0
    units: UnitSystem = field(default_factory=UnitSystem)
    ic_dir: str = ""

    # ---------------- Hydro scheme ------------------------------------------------
    CFL: float = 0.8
    ORDER: int = 1                        # 0 Godunov | 1 PLM
    LIMITER: int = 0                      # 0 minmod | 1 MC | 2 van Leer
    limiter_tangent_eps: float = 0.0
    riemann_solver: str = "hllc"          # hllc | hll
    hydro_kernel: str = "xla"             # xla | pallas (refuses what it cannot run) | auto (pallas on a GPU where it can)
    hydro_fused: bool = False
    hydro_recon32: bool = False
    preserve_face_states: bool = False    # RAMSES floors: primitives, face trace guard, Riemann inputs, dt (U is not floored)
    hydro_smallr: float = 1e-10           # preserve_face_states density floor, code units (RAMSES smallr)
    hydro_smallc: float = 1e-10           # preserve_face_states sound-speed floor, code units (RAMSES smallc)
    allow_first_order_faces: bool = False  # audit C8: unphysical Hancock face (rho or p <= 0) -> cell-centre state, logged; False: the run stops
    boundary_mass_budget: bool = False

    # ---------------- Time and precision ------------------------------------------
    t_start: float = 0.0
    t_end: float = 0.245
    max_steps: int = 0
    enable_x64: bool = False

    # ---------------- Output and diagnostics --------------------------------------
    write_snapshots: bool = True
    time_bet_snaps: float = 0.01
    output_dir: str = "snapshots"
    base_name: str = "snap"
    clear_output_dir: bool = False
    PLOT_AFTER_SIM: bool = True
    output_cadence: int = 10
    enable_diagnostics: bool = True
    diag_interval: int = 100
    timing_every: int = 0

    # ---------------- Restart and checkpoints -------------------------------------
    restart: bool = False
    restart_file: str = None
    checkpoint_dir: str = None
    enable_checkpointing: bool = True
    checkpoint_wall_time_interval: float = 7200.0
    max_checkpoints: int = 5

    # ---------------- Devices -----------------------------------------------------
    MESH_SHAPE: tuple = None
    decomposition: str = "auto"           # auto | dense | blocks | amr_blocks
    dm_pdims: Optional[tuple] = None

    # ---------------- What kind of run --------------------------------------------
    run: str = "gas"                      # gas | dm | gas+dm
    CosmologicalIntegration: bool = False

    # ---------------- Self-gravity and the Poisson solve --------------------------
    SELF_GRAVITY: bool = False
    GRAVITATIONAL_G: float = 1.0
    gravity_warm_vcycles: int = 2
    gravity_warm_tol: float = 1e-4        # 0: gravity_warm_vcycles fixed cycles
    gravity_warm_max_cycles: int = 10
    gravity_warm_extrapolate: bool = True
    gravity_warm_dt_ratio_max: float = 4.0
    gravity_proper_mixed: bool = True
    POISSON_N_PRE: int = 2
    POISSON_N_POST: int = 2
    POISSON_N_COARSEST: int = 50          # None: 4 on a whole periodic grid of side <= 4, else 50
    POISSON_N_LEVELS: int = None
    POISSON_FMG_VCYCLES_PER_LEVEL: int = 1
    POISSON_FMG_EXTRA_VCYCLES: int = 0
    POISSON_BOUNDARY: int = None          # None (BOUNDARY) | 0 | 1 | 2 | 3 isolated
    gravity_precision: str = "mixed"      # double | mixed | single
    gravity_deposit_fixed: bool = True
    gravity_mg_ca: str = "auto"           # auto (= 0) | 0 | 1 communication-avoiding smoother
    gravity_mg_agg_n: int = 128           # sharded solve: gather the root grid at about this side
    gravity_mg_smoother: str = "rbgs"     # rbgs | cheb
    gravity_kernel: str = "xla"           # xla | pallas | auto (pallas on a GPU)
    gravity_energy: str = "auto"          # auto: flux where supported, otherwise kick | kick | flux

    # ---------------- Cosmology (comoving runs) -----------------------------------
    cosmo_Omega_m: float = 1.0
    cosmo_Omega_L: float = 0.0
    cosmo_H0: float = 1.0
    a_init: float = 0.3
    a_end: float = 1.0
    dloga_max: float = 0.01
    music_ic_path: str = ""
    cosmo_checkpoint_every: int = 0
    cosmo_snapshot_a: tuple = ()
    cosmo_p_floor: float = 0.0
    cosmo_jeans_ncell: float = 0.0
    cosmo_T_floor: float = 0.0
    cosmo_rho_floor: float = 0.0
    cosmo_vmax: float = 0.0
    cosmo_Tmax: float = 0.0
    cosmo_floor_buffer: float = 1.0
    cosmo_T_init: float = 0.0
    cosmo_mu: float = 1.22

    # ---------------- Dark-matter particles ---------------------------------------
    dm_softening: float = 0.0
    dm_eta: float = 0.025
    dm_capacity_factor: float = 1.5
    dm_warm_vcycles: int = 1
    dm_warm_n_pre: int = 1
    dm_warm_n_post: int = 1
    dm_resort_every: int = 4
    dm_face_capacity_factor: float = 0.0

    # ---------------- Equation of state -------------------------------------------
    isothermal_cs2: float = 0.0
    barotropic_eos: bool = False
    barotropic_T0_K: float = 10.0
    barotropic_rho_AD_cgs: float = 1.0e-14
    barotropic_mu: float = 2.35

    # ---------------- Radiative cooling -------------------------------------------
    cooling_enabled: bool = False
    cooling_table: str = "sd93_solar"     # sd93_solar | koyama_inutsuka | combined
    cooling_n_sub: int = 16
    cooling_dt_frac: float = 0.1
    cooling_T_floor: float = 1.0e4
    cooling_X_H: float = 0.7154
    cooling_ne_per_nH: float = 1.189
    cooling_mu: float = 0.6

    # ---------------- Turbulent driving and the learned flux corrector ------------
    ou_config: Optional[OUDriverConfig] = None
    ou_seed: int = 7
    sil_weights: str = ""
    insitu: Optional[InsituConfig] = None  # in-situ PDFs, spectra, structure functions, snipshots, frames

    # ---------------- CC85 starburst wind -----------------------------------------
    cc85_enabled: bool = False
    cc85_R_star_pc: float = 300.0
    cc85_SFR_Msun_per_yr: float = 12.0
    cc85_eta_M: float = 1.0
    cc85_eta_E: float = 0.6

    # ---------------- M82 external potential --------------------------------------
    m82_use_external_gravity: bool = False
    m82_stellar_MN_M: float = 1.0
    m82_stellar_MN_a_kpc: float = 0.8
    m82_stellar_MN_b_kpc: float = 0.15
    m82_halo_NFW_M_vir: float = 50.0
    m82_halo_NFW_c: float = 20.0

    # ---------------- Analytic external potential (1-3D) ---------------------------
    external_potential: str = ""          # '' | sin | uniform | point_mass
    external_axis: int = 0                # sin, uniform: the axis
    external_phi0: float = 1.0            # sin: phi = phi0 sin(2 pi x / wavelength)
    external_wavelength: float = 1.0      # sin
    external_g: float = -1.0              # uniform: acceleration along +axis
    external_GM: float = 1.0              # point_mass: phi = -GM / sqrt(r^2 + softening^2)
    external_softening: float = 0.0       # point_mass
    external_center: tuple = None         # point_mass; None: box centre

    # ---------------- Thermochemistry ---------------------------------------------
    chemistry: str = ""                   # '' | net5
    chem_g0: float = 1.7
    chem_zeta: float = 3e-17
    chem_x0: tuple = (0.0, 1e-2, 0.0)
    chem_tdust0: float = 10.0
    chem_chunk_log2: int = 20
    chem_dust_rtol: float = 1e-2

    # ---------------- M1 radiation ------------------------------------------------
    radiation_enabled: bool = False
    rad_groups: int = 1
    rad_chat: float = 0.0
    rad_c_true: float = 0.0
    rad_E0: float = 0.0
    rad_kappa_a: float = 0.0
    rad_kappa_s: float = 0.0
    rad_source: float = 0.0
    rad_kappa_a_groups: tuple = ()
    rad_kappa_s_groups: tuple = ()
    rad_source_groups: tuple = ()
    rad_substeps: int = 10
    rad_cfl: float = 0.4
    rad_order: int = 2                    # 1 donor cell | 2 MUSCL
    rad_ion_solver: str = "implicit"      # implicit | split
    rad_ion_subcycles: int = 1
    rad_scheme: str = "hancock_split"     # hancock_split | rk2
    rad_kernel: str = "xla"               # xla | pallas | auto (pallas on a GPU)
    rad_closure: str = "m1"               # m1 | m2_learned | m2_entropy
    rad_closure_params: str = ""
    rad_force: bool = True
    rad_boundary: int = None              # None (BOUNDARY) | 0 periodic | 1 zero-gradient
    rad_remat: bool = True

    # ---------------- Hydrogen ionization -----------------------------------------
    rad_ionization: bool = False
    rad_sigma: float = 0.0
    rad_alpha_B: float = 0.0
    rad_photon_rate: float = 0.0
    rad_source_radius: float = 0.0
    rad_m_H: float = 1.0
    rad_cs2_neutral: float = 0.0
    rad_cs2_ionized: float = 0.0
    rad_neutral_eos_table: str = ""
    rad_x0: float = 0.0

    # ---------------- Point masses (disc-planet) ----------------------------------
    particles_enabled: bool = False
    particle_softening: float = 0.0
    particle_star_mass: float = 1.0
    particle_planet_q: float = 1.0e-3
    particle_planet_a: float = 1.0
    particle_sink_enabled: bool = False
    particle_sink_radius: float = 0.0
    particle_sink_rate: float = 0.0
    particle_sink_on_star: bool = False
    particle_star_sink_radius: float = 0.0
    particle_sink_vdamp: float = 0.0
    disc_damping_enabled: bool = False
    disc_damp_R_in: float = 0.5
    disc_damp_R_out: float = 2.2
    disc_damp_rate: float = 5.0

    # ---------------- Sink particles ----------------------------------------------
    sinks_enabled: bool = False
    sink_max_count: int = 4
    sink_sink_gravity: bool = True
    sink_creation_density: float = 0.0
    sink_accretion_radius: float = 0.0
    sink_softening: float = 0.0
    sink_creation_separation: float = 0.0
    sink_accretion: str = "threshold"     # threshold | federrath
    sink_creation_checks: str = "luna"    # luna | flash
    sink_merging: bool = False
    sink_create_frac: float = 0.0

    # ---------------- Coupled stars (Torch) ---------------------------------------
    star_sink_gravity: bool = True
    particle_boundary: str = ""           # '' (as the gas) | periodic | outflow
    star_max_count: int = 1024
    sink_star_T_max: float = 0.0
    sink_star_mu: float = 1.27
    wind_target_temp_K: float = 0.0
    feedback_radius: float = 0.0
    wind_add_heat: bool = False
    wind_mass_load_mode: str = "conservative"  # conservative | torch | sink
    wind_cons_quant: str = "momentum"     # momentum | energy
    wind_var_radius: bool = False
    wind_ref_radius: float = 0.0

    # ---------------- Block AMR ---------------------------------------------------
    AMR: bool = False
    amr_levels: int = 2
    amr_block_shape: int = 16
    amr_thresholds: list = None
    amr_criterion: str = "density"        # density | overdensity | pressure | vorticity | prho | jeans | lohner | static
    amr_capacities: list = None
    amr_max_blocks: int = 0
    amr_rho_floor: float = 0.0
    amr_p_floor: float = 0.0
    amr_rho_floor_frac: float = 0.0
    amr_dense_snapshots: bool = False
    amr_regrid_interval: int = 4
    amr_block_bucket: int = 256
    amr_subcycle: bool = True
    amr_shard: bool = False
    amr_reflux: str = "repl"              # repl | sharded | iface
    amr_shard_storage: bool = True
    amr_grav_vcycles: int = 3
    amr_n_snaps: int = 4

    # ---------------- Autodiff ----------------------------------------------------
    autodiff_mode: bool = False
    autodiff_n_steps: int = 100
    autodiff_fixed_dt: float = 0.0
    autodiff_dt_mode: str = "stopgrad"    # stopgrad | diff
    autodiff_grad_mode: str = "reverse"   # reverse | forward
    autodiff_checkpoint: str = "step"     # step | revolve
    autodiff_snapshots: int = 32
    autodiff_snapshot_gib: float = 0.0
    autodiff_task: str = "ic_inversion"   # eta_recovery | ic_inversion | luna_emergence | corrector_training

    def __post_init__(self):
        from luna.ic_generation.ic import problem_params
        object.__setattr__(self, "problem", problem_params(self.test_problem, self.problem))

    @cached_property
    def snapshot_times(self) -> np.ndarray:
        """Snapshot times, in numpy so no device array is made."""
        return np.arange(self.t_start, self.t_end + self.time_bet_snaps/2, self.time_bet_snaps)

    def validate(self):
        """Fail fast on the known config traps (called by main.run)."""
        errs = []
        if not 0.0 < self.CFL <= 1.0:
            errs.append(f"CFL={self.CFL}: the unsplit step is unstable above 1 "
                        "(docs/HYDRO_TIMESTEP.md; 0.8 is the default)")
        if self.ORDER == 2:
            errs.append("ORDER=2: there is no PPM; ORDER=2 always ran PLM, set ORDER=1")
        if self.BOUNDARY == 3:
            errs.append("BOUNDARY=3: isolated is a gravity boundary only; set "
                        "POISSON_BOUNDARY=3 and a gas BOUNDARY (4 diode outflow)")
        elif self.BOUNDARY not in (0, 1, 2, 4):
            errs.append(f"BOUNDARY={self.BOUNDARY}: valid codes are "
                        "0 periodic, 1 zero-gradient, 2 reflective, 4 diode")
        if self.external_potential not in ("", "sin", "uniform", "point_mass"):
            errs.append(f"external_potential={self.external_potential!r}: expected '', 'sin', "
                        "'uniform' or 'point_mass'")
        elif self.external_potential and self.m82_use_external_gravity:
            errs.append("external_potential and m82_use_external_gravity are exclusive")
        elif self.external_potential and not 0 <= self.external_axis < self.SPATIAL_DIMENSION:
            errs.append(f"external_axis={self.external_axis} outside SPATIAL_DIMENSION={self.SPATIAL_DIMENSION}")
        if self.chemistry and self.SPATIAL_DIMENSION != 3:
            errs.append(f"chemistry={self.chemistry!r} needs SPATIAL_DIMENSION=3 "
                        "(src/luna/sources/chemistry.py reads the 3D momentum and energy rows)")
        if self.cooling_table in ("ki", "ki02", "ki_sd93"):
            errs.append(f"cooling_table={self.cooling_table!r}: spell it "
                        "'koyama_inutsuka' or 'combined'")
        if self.gravity_precision not in ("double", "mixed", "single"):
            errs.append(f"gravity_precision={self.gravity_precision!r}: "
                        "'double', 'mixed' or 'single'")
        if self.gravity_warm_tol < 0 or self.gravity_warm_max_cycles < 1 \
                or self.gravity_warm_dt_ratio_max < 1:
            errs.append("gravity_warm_tol >= 0, gravity_warm_max_cycles >= 1 and "
                        "gravity_warm_dt_ratio_max >= 1")
        if int(self.gravity_mg_agg_n) < 0:
            errs.append(f"gravity_mg_agg_n={self.gravity_mg_agg_n}: a grid side >= 0")
        if self.run in ("dm", "gas+dm") and not self.music_ic_path:
            errs.append(f"run={self.run!r} requires music_ic_path")
        if self.timing_every < 0:
            errs.append(f"timing_every={self.timing_every}: 0 (off) or every N-th step")
        if self.limiter_tangent_eps:
            if self.limiter_tangent_eps < 0 or self.LIMITER != 0 \
                    or not self.autodiff_mode or self.AMR:
                errs.append("limiter_tangent_eps > 0 needs autodiff_mode, LIMITER=0 "
                            "and no AMR (it changes only minmod's derivative)")
        if self.riemann_solver == "lhllc":
            errs.append("riemann_solver='lhllc': the L-HLLC solver was removed "
                        "from LUNA; use 'hllc' (default) or 'hll'")
        elif self.riemann_solver not in ("hllc", "hll"):
            errs.append(f"riemann_solver={self.riemann_solver!r}: expected one of "
                        "('hllc', 'hll')")
        for name, allowed in (
            ("ORDER", (0, 1)),
            ("LIMITER", (0, 1, 2)),
            ("POISSON_BOUNDARY", (None, 0, 1, 2, 3)),
            ("decomposition", ("auto", "dense", "blocks", "amr_blocks")),
            ("cooling_table", ("sd93_solar", "koyama_inutsuka", "combined")),
            ("hydro_kernel", ("auto", "xla", "pallas")),
            ("gravity_kernel", ("auto", "xla", "pallas")),
            ("gravity_energy", ("auto", "kick", "flux")),
            ("rad_kernel", ("auto", "xla", "pallas")),
            ("rad_scheme", ("hancock_split", "rk2")),
            ("rad_order", (1, 2)),
            ("rad_ion_solver", ("implicit", "split")),
            ("gravity_mg_ca", ("auto", "0", "1")),
            ("gravity_mg_smoother", ("rbgs", "cheb")),
            ("particle_boundary", ("", "periodic", "outflow")),
            ("amr_reflux", ("repl", "sharded", "iface")),
            ("autodiff_task", ("eta_recovery", "ic_inversion",
                               "luna_emergence", "corrector_training")),
            ("autodiff_dt_mode", ("stopgrad", "diff")),
            ("autodiff_grad_mode", ("reverse", "forward")),
            ("autodiff_checkpoint", ("step", "revolve")),
            ("sink_accretion", ("threshold", "federrath")),
            ("sink_creation_checks", ("luna", "flash")),
            ("amr_criterion", ("density", "overdensity", "vorticity", "pressure",
                               "pressure_density", "prho", "jeans", "lohner", "static")),
        ):
            val = getattr(self, name)
            if val not in allowed and not any(e.startswith(f"{name}=") for e in errs):
                errs.append(f"{name}={val!r}: expected one of {allowed}")
        if self.gravity_energy == "flux":
            from luna.setup.gravity_energy import flux_requirements
            why = flux_requirements(self)
            if why:
                errs.append("gravity_energy='flux' needs " + ", ".join(why))
        if self.hydro_kernel == "pallas":
            # the step refuses the same (muscl_hancock_pallas.select); caught here before any work
            from luna.hydro.muscl_hancock_pallas import unsupported
            from luna.setup.hydro import chemistry_scalars, hydro_limiter
            why = ("AMR (the refinement runner's hydro is XLA only)" if self.AMR else unsupported(
                spatial_dim=self.SPATIAL_DIMENSION, order=self.ORDER, limiter=hydro_limiter(self),
                riemann_solver=self.riemann_solver, boundary=self.BOUNDARY,
                fused=self.hydro_fused, recon32=self.hydro_recon32,
                autodiff=bool(self.autodiff_mode), n_scalars=chemistry_scalars(self)))
            if why:
                errs.append(f"hydro_kernel='pallas' cannot run {why}; use hydro_kernel='xla', "
                            "or 'auto' (XLA where Pallas cannot run)")
        if self.allow_first_order_faces and self.autodiff_mode:
            # the rollout is one lax.scan under jit/grad: no host between steps reads a face report, so the
            # first-order faces would be silent; refused (the default is a NaN, and the loss is NaN)
            errs.append("allow_first_order_faces with autodiff_mode: the differentiable rollout cannot report "
                        "first-order faces (audit C8); unphysical faces are refused there (NaN)")
        if self.preserve_face_states:
            if self.AMR or self.ORDER < 1:
                errs.append("preserve_face_states: the dense MUSCL-Hancock step only (ORDER=1, no AMR; "
                            "the refinement runner has its own hydro)")
            if not (float(self.hydro_smallr) > 0 and float(self.hydro_smallc) > 0):
                errs.append(f"preserve_face_states: hydro_smallr={self.hydro_smallr} and "
                            f"hydro_smallc={self.hydro_smallc} must be > 0 (code units)")
        if float(self.sink_star_T_max) < 0.0:
            errs.append(f"sink_star_T_max={self.sink_star_T_max}: a temperature [K], "
                        "or 0 for no cold-gas condition")
        if int(self.autodiff_snapshots) < 1:
            errs.append(f"autodiff_snapshots={self.autodiff_snapshots}: at least "
                        "one (the initial state)")
        if self.radiation_enabled:
            if float(self.rad_chat) <= 0.0:
                errs.append("radiation_enabled needs rad_chat > 0: the reduced "
                            "speed of light in code units")
            if int(self.rad_groups) < 1:
                errs.append(f"rad_groups={self.rad_groups}: at least one group")
            if self.rad_boundary is not None and int(self.rad_boundary) not in (0, 1):
                errs.append(f"rad_boundary={self.rad_boundary}: the M1 transport "
                            "has 0 periodic and 1 zero-gradient")
            for nm in ("rad_kappa_a_groups", "rad_kappa_s_groups", "rad_source_groups"):
                v = getattr(self, nm, ())
                if v and len(v) != int(self.rad_groups):
                    errs.append(f"{nm} has {len(v)} values but "
                                f"rad_groups={self.rad_groups}")
            closure = self.rad_closure
            if closure not in ("m1", "m2_learned", "m2_entropy"):
                errs.append(f"rad_closure={closure!r}: 'm1', 'm2_learned' or 'm2_entropy'")
            elif closure != "m1" and (self.SPATIAL_DIMENSION != 3
                                      or self.rad_scheme != "hancock_split"
                                      or self.autodiff_mode):
                errs.append(f"rad_closure={closure!r} needs SPATIAL_DIMENSION=3 and "
                            "rad_scheme='hancock_split', and has no adjoint path yet")
            elif closure == "m2_entropy" and self.rad_kernel == "pallas":
                errs.append("rad_closure='m2_entropy' with rad_kernel='pallas': the fused M2 kernel "
                            "has no entropy solve; rad_kernel='xla' or 'auto' (XLA for this closure)")
            if self.rad_ionization and self.cooling_enabled:
                errs.append(
                    "rad_ionization with cooling_enabled: these two write the "
                    "same variable and neither knows about the other. "
                    "`two_temperature` does not HEAT the gas -- it overwrites "
                    "the pressure with rho*cs^2(x), interpolating between the "
                    "neutral and ionized sound speeds, which is STARBENCH's "
                    "prescription for isolating hydrodynamics from "
                    "thermodynamics. Run with cooling it silently fights the "
                    "cooling stage, last writer per cell wins, and the answer "
                    "looks plausible. Turn one off. Lifting this needs a real "
                    "photoheating rate so the radiation heats instead of "
                    "prescribing (docs/RADIATION.md).")
        # warnings: configurations that run but hit a known-open bug or an ignored setting
        from luna.setup.run_kind import check_modifiers, run_kind

        def kind_or_none(cfg):
            try:
                return run_kind(cfg)
            except ValueError:
                return None
        try:
            check_modifiers(self)
        except ValueError as e:
            errs.append(str(e))
        import warnings as _warnings
        warns = []
        if self.restart and self.autodiff_mode:
            warns.append("restart=True is not honoured by the autodiff mode -- the run "
                         "starts from the initial condition")
        from luna.setup.run_kind import DM, COUPLED
        if (self.gravity_deposit_fixed
                and self.gravity_precision != "double"
                and max(self.Nx, self.Ny, self.Nz) >= 512
                and kind_or_none(self) in (DM, COUPLED)):
            warns.append("gravity_deposit_fixed with an int32 fixed-point scatter at "
                         f"N>={max(self.Nx, self.Ny, self.Nz)}: 1+delta_max can overflow "
                         "the integer ceiling in clustered runs (the runner raises when "
                         "within 2x); gravity_precision='double' uses int64")
        if not self.enable_x64 and (self.run != "gas" or self.CosmologicalIntegration):
            warns.append("enable_x64=False with a cosmological run: the gravity and "
                         "growth gates assume float64 (AMRConfig asserts it)")
        if self.cooling_enabled:
            try:
                from luna.units import HYDROGEN_MASSFRAC
                if abs(float(self.cooling_X_H) - HYDROGEN_MASSFRAC) > 1e-6:
                    warns.append(f"cooling_X_H={self.cooling_X_H} differs "
                                 f"from units.HYDROGEN_MASSFRAC={HYDROGEN_MASSFRAC}: n_H in the "
                                 "cooling rate and the units module disagree (audit sec.3)")
            except ImportError:
                pass
        if self.insitu is not None:
            from luna.analysis.insitu import check as _insitu_check
            errs += _insitu_check(self)
        if warns:
            _warnings.warn("Config warnings:\n  - " + "\n  - ".join(warns), stacklevel=2)
        if errs:
            raise ValueError("Config validation failed:\n  - "
                             + "\n  - ".join(errs))
