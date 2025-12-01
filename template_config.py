# config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:

    # -------------------------------------------------------------------------
    # Test problem selection
    # -------------------------------------------------------------------------
    # Available options:
    #   1D:
    #       "Sod-Shock" (tested: works)
    #       "Woodward-Colella" (tested: works)
    #       "Shu-Osher" (tested: works)
    #       "Soundwave"
    #       "Soundwave-Advected"
    #
    #   Einfeldt (2D - effectively 1D):
    #       "Einfeldt-Strong"       (1-2-0-3)
    #       "Einfeldt-Shear"        (1-1-0-5)
    #       "Einfeldt-Transverse"   (1-1-2-5)
    #       NOTE: These are 2D tests with variation only in x-direction.
    #             Set SPATIAL_DIMENSION=2, Ny=1, and Ly to a small value (e.g., 0.01)
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
    # 3 = 3D
    SPATIAL_DIMENSION: int = 1
    # 0 = Periodic
    # 1 = von Neumann 
    # 2 = Reflective
    BOUNDARY: int = 1 

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

    # 0 = RK2
    # 1 = Predictor Corrector (Hancock)
    # 2 = RK4
    TIME_SOLVER: int = 2  # RK4 for better accuracy

    # ---------------- Riemann Solver ----------------
    # Reconstruction order:
    # 0 = first order Godunov
    # 1 = PLM (second-order MUSCL)
    # 2 = PPM (future, not implemented yet)
    ORDER: int = 1  # TESTING: First-order (no slope limiters) to isolate issue

    # Slope limiter (used only if ORDER == 1)
    # 0 = Minmod
    # 1 = MonCen
    # 2 = VanLeer
    LIMITER: int = 0

    # ---------------- Time control ----------------
    t_start: float = 0.0
    t_end: float = 0.245

    # Time interval between snapshots (code units)
    time_bet_snaps: float = 0.000038

    # ---------------- Output ----------------
    output_dir: str = "snapshots_sod_autodiff"
    base_name: str = "snap"
    PLOT_AFTER_SIM: bool = True
    output_cadence: int = 1  # Print diagnostics every N steps
    check_memory: bool = True

    # ---------------- Autodiff Options ----------------
    # Enable smooth (differentiable) slope limiters for automatic differentiation
    # Set to True when using main_autodiff.py or when computing gradients
    use_smooth_limiters: bool = True  # REQUIRED for autodiff - hard limiters are not differentiable!

    # Controls smoothness of differentiable limiters (used only if use_smooth_limiters=True)
    # Higher values = closer to original limiters (sharper transitions)
    # Lower values = smoother gradients (better for optimization)
    # Typical range: 5.0 (very smooth) to 50.0 (nearly original)
    # Recommended: 10.0 for smooth test problems (soundwave, advection)
    smooth_limiter_sharpness: float = 10.0   

