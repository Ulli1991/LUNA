# config.py
from dataclasses import dataclass

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
    test_problem: str = "Shu-Osher"

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

    Nx: int = 1024
    Ny: int = 1024
    Nz: int = 1024
    Lx: float = 10
    Ly: float = 10
    Lz: float = 10

    # ---------------- Physics ----------------
    #gamma: float = 5.0 / 3.0
    gamma: float = 1.4

    # ---------------- Time Integration ----------------
    CFL: float = 0.8

    # 0 = RK2
    # 1 = Predictor Corrector (Hancock)
    # 2 = RK4
    TIME_SOLVER: int = 0

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
    LIMITER: int = 1

    # ---------------- Time control ----------------
    t_start: float = 0.0
    t_end: float = 1.8

    # Time interval between snapshots (code units)
    time_bet_snaps: float = 0.005

    # ---------------- Output ----------------
    output_dir: str = "snapshots_shu"
    base_name: str = "snap"
    PLOT_AFTER_SIM: bool = True

