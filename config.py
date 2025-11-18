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
    test_problem: str = "Woodward-Colella"

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
    BOUNDARY: int = 2 

    Nx: int = 1024
    Ny: int = 1024
    Nz: int = 1024
    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 1.0

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
    t_end: float = 0.038

    # Time interval between snapshots (code units)
    time_bet_snaps: float = 0.001

    # ---------------- Output ----------------
    output_dir: str = "snapshots_wcb"
    base_name: str = "snap"
    PLOT_AFTER_SIM: bool = True

