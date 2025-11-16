# config.py
from dataclasses import dataclass

@dataclass
class Config:

    # ---------------- Device ----------------
    
    # 0 = CPU
    # 1 = GPU
    # 2 = TPU
    DEVICE: int = 1 

    # ---------------- Domain ----------------
    Nx: int = 512
    Ny: int = 512
    Lx: float = 2.0
    Ly: float = 2.0

    # ---------------- Physics ----------------
    gamma: float = 5.0 / 3.0

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
    t_end: float = 5

    # Save a snapshot every N steps
    n_snapshots: int = 500

    # ---------------- Output ----------------
    output_dir: str = "snapshots_512"
    base_name: str = "snap"
    PLOT_AFTER_SIM: bool = True

