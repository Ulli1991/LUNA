"""Linear Jeans instability — Phase 1 acceptance test for self-gravity.

Uniform periodic box (γ-law gas) with a small sinusoidal density
perturbation. For wavenumber k and background sound speed c_s, the
linear theory growth rate is

    ω² = c_s² k² - 4πGρ₀

For k < k_J = √(4πGρ₀)/c_s the mode is unstable: ω² < 0 and δρ grows
exponentially at rate σ = √|ω²|. We integrate with LUNA + multigrid
self-gravity and measure σ_measured from the time series of max|δρ|/ρ₀.

Default parameters give σ_analytic ≈ 3.27 → growth time 0.31, so 2.0
code time covers ~6 e-folds while keeping the perturbation linear
(A_0 = 1e-4 → A(t_end) ~ 7%).

Env vars:
    LUNA_JEANS_NX     : grid resolution (default 64)
    LUNA_JEANS_TEND   : end time in code units (default 2.0)
"""
import os
import numpy as np

from scripts.validation.runner import run_one


def main():
    Nx = int(os.environ.get("LUNA_JEANS_NX", "64"))
    t_end = float(os.environ.get("LUNA_JEANS_TEND", "2.0"))

    # Linear-theory parameters (code units): ρ₀ = 1, c_s² = γP/ρ.
    GAMMA = 5.0 / 3.0
    RHO0 = 1.0
    P0 = 1.0
    CS2 = GAMMA * P0 / RHO0  # = 5/3
    # Pick G so the lowest periodic mode (k = 2π/L) is unstable with σ ≈ 5
    # → ~5 e-folds at t_end = 1.0. Need 4πGρ₀ > c_s² k².
    G = 7.5                  # gravitational constant in code units
    FOURPIG = 4.0 * np.pi * G
    L = 1.0
    k = 2.0 * np.pi / L

    omega_sq = CS2 * k * k - FOURPIG * RHO0
    growth_rate = np.sqrt(-omega_sq) if omega_sq < 0 else 0.0
    if growth_rate <= 0:
        raise SystemExit(
            f"Mode k={k:.3f} is stable (ω²={omega_sq:.3f} ≥ 0). "
            "Adjust G or k for an unstable Jeans test."
        )

    overrides = dict(
        test_problem="Jeans-Instability-3D",
        SPATIAL_DIMENSION=3,
        BOUNDARY=0,                 # periodic
        Nx=Nx, Ny=Nx, Nz=Nx,
        Lx=L, Ly=L, Lz=L,
        gamma=GAMMA,
        TIME_SOLVER=1,              # MUSCL-Hancock
        ORDER=1,
        LIMITER=1,
        CFL=0.3,
        t_end=t_end,
        time_bet_snaps=t_end / 20,  # 20 snapshots
        # Self-gravity wiring
        SELF_GRAVITY=os.environ.get("LUNA_SELF_GRAVITY", "1") == "1",
        GRAVITATIONAL_G=G,
        POISSON_TOL=1e-6,
        POISSON_MAX_CYCLES=15,
        POISSON_FMG_VCYCLES_PER_LEVEL=2,
        POISSON_FMG_EXTRA_VCYCLES=5,
        # IC parameters — these go into the IC builder (must be added there).
        jeans_rho0=RHO0,
        jeans_P0=P0,
        jeans_amp=1.0e-4,
        jeans_kx=k, jeans_ky=0.0, jeans_kz=0.0,  # plane-wave perturbation in x
        PLOT_AFTER_SIM=False,
        production_mode=True,
        output_dir=f"snapshots/jeans_Nx{Nx}",
        enable_x64=True,            # tight test — fp64 for clean growth rate
        enable_checkpointing=False,
        USE_SHARDMAP=os.environ.get("LUNA_USE_SHARDMAP", "0") == "1",
        MESH_SHAPE=(
            tuple(int(s) for s in os.environ["LUNA_MESH_SHAPE"].split(","))
            if os.environ.get("LUNA_MESH_SHAPE") else None
        ),
    )

    print(f"Jeans instability test: Nx={Nx}, k={k:.3f}, c_s²={CS2:.3f}, G={G:.3f}")
    print(f"  ω² = {omega_sq:.4f}  →  σ_analytic = {growth_rate:.4f}  (growth time {1/growth_rate:.4f})")
    print(f"  Running to t_end = {t_end:.3f}  →  {t_end * growth_rate:.2f} e-folds expected")
    run_one(overrides)


if __name__ == "__main__":
    main()
