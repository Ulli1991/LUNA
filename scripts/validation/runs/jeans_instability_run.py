"""Linear Jeans instability — Phase 1 acceptance test for self-gravity.

Uniform periodic box (γ-law gas) with a small sinusoidal density
perturbation. For wavenumber k and background sound speed c_s, the
linear theory growth rate is

    ω² = c_s² k² - 4πGρ₀

For k < k_J = √(4πGρ₀)/c_s the mode is unstable: ω² < 0. From a STATIC
(v=0) cosine density perturbation the mode grows as δρ(t) = A₀ cosh(σt)
with σ = √|ω²| (equal projection onto the growing and decaying
eigenmodes). We integrate with LUNA + multigrid self-gravity, read the
snapshots back, and fit σ_measured from the log-slope of max|δρ|/ρ₀ over
the linear window, then emit figures/jeans_growth_Nx{N}.png comparing the
measured growth to the analytic A₀cosh(σt).

Default parameters: σ_analytic ≈ 5.33 (growth time 0.19); t_end = 1.2
covers ~6.4 e-folds while keeping the perturbation linear
(A_0 = 1e-4 → A(t_end) ~ 3%, comfortably inside linear theory).

Env vars:
    LUNA_JEANS_NX     : grid resolution (default 64)
    LUNA_JEANS_TEND   : end time in code units (default 1.2)
"""
import os
import numpy as np

from dataclasses import replace
from template_config import Config
from setup.distributed_setup import initialize_jax_distributed
from main import run


def main():
    Nx = int(os.environ.get("LUNA_JEANS_NX", "64"))
    t_end = float(os.environ.get("LUNA_JEANS_TEND", "1.2"))
    A0 = 1.0e-4   # initial relative density amplitude (jeans_amp)

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
        ORDER=1,
        LIMITER=1,
        CFL=0.3,
        t_end=t_end,
        time_bet_snaps=t_end / 40,  # 40 snapshots → clean log-slope fit
        # Self-gravity wiring
        SELF_GRAVITY=os.environ.get("LUNA_SELF_GRAVITY", "1") == "1",
        GRAVITATIONAL_G=G,
        POISSON_FMG_VCYCLES_PER_LEVEL=2,
        POISSON_FMG_EXTRA_VCYCLES=5,
        # IC parameters — these go into the IC builder (must be added there).
        jeans_rho0=RHO0,
        jeans_P0=P0,
        jeans_amp=A0,
        jeans_kx=k, jeans_ky=0.0, jeans_kz=0.0,  # plane-wave perturbation in x
        PLOT_AFTER_SIM=False,
        production_mode=True,
        clear_output_dir=True,           # fresh run: no stale snapshots in analysis
        output_dir=f"snapshots/jeans_Nx{Nx}",
        enable_x64=True,            # tight test — fp64 for clean growth rate
        enable_checkpointing=False,
        MESH_SHAPE=(
            tuple(int(s) for s in os.environ["LUNA_MESH_SHAPE"].split(","))
            if os.environ.get("LUNA_MESH_SHAPE") else None
        ),
    )

    print(f"Jeans instability test: Nx={Nx}, k={k:.3f}, c_s²={CS2:.3f}, G={G:.3f}")
    print(f"  ω² = {omega_sq:.4f}  →  σ_analytic = {growth_rate:.4f}  (growth time {1/growth_rate:.4f})")
    print(f"  Running to t_end = {t_end:.3f}  →  {t_end * growth_rate:.2f} e-folds expected")
    out_dir = overrides["output_dir"]
    initialize_jax_distributed(enable_x64=bool(overrides.get("enable_x64", False)))
    overrides = {k: v for k, v in overrides.items() if k != "enable_x64"}
    run(replace(Config(), **overrides))

    # ---- measure σ from the snapshots & emit the analytic-comparison figure ----
    import jax
    if jax.process_index() != 0:
        return
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from file_io.read_snapshot import read_snapshot
    from utils.figstyle import apply_style, decorate, FIGSIZE
    apply_style()

    snapdirs = sorted(glob.glob(os.path.join(out_dir, "snapdir_*")))
    ts, amps = [], []
    for d in snapdirs:
        snap = read_snapshot(d)
        # Fourier-mode amplitude of the kx perturbation: project the
        # y,z-averaged δρ(x) onto cos/sin(kx). This is the physical linear-mode
        # amplitude — robust to limiter crest-clipping and single-cell noise
        # (unlike max|δρ|, which sits on the clipped extremum).
        xg = snap["x"]
        drho_bar = snap["rho"].mean(axis=(1, 2)) - RHO0
        c = 2.0 * np.mean(drho_bar * np.cos(k * xg)) / RHO0
        s = 2.0 * np.mean(drho_bar * np.sin(k * xg)) / RHO0
        amps.append(float(np.hypot(c, s)))
        ts.append(float(snap["t"]))
    ts = np.asarray(ts); amps = np.asarray(amps)
    order = np.argsort(ts); ts, amps = ts[order], amps[order]

    # The growing Jeans eigenmode IC gives a clean exponential δρ(t)=A₀e^{σt},
    # so a simple log-linear fit over the linear regime recovers σ directly.
    mask = (amps > 3.0 * A0) & (amps < 0.1) & np.isfinite(amps)
    if mask.sum() >= 2:
        slope, intercept = np.polyfit(ts[mask], np.log(amps[mask]), 1)
        sigma_meas = float(slope)
    else:
        slope = intercept = np.nan
        sigma_meas = float("nan")
    rel = abs(sigma_meas - growth_rate) / growth_rate if np.isfinite(sigma_meas) else float("nan")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    tt = np.linspace(float(ts.min()), float(ts.max()), 200)
    ax.semilogy(tt, A0 * np.exp(growth_rate * tt), "-", color="k",
                label=fr"analytic  $A_0 e^{{\sigma t}}$, $\sigma$={growth_rate:.3f}")
    if np.isfinite(sigma_meas):
        ax.semilogy(ts[mask], np.exp(intercept + slope * ts[mask]), "-",
                    color="C3", label=fr"fit  $\sigma$={sigma_meas:.3f} (rel {rel:.1%})")
    ax.semilogy(ts, amps, "o", ms=6, color="C0", zorder=5,
                label=r"LUNA  $|\delta\rho_k|/\rho_0$")
    ax.set_xlabel("t  [code units]"); ax.set_ylabel(r"mode amplitude $|\delta\rho_k|/\rho_0$")
    ax.legend(loc="upper left")
    decorate(ax)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    figpath = f"figures/jeans_growth_Nx{Nx}.png"
    fig.savefig(figpath, dpi=140)
    print(f"RESULT jeans: sigma_meas={sigma_meas:.4f} sigma_analytic={growth_rate:.4f} "
          f"rel={rel:.3e}  ({int(mask.sum())} pts in linear window)", flush=True)
    print(f"  figure -> {figpath}", flush=True)


if __name__ == "__main__":
    main()
