"""Linear Jeans instability: the self-gravity acceptance test.

    python -m scripts.validation.runs.jeans_instability_run nx=64 t_end=1.2
    python -m scripts.validation.runs.jeans_instability_run nx=32 AMR=1 amr_levels=1

A cosine perturbation in a periodic box grows as A0 exp(sigma t) with
sigma = sqrt(4 pi G rho0 - cs^2 k^2); sigma is fitted from the snapshots and
compared (default sigma ~ 5.33, ~6 e-folds, stays linear).

Parameters: nx, t_end, amr_levels, amr_thr and amr_regrid, which build the
per-level threshold list. AMR=1 runs the same problem through the refinement
runner: amr_levels=0 exercises the block machinery at base resolution, and
amr_levels=1 with a tiny threshold refines the whole box, so nx=32 with one
level compares with the dense nx=64 run. SELF_GRAVITY=0 is the null run.
MESH_SHAPE, output_dir and the other Config fields follow as field=value.
"""
import os
from dataclasses import replace

import numpy as np

from luna.setup.cli import script_main
from luna.ic_generation import jeans3d
from luna.template_config import Config

A0 = 1.0e-4                              # initial relative density amplitude
GAMMA, RHO0, P0 = 5.0 / 3.0, 1.0, 1.0
CS2 = GAMMA * P0 / RHO0
G = 7.5                                  # lowest periodic mode unstable, sigma ~ 5
L = 1.0
K = 2.0 * np.pi / L


def growth_rate():
    """sigma of this setup, from the IC module's dispersion relation."""
    cfg = replace(Config(), test_problem="Jeans-Instability-3D", Lx=L, gamma=GAMMA,
                  GRAVITATIONAL_G=G, problem=jeans3d.Params(rho0=RHO0, P0=P0, amp=A0, kx=K))
    sigma2 = float(jeans3d.sigma_squared(cfg))
    if sigma2 <= 0:
        raise SystemExit(f"Mode k={K:.3f} is stable (sigma^2={sigma2:.3f} <= 0); "
                         "adjust G or k for an unstable Jeans test")
    return np.sqrt(sigma2)


def mode_amplitudes(out_dir):
    """(t, |delta rho_k| / rho0, nx) of the kx mode of the y,z-averaged density, per snapshot.

    The Fourier amplitude is robust to limiter crest-clipping, unlike max|delta rho|."""
    import glob
    from luna.file_io.read_snapshot import read_snapshot
    ts, amps = [], []
    for d in sorted(glob.glob(os.path.join(out_dir, "snapdir_*"))):
        snap = read_snapshot(d)
        xg = snap["x"]
        drho = snap["rho"].mean(axis=(1, 2)) - RHO0
        c = 2.0 * np.mean(drho * np.cos(K * xg)) / RHO0
        s = 2.0 * np.mean(drho * np.sin(K * xg)) / RHO0
        amps.append(float(np.hypot(c, s)))
        ts.append(float(snap["t"]))
    order = np.argsort(ts)
    return np.asarray(ts)[order], np.asarray(amps)[order], len(xg)


def fit_sigma(ts, amps):
    """(slope, intercept, mask) of log amplitude vs t over the linear range 3 A0 < A < 0.1."""
    mask = (amps > 3.0 * A0) & (amps < 0.1) & np.isfinite(amps)
    if mask.sum() >= 2:
        slope, intercept = np.polyfit(ts[mask], np.log(amps[mask]), 1)
        return float(slope), float(intercept), mask
    return float("nan"), float("nan"), mask


def config(nx=64, t_end=1.2, amr_levels=1, amr_thr=1e-9, amr_regrid=0):
    sigma = growth_rate()
    print(f"Jeans instability test: Nx={nx}, k={K:.3f}, cs2={CS2:.3f}, G={G:.3f}")
    print(f"  sigma_analytic = {sigma:.4f} (growth time {1 / sigma:.4f}); "
          f"t_end = {t_end:.3f} -> {t_end * sigma:.2f} e-folds expected")
    return replace(
        Config(),
        test_problem="Jeans-Instability-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=nx, Ny=nx, Nz=nx, Lx=L, Ly=L, Lz=L,
        gamma=GAMMA, ORDER=1, LIMITER=1,
        t_end=t_end, time_bet_snaps=t_end / 40,      # 40 snapshots: clean log-slope fit
        SELF_GRAVITY=True, GRAVITATIONAL_G=G,
        POISSON_FMG_VCYCLES_PER_LEVEL=2, POISSON_FMG_EXTRA_VCYCLES=5,
        problem=jeans3d.Params(rho0=RHO0, P0=P0, amp=A0,
                               kx=K, ky=0.0, kz=0.0),       # plane wave in x
        PLOT_AFTER_SIM=False, clear_output_dir=True,
        output_dir=f"snapshots/jeans_Nx{nx}",
        amr_levels=amr_levels, amr_block_shape=16, amr_criterion="density",
        amr_thresholds=[amr_thr] * max(1, amr_levels), amr_regrid_interval=amr_regrid,
        amr_subcycle=False, amr_shard=False,
        enable_x64=True, enable_checkpointing=False,
    )


def report(res):
    """Fit sigma from the snapshots and draw the analytic comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from luna.utils.figstyle import FIGSIZE, apply_style, decorate
    apply_style()
    sigma = growth_rate()
    out_dir = res.output_dir
    ts, amps, nx = mode_amplitudes(out_dir)
    slope, intercept, mask = fit_sigma(ts, amps)
    rel = abs(slope - sigma) / sigma if np.isfinite(slope) else float("nan")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    tt = np.linspace(float(ts.min()), float(ts.max()), 200)
    ax.semilogy(tt, A0 * np.exp(sigma * tt), "-", color="k",
                label=fr"analytic  $A_0 e^{{\sigma t}}$, $\sigma$={sigma:.3f}")
    if np.isfinite(slope):
        ax.semilogy(ts[mask], np.exp(intercept + slope * ts[mask]), "-",
                    color="C3", label=fr"fit  $\sigma$={slope:.3f} (rel {rel:.1%})")
    ax.semilogy(ts, amps, "o", ms=6, color="C0", zorder=5,
                label=r"LUNA  $|\delta\rho_k|/\rho_0$")
    ax.set_xlabel("t  [code units]")
    ax.set_ylabel(r"mode amplitude $|\delta\rho_k|/\rho_0$")
    ax.legend(loc="upper left")
    decorate(ax)
    fig.tight_layout()
    os.makedirs("code_paper_figs", exist_ok=True)
    figpath = f"code_paper_figs/jeans_growth_Nx{nx}.png"
    fig.savefig(figpath, dpi=140)
    print(f"RESULT jeans: sigma_meas={slope:.4f} sigma_analytic={sigma:.4f} "
          f"rel={rel:.3e}  ({int(mask.sum())} pts in linear window)", flush=True)
    print(f"  figure -> {figpath}", flush=True)


if __name__ == "__main__":
    script_main(config, report=report)
