"""Jeans instability growth rate vs analytic linear theory — resolution
convergence. Overlays the measured growing-mode amplitude δρ_k(t)/ρ₀ for every
LUNA resolution on disk against the analytic exponential A₀e^{σt}. Reads the
existing snapshots/jeans_Nx{N} runs; no in-figure title (per the figstyle rule).
"""
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from file_io.read_snapshot import read_snapshot
from utils.figstyle import apply_style, decorate, FIGSIZE

apply_style()

# linear-theory parameters (code units) — must match jeans_instability_run.py
RHO0 = 1.0
A0 = 1.0e-4
GAMMA = 5.0 / 3.0
P0 = 1.0
CS2 = GAMMA * P0 / RHO0
G = 7.5
L = 1.0
k = 2.0 * np.pi / L
GROWTH = np.sqrt(4.0 * np.pi * G * RHO0 - CS2 * k * k)   # σ_analytic ≈ 5.33

RES = [32, 64, 128, 256]
COLORS = {32: "C0", 64: "C1", 128: "C2", 256: "C3"}


def amps_for(nx):
    dirs = sorted(glob.glob(f"snapshots/jeans_Nx{nx}/snapdir_*"))
    ts, amps = [], []
    for d in dirs:
        snap = read_snapshot(d)
        xg = snap["x"]
        drho_bar = snap["rho"].mean(axis=(1, 2)) - RHO0
        c = 2.0 * np.mean(drho_bar * np.cos(k * xg)) / RHO0
        s = 2.0 * np.mean(drho_bar * np.sin(k * xg)) / RHO0
        amps.append(float(np.hypot(c, s)))
        ts.append(float(snap["t"]))
    o = np.argsort(ts)
    return np.asarray(ts)[o], np.asarray(amps)[o]


def sigma_fit(ts, amps):
    m = (amps > 3.0 * A0) & (amps < 0.1) & np.isfinite(amps)
    if m.sum() >= 2:
        slope, _ = np.polyfit(ts[m], np.log(amps[m]), 1)
        return float(slope)
    return float("nan")


def main():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    tmax = 0.0
    curves = []
    for nx in RES:
        if not glob.glob(f"snapshots/jeans_Nx{nx}/snapdir_*"):
            print(f"  [skip] no snapshots for Nx={nx}")
            continue
        ts, amps = amps_for(nx)
        curves.append((nx, ts, amps, sigma_fit(ts, amps)))
        tmax = max(tmax, float(ts.max()))

    tt = np.linspace(0.0, tmax, 200)
    ax.semilogy(tt, A0 * np.exp(GROWTH * tt), "-", color="k", lw=2.0,
                label=fr"analytic  $A_0 e^{{\sigma t}}$, $\sigma$={GROWTH:.3f}")
    for nx, ts, amps, sm in curves:
        ax.semilogy(ts, amps, "o", ms=5, color=COLORS.get(nx, None),
                    label=fr"LUNA $N={nx}^3$  ($\sigma$={sm:.2f})")
    ax.set_xlabel("t  [code units]")
    ax.set_ylabel(r"mode amplitude $|\delta\rho_k|/\rho_0$")
    ax.legend(loc="upper left")
    decorate(ax)
    fig.tight_layout()
    out = "figures/jeans_growth.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}  (σ_analytic={GROWTH:.3f}; "
          f"σ_meas={[round(c[3], 2) for c in curves]})")


if __name__ == "__main__":
    main()
