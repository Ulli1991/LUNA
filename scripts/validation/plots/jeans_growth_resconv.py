"""Jeans instability growth rate vs analytic linear theory — resolution
convergence. Overlays the measured growing-mode amplitude δρ_k(t)/ρ₀ for every
LUNA resolution on disk against the analytic exponential A₀e^{σt}. Reads the
existing snapshots/jeans_Nx{N} runs; no in-figure title (per the figstyle rule).
"""
import glob

import numpy as np
from scripts.validation.plots._plotkit import mpl, save

plt = mpl()
from luna.utils.figstyle import decorate, FIGSIZE

from scripts.validation.runs.jeans_instability_run import A0, fit_sigma, growth_rate, mode_amplitudes

GROWTH = growth_rate()                     # sigma_analytic ~ 5.33

RES = [32, 64, 128, 256]
COLORS = {32: "C0", 64: "C1", 128: "C2", 256: "C3"}


def amps_for(nx):
    return mode_amplitudes(f"snapshots/jeans_Nx{nx}")[:2]


def sigma_fit(ts, amps):
    return fit_sigma(ts, amps)[0]


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
    out = "code_paper_figs/jeans_growth.png"
    save(fig, out, dpi=140)
    print(f"wrote {out}  (σ_analytic={GROWTH:.3f}; "
          f"σ_meas={[round(c[3], 2) for c in curves]})")


if __name__ == "__main__":
    main()
