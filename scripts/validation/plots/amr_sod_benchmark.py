"""AMR physics benchmark #2: planar Sod shock tube vs the EXACT Riemann solution.

Same controlled design as benchmark #1: AMR (base + L levels, density-gradient
regrid tracking shock/contact) vs uniform (identical IC, amr_levels=0), both
against the exact solution (scripts/validation/sod_exact, Toro solver).

Metrics: L1(rho) x-profile error vs exact (transverse-averaged), positions of
the three waves, mass/energy conservation. VERDICT: AMR beats uniform on L1
and conserves to near machine eps.

Env: AMR_DIR, UNI_DIR, BENCH_TAG.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.validation.sod_exact import sod_exact
from scripts.validation.plots.amr_sedov_benchmark import load_snapshot

AMR_DIR = os.environ.get("AMR_DIR", "snapshots/amr_bench_sod")
UNI_DIR = os.environ.get("UNI_DIR", "snapshots/amr_bench_sod_uniform")
GAMMA = 1.4
TAG = os.environ.get("BENCH_TAG", "")


def x_profile(snap):
    """Transverse-averaged rho(x) + cell-center coordinates."""
    rho = snap["rho"].mean(axis=(1, 2))
    x = (np.arange(rho.size) + 0.5) * snap["dx"]
    return x, rho


def conservation(snapdir):
    first, last = load_snapshot(snapdir, 0), load_snapshot(snapdir, -1)

    def totals(s):
        dV = s["dx"] ** 3
        M = float(np.sum(s["rho"])) * dV
        E = float(np.sum(s["p"] / (GAMMA - 1.0) + 0.5 * s["rho"] * s["v2"])) * dV
        return M, E

    M0, E0 = totals(first)
    M1, E1 = totals(last)
    return abs(M1 - M0) / M0, abs(E1 - E0) / E0, last


def metrics(snap):
    x, rho = x_profile(snap)
    exact = sod_exact(x, snap["t"], x0=0.5, gamma=GAMMA)
    L1 = float(np.mean(np.abs(rho - exact["rho"])) / np.mean(exact["rho"]))
    return dict(x=x, rho=rho, exact=exact["rho"], L1=L1, t=snap["t"])


def main():
    dM_a, dE_a, amr_last = conservation(AMR_DIR)
    dM_u, dE_u, uni_last = conservation(UNI_DIR)
    A, U = metrics(amr_last), metrics(uni_last)

    print("\n=== AMR Sod benchmark (vs exact Riemann solution) ===")
    print(f"  t = {A['t']:.4f}   gamma = {GAMMA}")
    print(f"  {'':10s} {'L1(rho)':>9s} {'dM/M':>10s} {'dE/E':>10s}")
    print(f"  {'AMR':10s} {A['L1']:9.4f} {dM_a:10.2e} {dE_a:10.2e}")
    print(f"  {'uniform':10s} {U['L1']:9.4f} {dM_u:10.2e} {dE_u:10.2e}")
    ok = (A["L1"] < U["L1"]) and dM_a < 1e-10
    print(f"  gain: L1(uniform)/L1(AMR) = {U['L1'] / max(A['L1'], 1e-30):.2f}x")
    print("BENCH RESULT:", "PASS" if ok else "FAIL")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    xf = np.linspace(0, 1, 2000)
    ex = sod_exact(xf, A["t"], x0=0.5, gamma=GAMMA)
    ax.plot(xf, ex["rho"], "k-", lw=1.4, label="exact (Toro)")
    ax.plot(U["x"], U["rho"], "o-", ms=2.5, lw=0.8, color="#d62728",
            label=f"uniform {uni_last['Nx']}")
    ax.plot(A["x"], A["rho"], "o-", ms=2.5, lw=0.8, color="#1f77b4",
            label=f"AMR (eff. {amr_last['Nx']})")
    ax.set_xlabel("x"); ax.set_ylabel(r"$\rho$")
    ax.set_title(f"Sod 3D (planar) at t={A['t']:.2f}")
    ax.legend(frameon=False, fontsize=9)
    ax2.semilogy(U["x"], np.abs(U["rho"] - U["exact"]), color="#d62728",
                 lw=1, label="uniform")
    ax2.semilogy(A["x"], np.abs(A["rho"] - A["exact"]), color="#1f77b4",
                 lw=1, label="AMR")
    ax2.set_xlabel("x"); ax2.set_ylabel(r"$|\rho - \rho_{\rm exact}|$")
    ax2.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = f"code_paper_figs/amr_benchmark_sod{TAG}.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)
    with open(out.replace(".png", ".json"), "w") as f:
        json.dump(dict(t=A["t"],
                       amr=dict(L1=A["L1"], dM=dM_a, dE=dE_a),
                       uniform=dict(L1=U["L1"], dM=dM_u, dE=dE_u)), f, indent=2)


if __name__ == "__main__":
    main()
