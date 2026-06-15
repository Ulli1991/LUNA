"""Strong + weak scaling: LUNA (shard_map) vs AthenaK (RK2), 3D Sedov blast.

Fair 2nd-order comparison: both PLM + HLLC + 2nd-order time (LUNA MUSCL-Hancock,
AthenaK RK2). Metric: aggregate Giga cell-updates / second.

  LUNA    : per-step "perf = X Mcells/s (global grid)" from the runner log,
            median over steady-state steps (skip JIT-compile warmup).
  AthenaK : "zone-cycles/cpu_second" reported at end of run.

Configs (separate SLURM jobs):
  strong (fixed 1024^3): 16, 32, 64, 128 GPU
  weak   (grow w/ GPU) : 256^3@4, 512^3@8, 1024^3@16, 2048^3@32 GPU

Reads the newest slurm_logs/scal_{luna,ak}_<NX>_<GPU>.out.* per config.
"""
import glob
import os
import re

import numpy as np

STRONG = [(1024, 16), (1024, 32), (1024, 64), (1024, 128)]
WEAK = [(256, 4), (512, 8), (1024, 16), (2048, 32)]


def _newest(pattern):
    fs = glob.glob(pattern)
    return max(fs, key=os.path.getmtime) if fs else None


def luna_gcells(nx, gpu):
    """Median steady-state aggregate Gcell/s from a LUNA runner log."""
    f = _newest(f"slurm_logs/scal_luna_{nx}_{gpu}.out.*")
    if not f:
        return None
    perf = []
    for ln in open(f, errors="ignore"):
        m = re.search(r"perf\s*=\s*([0-9.eE+-]+)\s*Mcells/s", ln)
        if m:
            v = float(m.group(1))
            if v > 0:
                perf.append(v)
    if len(perf) < 5:
        return None
    perf = perf[2:]                       # drop first couple (compile-tainted)
    return float(np.median(perf)) / 1e3   # Mcell/s -> Gcell/s


def ak_gcells(nx, gpu):
    """Aggregate Gcell/s from AthenaK zone-cycles/cpu_second."""
    f = _newest(f"slurm_logs/scal_ak_{nx}_{gpu}.out.*")
    if not f:
        return None
    val = None
    for ln in open(f, errors="ignore"):
        m = re.search(r"zone-cycles/cpu_second\s*=\s*([0-9.eE+-]+)", ln)
        if m:
            val = float(m.group(1))
    return val / 1e9 if val else None      # zone-cycles/s -> Gcell/s


def _collect(cfgs):
    rows = []
    for nx, gpu in cfgs:
        rows.append((nx, gpu, luna_gcells(nx, gpu), ak_gcells(nx, gpu)))
    return rows


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.figstyle import apply_style, FIGSIZE
    apply_style()

    strong = _collect(STRONG)
    weak = _collect(WEAK)

    # write a data record
    os.makedirs("figures", exist_ok=True)
    with open("figures/scaling_data.csv", "w") as fh:
        fh.write("test,Nx,GPUs,LUNA_Gcells_s,AthenaK_Gcells_s\n")
        for tag, rows in (("strong", strong), ("weak", weak)):
            for nx, gpu, l, a in rows:
                fh.write(f"{tag},{nx},{gpu},{'' if l is None else f'{l:.3f}'},"
                         f"{'' if a is None else f'{a:.3f}'}\n")

    fig, ax = plt.subplots(1, 2, figsize=(FIGSIZE[0] * 1.9, FIGSIZE[1]))

    # ---- strong scaling (1024^3) ----
    g = [r[1] for r in strong]
    lL = [r[2] for r in strong]; lA = [r[3] for r in strong]
    ax[0].plot(g, lL, "o-", color="#1f77b4", label="LUNA (shard_map)")
    ax[0].plot(g, lA, "s--", color="#d62728", label="AthenaK (RK2)")
    g0 = g[0]
    base = next((v for v in lL if v is not None), None)
    if base:
        ax[0].plot(g, [base * gg / g0 for gg in g], ":", color="gray",
                   label="ideal (linear)")
    ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
    ax[0].set_xticks(g); ax[0].set_xticklabels([str(x) for x in g])
    ax[0].set_xlabel("GPUs (A100)")
    ax[0].set_ylabel("aggregate Gcell-updates / s")
    ax[0].text(0.04, 0.96, "strong scaling — Sedov 1024$^3$",
               transform=ax[0].transAxes, va="top", ha="left", fontsize=11)
    ax[0].legend(fontsize=9, loc="lower right")

    # ---- weak scaling ----
    g = [r[1] for r in weak]
    lL = [r[2] for r in weak]; lA = [r[3] for r in weak]
    ax[1].plot(g, lL, "o-", color="#1f77b4", label="LUNA (shard_map)")
    ax[1].plot(g, lA, "s--", color="#d62728", label="AthenaK (RK2)")
    ax[1].set_xscale("log", base=2); ax[1].set_yscale("log")
    ax[1].set_xticks(g)
    ax[1].set_xticklabels([f"{nx}$^3$\n{gpu}" for nx, gpu in weak])
    ax[1].set_xlabel("resolution / GPUs")
    ax[1].set_ylabel("aggregate Gcell-updates / s")
    ax[1].text(0.04, 0.96, "weak scaling — min GPUs per resolution",
               transform=ax[1].transAxes, va="top", ha="left", fontsize=11)
    ax[1].legend(fontsize=9, loc="lower right")

    for a in (ax[0], ax[1]):
        a.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out = "figures/scaling_luna_vs_athenak.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")
    print("data:")
    for tag, rows in (("strong", strong), ("weak", weak)):
        for nx, gpu, l, a in rows:
            print(f"  {tag:6s} {nx}^3/{gpu}gpu  LUNA={l}  AthenaK={a}")


if __name__ == "__main__":
    main()
