"""FP64 scaling: LUNA (shard_map) vs AthenaK (RK2), 3D Sedov blast.

Matched double precision (the only comparison we report). Fair 2nd-order:
both PLM + HLLC + 2nd-order time (LUNA MUSCL-Hancock, AthenaK RK2). Metric:
aggregate Giga cell-updates / second.

  LUNA    : per-step "perf = X Mcells/s" from the runner log, median over
            steady-state steps (skip the JIT-compile-tainted warmup steps).
            NOT the "Average performance (global grid)" line, which divides
            total cells by the whole runtime incl. ~20 s compile.
  AthenaK : "zone-cycles/cpu_second" reported at end of the main loop.

Reads slurm_logs/scal_{luna,ak}_fp64_<NX>_<GPU>.out.* (newest per config).
Missing configs (not yet run) are simply skipped.
"""
import glob
import os
import re

import numpy as np

# Weak ladder: grid grows with the GPU count. 2048^3 does NOT fit on 64 GPUs in
# FP64 (134M cells/GPU OOMs), so its rung is at 128 GPUs (67M cells/GPU, ~10 GB).
# The 2048^3/128 FP64 run (submit/scaling/luna_fp64_2048_128.sh) is picked up
# automatically once its log lands; until then it is skipped as a missing point.
WEAK = [(32, 1), (64, 2), (128, 4), (256, 8), (512, 16), (1024, 32), (2048, 128)]
# Strong ladder: fixed 1024^3, increasing GPU count.
STRONG = [(1024, 32), (1024, 64), (1024, 128)]


def _newest(pattern):
    fs = glob.glob(pattern)
    return max(fs, key=os.path.getmtime) if fs else None


def luna_gcells(nx, gpu):
    """Median steady-state aggregate Gcell/s from a LUNA runner log."""
    f = _newest(f"slurm_logs/scal_luna_fp64_{nx}_{gpu}.out.*")
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
    f = _newest(f"slurm_logs/scal_ak_fp64_{nx}_{gpu}.out.*")
    if not f:
        return None
    val = None
    for ln in open(f, errors="ignore"):
        m = re.search(r"zone-cycles/cpu_second\s*=\s*([0-9.eE+-]+)", ln)
        if m:
            val = float(m.group(1))
    return val / 1e9 if val else None      # zone-cycles/s -> Gcell/s


def _collect(cfgs):
    return [(nx, gpu, luna_gcells(nx, gpu), ak_gcells(nx, gpu))
            for nx, gpu in cfgs]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.figstyle import apply_style, FIGSIZE
    apply_style()

    weak_all = _collect(WEAK)
    strong_all = _collect(STRONG)

    os.makedirs("figures", exist_ok=True)
    os.makedirs("code_paper_figs", exist_ok=True)
    with open("figures/scaling_fp64_data.csv", "w") as fh:
        fh.write("test,Nx,GPUs,LUNA_Gcells_s,AthenaK_Gcells_s\n")
        for tag, rows in (("weak", weak_all), ("strong", strong_all)):
            for nx, gpu, l, a in rows:
                fh.write(f"{tag},{nx},{gpu},{'' if l is None else f'{l:.3f}'},"
                         f"{'' if a is None else f'{a:.3f}'}\n")

    # Plot only configs with a measured LUNA point (skip crashed / not-yet-run),
    # so the paper figure shows no dangling empty ticks or broken lines.
    weak = [r for r in weak_all if r[2] is not None]
    strong = [r for r in strong_all if r[2] is not None]

    def _panel(rows, xs, xlabels, xlabel, out):
        """One standalone scaling plot (LUNA vs AthenaK + ideal-linear guide)."""
        fig, a_ = plt.subplots(figsize=FIGSIZE)
        lL = [r[2] for r in rows]
        lA = [r[3] for r in rows]
        a_.plot(xs, lL, "o-", color="#1f77b4", label="LUNA (shard-map)")
        a_.plot(xs, lA, "s--", color="#d62728", label="AthenaK (RK2)")
        a_.set_xscale("log", base=2)
        a_.set_yscale("log")
        a_.set_xticks(xs)
        a_.set_xticklabels(xlabels)
        # ideal linear scaling anchored at the first measured LUNA point
        g0i = next((i for i, v in enumerate(lL) if v is not None), None)
        if g0i is not None:
            base, gb = lL[g0i], xs[g0i]
            a_.plot(xs, [base * gg / gb for gg in xs], ":", color="gray",
                    label="ideal (linear)")
        a_.set_xlabel(xlabel)
        a_.set_ylabel("aggregate Gcell-updates / s")
        a_.legend(fontsize=9, loc="lower right")
        a_.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"wrote {out}")

    gw = [r[1] for r in weak]
    _panel(weak, gw, [f"{nx}$^3$\n{gpu}" for nx, gpu, *_ in weak],
           "resolution / GPUs (A100)", "code_paper_figs/scaling_weak_fp64.png")
    gs = [r[1] for r in strong]
    _panel(strong, gs, [str(x) for x in gs],
           "GPUs (A100)", "code_paper_figs/scaling_strong_fp64.png")

    for tag, rows in (("weak", weak_all), ("strong", strong_all)):
        for nx, gpu, l, a in rows:
            print(f"  {tag:6s} {nx}^3/{gpu}gpu  LUNA={l}  AthenaK={a}")


if __name__ == "__main__":
    main()
