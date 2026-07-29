"""FP64 scaling figure: LUNA (shard_map) vs AthenaK (RK2), 3D Sedov, A100 + MI300A.

LUNA metric = median per-step "perf = X Mcells/s" over steady-state steps --
NOT the "Average performance" line (that includes JIT compile). AthenaK =
end-of-loop zone-cycles/cpu_second. A100 numbers from
code_paper_figs/scaling_fp64_data.csv; MI300A parsed live from scaling/runs/<label>/.
Missing configs are skipped, so the figure fills in as points land."""
import glob
import os
import re

import numpy as np

from scripts.validation.plots._plotkit import save

# Weak ladder: grid grows with the GPU count.
#  A100 : 2048^3 does NOT fit on 64 GPUs in FP64 (134M cells/GPU OOMs on 40GB),
#         so its rung is at 128 GPUs.
#  MI300A (128GB): 2048^3 fits at 64 GPUs, and we push a 4096^3 rung at 128.
A100_WEAK = [(32, 1), (64, 2), (128, 4), (256, 8), (512, 16), (1024, 32), (2048, 128)]
# Original diagonal ladder — truncated at 2048^3 (4096^3 OOMs in FP64 at 128 GPU).
AMD_WEAK  = [(32, 1), (64, 2), (128, 4), (256, 8), (512, 16), (1024, 32), (2048, 64)]
A100_STRONG = [(1024, 32), (1024, 64), (1024, 128)]
AMD_STRONG  = [(1024, 32), (1024, 64), (1024, 128), (1024, 256)]
# Saturated weak ladder: bigger per-GPU tiles (~161^3 -> 645^3, all >=90% of the
# single-GPU roofline), z=1 slab meshes (race-free). MI300A only (no A100 ref).
SAT_WEAK = [(512, 32), (1024, 64), (2048, 128), (4096, 256)]


def _newest(pattern):
    fs = glob.glob(pattern)
    return max(fs, key=os.path.getmtime) if fs else None


def _luna_perf_median(logfile, min_samples=1):
    """Median steady-state aggregate Mcell/s from a LUNA runner log (or None).

    30-step scaling runs emit only ~3 `perf =` lines, so we drop just the first
    (compile-warm) sample and take the median of the rest.
    """
    if not logfile:
        return None
    perf = []
    for ln in open(logfile, errors="ignore"):
        m = re.search(r"perf\s*=\s*([0-9.eE+-]+)\s*Mcells/s", ln)
        if m and float(m.group(1)) > 0:
            perf.append(float(m.group(1)))
    if len(perf) < min_samples:
        return None
    steady = perf[1:] if len(perf) > 1 else perf   # drop the compile-warm sample
    return float(np.median(steady))


def _ak_zonecyc(logfile):
    """Aggregate zone-cycles/cpu_second from an AthenaK log (or None)."""
    if not logfile:
        return None
    val = None
    for ln in open(logfile, errors="ignore"):
        m = re.search(r"zone-cycles/cpu_second\s*=\s*([0-9.eE+-]+)", ln)
        if m:
            val = float(m.group(1))
    return val


# ---------------------------------------------------------------------------
# A100 (Raven) — reference. slurm_logs live on Raven; fall back to the cached
# CSV (code_paper_figs/scaling_fp64_data.csv) which is what persists on viper.
# ---------------------------------------------------------------------------
def a100_ref():
    """Return {(test, nx, gpu): (luna_Gcells, ak_Gcells)} from the cached CSV."""
    out = {}
    path = "code_paper_figs/scaling_fp64_data.csv"
    if not os.path.exists(path):
        return out
    for ln in open(path):
        p = ln.strip().split(",")
        if len(p) != 5 or p[0] == "test":
            continue
        test, nx, gpu, l, a = p
        out[(test, int(nx), int(gpu))] = (
            float(l) if l else None, float(a) if a else None)
    return out


# ---------------------------------------------------------------------------
# MI300A (viper) — LUNA in scaling/runs/, AthenaK in athenak/viper_run/runs/.
# Per-config run-dir label maps (labels are not uniform across weak/strong).
# ---------------------------------------------------------------------------
_LUNA_ROOT = "scaling/runs"
_AK_ROOT = "/viper/u2/uli/athenak/viper_run/runs"

AMD_LUNA_DIR = {
    ("weak", 32, 1): "weak_g1_32",   ("weak", 64, 2): "weak_g2_64",
    ("weak", 128, 4): "weak_g4_128", ("weak", 256, 8): "weak_g8_256",
    ("weak", 512, 16): "weak_g16_512", ("weak", 1024, 32): "weak_g32_1024",
    ("weak", 2048, 64): "weak_g64_2048_881", ("weak", 4096, 128): "weak_g128_4096_z2",
    ("strong", 1024, 32): "weak_g32_1024",   ("strong", 1024, 64): "zsplit_g64_881",
    ("strong", 1024, 128): "strong_g128_z2", ("strong", 1024, 256): "strong_g256_z2",
    # saturated weak ladder (z=1 slabs)
    ("weak_sat", 512, 32): "sat_g32_512",      ("weak_sat", 1024, 64): "zsplit_g64_881",
    ("weak_sat", 2048, 128): "sat_g128_2048",  ("weak_sat", 4096, 256): "sat_g256_4096",
}
AMD_AK_DIR = {
    ("weak", 32, 1): "ak_weak_g1_32",   ("weak", 64, 2): "ak_weak_g2_64",
    ("weak", 128, 4): "ak_weak_g4_128", ("weak", 256, 8): "ak_weak_g8_256",
    ("weak", 512, 16): "ak_weak_g16_512", ("weak", 1024, 32): "ak_weak_g32_1024",
    ("weak", 2048, 64): "ak_weak_g64_2048", ("weak", 4096, 128): "ak_weak_g128_4096",
    ("strong", 1024, 32): "ak_strong_g32",   ("strong", 1024, 64): "ak_strong_g64",
    ("strong", 1024, 128): "ak_strong_g128", ("strong", 1024, 256): "ak_strong_g256",
    # saturated weak ladder (matched meshblocks)
    ("weak_sat", 512, 32): "ak_sat_g32_512",     ("weak_sat", 1024, 64): "ak_strong_g64",
    ("weak_sat", 2048, 128): "ak_sat_g128_2048", ("weak_sat", 4096, 256): "ak_sat_g256_4096",
}


# Portable fallback: when the live viper run-dirs are not reachable (e.g. running
# on Raven), read the AMD points from the snapshot CSV instead.
def _amd_csv():
    out = {}
    path = "code_paper_figs/scaling_fp64_amd_data.csv"
    if not os.path.exists(path):
        return out
    for ln in open(path):
        p = ln.strip().split(",")
        if len(p) != 7 or p[0] == "test":
            continue
        test, nx, gpu, lA, aA, _l1, _a1 = p
        out[(test, int(nx), int(gpu))] = (
            float(lA) if lA else None, float(aA) if aA else None)
    return out


_AMD_SNAP = _amd_csv()


def amd_luna(test, nx, gpu):
    lbl = AMD_LUNA_DIR.get((test, nx, gpu))
    v = _luna_perf_median(_newest(f"{_LUNA_ROOT}/{lbl}/slurm.out.*")) if lbl else None
    if v:
        return v / 1e3    # Mcell/s -> Gcell/s
    return _AMD_SNAP.get((test, nx, gpu), (None, None))[0]   # CSV fallback


def amd_ak(test, nx, gpu):
    lbl = AMD_AK_DIR.get((test, nx, gpu))
    v = _ak_zonecyc(_newest(f"{_AK_ROOT}/{lbl}/slurm.out.*")) if lbl else None
    if v:
        return v / 1e9    # zone-cyc/s -> Gcell/s
    return _AMD_SNAP.get((test, nx, gpu), (None, None))[1]   # CSV fallback


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from utils.figstyle import apply_style, FIGSIZE
        apply_style()
    except Exception:
        FIGSIZE = (6, 4.5)

    ref = a100_ref()

    def collect(test, amd_cfgs, a100_cfgs):
        rows = []
        for nx, gpu in amd_cfgs:
            lA, aA = amd_luna(test, nx, gpu), amd_ak(test, nx, gpu)
            l1, a1 = ref.get((test, nx, gpu), (None, None))
            rows.append((nx, gpu, lA, aA, l1, a1))
        # A100 points at GPU counts not in the AMD ladder (e.g. weak 2048@128)
        amd_gpus = {g for _, g in amd_cfgs}
        for nx, gpu in a100_cfgs:
            if gpu not in amd_gpus:
                l1, a1 = ref.get((test, nx, gpu), (None, None))
                rows.append((nx, gpu, None, None, l1, a1))
        return sorted(rows, key=lambda r: r[1])

    weak = collect("weak", AMD_WEAK, A100_WEAK)
    strong = collect("strong", AMD_STRONG, A100_STRONG)
    weak_sat = collect("weak_sat", SAT_WEAK, [])   # saturated ladder, MI300A only

    os.makedirs("figures", exist_ok=True)
    os.makedirs("code_paper_figs", exist_ok=True)
    with open("code_paper_figs/scaling_fp64_amd_data.csv", "w") as fh:
        fh.write("test,Nx,GPUs,LUNA_MI300A,AthenaK_MI300A,LUNA_A100,AthenaK_A100\n")
        for tag, rows in (("weak", weak), ("strong", strong), ("weak_sat", weak_sat)):
            for nx, gpu, lA, aA, l1, a1 in rows:
                f = lambda v: "" if v is None else f"{v:.3f}"
                fh.write(f"{tag},{nx},{gpu},{f(lA)},{f(aA)},{f(l1)},{f(a1)}\n")

    def _panel(rows, xlabels, xlabel, out, title):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.set_title(title, fontsize=10)
        xs = [r[1] for r in rows]

        def _plot(idx, style, lbl, **kw):
            ys = [r[idx] for r in rows]
            if any(v is not None for v in ys):   # skip series with no data
                ax.plot(xs, ys, style, label=lbl, **kw)

        # MI300A (AMD) — primary, solid, bold
        _plot(2, "o-", "LUNA  · MI300A (AMD)", color="#1f77b4", lw=2)
        _plot(3, "s-", "AthenaK · MI300A (AMD)", color="#d62728", lw=2)
        # A100 — reference, faded, dashed (skipped automatically if absent)
        _plot(4, "o--", "LUNA  · A100 (NVIDIA, ref)", color="#1f77b4", lw=1.2,
              alpha=0.45, mfc="none")
        _plot(5, "s--", "AthenaK · A100 (NVIDIA, ref)", color="#d62728", lw=1.2,
              alpha=0.45, mfc="none")
        # ideal-linear guide anchored at the first measured MI300A LUNA point
        lL = [r[2] for r in rows]
        g0 = next((i for i, v in enumerate(lL) if v is not None), None)
        if g0 is not None:
            base, gb = lL[g0], xs[g0]
            ax.plot(xs, [base * g / gb for g in xs], ":", color="gray",
                    lw=1, label="ideal (linear)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("aggregate Gcell-updates / s")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        save(fig, out, dpi=140, close=True)
        print(f"wrote {out}")

    _panel(weak, [f"{nx}$^3$\n{gpu}" for nx, gpu, *_ in weak],
           "resolution / GPUs", "code_paper_figs/scaling_weak_fp64.png",
           "FP64 weak scaling — MI300A (AMD) vs A100 (NVIDIA)")
    _panel(strong, [str(r[1]) for r in strong],
           "GPUs (1024$^3$)", "code_paper_figs/scaling_strong_fp64.png",
           "FP64 strong scaling (1024$^3$) — MI300A (AMD) vs A100 (NVIDIA)")
    _panel(weak_sat, [f"{nx}$^3$\n{gpu}" for nx, gpu, *_ in weak_sat],
           "resolution / GPUs", "code_paper_figs/scaling_weak_saturated_fp64.png",
           "FP64 weak scaling, saturated regime (MI300A) — LUNA vs AthenaK")

    print("\n  config            LUNA_MI300A  AthenaK_MI300A   LUNA_A100  AthenaK_A100")
    for tag, rows in (("weak", weak), ("strong", strong), ("weak_sat", weak_sat)):
        for nx, gpu, lA, aA, l1, a1 in rows:
            fmt = lambda v: "   -   " if v is None else f"{v:7.3f}"
            print(f"  {tag:6s} {nx:>4}^3/{gpu:<3}gpu "
                  f"{fmt(lA)}     {fmt(aA)}      {fmt(l1)}   {fmt(a1)}")


if __name__ == "__main__":
    main()
