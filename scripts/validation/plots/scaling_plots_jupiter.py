"""JUPITER GH200 scaling figures from the measured numbers in
scaling/jupiter/RESULTS.md: single-GPU hydro + gravity rooflines, hydro
weak scaling (512^3/GPU; fp64 on the 516^3 tile), hydro strong scaling.
Outputs to code_paper_figs/jupiter_*.pdf (+.png).
"""
import sys

sys.path.insert(0, ".")
import numpy as np

from scripts.validation.plots._plotkit import mpl, save

plt = mpl()

# same tab colors as the shipped scaling_plot_fp64.py
C_FP32, C_RECON, C_FP64, C_G = "#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"
IDEAL = dict(color="gray", linestyle=":", linewidth=1.4, zorder=1)

# ---- data (RESULTS.md) ----------------------------------------------------
# single-GPU hydro roofline: N -> Mcell/s
ROOF_N = [32, 64, 128, 256, 512]
ROOF = {"fp32":    [91, 638, 1371, 2117, 2193],
        "recon32": [88, 608, 1379, 1978, 2104],
        "fp64":    [86, 524, 904, 1251, 1223]}   # 512^3 = fused (plain NaNs)

# single-GPU gravity kernel roofline (mixed precision, fixed-point CIC)
GRAV_N = [32, 64, 128, 256, 512]
GRAV_PART = {"deposit": [498, 2602, 8033, 11044, 11717],
             "gather":  [471, 2145, 5096, 6137, 6096],
             "force (warm)": [82, 594, 2272, 2957, 2961],
             "force (cold)": [24, 141, 804, 1895, 2106]}   # Mpart/s
GRAV_CELL = {"V-cycle": [89, 648, 4278, 9319, 10080],
             "FMG": [23, 147, 866, 3025, 4357]}            # Mcell/s

# hydro weak scaling, 512^3/GPU tile: GPUs -> aggregate Mcell/s
WEAK_G = [1, 8, 64, 512, 4096]
WEAK = {"fp32":    [2193, 17423, 134209, 1018884, 7986287],
        "recon32": [2104, 15863, 121680, 936555, 7616118]}
# fp64 on the 516^3 tile (uint32-wrap-safe), to 512 GPUs
WEAK64_G = [1, 8, 64, 512]
WEAK64 = [1333, 10045, 79446, 624175]

# hydro strong scaling, fp32: grid -> [(GPUs, Mcell/s)]
STRONG = {512:  [(1, 2193), (8, 15523), (64, 67214)],
          1024: [(8, 17423), (64, 118916), (512, 515292)],
          2048: [(64, 134209), (512, 930811), (4096, 1274088)],
          4096: [(512, 1018884), (4096, 6646843)]}
STRONG_C = {512: C_FP32, 1024: C_RECON, 2048: C_G, 4096: C_FP64}

OUT = "code_paper_figs"


def _logx_cells(ax, ns):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([f"${n}^3$" for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("mesh size")


# ---- 1: single-GPU hydro roofline ----------------------------------------
fig, ax = plt.subplots()
for name, col, mk in [("fp32", C_FP32, "o"), ("recon32", C_RECON, "s"),
                      ("fp64", C_FP64, "^")]:
    ax.plot(ROOF_N, np.array(ROOF[name]) / 1e3, marker=mk, ms=6, lw=2,
            color=col, label=name)
    ax.annotate(name, (ROOF_N[-1], ROOF[name][-1] / 1e3),
                xytext=(6, 0), textcoords="offset points", color="0.25",
                fontsize=11, va="center")
_logx_cells(ax, ROOF_N)
ax.set_ylabel("Gcell / s (1 GH200)")
ax.set_xlim(28, 800)
ax.legend(loc="upper left", frameon=False)
save(fig, f"{OUT}/jupiter_hydro_roofline.pdf", also_png=True, close=True)

# ---- 2: single-GPU gravity kernel roofline (two units, two panels) -------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
for (name, v), mk in zip(GRAV_PART.items(), "os^D"):
    col = {"deposit": C_FP32, "gather": C_RECON, "force (warm)": C_G,
           "force (cold)": C_FP64}[name]
    a1.plot(GRAV_N, np.array(v) / 1e3, marker=mk, ms=6, lw=2, color=col,
            label=name)
a1.set_ylabel("Gpart / s (1 GH200)")
a1.set_yscale("log")
a1.legend(loc="lower right", frameon=False, fontsize=11)
for (name, v), col, mk in zip(GRAV_CELL.items(), [C_FP32, C_FP64], "os"):
    a2.plot(GRAV_N, np.array(v) / 1e3, marker=mk, ms=6, lw=2, color=col,
            label=name)
a2.set_ylabel("Gcell / s (1 GH200)")
a2.set_yscale("log")
a2.legend(loc="lower right", frameon=False, fontsize=11)
for ax in (a1, a2):
    _logx_cells(ax, GRAV_N)
save(fig, f"{OUT}/jupiter_gravity_roofline.pdf", also_png=True, close=True)

# ---- 3: hydro weak scaling -----------------------------------------------
fig, ax = plt.subplots()
g = np.array(WEAK_G, float)
ax.plot(g, WEAK["fp32"][0] * g / 1e3, **IDEAL)
ax.annotate("ideal", (g[-2], WEAK["fp32"][0] * g[-2] / 1e3),
            xytext=(-4, 10), textcoords="offset points", color="0.45",
            fontsize=11, ha="right")
for name, col, mk in [("fp32", C_FP32, "o"), ("recon32", C_RECON, "s")]:
    ax.plot(WEAK_G, np.array(WEAK[name]) / 1e3, marker=mk, ms=6, lw=2,
            color=col, label=f"{name} ($512^3$/GPU)")
ax.plot(WEAK64_G, np.array(WEAK64) / 1e3, marker="^", ms=6, lw=2,
        color=C_FP64, label="fp64 ($516^3$/GPU)")
eff32 = WEAK["fp32"][-1] / (WEAK["fp32"][0] * WEAK_G[-1])
ax.annotate(f"{WEAK['fp32'][-1]/1e6:.2f} Tcell/s\n{eff32:.0%} efficiency",
            (WEAK_G[-1], WEAK["fp32"][-1] / 1e3), xytext=(-8, -34),
            textcoords="offset points", ha="right", color="0.25", fontsize=11)
eff64 = WEAK64[-1] / (WEAK64[0] * WEAK64_G[-1])
ax.annotate(f"{eff64:.0%}", (WEAK64_G[-1], WEAK64[-1] / 1e3),
            xytext=(4, -14), textcoords="offset points", color="0.25",
            fontsize=11)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(WEAK_G)
ax.set_xticklabels([str(v) for v in WEAK_G])
ax.set_xlabel("GH200 GPUs")
ax.set_ylabel("aggregate Gcell / s")
ax.legend(loc="upper left", frameon=False, fontsize=11)
save(fig, f"{OUT}/jupiter_hydro_weak.pdf", also_png=True, close=True)

# ---- 4: hydro strong scaling (fp32) --------------------------------------
fig, ax = plt.subplots()
for nx, pts in STRONG.items():
    gs = np.array([p[0] for p in pts], float)
    ys = np.array([p[1] for p in pts]) / 1e3
    ax.plot(gs, ys[0] * gs / gs[0], **IDEAL)
    ax.plot(gs, ys, marker="o", ms=6, lw=2, color=STRONG_C[nx],
            label=f"${nx}^3$")
    ax.annotate(f"${nx}^3$", (gs[-1], ys[-1]), xytext=(6, -2),
                textcoords="offset points", color="0.25", fontsize=11)
ax.set_xscale("log")
ax.set_yscale("log")
gticks = [1, 8, 64, 512, 4096]
ax.set_xticks(gticks)
ax.set_xticklabels([str(v) for v in gticks])
ax.set_xlabel("GH200 GPUs")
ax.set_ylabel("aggregate Gcell / s")
ax.legend(loc="upper left", frameon=False, fontsize=11, title="grid")
save(fig, f"{OUT}/jupiter_hydro_strong.pdf", also_png=True, close=True)

print("wrote code_paper_figs/jupiter_{hydro_roofline,gravity_roofline,"
      "hydro_weak,hydro_strong}.{pdf,png}")
