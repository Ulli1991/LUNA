"""JUPITER GH200 scaling figures from the measured numbers in
scaling/jupiter/RESULTS.md: single-GPU hydro + gravity rooflines, hydro
weak scaling (512^3/GPU; fp64 on the 516^3 tile), hydro strong scaling.
The single-GPU hydro roofline also carries the viper MI300A ladder, so the
two machines are read off one axis.
Outputs to code_paper_figs/jupiter_*.pdf (+.png).
"""

import numpy as np
from matplotlib.lines import Line2D

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
# viper MI300A, same runner/mesh axis (jobs 10958599-601, 2026-08-19).
# fp64 512^3 is PLAIN here — the uint32-wrap
# bug that forces the fused variant on CUDA does not occur on ROCm.
ROOF_MI = {"fp32":    [78, 544, 1706, 2262, 2437],
           "recon32": [83, 554, 1581, 1990, 2181],
           "fp64":    [84, 448, 927, 1131, 1082]}

# "recon32" = FP32 slopes/predictor/face states, FP64 conserved state+fluxes.
PREC_LABEL = {"fp32": "fp32",
              "recon32": "mixed (fp32 flux reconstruction)",
              "fp64": "fp64"}
PREC_STYLE = [("fp32", C_FP32, "o"), ("recon32", C_RECON, "s"),
              ("fp64", C_FP64, "^")]

# single-GPU gravity kernel roofline (mixed precision, fixed-point CIC)
GRAV_N = [32, 64, 128, 256, 512]
GRAV_PART = {"deposit": [498, 2602, 8033, 11044, 11717],
             "gather":  [471, 2145, 5096, 6137, 6096],
             "force (warm)": [82, 594, 2272, 2957, 2961],
             "force (cold)": [24, 141, 804, 1895, 2106]}   # Mpart/s
GRAV_CELL = {"V-cycle": [89, 648, 4278, 9319, 10080],
             "FMG": [23, 147, 866, 3025, 4357]}            # Mcell/s
# viper MI300A, matched to the GH200 setup — one particle per cell (N = nx^3)
# and a 100 Mpc/h box, from monofonIC unigrid ICs generated for this sweep
# (job 10958976, 2026-08-19). The l9 IC used elsewhere is NOT valid here: it
# holds 134M particles at every mesh, which is a different experiment.
GRAV_PART_MI = {"deposit": [400, 3097, 9172, 10723, 10650],
                "gather":  [409, 2669, 9274, 10000, 9439],
                "force (warm)": [49, 352, 2034, 3814, 4071],
                "force (cold)": [13, 80, 490, 1977, 2748]}
GRAV_CELL_MI = {"V-cycle": [50, 379, 2755, 9836, 13323],
                "FMG": [13, 79, 479, 2371, 5246]}

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


# ---- 1: single-GPU hydro roofline, GH200 vs MI300A ------------------------
# Precision is carried by colour, machine by line style + open/filled markers,
# so the two legends stay short and nothing is annotated onto the curves.
fig, ax = plt.subplots()
for name, col, mk in PREC_STYLE:
    ax.plot(ROOF_N, np.array(ROOF[name]) / 1e3, marker=mk, ms=6, lw=2,
            color=col, linestyle="-")
    ax.plot(ROOF_N, np.array(ROOF_MI[name]) / 1e3, marker=mk, ms=6, lw=2,
            color=col, linestyle="--", markerfacecolor="none")
_logx_cells(ax, ROOF_N)
ax.set_ylabel("Gcell / s per GPU")
ax.set_xlim(28, 640)

prec_leg = ax.legend(
    handles=[Line2D([], [], color=col, lw=2, marker=mk, ms=6,
                    label=PREC_LABEL[name]) for name, col, mk in PREC_STYLE],
    loc="upper left", frameon=False, fontsize=11)
ax.add_artist(prec_leg)
ax.legend(
    handles=[Line2D([], [], color="0.35", lw=2, linestyle="-", label="GH200"),
             Line2D([], [], color="0.35", lw=2, linestyle="--",
                    label="MI300A")],
    loc="lower right", frameon=False, fontsize=11)
save(fig, f"{OUT}/jupiter_hydro_roofline.pdf", also_png=True, close=True)

# ---- 2: single-GPU gravity kernel roofline (two units, two panels) -------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
PART_C = {"deposit": C_FP32, "gather": C_RECON, "force (warm)": C_G,
          "force (cold)": C_FP64}
for (name, v), mk in zip(GRAV_PART.items(), "os^D"):
    a1.plot(GRAV_N, np.array(v) / 1e3, marker=mk, ms=6, lw=2,
            color=PART_C[name], label=name)
    a1.plot(GRAV_N, np.array(GRAV_PART_MI[name]) / 1e3, marker=mk, ms=6, lw=2,
            color=PART_C[name], linestyle="--", markerfacecolor="none")
a1.set_ylabel("Gpart / s per GPU")
a1.set_yscale("log")
a1.legend(loc="lower right", frameon=False, fontsize=11)
for (name, v), col, mk in zip(GRAV_CELL.items(), [C_FP32, C_FP64], "os"):
    a2.plot(GRAV_N, np.array(v) / 1e3, marker=mk, ms=6, lw=2, color=col,
            label=name)
    a2.plot(GRAV_N, np.array(GRAV_CELL_MI[name]) / 1e3, marker=mk, ms=6, lw=2,
            color=col, linestyle="--", markerfacecolor="none")
a2.set_ylabel("Gcell / s per GPU")
a2.set_yscale("log")
for ax in (a1, a2):
    _logx_cells(ax, GRAV_N)
# Kernel names per panel; the machine key sits once, in the panel with room.
a2.legend(handles=a2.get_legend_handles_labels()[0] +
          [Line2D([], [], color="0.35", lw=2, linestyle="-", label="GH200"),
           Line2D([], [], color="0.35", lw=2, linestyle="--", label="MI300A")],
          loc="lower right", frameon=False, fontsize=11)
save(fig, f"{OUT}/jupiter_gravity_roofline.pdf", also_png=True, close=True)

# ---- 3: hydro weak scaling -----------------------------------------------
# Curves climb left-to-right on log-log, so the legend takes the empty upper
# left and the endpoint summary the empty lower right: no floating callouts.
fig, ax = plt.subplots()
g = np.array(WEAK_G, float)
ax.plot(g, WEAK["fp32"][0] * g / 1e3, label="ideal", **IDEAL)
for name, col, mk in [("fp32", C_FP32, "o"), ("recon32", C_RECON, "s")]:
    ax.plot(WEAK_G, np.array(WEAK[name]) / 1e3, marker=mk, ms=6, lw=2,
            color=col, label=f"{PREC_LABEL[name]}, $512^3$/GPU")
ax.plot(WEAK64_G, np.array(WEAK64) / 1e3, marker="^", ms=6, lw=2,
        color=C_FP64, label="fp64, $516^3$/GPU")

eff = lambda v, gs: v[-1] / (v[0] * gs[-1])
ax.text(0.97, 0.04,
        f"at {WEAK_G[-1]} GPUs\n"
        f"fp32   {WEAK['fp32'][-1] / 1e6:.2f} Tcell/s  ({eff(WEAK['fp32'], WEAK_G):.0%})\n"
        f"mixed  {WEAK['recon32'][-1] / 1e6:.2f} Tcell/s  ({eff(WEAK['recon32'], WEAK_G):.0%})\n"
        f"fp64 at {WEAK64_G[-1]} GPUs: {WEAK64[-1] / 1e6:.2f} Tcell/s "
        f"({eff(WEAK64, WEAK64_G):.0%})",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
        color="0.25", linespacing=1.5)

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
# The grid label lives in the legend only — annotating curve ends as well
# collides once two curves finish at the same GPU count.
for i, (nx, pts) in enumerate(STRONG.items()):
    gs = np.array([p[0] for p in pts], float)
    ys = np.array([p[1] for p in pts]) / 1e3
    ax.plot(gs, ys[0] * gs / gs[0], label="ideal" if i == 0 else None, **IDEAL)
    ax.plot(gs, ys, marker="o", ms=6, lw=2, color=STRONG_C[nx],
            label=f"${nx}^3$")
ax.set_xscale("log")
ax.set_yscale("log")
gticks = [1, 8, 64, 512, 4096]
ax.set_xticks(gticks)
ax.set_xticklabels([str(v) for v in gticks])
ax.set_xlabel("GH200 GPUs")
ax.set_ylabel("aggregate Gcell / s")
ax.text(0.97, 0.04, "fp32; fixed global grid,\nper-GPU tile shrinks to the right",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
        color="0.25", linespacing=1.5)
ax.legend(loc="upper left", frameon=False, fontsize=11, title="global grid",
          ncol=2)
save(fig, f"{OUT}/jupiter_hydro_strong.pdf", also_png=True, close=True)

print("wrote code_paper_figs/jupiter_{hydro_roofline,gravity_roofline,"
      "hydro_weak,hydro_strong}.{pdf,png}")
