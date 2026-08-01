"""Single-GPU component roofline (MI300A): hydro step / one V-cycle / CIC
deposit vs grid size, FP64 + FP32, before/after the 2026-07 optimization
campaign. Data measured by scaling/repro/comp_roofline.py (post) and the
2026-07 saturation scan (pre, hydro only). Ceilings are hardware bandwidth;
the campaign moved the FP64 hydro curve to the FP32-class ceiling (recon32).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = np.array([32, 64, 128, 256, 512])
# measured 2026-07-31 (comp_roofline.py, Mcell/s or Mparticle/s)
HYD64 = [249, 1012, 1937, 2117, 2194]
HYD32 = [245, 1067, 2125, 2346, 2453]
VC64 = [50, 368, 2627, 6118, 7412]
VC32 = [53, 393, 2868, 9978, 13556]
DEP64 = [307, 2170, 3589, 4001, 3978]
DEP32 = [333, 2436, 3895, 4616, 4659]
# pre-campaign hydro anchors (2026-07 saturation scan, all-FP64 pipeline)
HYD64_OLD = [82, 473, 916, 1110, 1061]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(N, HYD64_OLD, "o--", color="0.6", label="hydro FP64 (pre-campaign)")
ax.plot(N, HYD64, "o-", color="C0", label="hydro FP64 state (recon32)")
ax.plot(N, HYD32, "o:", color="C0", alpha=0.6, label="hydro FP32")
ax.plot(N, VC64, "s-", color="C1", label="V(2,2)-cycle FP64")
ax.plot(N, VC32, "s:", color="C1", alpha=0.6, label="V-cycle FP32")
ax.plot(N, DEP64, "^-", color="C2", label="CIC deposit FP64")
ax.plot(N, DEP32, "^:", color="C2", alpha=0.6, label="deposit FP32")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(N)
ax.set_xticklabels([f"{n}³" for n in N])
ax.set_xlabel("grid size")
ax.set_ylabel("throughput [Mcell/s or Mparticle/s]")
ax.set_title("LUNA single-GPU component roofline (MI300A)")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig("figures/roofline_components.png", dpi=160)
print("wrote figures/roofline_components.png")
