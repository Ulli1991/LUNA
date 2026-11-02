"""JUPITER GH200 Sedov-3D scaling figure from scaling/jupiter/jupiter_numbers.csv (LUNA only)."""
import csv
from collections import defaultdict

from scripts.validation.plots._plotkit import mpl, save

DATA = "scaling/jupiter/jupiter_numbers.csv"
OUT = "code_paper_figs/scaling_jupiter.png"
# fp32 at 2048^3 on 4096 GPUs is a dying job's last window (RESULTS.md, note 1): not plotted.
NOT_CLEAN = {("fp32", 2048, 4096)}
STYLE = {"fp32": ("#1f77b4", "o", "FP32"), "recon32": ("#2ca02c", "s", "FP32 reconstruction, FP64 state"),
         "fp64": ("#d62728", "^", "FP64")}


def rows(table):
    with open(DATA) as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    return [r for r in csv.DictReader(lines) if r["table"] == table]


def main():
    plt = mpl()
    weak = defaultdict(list)
    for r in rows("hydro_weak"):
        weak[r["variant"]].append((int(r["gpus"]), float(r["value"]) / 1e6, float(r["eff"])))
    strong = defaultdict(list)
    for r in rows("hydro_strong"):
        key = (r["variant"], int(r["grid"]), int(r["gpus"]))
        if key not in NOT_CLEAN:
            strong[(r["variant"], int(r["grid"]))].append((int(r["gpus"]), float(r["value"]) / 1e6))

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for v, pts in weak.items():
        c, m, lbl = STYLE[v]
        g, t, e = zip(*sorted(pts))
        tile = "516" if v == "fp64" else "512"
        ax[0].plot(g, t, m + "-", color=c, label=f"{lbl}, {tile}$^3$ per GPU")
        ax[1].plot(g, [100 * x for x in e], m + "-", color=c, label=lbl)
    t0 = weak["fp32"][0][1]
    ax[0].plot([1, 4096], [t0, t0 * 4096], ":", color="gray", label="ideal (FP32)")
    ax[0].set(xscale="log", yscale="log", xlabel="GPUs (GH200)", ylabel="aggregate Tcell-updates / s")
    ax[1].set(xscale="log", ylim=(80, 102), xlabel="GPUs (GH200)", ylabel="weak-scaling efficiency [%]")
    grids = sorted({k[1] for k in strong})
    cmap = plt.get_cmap("viridis")
    for i, n in enumerate(grids):
        col = cmap(i / max(len(grids) - 1, 1))
        for v, ls in (("fp32", "-"), ("fp64", "--")):
            pts = sorted(strong.get((v, n), []))
            if pts:
                ax[2].plot(*zip(*pts), ls, marker="o" if v == "fp32" else "^", color=col,
                           label=f"{n}$^3$ {STYLE[v][2]}")
    ax[2].set(xscale="log", yscale="log", xlabel="GPUs (GH200)", ylabel="aggregate Tcell-updates / s")
    for a, tag in zip(ax, "abc"):
        a.text(0.5 if tag == "b" else 0.02, 0.95, f"({tag})", transform=a.transAxes, va="top")
        a.legend(fontsize=9, loc="lower right" if tag != "b" else "lower left")
        a.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    save(fig, OUT, dpi=160, close=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
