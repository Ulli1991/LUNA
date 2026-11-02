"""Tables and figures of a run_turb_chem.py run: per-stage cost and substeps over time, phase diagrams.
    python tools/chem_coupled/summarize.py DIR [--bin-myr 1.0]
"""
import argparse
import glob
import json
import os

import numpy as np

STAGES = ("hydro", "gravity", "forcing", "chemistry")


def load(run):
    """Every step record except compiling first ones; a restart's repeat of a step replaces it."""
    recs = {}
    for line in open(os.path.join(run, "steps.jsonl")):
        r = json.loads(line)
        if not r["first"]:
            recs[r["step"]] = r
    return [recs[k] for k in sorted(recs)]


def table(recs, bin_myr, ncell):
    """Median per-stage ms, chemistry share and substep stats per bin of simulated time."""
    t = np.array([r["t_myr"] for r in recs])
    edges = np.arange(0.0, t.max() + bin_myr, bin_myr)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r, tt in zip(recs, t) if lo <= tt < hi]
        if not sel:
            continue
        med = {s: 1e3 * float(np.median([r.get(f"t_{s}", 0.0) for r in sel])) for s in STAGES}
        tot = sum(med.values()) + 1e3 * float(np.median([r["t_dt"] for r in sel]))
        rows.append(dict(
            t=f"{lo:.1f}-{hi:.1f}", steps=len(sel), dt_yr=float(np.median([r["dt_yr"] for r in sel])),
            **{f"{s}_ms": med[s] for s in STAGES}, chem_ns_cell=med["chemistry"] * 1e6 / ncell,
            chem_pct=100.0 * med["chemistry"] / tot,
            mean=float(np.mean([r["mean"] for r in sel])), p50=max(r["p50"] for r in sel),
            p90=max(r["p90"] for r in sel), p99=max(r["p99"] for r in sel),
            max=max(r["max"] for r in sel),
            multi_frac=float(np.mean([r["multi_attempt"] for r in sel])) / ncell,
            failed=sum(r["failed"] for r in sel), floored=sum(r.get("floored", 0) for r in sel)))
    return rows


def print_table(rows):
    """Markdown table of the rows."""
    keys = list(rows[0])
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for r in rows:
        print("| " + " | ".join(f"{v:.3g}" if isinstance(v, float) else str(v)
                                for v in r.values()) + " |")


def phase_figure(run, out, n_panels=4):
    """Mass-weighted (n_H, T) histograms at evenly spread saved steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted(glob.glob(os.path.join(run, "phase_*.npz")))
    pick = [files[int(i)] for i in np.linspace(0, len(files) - 1, min(n_panels, len(files)))]
    fig, axes = plt.subplots(1, len(pick), figsize=(3.6 * len(pick), 3.4), sharey=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, pick):
        d = np.load(f)
        m = d["mass"] / d["mass"].sum()
        im = ax.imshow(np.log10(np.where(m > 0, m, np.nan)).T, origin="lower",
                       extent=(-3, 5, 0, 8), aspect="auto", cmap="Blues", vmin=-7, vmax=-1)
        ax.set_title(f"t = {float(d['t_myr']):.1f} Myr", fontsize=10)
        ax.set_xlabel(r"log$_{10}$ n$_H$ [cm$^{-3}$]")
        ax.set_xlim(-2, 4)
        ax.set_ylim(0.5, 7)
        ax.grid(color="0.85", lw=0.5)
    axes[0].set_ylabel("log$_{10}$ T [K]")
    fig.colorbar(im, ax=axes, label="log$_{10}$ mass fraction per bin", shrink=0.9)
    fig.savefig(out, dpi=150)
    return [os.path.basename(f) for f in pick]


def cost_figure(recs, out, ncell):
    """Per-step stage times and chemistry substeps against simulated time (two panels, one axis each)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([r["t_myr"] for r in recs])
    fig, (a, b) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, constrained_layout=True)
    for s, col in zip(STAGES, ("#3b6fb6", "#d17a22", "#7a7a7a", "#2a9d5c")):
        a.plot(t, [1e3 * r.get(f"t_{s}", 0.0) for r in recs], lw=1.5, color=col, label=s)
    a.set_yscale("log")
    a.set_ylabel("ms per step")
    a.legend(frameon=False, ncol=4, fontsize=9)
    a.grid(color="0.9", lw=0.5)
    for k, col in (("mean", "#2a9d5c"), ("p99", "#3b6fb6"), ("max", "#7a7a7a")):
        b.plot(t, [r[k] for r in recs], lw=1.5, color=col, label=k)
    b.set_yscale("log")
    b.set_ylabel("accepted substeps per cell")
    b.set_xlabel("t [Myr]")
    b.legend(frameon=False, ncol=3, fontsize=9)
    b.grid(color="0.9", lw=0.5)
    fig.savefig(out, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--bin-myr", type=float, default=1.0)
    a = ap.parse_args()
    setup = json.load(open(os.path.join(a.run, "setup.json")))
    ncell = setup["n"] ** 3
    recs = load(a.run)
    print_table(table(recs, a.bin_myr, ncell))
    print("phase panels:", phase_figure(a.run, os.path.join(a.run, "phase_diagram.png")))
    cost_figure(recs, os.path.join(a.run, "cost_vs_time.png"), ncell)


if __name__ == "__main__":
    main()
