"""R(t) of the H II region against the STARBENCH reference solutions.

    python -m scripts.validation.plots.starbench_hii_plot snapshots/starbench_early_nx96

Reads the snapshots of a `starbench_hii_run` run, recovers the ionization
fraction from the gas -- the temperature prescription makes it exact,
x = (P/rho - cs_o^2) / (cs_i^2 - cs_o^2) -- and measures the front as the
volume-equivalent radius of the ionized region. Writes a figure and the
numbers beside it.

The comparison is the pair of classical solutions, Spitzer and
Hosokawa-Inutsuka (Bisbas et al. 2015, eqs. 9 and 13), which bracket what
the twelve codes in the benchmark found: the codes track Hosokawa-Inutsuka
early and fall towards Spitzer late, and STARBENCH's own fit lies between
them.
"""
import json
import os
import sys

import numpy as np


from luna.file_io.read_snapshot import read_snapshot          # noqa: E402
from luna.radiation.ionization import (hosokawa_inutsuka_radius,  # noqa: E402
                                       spitzer_radius, starbench_units)


def series(outdir, phase="early"):
    """[(t, R_front)] over every snapshot in the directory."""
    u = starbench_units(phase)
    cs2_o, cs2_i = u["cs2_neutral"], u["cs2_ionized"]
    rows = []
    for name in sorted(os.listdir(outdir)):
        if not name.startswith("snapdir_"):
            continue
        snap = read_snapshot(os.path.join(outdir, name))
        rho, P = np.asarray(snap["rho"]), np.asarray(snap["P"])
        x = (P / rho - cs2_o) / (cs2_i - cs2_o)
        cell = snap["dx"] * snap["dy"] * snap["dz"]
        volume = float(np.sum(x > 0.5)) * cell
        rows.append((float(snap["t"]),
                     (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0),
                     float(np.sum(x) * cell)))
    return u, sorted(rows)


def main(argv):
    outdir = argv[0] if argv else "snapshots/starbench_early_nx96"
    phase = argv[1] if len(argv) > 1 else "early"
    u, rows = series(outdir, phase)
    c_i, R_St = u["cs2_ionized"] ** 0.5, u["R_St"]
    t = np.array([r[0] for r in rows])
    R = np.array([r[1] for r in rows])
    sp = spitzer_radius(t, R_St, c_i)
    hi = hosokawa_inutsuka_radius(t, R_St, c_i)

    print(f"{'t [Myr]':>10} {'R [pc]':>9} {'Spitzer':>9} {'HI':>9} "
          f"{'R/Sp':>7} {'R/HI':>7}")
    for i in range(len(t)):
        print(f"{t[i]:10.5f} {R[i]:9.4f} {sp[i]:9.4f} {hi[i]:9.4f} "
              f"{R[i] / sp[i]:7.3f} {R[i] / hi[i]:7.3f}")
    with open(os.path.join(outdir, "starbench_radius.json"), "w") as f:
        json.dump({"R_St": R_St, "c_i": c_i, "phase": phase,
                   "t": t.tolist(), "R": R.tolist(),
                   "spitzer": sp.tolist(), "hosokawa_inutsuka": hi.tolist()},
                  f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.9))
    a.plot(t, hi, "k--", lw=1.2, label="Hosokawa-Inutsuka")
    a.plot(t, sp, "k:", lw=1.2, label="Spitzer")
    a.plot(t, R, "o-", ms=3.5, color="tab:red", label="LUNA")
    a.axhline(R_St, color="0.8", lw=0.8)
    a.set_xlabel("t [Myr]")
    a.set_ylabel("ionization front radius [pc]")
    a.legend(fontsize=8)
    m = t > 0.0                    # t = 0 has no front yet, and no ratio
    b.plot(t[m], (R / sp)[m], "o-", ms=3.5, color="tab:blue",
           label="R / Spitzer")
    b.plot(t[m], (R / hi)[m], "s-", ms=3.5, color="tab:green",
           label="R / Hosokawa-Inutsuka")
    b.axhline(1.0, color="0.7", lw=0.8)
    b.set_xlabel("t [Myr]")
    b.set_ylabel("ratio")
    b.legend(fontsize=8)
    fig.suptitle("STARBENCH D-type expansion, " + phase + " phase", fontsize=10)
    fig.tight_layout()
    path = os.path.join(outdir, "starbench_radius.png")
    fig.savefig(path, dpi=140)
    print("figure ->", path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
