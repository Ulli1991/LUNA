"""2x2 physical-space comparison panels for the three 1D shock tests.

For each of {Sod, Shu-Osher, Woodward-Colella}:
  - Run LUNA at Nx ∈ {256, 512, 1024, 2048}
  - Load the high-resolution AthenaK PPM 8192 reference
  - Produce a 2x2 panel figure: density, vx, pressure, internal energy
    as functions of x. AthenaK reference plotted as a thick black curve;
    LUNA at each resolution plotted as a colored curve.

Saves figures/shock_panels_{problem}.pdf (one file per problem).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from scripts.validation.runner import sweep
from scripts.validation.athenak_io import load_1d


# Per-problem config: LUNA test_problem name, t_end, AthenaK reference path.
PROBLEMS = {
    "Sod-Shock": dict(
        t_end=0.2,
        ak_ref=Path("reference/athenak/sod_ppm_8192.athdf"),
    ),
    "Shu-Osher": dict(
        t_end=1.8,
        ak_ref=Path("reference/athenak/shu_osher_ppm_8192.athdf"),
    ),
    "Woodward-Colella": dict(
        t_end=0.038,
        ak_ref=Path("reference/athenak/woodward_colella_ppm_8192.athdf"),
    ),
}

NX_SCAN = [256, 512, 1024, 2048]


def _internal_energy(rho, P, gamma):
    return P / ((gamma - 1.0) * rho)


def plot_one(problem_name, t_end, ak_ref_path):
    base = dict(
        test_problem=problem_name,
        SPATIAL_DIMENSION=1,
        t_end=t_end,
        TIME_SOLVER=1,
        ORDER=1,
        PLOT_AFTER_SIM=False,
        production_mode=True,
    )
    results = sweep(base, "Nx", NX_SCAN,
                    output_dir_template=f"snapshots/{problem_name.lower()}_Nx{{}}")

    ref = load_1d(ak_ref_path)
    gamma = ref.get("gamma", 5.0 / 3.0)
    e_int_ref = _internal_energy(ref["rho"], ref["P"], gamma)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    panels = [
        (axes[0, 0], "rho", "density"),
        (axes[0, 1], "vx",  "v_x"),
        (axes[1, 0], "P",   "pressure"),
        (axes[1, 1], "e",   "internal energy"),
    ]

    # AthenaK reference
    for ax, key, label in panels:
        y = e_int_ref if key == "e" else ref[key]
        ax.plot(ref["x"], y, "k-", lw=2.0, label="AthenaK PPM 8192")
        ax.set_ylabel(label)
    axes[1, 0].set_xlabel("x")
    axes[1, 1].set_xlabel("x")

    # LUNA overlays
    cmap = plt.get_cmap("viridis")
    for i, (Nx, outdir) in enumerate(results):
        color = cmap(0.15 + 0.7 * i / max(1, len(results) - 1))
        # TODO: load final LUNA snapshot and extract rho, vx, P.
        raise NotImplementedError(
            "Hook this up to the LUNA snapshot reader once the snapshot "
            "format / final-snapshot location is locked in."
        )

    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{problem_name} at t = {t_end}")
    fig.tight_layout()

    Path("figures").mkdir(exist_ok=True)
    fname = f"figures/shock_panels_{problem_name.lower().replace('-', '_')}.pdf"
    fig.savefig(fname)
    print(f"Saved {fname}")


def main():
    for name, cfg in PROBLEMS.items():
        plot_one(name, cfg["t_end"], cfg["ak_ref"])


if __name__ == "__main__":
    main()
