"""2x2 physical-space comparison panels for the three 1D shock tests.

Runs LUNA at Nx ∈ {256, 512, 1024, 2048} for each of:
    Sod-Shock, Shu-Osher, Woodward-Colella

Produces one 2x2 figure per problem: density, vx, pressure, internal energy
vs x. LUNA at each resolution is plotted as a colored curve. If an AthenaK
PPM 8192 reference .athdf exists at `reference/athenak/<slug>_ppm_8192.athdf`,
it is overlaid as a thick black curve; otherwise the LUNA-only plot is
still useful as a visual convergence check.

Saves figures/shock_panels_{problem}.pdf per problem.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from scripts.validation.runner import sweep
from scripts.validation.athenak_io import load_1d
from file_io.read_snapshot import read_snapshot, latest_snapshot_dir


# Per-problem: t_end, AthenaK reference slug.
PROBLEMS = {
    "Sod-Shock": dict(t_end=0.2,   slug="sod"),
    "Shu-Osher": dict(t_end=1.8,   slug="shu_osher"),
    "Woodward-Colella": dict(t_end=0.038, slug="woodward_colella"),
}

NX_SCAN = [256, 512, 1024, 2048]
AK_REF_DIR = Path("reference/athenak")


def _internal_energy(rho, P, gamma):
    return P / ((gamma - 1.0) * rho)


def plot_one(problem_name, t_end, slug):
    base = dict(
        test_problem=problem_name,
        SPATIAL_DIMENSION=1,
        Lx=1.0,
        BOUNDARY=1,  # zero-gradient for shock tubes
        t_end=t_end,
        time_bet_snaps=t_end,
        TIME_SOLVER=1,    # MUSCL-Hancock (the only validated integrator)
        ORDER=1,
        LIMITER=0,
        CFL=0.4,
        PLOT_AFTER_SIM=False,
        production_mode=True,
    )
    out_template = f"snapshots/{slug}_Nx{{}}"
    results = sweep(base, "Nx", NX_SCAN, output_dir_template=out_template)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    panels = [
        (axes[0, 0], "rho",   "density"),
        (axes[0, 1], "vx",    "v_x"),
        (axes[1, 0], "P",     "pressure"),
        (axes[1, 1], "e_int", "internal energy"),
    ]

    # AthenaK overlay if available
    ak_path = AK_REF_DIR / f"{slug}_ppm_8192.athdf"
    if ak_path.is_file():
        try:
            ref = load_1d(ak_path)
            gamma = ref.get("gamma", 5.0 / 3.0)
            ref_e = _internal_energy(ref["rho"], ref["P"], gamma)
            for ax, key, _ in panels:
                y = ref_e if key == "e_int" else ref[key]
                ax.plot(ref["x"], y, "k-", lw=2.0, label="AthenaK PPM 8192")
        except Exception as exc:
            print(f"WARN: could not load AthenaK reference {ak_path}: {exc}")
    else:
        print(f"Note: no AthenaK reference at {ak_path} — LUNA-only plot.")

    # LUNA overlays
    cmap = plt.get_cmap("viridis")
    for i, (Nx, outdir) in enumerate(results):
        snap = read_snapshot(latest_snapshot_dir(outdir))
        gamma = snap["gamma"]
        e_int = snap.get("e_int")
        if e_int is None:
            e_int = _internal_energy(snap["rho"], snap["P"], gamma)
        fields = {"rho": snap["rho"], "vx": snap["vx"], "P": snap["P"], "e_int": e_int}
        color = cmap(0.15 + 0.7 * i / max(1, len(results) - 1))
        for ax, key, _ in panels:
            ax.plot(snap["x"], fields[key], color=color, lw=1.0,
                    label=f"LUNA Nx={Nx}")

    for ax, _, label in panels:
        ax.set_ylabel(label)
    axes[1, 0].set_xlabel("x")
    axes[1, 1].set_xlabel("x")
    axes[0, 0].legend(loc="best", fontsize=8)

    fig.suptitle(f"{problem_name} at t = {t_end}  —  MUSCL-Hancock")
    fig.tight_layout()

    Path("figures").mkdir(exist_ok=True)
    fname = f"figures/shock_panels_{slug}.pdf"
    fig.savefig(fname)
    print(f"Saved {fname}")
    plt.close(fig)


def main():
    for name, params in PROBLEMS.items():
        plot_one(name, params["t_end"], params["slug"])


if __name__ == "__main__":
    main()
