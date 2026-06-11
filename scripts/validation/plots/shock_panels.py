"""4-panel comparison plots for the three 1D shock tests.

LUNA (PLM + MUSCL-Hancock + minmod, HLLC) at Nx ∈ {256, 512, 1024, 2048}
vs AthenaK PPMX (RK2) at Nx=8192 as the high-resolution reference.

For each of Sod / Shu-Osher / Woodward-Colella we plot density, velocity,
pressure, and specific internal energy as functions of x.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from scripts.validation.runner import sweep
from scripts.validation.athenak_io import load_1d as load_ak
from file_io.read_snapshot import read_snapshot, latest_snapshot_dir


# Per-problem: t_end, slug, physical domain, BC (1 = zero-gradient/outflow,
# 2 = reflective — Woodward-Colella requires reflective walls).
PROBLEMS = {
    "Sod-Shock": dict(
        slug="sod",
        t_end=0.2,
        x_min=0.0, x_max=1.0, lx_runner=1.0,
        boundary=1,
        gamma=5.0 / 3.0,             # Sod traditionally γ=5/3
    ),
    "Shu-Osher": dict(
        slug="shu_osher",
        t_end=1.8,
        x_min=-5.0, x_max=5.0, lx_runner=10.0,
        boundary=1,
        gamma=1.4,                   # textbook Shu-Osher uses γ=1.4
    ),
    "Woodward-Colella": dict(
        slug="woodward_colella",
        t_end=0.038,
        x_min=0.0, x_max=1.0, lx_runner=1.0,
        boundary=2,                  # reflecting walls — the gas is contained
        gamma=1.4,                   # textbook WC uses γ=1.4 (peak ρ ≈ 6)
    ),
}

NX_SCAN = [256, 512, 1024, 2048]
AK_REF_DIR = Path("reference/athenak")
# Per-problem γ is set in PROBLEMS below — Sod uses γ=5/3, Shu-Osher and
# Woodward-Colella use the standard γ=1.4.


def _internal_energy(rho, P, gamma):
    return P / ((gamma - 1.0) * rho)


def plot_one(problem_name, params):
    slug    = params["slug"]
    t_end   = params["t_end"]
    x_min   = params["x_min"]
    lx      = params["lx_runner"]
    bnd     = params["boundary"]   # 1 = zero-gradient, 2 = reflecting
    gam     = params["gamma"]

    base = dict(
        test_problem=problem_name,
        SPATIAL_DIMENSION=1,
        Lx=lx,
        BOUNDARY=bnd,
        gamma=gam,
        t_end=t_end,
        time_bet_snaps=t_end,
        ORDER=1,
        LIMITER=0,                 # minmod, per user spec
        CFL=0.4,
        PLOT_AFTER_SIM=False,
        production_mode=True,
        enable_x64=True,
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

    # AthenaK PPMX 8192 reference (load .bin via bin_convert; physical x in
    # the file metadata)
    ak_path = AK_REF_DIR / f"{slug}_ppmx_Nx8192.bin"
    if ak_path.is_file():
        try:
            ref = load_ak(ak_path, gamma=gam)
            ref_e = _internal_energy(ref["rho"], ref["P"], gam)
            ref_fields = {"rho": ref["rho"], "vx": ref["vx"],
                          "P": ref["P"], "e_int": ref_e}
            for ax, key, _ in panels:
                ax.plot(ref["x"], ref_fields[key], "k-", lw=2.0,
                        label="AthenaK PPMX (Nx=8192)")
        except Exception as exc:
            print(f"WARN: could not load AthenaK reference {ak_path}: {exc}")
    else:
        print(f"Note: no AthenaK reference at {ak_path} — LUNA-only plot.")

    # LUNA curves. NOTE: LUNA's snap["e_int"] is internal energy DENSITY
    # (P/(γ-1)), not specific internal energy — recompute specific to match
    # the AthenaK panel above.
    cmap = plt.get_cmap("viridis")
    for i, (Nx, outdir) in enumerate(results):
        snap = read_snapshot(latest_snapshot_dir(outdir))
        x_phys = snap["x"] + x_min  # shift to physical domain
        gamma = snap["gamma"]
        e_int = _internal_energy(snap["rho"], snap["P"], gamma)
        fields = {"rho": snap["rho"], "vx": snap["vx"],
                  "P": snap["P"], "e_int": e_int}
        color = cmap(0.15 + 0.7 * i / max(1, len(results) - 1))
        for ax, key, _ in panels:
            ax.plot(x_phys, fields[key], color=color, lw=1.0,
                    label=f"LUNA Nx={Nx}")

    for ax, _, label in panels:
        ax.set_ylabel(label)
        ax.grid(True, which="major", alpha=0.3)
        ax.tick_params(which="both", top=True, right=True, direction="in")
        ax.minorticks_on()
    axes[1, 0].set_xlabel("x")
    axes[1, 1].set_xlabel("x")
    axes[0, 0].legend(loc="best", fontsize=8, frameon=False)

    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fname_pdf = f"figures/shock_panels_{slug}.pdf"
    fname_png = f"figures/shock_panels_{slug}.png"
    fig.savefig(fname_pdf)
    fig.savefig(fname_png, dpi=200)
    print(f"Saved {fname_pdf} and {fname_png}")
    plt.close(fig)


def main():
    for name, params in PROBLEMS.items():
        plot_one(name, params)


if __name__ == "__main__":
    main()
