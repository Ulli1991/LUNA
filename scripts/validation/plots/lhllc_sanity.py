"""Quick L-HLLC sanity check: re-run Sod with the L-HLLC solver and overlay
the existing HLLC baseline.

For a transsonic shock test like Sod the local Mach number is order-1 across
most of the domain, so phi ~ 0.9-1.0; L-HLLC should differ from HLLC by only
a few percent. The point is to confirm the new solver doesn't NaN, gives a
sensible shock/contact/rarefaction structure, and degenerates correctly to
standard HLLC at sonic speeds.

Saves figures/lhllc_sanity.{pdf,png}.
"""
from pathlib import Path
import matplotlib.pyplot as plt

from scripts.validation.runner import run_one
from file_io.read_snapshot import read_snapshot, latest_snapshot_dir


NX = 512
T_END = 0.2
COMMON = dict(
    test_problem="Sod-Shock",
    SPATIAL_DIMENSION=1,
    BOUNDARY=1,
    Nx=NX,
    gamma=5.0 / 3.0,
    t_end=T_END,
    TIME_SOLVER=1,
    ORDER=1,
    LIMITER=0,
    CFL=0.4,
    PLOT_AFTER_SIM=False,
    production_mode=True,
)


def main():
    overrides_hllc = dict(COMMON, riemann_solver="hllc",
                          output_dir=f"snapshots/sod_hllc_Nx{NX}")
    overrides_lhllc = dict(COMMON, riemann_solver="lhllc",
                           output_dir=f"snapshots/sod_lhllc_Nx{NX}")

    run_one(overrides_hllc, jax_already_initialized=False)
    run_one(overrides_lhllc, jax_already_initialized=True)

    sn_h = read_snapshot(latest_snapshot_dir(overrides_hllc["output_dir"]))
    sn_l = read_snapshot(latest_snapshot_dir(overrides_lhllc["output_dir"]))
    x = sn_h["x"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key, label in zip(axes,
                              ["rho", "vx", "P"],
                              [r"$\rho$", r"$v_x$", r"$P$"]):
        ax.plot(x, sn_h[key], "-", lw=1.2, color="C0", label="HLLC")
        ax.plot(x, sn_l[key], "--", lw=1.2, color="C3", label="L-HLLC")
        ax.set_xlabel("x"); ax.set_ylabel(label)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, which="major", alpha=0.3)
        ax.tick_params(top=True, right=True, direction="in")
    fig.suptitle(f"Sod shock-tube at Nx={NX}, t={T_END}: HLLC vs L-HLLC", fontsize=11)
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/lhllc_sanity.pdf")
    fig.savefig("figures/lhllc_sanity.png", dpi=200)
    print("Saved figures/lhllc_sanity.{pdf,png}")


if __name__ == "__main__":
    main()
