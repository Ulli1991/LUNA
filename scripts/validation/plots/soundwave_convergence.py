"""Sound-wave L1/L2 convergence plot — demonstrates 2nd order in smooth flow.

Reference: the analytic IC. The stationary sound-wave problem
(`soundwave.py`: ρ=ρ₀(1+ε sin(kx)), p=p₀(1+ε sin(kx)), u=0) decomposes into
a left- and right-going wave of equal amplitude; both return exactly to
the IC at one acoustic period t = Lx * sqrt(ρ₀/(γ·p₀)). So at t_end equal
to one period, the analytic solution equals the IC.

We sweep Nx ∈ {128, 256, 512, 1024, 2048, 4096}, take the final-snapshot
density and pressure, and compute L1/L2 errors against the IC analytically
recomputed on each LUNA grid. AthenaK PPM 8192 will be added as a second
reference once a sample .athdf file is in place.

Saves figures/soundwave_convergence.pdf.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils.figstyle import apply_style, decorate, FIGSIZE

apply_style()

from scripts.validation.runner import sweep
from scripts.validation.metrics import l1_error, l2_error, convergence_slope
from scripts.validation.athenak_io import load_1d as load_athdf_1d
from file_io.read_snapshot import read_snapshot, latest_snapshot_dir


NX_SCAN = [32, 64, 128, 256, 512]
LX = 1.0
GAMMA = 5.0 / 3.0
RHO0 = 1.0
P0 = 0.6        # with γ=5/3 → cs² = 1, period = Lx (matches AthenaK setup)
AMP = 1e-4      # eigenmode amplitude — must match ic_generation/soundwave.py
M = 1
CS = np.sqrt(GAMMA * P0 / RHO0)
T_END = LX / CS  # one acoustic period = 1.0


def analytic_ic_density(x):
    """Right-going acoustic eigenmode density profile (returns to IC after one period)."""
    k = 2.0 * np.pi * M / LX
    return RHO0 + AMP * np.sin(k * x)


def main():
    base = dict(
        test_problem="Soundwave",
        SPATIAL_DIMENSION=1,
        BOUNDARY=0,  # periodic
        Lx=LX,
        gamma=GAMMA,
        t_end=T_END,
        time_bet_snaps=T_END,  # one snapshot at t_end
        CFL=0.4,
        ORDER=1,
        LIMITER=1,  # MonCen: doesn't clip slopes at smooth extrema (sine peaks)
        PLOT_AFTER_SIM=False,
        production_mode=True,
        enable_x64=True,       # float64: perturbation is small, need precision
        clear_output_dir=True,
    )
    results = sweep(base, "Nx", NX_SCAN,
                    output_dir_template="snapshots/soundwave_Nx{}")

    l1s, l2s = [], []
    for Nx, outdir in results:
        snap = read_snapshot(latest_snapshot_dir(outdir))
        rho_luna = snap["rho"]
        rho_ref = analytic_ic_density(snap["x"])
        l1s.append(l1_error(rho_luna, rho_ref))
        l2s.append(l2_error(rho_luna, rho_ref))
        print(f"LUNA   Nx={Nx:5d}  L1={l1s[-1]:.3e}  L2={l2s[-1]:.3e}")

    slope_l1 = convergence_slope(NX_SCAN, l1s)
    slope_l2 = convergence_slope(NX_SCAN, l2s)
    print(f"LUNA convergence slope: L1 = {slope_l1:.3f}, L2 = {slope_l2:.3f}")

    # AthenaK overlay: parse the per-run errs.dat files which AthenaK writes
    # in double precision (computed before float32 binary storage). This
    # avoids the ~3e-8 plateau we'd hit if we computed L1 from the float32
    # .bin output. errs.dat row 2 has columns:
    #   Nx1 Nx2 Nx3 Ncycle RMS-L1 L-infty d_L1 M1_L1 M2_L1 M3_L1 E_L1
    # We use d_L1 (density L1, column index 6) for the AthenaK convergence curve.
    ak_dir = Path("reference/athenak")
    ak_results = {}  # scheme -> (nx_list, l1_list)
    for scheme in ("plm", "ppmx"):
        nx_list, l1_list = [], []
        for Nx in NX_SCAN:
            errs_path = ak_dir / f"soundwave_Nx{Nx}_{scheme}-errs.dat"
            if not errs_path.is_file():
                continue
            try:
                data = np.loadtxt(errs_path, comments="#")
                if data.ndim == 1:
                    data = data[np.newaxis, :]
                # Last row is the latest run; take it.
                d_L1 = float(data[-1, 6])
                nx_list.append(Nx)
                l1_list.append(d_L1)
                print(f"AthenaK {scheme:5s} Nx={Nx:5d}  L1={d_L1:.3e}")
            except Exception as exc:
                print(f"WARN: could not parse {errs_path}: {exc}")
        if nx_list:
            ak_results[scheme] = (nx_list, l1_list)

    # Compute a per-segment slope for the converging part (Nx ≤ 256), to put
    # a meaningful number in the legend rather than a global fit that the
    # plateau dilutes.
    def converging_slope(nx, l1, nx_max=256):
        nx_arr = np.asarray(nx, dtype=float)
        l1_arr = np.asarray(l1, dtype=float)
        mask = nx_arr <= nx_max
        if mask.sum() < 2:
            return convergence_slope(nx_arr, l1_arr)
        return convergence_slope(nx_arr[mask], l1_arr[mask])

    luna_conv = converging_slope(NX_SCAN, l1s)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.loglog(NX_SCAN, l1s, "o-", color="C0", lw=2.0, ms=7,
              label="LUNA  PLM + MUSCL-Hancock + MonCen")

    scheme_styles = {
        "plm":  dict(color="C1", label="AthenaK  PLM + RK2"),
        "ppmx": dict(color="C3", label="AthenaK  PPMX + RK2"),
    }
    for scheme, (nx, ak_l1) in ak_results.items():
        s = scheme_styles[scheme]
        ax.loglog(nx, ak_l1, "s--", color=s["color"], lw=1.5, ms=6,
                  label=s["label"])

    # 2nd-order reference, anchored to LUNA's first data point.
    nx_ref = np.array([NX_SCAN[0], 512], dtype=float)
    ax.loglog(nx_ref, l1s[0] * (nx_ref / nx_ref[0]) ** -2,
              color="0.4", linestyle=":", lw=1.5, label=r"slope $-2$ (2nd order)")

    # Annotate the nonlinear-physics plateau (no slope label, just a horizontal line)
    plateau = float(min(l1s))
    ax.axhline(plateau, color="0.75", linestyle="-", lw=0.8, alpha=0.6, zorder=0)

    ax.set_xlabel("resolution Nx", fontsize=11)
    ax.set_ylabel(r"L1 error in density (vs linear analytic)", fontsize=11)
    # Zoom y-axis to the data range
    ymin = 0.3 * plateau
    ymax = 3.0 * max(max(l1s), max(max(v[1]) for v in ak_results.values()) if ak_results else 0)
    ax.set_ylim(ymin, ymax)
    ax.legend(loc="upper right")
    decorate(ax)
    fig.tight_layout()

    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/soundwave_convergence.pdf")
    fig.savefig("figures/soundwave_convergence.png", dpi=200)
    print("Saved figures/soundwave_convergence.pdf and .png")


if __name__ == "__main__":
    main()
