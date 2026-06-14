"""Standing-wave (u=0 superposition) convergence test for LUNA vs AthenaK.

Sister plot to soundwave_convergence.py (which uses a single right-going
eigenmode). The standing wave avoids the O(amp^2) nonlinear-steepening
floor visible in the eigenmode plot — symmetry of the standing wave makes
the nonlinear contributions cancel — so we get a textbook -2 slope all
the way down the scan.

IC for both codes:
    rho(x, 0) = rho0 + amp * sin(2π x / Lx)
    u  (x, 0) = 0
    p  (x, 0) = p0   + amp * cs² * sin(2π x / Lx)

Linear superposition of equal-amplitude left+right going acoustic waves;
returns exactly to IC at t = Lx / cs (one period) under linear theory.

LUNA side: same Soundwave problem with soundwave_single_mode=False.
AthenaK side: custom soundwave_uzero pgen built with -DPROBLEM=soundwave_uzero;
reference snapshots at reference/athenak/standing_Nx{Nx}_plm.bin.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils.figstyle import apply_style, decorate, FIGSIZE

apply_style()

from scripts.validation.runner import sweep
from scripts.validation.athenak_io import load_1d as load_athenak_bin
from scripts.validation.metrics import l1_error, l2_error, convergence_slope
from file_io.read_snapshot import read_snapshot, latest_snapshot_dir


NX_SCAN = [32, 64, 128, 256, 512]
LX = 1.0
GAMMA = 5.0 / 3.0
RHO0 = 1.0
P0 = 0.6
AMP = 1e-4
M = 1
CS = np.sqrt(GAMMA * P0 / RHO0)
T_END = LX / CS


def analytic_ic_density(x):
    k = 2.0 * np.pi * M / LX
    return RHO0 + AMP * np.sin(k * x)


def main():
    base = dict(
        test_problem="Soundwave",
        SPATIAL_DIMENSION=1,
        BOUNDARY=0,
        Lx=LX,
        gamma=GAMMA,
        t_end=T_END,
        time_bet_snaps=T_END,
        CFL=0.4,
        ORDER=1,
        LIMITER=1,
        PLOT_AFTER_SIM=False,
        production_mode=True,
        enable_x64=True,
        soundwave_single_mode=False,   # ← u=0 superposition (standing wave)
        clear_output_dir=True,
    )

    results = sweep(base, "Nx", NX_SCAN,
                    output_dir_template="snapshots/standing_Nx{}")

    l1s, l2s = [], []
    for Nx, outdir in results:
        snap = read_snapshot(latest_snapshot_dir(outdir))
        rho_ref = analytic_ic_density(snap["x"])
        l1s.append(l1_error(snap["rho"], rho_ref))
        l2s.append(l2_error(snap["rho"], rho_ref))
        print(f"LUNA   Nx={Nx:5d}  L1={l1s[-1]:.3e}  L2={l2s[-1]:.3e}")

    slope_l1 = convergence_slope(NX_SCAN, l1s)
    slope_l2 = convergence_slope(NX_SCAN, l2s)
    print(f"LUNA slope: L1={slope_l1:.3f}  L2={slope_l2:.3f}")

    # AthenaK standing-wave reference: use errs.dat (DOUBLE precision L1
    # computed inside AthenaK before the float32 .bin cast). .bin values are
    # not usable here because they bottom out at the float32 storage floor.
    ak_dir = Path("reference/athenak")
    ak_nx, ak_l1 = [], []
    for Nx in NX_SCAN:
        ep = ak_dir / f"standing_Nx{Nx}_plm-errs.dat"
        if not ep.is_file():
            continue
        data = np.loadtxt(ep, comments="#")
        if data.ndim == 1:
            data = data[np.newaxis, :]
        d_L1 = float(data[-1, 6])
        ak_nx.append(Nx)
        ak_l1.append(d_L1)
        print(f"AthenaK Nx={Nx:5d}  L1={d_L1:.3e}")
    if ak_nx:
        ak_slope = convergence_slope(ak_nx, ak_l1)
        print(f"AthenaK slope: L1={ak_slope:.3f}")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.loglog(NX_SCAN, l1s, "o-", color="C0", lw=2.0, ms=7,
              label="LUNA  PLM+MUSCL-Hancock")
    if ak_nx:
        ax.loglog(ak_nx, ak_l1, "s--", color="C1", lw=1.5, ms=6,
                  label="AthenaK  PLM+RK2")

    # 2nd-order reference, anchored to LUNA's first point
    nx_ref = np.array(NX_SCAN, dtype=float)
    ax.loglog(nx_ref, l1s[0] * (nx_ref / nx_ref[0]) ** -2,
              color="0.4", linestyle=":", lw=1.5, label=r"slope $-2$ (2nd order)")

    ax.set_xlabel("resolution Nx", fontsize=11)
    ax.set_ylabel(r"L1 error in density (vs analytic)", fontsize=11)
    ax.legend(loc="upper right")
    decorate(ax)
    fig.tight_layout()

    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/soundwave_standing_convergence.pdf")
    fig.savefig("figures/soundwave_standing_convergence.png", dpi=200)
    print("Saved figures/soundwave_standing_convergence.{pdf,png}")


if __name__ == "__main__":
    main()
