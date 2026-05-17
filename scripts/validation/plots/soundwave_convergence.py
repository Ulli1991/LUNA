"""Sound-wave L1/L2 convergence plot for LUNA vs AthenaK PPM 8192 reference.

Usage:
    python -m scripts.validation.plots.soundwave_convergence

Drives LUNA at Nx ∈ {128, 256, 512, 1024, 2048, 4096} for the stationary
sound-wave problem, one acoustic period (t_end = Lx * sqrt(rho0/(gamma*p0))).
For each resolution, computes L1 and L2 errors of density vs the
high-resolution AthenaK PPM 8192 reference (interpolated onto the LUNA
grid). Saves a log-log plot with reference slope -2.

Requires: AthenaK reference at AK_REF (see top of file).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from scripts.validation.runner import sweep
from scripts.validation.athenak_io import load_1d
from scripts.validation.metrics import l1_error, l2_error, interp_1d
from file_io.read_checkpoint import read_checkpoint  # TODO: read snapshot instead


# Path to the user-provided high-resolution AthenaK reference snapshot.
# Fill in once the .athdf file is available.
AK_REF = Path("reference/athenak/soundwave_ppm_8192.athdf")

NX_SCAN = [128, 256, 512, 1024, 2048, 4096]
LX = 1.0
GAMMA = 5.0 / 3.0
RHO0 = 1.0
P0 = 1.0
T_END = LX * np.sqrt(RHO0 / (GAMMA * P0))  # one acoustic period ≈ 0.7746


def main():
    # 1. Run LUNA scan
    base = dict(
        test_problem="Soundwave",
        SPATIAL_DIMENSION=1,
        BOUNDARY=0,  # periodic
        Lx=LX,
        gamma=GAMMA,
        t_end=T_END,
        TIME_SOLVER=1,  # MUSCL-Hancock
        CFL=0.4,
        ORDER=1,
        PLOT_AFTER_SIM=False,
        production_mode=True,
    )
    results = sweep(base, "Nx", NX_SCAN,
                    output_dir_template="snapshots/soundwave_Nx{}")

    # 2. Load reference
    ref = load_1d(AK_REF)

    # 3. Compute errors
    l1s, l2s = [], []
    for Nx, outdir in results:
        # TODO: load final LUNA snapshot. Replace with the actual snapshot
        # reader once we confirm the format. Tentatively: density at t=T_END
        # from the last snapshot file in outdir.
        raise NotImplementedError(
            "Hook this up to the LUNA snapshot reader once the snapshot "
            "format / final-snapshot location is locked in."
        )

    # 4. Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(NX_SCAN, l1s, "o-", label="L1")
    ax.loglog(NX_SCAN, l2s, "s-", label="L2")
    nx_ref = np.array(NX_SCAN, dtype=float)
    ax.loglog(nx_ref, l1s[0] * (nx_ref / nx_ref[0]) ** -2,
              "k--", label="slope -2")
    ax.set_xlabel("Nx")
    ax.set_ylabel("error norm (density)")
    ax.set_title("Soundwave convergence vs AthenaK PPM 8192")
    ax.legend()
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/soundwave_convergence.pdf")
    print("Saved figures/soundwave_convergence.pdf")


if __name__ == "__main__":
    main()
