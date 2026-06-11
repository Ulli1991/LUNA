"""3D Sedov radial profile plot vs analytic Sedov-Taylor solution.

Runs Sedov-Blast-3D at the largest resolution that fits, extracts spherically-
averaged radial profiles of density, radial velocity, and pressure, and
overlays the analytic Sedov-Taylor self-similar curves.

Saves figures/sedov_radial.pdf.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from scripts.validation.runner import run_one
from scripts.validation.metrics import radial_profile
from scripts.validation.sedov_analytic import sedov_taylor_3d


# Push to the largest grid that fits on the available GPUs.
NX = 1024
T_END = 0.05
E0 = 1.0
RHO0 = 1.0
GAMMA = 5.0 / 3.0


def main():
    overrides = dict(
        test_problem="Sedov-Blast-3D",
        SPATIAL_DIMENSION=3,
        BOUNDARY=1,  # zero-gradient (Sedov is not periodic)
        Nx=NX, Ny=NX, Nz=NX,
        t_end=T_END,
        gamma=GAMMA,
        ORDER=1,
        PLOT_AFTER_SIM=False,
        production_mode=True,
        output_dir=f"snapshots/sedov3d_Nx{NX}",
    )
    _, _, outdir = run_one(overrides)

    # TODO: load final-time conservative state U_final from outdir.
    # Convert to primitives: rho, vx, vy, vz, P. Compute v_r from velocities
    # relative to the geometric center.
    raise NotImplementedError(
        "Hook this up to the LUNA snapshot reader once the snapshot "
        "format / final-snapshot location is locked in."
    )

    # Spherically-averaged radial profiles
    # dx = Lx / NX (defaults to 1/NX)
    # r_centers, rho_r = radial_profile(rho, dx, dx, dx)
    # _,         vr_r  = radial_profile(v_r, dx, dx, dx)
    # _,         p_r   = radial_profile(P,   dx, dx, dx)

    # Analytic curves on the same radial grid
    # ref = sedov_taylor_3d(r_centers, t=T_END, E0=E0, rho0=RHO0, gamma=GAMMA)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    # axes[0].plot(r_centers, rho_r, "C0o", ms=2, label=f"LUNA Nx={NX}")
    # axes[0].plot(r_centers, ref["rho"], "k-", label="Sedov-Taylor")
    # axes[0].set_xlabel("r"); axes[0].set_ylabel("density"); axes[0].legend()
    # axes[1].plot(r_centers, vr_r,  "C0o", ms=2)
    # axes[1].plot(r_centers, ref["v_r"], "k-")
    # axes[1].set_xlabel("r"); axes[1].set_ylabel("v_r")
    # axes[2].plot(r_centers, p_r,   "C0o", ms=2)
    # axes[2].plot(r_centers, ref["P"], "k-")
    # axes[2].set_xlabel("r"); axes[2].set_ylabel("pressure")

    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/sedov_radial.pdf")
    print("Saved figures/sedov_radial.pdf")


if __name__ == "__main__":
    main()
