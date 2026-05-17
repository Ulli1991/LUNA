"""3D Sedov central-slice render.

Takes a 2D slice through the (z = Lz/2) plane of the final-time density
field and renders it as a pcolormesh — shows the spherical shock as a ring
with the rarefied interior. Companion to sedov_radial.py for the visual
showcase of the highest-resolution run.

Saves figures/sedov_slice.pdf.
"""
from pathlib import Path
import matplotlib.pyplot as plt

from scripts.validation.runner import run_one


NX = 1024
T_END = 0.05


def main():
    overrides = dict(
        test_problem="Sedov-Blast-3D",
        SPATIAL_DIMENSION=3,
        BOUNDARY=1,
        Nx=NX, Ny=NX, Nz=NX,
        t_end=T_END,
        TIME_SOLVER=1,
        ORDER=1,
        PLOT_AFTER_SIM=False,
        production_mode=True,
        output_dir=f"snapshots/sedov3d_Nx{NX}",
    )
    _, _, outdir = run_one(overrides)

    # TODO: load final density from outdir, take the z=Nz//2 slice.
    raise NotImplementedError(
        "Hook this up to the LUNA snapshot reader once the snapshot "
        "format / final-snapshot location is locked in."
    )

    # fig, ax = plt.subplots(figsize=(6, 5))
    # im = ax.pcolormesh(rho_slice, cmap="inferno", shading="auto")
    # ax.set_aspect("equal")
    # fig.colorbar(im, ax=ax, label="density")
    # ax.set_title(f"Sedov-3D, central z-slice, Nx={NX}, t={T_END}")
    # fig.tight_layout()

    Path("figures").mkdir(exist_ok=True)
    # fig.savefig("figures/sedov_slice.pdf")
    print("Saved figures/sedov_slice.pdf")


if __name__ == "__main__":
    main()
