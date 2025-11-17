import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def plot_density_from_hdf5_all(output_dir, base_name):
    """
    Load all HDF5 snapshots and produce density plots.
    """
    # Find all snapshot files
    files = sorted(
        [f for f in os.listdir(output_dir) if f.startswith(base_name) and f.endswith(".hdf5")]
    )

    if not files:
        print("No HDF5 snapshots found!")
        return

    for fname in files:
        fpath = os.path.join(output_dir, fname)

        with h5py.File(fpath, "r") as f:
            # Only one group inside, e.g. time_000, time_001, ...
            group = list(f.keys())[0]
            g = f[group]

            rho = g["density"][:]  # (Nx, Ny)
            t   = g.attrs["time"]
            Lx  = g.attrs["Lx"]
            Ly  = g.attrs["Ly"]

        # ---- Plot figure ----
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        im = ax.imshow(
            rho.T,
            origin="lower",
            extent=[0, Lx, 0, Ly],
            cmap="magma",
            vmin=0.75,
            vmax=2.25,
            interpolation='nearest',
        )

        # Add time text inside frame
        ax.text(0.02, 0.95, f"t={t:.3f}", color="white", transform=ax.transAxes,
                fontsize=12, ha="left", va="top")

        # Ticks inside, all four sides
        ax.tick_params(direction="in", top=True, right=True)

        plt.colorbar(im, ax=ax, label="density")

        # Save PNG next to the HDF5 file
        png_name = fname.replace(".hdf5", ".png")
        plt.savefig(os.path.join(output_dir, png_name), dpi=300)
        plt.close(fig)

        print(f"Saved density plot for {fname}")

