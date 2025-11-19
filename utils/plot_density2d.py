import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def plot_density2d(output_dir, base_name):
    """
    Load all HDF5 snapshots and produce density plots for 2D fields.
    Uses the density range from the last snapshot for consistent color scaling.
    """
    # Find all snapshot files
    files = sorted(
        [f for f in os.listdir(output_dir) if f.startswith(base_name) and f.endswith(".hdf5")]
    )

    if not files:
        print("No HDF5 snapshots found!")
        return

    # Initialize min/max density values
    min_density = float('inf')
    max_density = float('-inf')

    # First, check the last snapshot for the dynamic colorbar limits
    last_fpath = os.path.join(output_dir, files[-1])

    png_dir = os.path.join(output_dir, "png")
    os.makedirs(png_dir, exist_ok=True)

    with h5py.File(last_fpath, "r") as f:
        cells = f["cells"]

        # Fields
        rho = np.array(cells["density"])  # (Nx, Ny)
        t = cells.attrs["time"]
        Lx = cells.attrs["Lx"]
        Ly = cells.attrs["Ly"]

        # Update the min/max values from the last snapshot
        min_density = np.min(rho)
        max_density = np.max(rho)

    # Now, iterate through all snapshot files and plot
    for fname in files:
        fpath = os.path.join(output_dir, fname)

        with h5py.File(fpath, "r") as f:
            cells = f["cells"]

            # Fields
            rho = np.array(cells["density"])  # (Nx, Ny)
            x = np.array(cells["x_positions"])  # (Nx,)
            y = np.array(cells["y_positions"])  # (Ny,)
            t = cells.attrs["time"]
            Lx = cells.attrs["Lx"]
            Ly = cells.attrs["Ly"]

        # ---- Plot figure ----
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        im = ax.imshow(
            rho.T,  # Transpose to match the x and y axes
            origin="lower",
            extent=[0, Lx, 0, Ly],
            cmap="magma",
            vmin=min_density,  # Dynamically set min/max density from the last snapshot
            vmax=max_density,
            interpolation='nearest',
        )

        # Add time text inside frame
        ax.text(0.02, 0.95, f"t={t:.3f}", color="white", transform=ax.transAxes,
                fontsize=12, ha="left", va="top")

        # Ticks inside, all four sides
        ax.tick_params(direction="in", top=True, right=True)

        # Add colorbar with label "Density"
        cbar = plt.colorbar(im, ax=ax, label="Density")

        # Save PNG next to the HDF5 file
        png_name = fname.replace(".hdf5", ".png")
        plt.savefig(os.path.join(png_dir, png_name), dpi=300)
        plt.close(fig)

        print(f"Saved density plot for {fname}")


