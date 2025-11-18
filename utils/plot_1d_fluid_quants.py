import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_1d_fluid_quantities(output_dir, base_name):

    # --- find all snapshot files ---
    files = sorted(
        [f for f in os.listdir(output_dir) if f.startswith(base_name) and f.endswith(".hdf5")]
    )

    if not files:
        print("No HDF5 snapshots found!")
        return

    # --- create png directory ---
    png_dir = os.path.join(output_dir, "png")
    os.makedirs(png_dir, exist_ok=True)

    for fname in files:
        fpath = os.path.join(output_dir, fname)

        with h5py.File(fpath, "r") as f:
            cells = f["cells"]

            # Fields
            x = np.array(cells["x_positions"])
            density = np.array(cells["density"])
            velocity_x = np.array(cells["velocity_x"])
            internal_energy = np.array(cells["internal_energy"])

            gamma = cells.attrs["gamma"]
            pressure = (gamma - 1.0) * internal_energy

            t = cells.attrs["time"]

        # --- plotting ---
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.ravel()

        # Main curves
        axs[0].plot(x, density, lw=1.4, color="k")
        axs[1].plot(x, velocity_x, lw=1.4, color="k")
        axs[2].plot(x, pressure, lw=1.4, color="k")
        axs[3].plot(x, internal_energy, lw=1.4, color="k")

        # Y-axis labels (no titles)
        axs[0].set_ylabel("density")
        axs[1].set_ylabel("v$_{x}$")
        axs[2].set_ylabel("pressure")
        axs[3].set_ylabel("internal energy")

        # X-axis labels only on bottom row
        axs[2].set_xlabel("x")
        axs[3].set_xlabel("x")

        axs[0].set_xlim(0,1)
        axs[1].set_xlim(0,1)
        axs[2].set_xlim(0,1)
        axs[3].set_xlim(0,1)

        axs[0].set_ylim(0,1.1)
        axs[1].set_ylim(0,1.1)
        axs[2].set_ylim(0,1.1)
        axs[3].set_ylim(0,1.6)

        # Remove titles
        for ax in axs:
            ax.set_title("")

        # Tick labels inside
        for ax in axs:
            ax.tick_params(
                direction="in",
                which="both",
                labelsize=10
            )
            ax.grid(False)

        # Time annotation in top-left panel
        axs[0].text(
            0.80, 0.92, f"t = {t:.4f}",
            transform=axs[0].transAxes,
            fontsize=12,
            color="black",
            ha="left", va="top",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
        )

        plt.tight_layout()

        # Save file inside /png
        png_name = fname.replace(".hdf5", ".png")
        png_path = os.path.join(png_dir, png_name)
        plt.savefig(png_path, dpi=300)
        plt.close()

        print(f"Saved plot: {png_path}")

