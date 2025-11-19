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

    # Initialize variables for min and max values across all files
    min_density = float('inf')
    max_density = float('-inf')
    min_velocity_x = float('inf')
    max_velocity_x = float('-inf')
    min_pressure = float('inf')
    max_pressure = float('-inf')
    min_internal_energy = float('inf')
    max_internal_energy = float('-inf')

    # First, check the last snapshot for the dynamic y-axis limits
    last_fpath = os.path.join(output_dir, files[-1])

    with h5py.File(last_fpath, "r") as f:
        cells = f["cells"]

        # Fields
        x = np.array(cells["x_positions"])
        density = np.array(cells["density"])
        velocity_x = np.array(cells["velocity_x"])
        internal_energy = np.array(cells["internal_energy"])

        gamma = cells.attrs["gamma"]
        pressure = (gamma - 1.0) * internal_energy

        # Update the min/max values from the last snapshot
        min_density = min(min_density, np.min(density))
        max_density = max(max_density, np.max(density))
        min_velocity_x = min(min_velocity_x, np.min(velocity_x))
        max_velocity_x = max(max_velocity_x, np.max(velocity_x))
        min_pressure = min(min_pressure, np.min(pressure))
        max_pressure = max(max_pressure, np.max(pressure))
        min_internal_energy = min(min_internal_energy, np.min(internal_energy))
        max_internal_energy = max(max_internal_energy, np.max(internal_energy))

    # Now iterate through all snapshot files for plotting
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
            Lx = cells.attrs["Lx"]

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

        axs[0].set_xlim(0,Lx)
        axs[1].set_xlim(0,Lx)
        axs[2].set_xlim(0,Lx)
        axs[3].set_xlim(0,Lx)

        # Dynamically set y-limits based on the last snapshot data
        # Check for identical min/max values and adjust the y-limits
        axs[0].set_ylim(min_density, max_density * 1.1 if max_density != min_density else min_density * 1.1)  # density
        axs[1].set_ylim(min_velocity_x, max_velocity_x * 1.1 if max_velocity_x != min_velocity_x else min_velocity_x * 1.1)  # velocity_x
        axs[2].set_ylim(min_pressure, max_pressure * 1.1 if max_pressure != min_pressure else min_pressure * 1.1)  # pressure
        axs[3].set_ylim(min_internal_energy, max_internal_energy * 1.1 if max_internal_energy != min_internal_energy else min_internal_energy * 1.1)  # internal energy

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

