import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_1d_fluid_quantities(output_dir, base_name):
    """
    Plot the fluid quantities (density, velocity, internal energy, pressure) vs. x for 1D simulations,
    and save the plots as PNG files in the output directory.
    
    Parameters:
    output_dir (str): The directory where the snapshot files are stored.
    base_name (str): The base name of the snapshot files.
    """
    # List all snapshots in the output directory
    snapshot_files = sorted([f for f in os.listdir(output_dir) if f.startswith(base_name) and f.endswith('.hdf5')])

    # Loop through the snapshot files and plot the desired quantities
    for snapshot_file in snapshot_files:
        snapshot_path = os.path.join(output_dir, snapshot_file)
        
        with h5py.File(snapshot_path, 'r') as f:
            cells_group = f['cells']
            
            # Extract the x positions (assumes 'x' attribute is present in each snapshot)
            x = np.array(cells_group['x'])  # x positions of the cells
            
            # Extract fluid quantities
            density = np.array(cells_group['density'])
            velocity_x = np.array(cells_group['velocity_x'])
            internal_energy = np.array(cells_group['internal_energy'])
            pressure = (cells_group.attrs['gamma'] - 1.0) * (internal_energy - 0.5 * velocity_x**2)  # Ideal gas equation

            # Create the plot
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            axs = axs.ravel()

            # Plot each quantity
            axs[0].plot(x, density)
            axs[0].set_xlabel('x')
            axs[0].set_ylabel('Density')
            axs[0].set_title('Density vs. x')

            axs[1].plot(x, velocity_x)
            axs[1].set_xlabel('x')
            axs[1].set_ylabel('Velocity (x)')
            axs[1].set_title('Velocity (x) vs. x')

            axs[2].plot(x, pressure)
            axs[2].set_xlabel('x')
            axs[2].set_ylabel('Pressure')
            axs[2].set_title('Pressure vs. x')

            axs[3].plot(x, internal_energy)
            axs[3].set_xlabel('x')
            axs[3].set_ylabel('Internal Energy')
            axs[3].set_title('Internal Energy vs. x')

            # Set a tight layout
            plt.tight_layout()

            # Add a legend with the time step in the figure (if needed)
            time_step = cells_group.attrs['time']
            plt.suptitle(f"Snapshot at t = {time_step:.2f}", fontsize=16)

            # Save the plot as a PNG file in the output directory
            plot_filename = os.path.join(output_dir, f"{base_name}_t{time_step:.2f}.png")
            plt.savefig(plot_filename)
            plt.close()  # Close the plot to avoid memory issues

            print(f"Saved plot for snapshot at t = {time_step:.2f} to {plot_filename}")

