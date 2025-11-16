import h5py
import os

def write_snapshot(U, cfg, t, step):
    """
    Save the full simulation state (density, velocity, internal energy, etc.) to an HDF5 file.
    
    Parameters:
    U (array): The full simulation state, which includes all fields (density, velocity, internal energy).
    cfg (Config): The configuration object containing simulation settings.
    t (float): The current time step.
    step (int): The current step number.
    """
    
    # Extract output directory and base name from the configuration
    output_dir = cfg.output_dir
    base_name = cfg.base_name
    
    # Create a zero-padded snapshot filename with 3 digits and .hdf5 extension
    snapshot_filename = os.path.join(output_dir, f"{base_name}_{step:03d}.hdf5")
    
    # Calculate internal energy from total energy and kinetic energy
    kinetic_energy = (0.5 / U[0]) * (U[1]**2 + U[2]**2)  # Kinetic energy per unit mass
    internal_energy = U[3] - kinetic_energy  # E_internal = E_total - E_kinetic

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the data in HDF5 format
    with h5py.File(snapshot_filename, 'a') as f:
        # Create a group for this time step
        time_group = f.create_group(f'time_{step}')
        
        # Save the fields to the HDF5 file
        time_group.create_dataset('density', data=U[0])  # Density (rho)
        time_group.create_dataset('velocity_x', data=U[1] / U[0])  # Velocity in x-direction (ux)
        time_group.create_dataset('velocity_y', data=U[2] / U[0])  # Velocity in y-direction (uy)
        time_group.create_dataset('internal_energy', data=internal_energy)  # Internal energy (E_internal)
        
        # Save simulation metadata as attributes
        time_group.attrs['time'] = t
        time_group.attrs['step'] = step
        time_group.attrs['Lx'] = cfg.Lx
        time_group.attrs['Ly'] = cfg.Ly
        time_group.attrs['dx'] = (cfg.Lx / cfg.Nx)
        time_group.attrs['dy'] = (cfg.Ly / cfg.Ny)
        time_group.attrs['order'] = cfg.ORDER
        time_group.attrs['limiter'] = cfg.LIMITER
        
        print(f"Snapshot saved at time {t}, step {step} to {snapshot_filename}")

