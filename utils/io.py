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
    
    # Calculate kinetic energy based on spatial dimensions
    rho = U[0]
    vx = U[1] / rho
    vy = U[2] / rho if cfg.SPATIAL_DIMENSION >= 2 else 0.0
    vz = U[3] / rho if cfg.SPATIAL_DIMENSION == 3 else 0.0

    # Kinetic energy based on spatial dimensions
    if cfg.SPATIAL_DIMENSION == 1:
        kinetic_energy = 0.5 * rho * vx**2  # 1D: only vx contributes
    elif cfg.SPATIAL_DIMENSION == 2:
        kinetic_energy = 0.5 * rho * (vx**2 + vy**2)  # 2D: vx and vy contribute
    elif cfg.SPATIAL_DIMENSION == 3:
        kinetic_energy = 0.5 * rho * (vx**2 + vy**2 + vz**2)  # 3D: vx, vy, and vz contribute

    # Internal energy = Total energy (U[-1]) - Kinetic energy
    internal_energy = U[-1] - kinetic_energy  # E_internal = E_total - E_kinetic

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the data in HDF5 format
    with h5py.File(snapshot_filename, 'a') as f:
        # Create a group for this time step
        cells_group = f.create_group('cells')

        # Save the fields to the HDF5 file
        cells_group.create_dataset('density', data=U[0])  # Density (rho)
        cells_group.create_dataset('velocity_x', data=U[1] / U[0])  # Velocity in x-direction (ux)
        
        if cfg.SPATIAL_DIMENSION >= 2:
            cells_group.create_dataset('velocity_y', data=U[2] / U[0])  # Velocity in y-direction (uy)

        if cfg.SPATIAL_DIMENSION == 3:
            cells_group.create_dataset('velocity_z', data=U[3] / U[0])  # Velocity in z-direction (uz)
        
        cells_group.create_dataset('internal_energy', data=internal_energy)  # Internal energy (E_internal)
        
        # Create position datasets for x, y, z (depends on the spatial dimension)
        # 1D: only x positions (midpoints)
        x_positions = jnp.linspace(0 + (cfg.Lx / cfg.Nx) / 2, cfg.Lx - (cfg.Lx / cfg.Nx) / 2, cfg.Nx)
        cells_group.create_dataset('x_positions', data=x_positions)

        if cfg.SPATIAL_DIMENSION >= 2:
            # 2D: both x and y positions (midpoints)
            y_positions = jnp.linspace(0 + (cfg.Ly / cfg.Ny) / 2, cfg.Ly - (cfg.Ly / cfg.Ny) / 2, cfg.Ny)
            cells_group.create_dataset('y_positions', data=y_positions)

        if cfg.SPATIAL_DIMENSION == 3:
            # 3D: x, y, and z positions (midpoints)
            z_positions = jnp.linspace(0 + (cfg.Lz / cfg.Nz) / 2, cfg.Lz - (cfg.Lz / cfg.Nz) / 2, cfg.Nz)
            cells_group.create_dataset('z_positions', data=z_positions)
        
        # Save simulation metadata as attributes
        cells_group.attrs['time'] = t
        cells_group.attrs['step'] = step
        cells_group.attrs['Lx'] = cfg.Lx
        cells_group.attrs['Ly'] = cfg.Ly
        cells_group.attrs['dx'] = (cfg.Lx / cfg.Nx)
        cells_group.attrs['dy'] = (cfg.Ly / cfg.Ny) if cfg.SPATIAL_DIMENSION >= 2 else None
        cells_group.attrs['dz'] = (cfg.Lz / cfg.Nz) if cfg.SPATIAL_DIMENSION == 3 else None
        
        # Additional simulation metadata
        cells_group.attrs['order'] = cfg.ORDER
        cells_group.attrs['limiter'] = cfg.LIMITER
        
        print(f"Snapshot saved at time {t}, step {step} to {snapshot_filename}")

