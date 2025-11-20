import os

import h5py
import jax.numpy as jnp

def write_snapshot(U, cfg, t, step):
    """
    Save the full simulation state (density, velocity, internal energy, etc.) to an HDF5 file.
    JAX is used for all numerical operations. h5py automatically converts JAX arrays to NumPy for writing.
    """

    # Build output path
    os.makedirs(cfg.output_dir, exist_ok=True)
    snapshot_filename = os.path.join(cfg.output_dir, f"{cfg.base_name}_{step:03d}.hdf5")

    # --- Extract fields ---
    rho = U[0]
    vx = U[1] / rho

    if cfg.SPATIAL_DIMENSION >= 2:
        vy = U[2] / rho
    else:
        vy = 0.0

    if cfg.SPATIAL_DIMENSION == 3:
        vz = U[3] / rho
    else:
        vz = 0.0

    # --- Kinetic energy ---
    vel_sq = vx**2
    if cfg.SPATIAL_DIMENSION >= 2:
        vel_sq = vel_sq + vy**2
    if cfg.SPATIAL_DIMENSION == 3:
        vel_sq = vel_sq + vz**2

    kinetic_energy = 0.5 * rho * vel_sq

    # --- Internal energy ---
    internal_energy = U[-1] - kinetic_energy

    # --- Cell-centered coordinates (always with jnp) ---
    # 1D
    x_positions = jnp.linspace(
        0 + (cfg.Lx / cfg.Nx) * 0.5,
        cfg.Lx - (cfg.Lx / cfg.Nx) * 0.5,
        cfg.Nx,
    )

    # 2D
    if cfg.SPATIAL_DIMENSION >= 2:
        y_positions = jnp.linspace(
            0 + (cfg.Ly / cfg.Ny) * 0.5,
            cfg.Ly - (cfg.Ly / cfg.Ny) * 0.5,
            cfg.Ny,
        )

    # 3D
    if cfg.SPATIAL_DIMENSION == 3:
        z_positions = jnp.linspace(
            0 + (cfg.Lz / cfg.Nz) * 0.5,
            cfg.Lz - (cfg.Lz / cfg.Nz) * 0.5,
            cfg.Nz,
        )

    # --- Write HDF5 file ---
    with h5py.File(snapshot_filename, "w") as f:
        cells = f.create_group("cells")

        # Fields
        cells.create_dataset("density", data=rho)
        cells.create_dataset("velocity_x", data=vx)
        if cfg.SPATIAL_DIMENSION >= 2:
            cells.create_dataset("velocity_y", data=vy)
        if cfg.SPATIAL_DIMENSION == 3:
            cells.create_dataset("velocity_z", data=vz)

        cells.create_dataset("internal_energy", data=internal_energy)

        # Coordinates
        cells.create_dataset("x_positions", data=x_positions)
        if cfg.SPATIAL_DIMENSION >= 2:
            cells.create_dataset("y_positions", data=y_positions)
        if cfg.SPATIAL_DIMENSION == 3:
            cells.create_dataset("z_positions", data=z_positions)

        # Metadata
        cells.attrs["time"] = float(t)
        cells.attrs["step"] = int(step)
        cells.attrs["Lx"] = float(cfg.Lx)
        cells.attrs["Ly"] = float(cfg.Ly) if cfg.SPATIAL_DIMENSION >= 2 else 0.0
        cells.attrs["Lz"] = float(cfg.Lz) if cfg.SPATIAL_DIMENSION == 3 else 0.0

        cells.attrs["dx"] = float(cfg.Lx / cfg.Nx)
        cells.attrs["dy"] = float(cfg.Ly / cfg.Ny) if cfg.SPATIAL_DIMENSION >= 2 else 0.0
        cells.attrs["dz"] = float(cfg.Lz / cfg.Nz) if cfg.SPATIAL_DIMENSION == 3 else 0.0

        cells.attrs["order"] = cfg.ORDER
        cells.attrs["limiter"] = cfg.LIMITER
        cells.attrs["gamma"] = cfg.gamma

    print(f"Snapshot saved at time {t}, step {step} to {snapshot_filename}")

