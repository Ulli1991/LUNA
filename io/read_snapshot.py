import h5py
import numpy as np

def read_snapshot(snapshot_filename, cfg):
    """
    Load the simulation state (density, velocity, internal energy, etc.) from an HDF5 snapshot file.
    Assumes the data was written using the same structure as the `write_snapshot` function.
    """
    # Open the snapshot file
    with h5py.File(snapshot_filename, "r") as f:
        cells = f["cells"]

        # Read fields (density, velocity components, internal energy)
        rho = np.array(cells["density"])
        vx = np.array(cells["velocity_x"])
        
        if cfg.SPATIAL_DIMENSION >= 2:
            vy = np.array(cells["velocity_y"])
        else:
            vy = 0.0

        if cfg.SPATIAL_DIMENSION == 3:
            vz = np.array(cells["velocity_z"])
        else:
            vz = 0.0

        internal_energy = np.array(cells["internal_energy"])

        # Read coordinates
        x_positions = np.array(cells["x_positions"])
        if cfg.SPATIAL_DIMENSION >= 2:
            y_positions = np.array(cells["y_positions"])
        else:
            y_positions = None
        
        if cfg.SPATIAL_DIMENSION == 3:
            z_positions = np.array(cells["z_positions"])
        else:
            z_positions = None

        # Read metadata (time, step, grid dimensions, etc.)
        time = cells.attrs["time"]
        step = cells.attrs["step"]
        Lx = cells.attrs["Lx"]
        Ly = cells.attrs["Ly"] if cfg.SPATIAL_DIMENSION >= 2 else 0.0
        Lz = cells.attrs["Lz"] if cfg.SPATIAL_DIMENSION == 3 else 0.0

        dx = cells.attrs["dx"]
        dy = cells.attrs["dy"] if cfg.SPATIAL_DIMENSION >= 2 else 0.0
        dz = cells.attrs["dz"] if cfg.SPATIAL_DIMENSION == 3 else 0.0
        
        # Read additional metadata (order, limiter, gamma)
        order = cells.attrs["order"]
        limiter = cells.attrs["limiter"]
        gamma = cells.attrs["gamma"]

    # Create a dictionary to return all loaded data in a structured way
    data = {
        "rho": rho,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "internal_energy": internal_energy,
        "x_positions": x_positions,
        "y_positions": y_positions,
        "z_positions": z_positions,
        "time": time,
        "step": step,
        "Lx": Lx,
        "Ly": Ly,
        "Lz": Lz,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "order": order,
        "limiter": limiter,
        "gamma": gamma
    }

    return data

