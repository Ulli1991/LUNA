import h5py
import jax.numpy as jnp

def read_checkpoint(checkpoint_filename):
    """
    Read a checkpoint file and return the complete simulation state for restart.

    Parameters
    ----------
    checkpoint_filename : str
        Path to the checkpoint file

    Returns
    -------
    checkpoint_data : dict
        Dictionary containing:
        - 'U': Conserved variables (Nvar, Nx, Ny, Nz)
        - 'time': Current simulation time
        - 'step': Current step number
        - 'grid': Dictionary with grid configuration (Nx, Ny, Nz, Lx, Ly, Lz, etc.)
        - 'physics': Dictionary with physics parameters (gamma)
        - 'numerics': Dictionary with numerical scheme parameters
        - 'time_control': Dictionary with time control parameters
    """

    with h5py.File(checkpoint_filename, "r") as f:
        # Read conserved variables
        U = jnp.array(f["conserved"]["U"])

        # Read simulation state
        time = float(f["state"].attrs["time"])
        step = int(f["state"].attrs["step"])

        # Read grid configuration
        grid = {
            "Nx": int(f["grid"].attrs["Nx"]),
            "Ny": int(f["grid"].attrs["Ny"]),
            "Nz": int(f["grid"].attrs["Nz"]),
            "Lx": float(f["grid"].attrs["Lx"]),
            "Ly": float(f["grid"].attrs["Ly"]),
            "Lz": float(f["grid"].attrs["Lz"]),
            "SPATIAL_DIMENSION": int(f["grid"].attrs["SPATIAL_DIMENSION"]),
            "BOUNDARY": int(f["grid"].attrs["BOUNDARY"]),
        }

        # Read physics parameters
        physics = {
            "gamma": float(f["physics"].attrs["gamma"]),
        }

        # Read numerical scheme parameters
        numerics = {
            "ORDER": int(f["numerics"].attrs["ORDER"]),
            "LIMITER": int(f["numerics"].attrs["LIMITER"]),
            "TIME_SOLVER": int(f["numerics"].attrs["TIME_SOLVER"]),
            "CFL": float(f["numerics"].attrs["CFL"]),
        }

        # Read time control
        time_control = {
            "t_start": float(f["time_control"].attrs["t_start"]),
            "t_end": float(f["time_control"].attrs["t_end"]),
            "time_bet_snaps": float(f["time_control"].attrs["time_bet_snaps"]),
        }

    checkpoint_data = {
        "U": U,
        "time": time,
        "step": step,
        "grid": grid,
        "physics": physics,
        "numerics": numerics,
        "time_control": time_control,
    }

    print(f"✓ Checkpoint loaded: {checkpoint_filename}")
    print(f"  - Time: {time:.5f}")
    print(f"  - Step: {step}")
    print(f"  - Grid: {grid['Nx']} x {grid['Ny']} x {grid['Nz']}")
    print(f"  - Spatial dimension: {grid['SPATIAL_DIMENSION']}")

    return checkpoint_data


def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """
    Find the latest checkpoint file in the checkpoint directory.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing checkpoint files (default: "checkpoints")

    Returns
    -------
    checkpoint_filename : str or None
        Path to the latest checkpoint file, or None if no checkpoints found
    """
    import os
    import glob

    if not os.path.exists(checkpoint_dir):
        return None

    # Find all checkpoint files
    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint_*.hdf5")
    checkpoint_files = glob.glob(checkpoint_pattern)

    if not checkpoint_files:
        return None

    # Sort by modification time (most recent first)
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)

    return checkpoint_files[0]
