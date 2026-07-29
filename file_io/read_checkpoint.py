"""Read HDF5 checkpoints for restart and locate the latest one."""
import h5py
import jax.numpy as jnp


def read_checkpoint(checkpoint_filename):
    """Read a checkpoint file into a restart dict with keys 'U' (Nvar, Nx,
    Ny, Nz), 'time', 'step', 'grid', 'physics', 'numerics', 'time_control',
    and 'units' (None for pre-units checkpoints)."""

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
            "CFL": float(f["numerics"].attrs["CFL"]),
        }

        # Read time control
        time_control = {
            "t_start": float(f["time_control"].attrs["t_start"]),
            "t_end": float(f["time_control"].attrs["t_end"]),
            "time_bet_snaps": float(f["time_control"].attrs["time_bet_snaps"]),
        }

        # Read unit system (backward compatible - may not exist)
        if "units" in f:
            units_data = {
                "UnitLength_in_cm": float(f["units"].attrs["UnitLength_in_cm"]),
                "UnitMass_in_g": float(f["units"].attrs["UnitMass_in_g"]),
                "UnitVelocity_in_cm_per_s": float(f["units"].attrs["UnitVelocity_in_cm_per_s"]),
            }
        else:
            units_data = None

    checkpoint_data = {
        "U": U,
        "time": time,
        "step": step,
        "grid": grid,
        "physics": physics,
        "numerics": numerics,
        "time_control": time_control,
        "units": units_data,
    }

    print(f"✓ Checkpoint loaded: {checkpoint_filename}")
    print(f"  - Time: {time:.5f}")
    print(f"  - Step: {step}")
    print(f"  - Grid: {grid['Nx']} x {grid['Ny']} x {grid['Nz']}")
    print(f"  - Spatial dimension: {grid['SPATIAL_DIMENSION']}")

    return checkpoint_data


def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Return the newest checkpoint path (by mtime) in checkpoint_dir,
    or None if there is none."""
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
