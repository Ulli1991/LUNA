"""Checkpoint housekeeping: rotation and config-compatibility checks."""
import os
import glob


def manage_checkpoints(checkpoint_dir, max_checkpoints):
    """Delete oldest checkpoints, keeping the newest max_checkpoints
    (max_checkpoints <= 0 keeps all)."""
    if max_checkpoints <= 0:
        return  # Keep all checkpoints

    # Find all checkpoint files
    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint_*.hdf5")
    checkpoint_files = glob.glob(checkpoint_pattern)

    if len(checkpoint_files) <= max_checkpoints:
        return  # Nothing to delete

    # Sort by modification time (oldest first)
    checkpoint_files.sort(key=os.path.getmtime)

    # Delete oldest checkpoints
    num_to_delete = len(checkpoint_files) - max_checkpoints
    for i in range(num_to_delete):
        os.remove(checkpoint_files[i])
        print(f"  Deleted old checkpoint: {os.path.basename(checkpoint_files[i])}")


def validate_checkpoint_compatibility(checkpoint_data, cfg):
    """Raise ValueError if checkpoint grid/domain/dimension mismatch cfg;
    gamma mismatch only warns (checkpoint gamma wins)."""
    grid = checkpoint_data["grid"]
    physics = checkpoint_data["physics"]

    # Check grid dimensions
    if grid["Nx"] != cfg.Nx or grid["Ny"] != cfg.Ny or grid["Nz"] != cfg.Nz:
        raise ValueError(
            f"Grid mismatch: checkpoint has ({grid['Nx']}, {grid['Ny']}, {grid['Nz']}), "
            f"config has ({cfg.Nx}, {cfg.Ny}, {cfg.Nz})"
        )

    # Check domain size
    if (
        abs(grid["Lx"] - cfg.Lx) > 1e-10
        or abs(grid["Ly"] - cfg.Ly) > 1e-10
        or abs(grid["Lz"] - cfg.Lz) > 1e-10
    ):
        raise ValueError(
            f"Domain size mismatch: checkpoint has ({grid['Lx']}, {grid['Ly']}, {grid['Lz']}), "
            f"config has ({cfg.Lx}, {cfg.Ly}, {cfg.Lz})"
        )

    # Check spatial dimension
    if grid["SPATIAL_DIMENSION"] != cfg.SPATIAL_DIMENSION:
        raise ValueError(
            f"Spatial dimension mismatch: checkpoint has {grid['SPATIAL_DIMENSION']}, "
            f"config has {cfg.SPATIAL_DIMENSION}"
        )

    # Check physics parameters
    if abs(physics["gamma"] - cfg.gamma) > 1e-10:
        print(f"WARNING: gamma mismatch: checkpoint has {physics['gamma']}, config has {cfg.gamma}")
        print(f"         Using checkpoint gamma: {physics['gamma']}")

    print("✓ Checkpoint compatibility validated")
