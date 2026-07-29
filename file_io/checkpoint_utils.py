"""Checkpoint housekeeping: rotation and config-compatibility checks."""
import os
import glob


def _write_meta(f, cfg, t, step, local_shape=None):
    """Write the state/grid/physics/numerics/time_control groups shared by the
    checkpoint writers. local_shape, if given, is stored as a grid attr."""
    state = f.create_group("state")
    state.attrs["time"] = float(t)
    state.attrs["step"] = int(step)

    grid = f.create_group("grid")
    grid.attrs["Nx"] = cfg.Nx
    grid.attrs["Ny"] = cfg.Ny
    grid.attrs["Nz"] = cfg.Nz
    grid.attrs["Lx"] = cfg.Lx
    grid.attrs["Ly"] = cfg.Ly
    grid.attrs["Lz"] = cfg.Lz
    grid.attrs["SPATIAL_DIMENSION"] = cfg.SPATIAL_DIMENSION
    grid.attrs["BOUNDARY"] = cfg.BOUNDARY
    if local_shape is not None:
        grid.attrs["local_shape"] = local_shape

    physics = f.create_group("physics")
    physics.attrs["gamma"] = cfg.gamma

    numerics = f.create_group("numerics")
    numerics.attrs["ORDER"] = cfg.ORDER
    numerics.attrs["LIMITER"] = cfg.LIMITER
    numerics.attrs["CFL"] = cfg.CFL

    time_control = f.create_group("time_control")
    time_control.attrs["t_start"] = cfg.t_start
    time_control.attrs["t_end"] = cfg.t_end
    time_control.attrs["time_bet_snaps"] = cfg.time_bet_snaps


def _write_units(f, cfg, full=False):
    """Write the units group if cfg.units is enabled; full=True adds the
    derived-unit attrs the snapshot writer stores."""
    if not cfg.units.is_enabled:
        return
    units_grp = f.create_group("units")
    units_grp.attrs["UnitLength_in_cm"] = cfg.units.UnitLength_in_cm
    units_grp.attrs["UnitMass_in_g"] = cfg.units.UnitMass_in_g
    units_grp.attrs["UnitVelocity_in_cm_per_s"] = cfg.units.UnitVelocity_in_cm_per_s
    if full:
        units_grp.attrs["UnitTime_in_s"] = cfg.units.UnitTime_in_s
        units_grp.attrs["UnitDensity_in_cgs"] = cfg.units.UnitDensity_in_cgs
        units_grp.attrs["UnitPressure_in_cgs"] = cfg.units.UnitPressure_in_cgs
        units_grp.attrs["UnitEnergy_in_cgs"] = cfg.units.UnitEnergy_in_cgs


def _shard_context(U_sharded, cfg, mesh, per_process=False):
    """Common preamble for the distributed writers: process ids, the
    (reconstructed) device mesh, and the local addressable shards.

    per_process=True sizes the reconstructed mesh by process_count in true
    multi-process runs (the distributed-checkpoint convention); otherwise by
    jax.device_count(). An empty shard list means the caller should bail (a
    warning has already been printed).
    """
    import jax
    from utils.distributed import create_device_mesh

    process_id = jax.process_index()
    process_count = jax.process_count()
    if mesh is None:
        n = process_count if (per_process and process_count > 1) else jax.device_count()
        mesh, _ = create_device_mesh(n, cfg.SPATIAL_DIMENSION)
    local_shards = U_sharded.addressable_shards
    if len(local_shards) == 0:
        print(f"Warning: Process {process_id} has no addressable shards")
    return process_id, process_count, mesh, local_shards


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
