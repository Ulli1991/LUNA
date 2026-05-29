import os
import h5py
import jax.numpy as jnp


def write_checkpoint(U, cfg, t, step, checkpoint_dir="checkpoints"):
    """
    Write a checkpoint file containing complete simulation state for restart.

    This saves the conserved variables U directly (not primitive variables)
    along with all metadata needed to restart the simulation exactly.

    Parameters
    ----------
    U : jax array
        Conserved variables (Nvar, Nx, Ny, Nz)
    cfg : Config
        Configuration object
    t : float
        Current simulation time
    step : int
        Current step number
    checkpoint_dir : str
        Directory to save checkpoints (default: "checkpoints")
    """

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Build checkpoint filename
    checkpoint_filename = os.path.join(checkpoint_dir, f"checkpoint_{step:06d}.hdf5")

    # Write checkpoint file
    with h5py.File(checkpoint_filename, "w") as f:
        # Create main group for conserved variables
        conserved = f.create_group("conserved")

        # Save conserved variables U directly (no conversion to primitives)
        conserved.create_dataset("U", data=U, compression="gzip", compression_opts=4)

        # Save simulation state
        state = f.create_group("state")
        state.attrs["time"] = float(t)
        state.attrs["step"] = int(step)

        # Save grid configuration
        grid = f.create_group("grid")
        grid.attrs["Nx"] = cfg.Nx
        grid.attrs["Ny"] = cfg.Ny
        grid.attrs["Nz"] = cfg.Nz
        grid.attrs["Lx"] = cfg.Lx
        grid.attrs["Ly"] = cfg.Ly
        grid.attrs["Lz"] = cfg.Lz
        grid.attrs["SPATIAL_DIMENSION"] = cfg.SPATIAL_DIMENSION
        grid.attrs["BOUNDARY"] = cfg.BOUNDARY

        # Save physics parameters
        physics = f.create_group("physics")
        physics.attrs["gamma"] = cfg.gamma

        # Save numerical scheme parameters
        numerics = f.create_group("numerics")
        numerics.attrs["ORDER"] = cfg.ORDER
        numerics.attrs["LIMITER"] = cfg.LIMITER
        numerics.attrs["TIME_SOLVER"] = cfg.TIME_SOLVER
        numerics.attrs["CFL"] = cfg.CFL

        # Save time control
        time_control = f.create_group("time_control")
        time_control.attrs["t_start"] = cfg.t_start
        time_control.attrs["t_end"] = cfg.t_end
        time_control.attrs["time_bet_snaps"] = cfg.time_bet_snaps

    print(f"✓ Checkpoint saved: {checkpoint_filename}")
    return checkpoint_filename
