import os
import h5py
import jax.numpy as jnp

def write_autodiff_result(U0, U_final, grads, observable_value, observable_name, cfg,
                         fd_grads=None, fd_sample_indices=None, filename="autodiff_result.hdf5"):
    """
    Save autodiff results to HDF5 file for post-processing and visualization.

    Saves:
    - Initial condition U0 (conservative variables)
    - Final state U_final (conservative variables)
    - Gradients dJ/dU0 (same shape as U0)
    - Finite difference gradients (if provided, for sampled points)
    - Observable value and metadata
    - Spatial coordinates for plotting

    Parameters
    ----------
    U0 : jax.numpy.ndarray
        Initial conservative variables, shape (Nvar, Nx) or (Nvar, Nx, Ny) or (Nvar, Nx, Ny, Nz)
    U_final : jax.numpy.ndarray
        Final conservative variables, same shape as U0
    grads : jax.numpy.ndarray
        Gradients dJ/dU0, same shape as U0
    observable_value : float
        Value of the observable J(U_final)
    observable_name : str
        Name of the observable function (e.g., 'density_at_point')
    cfg : Config
        Configuration object
    fd_grads : jax.numpy.ndarray, optional
        Finite difference gradients, shape (Nvar, num_samples)
    fd_sample_indices : jax.numpy.ndarray, optional
        Spatial indices where finite differences were computed
    filename : str
        Output filename (default: 'autodiff_result.hdf5')
    """

    # Build output path
    os.makedirs(cfg.output_dir, exist_ok=True)
    output_path = os.path.join(cfg.output_dir, filename)

    # Compute spatial coordinates (cell centers)
    dx = cfg.Lx / cfg.Nx
    x_positions = jnp.linspace(dx * 0.5, cfg.Lx - dx * 0.5, cfg.Nx)

    if cfg.SPATIAL_DIMENSION >= 2:
        dy = cfg.Ly / cfg.Ny
        y_positions = jnp.linspace(dy * 0.5, cfg.Ly - dy * 0.5, cfg.Ny)
    else:
        y_positions = None

    if cfg.SPATIAL_DIMENSION == 3:
        dz = cfg.Lz / cfg.Nz
        z_positions = jnp.linspace(dz * 0.5, cfg.Lz - dz * 0.5, cfg.Nz)
    else:
        z_positions = None

    # Write HDF5 file
    with h5py.File(output_path, "w") as f:

        # ===== Initial condition group =====
        initial = f.create_group("initial")
        initial.create_dataset("U", data=U0)  # Full conservative state

        # Decompose for convenience
        rho0 = U0[0]
        initial.create_dataset("density", data=rho0)
        initial.create_dataset("momentum_x", data=U0[1])
        if cfg.SPATIAL_DIMENSION >= 2:
            initial.create_dataset("momentum_y", data=U0[2])
        if cfg.SPATIAL_DIMENSION == 3:
            initial.create_dataset("momentum_z", data=U0[3])
        initial.create_dataset("energy", data=U0[-1])

        # ===== Final state group =====
        final = f.create_group("final")
        final.create_dataset("U", data=U_final)  # Full conservative state

        # Decompose for convenience
        rho_final = U_final[0]
        final.create_dataset("density", data=rho_final)
        final.create_dataset("momentum_x", data=U_final[1])
        if cfg.SPATIAL_DIMENSION >= 2:
            final.create_dataset("momentum_y", data=U_final[2])
        if cfg.SPATIAL_DIMENSION == 3:
            final.create_dataset("momentum_z", data=U_final[3])
        final.create_dataset("energy", data=U_final[-1])

        # ===== Gradients group =====
        gradients = f.create_group("gradients")
        gradients.create_dataset("dJ_dU0", data=grads)  # Full gradient array

        # Split by variable for easier plotting
        gradients.create_dataset("dJ_drho0", data=grads[0])
        gradients.create_dataset("dJ_dmomx0", data=grads[1])
        if cfg.SPATIAL_DIMENSION >= 2:
            gradients.create_dataset("dJ_dmomy0", data=grads[2])
        if cfg.SPATIAL_DIMENSION == 3:
            gradients.create_dataset("dJ_dmomz0", data=grads[3])
        gradients.create_dataset("dJ_dE0", data=grads[-1])

        # Gradient statistics (useful for quick checks)
        gradients.attrs["min"] = float(jnp.min(grads))
        gradients.attrs["max"] = float(jnp.max(grads))
        gradients.attrs["mean"] = float(jnp.mean(grads))
        gradients.attrs["std"] = float(jnp.std(grads))
        gradients.attrs["l2norm"] = float(jnp.linalg.norm(grads))

        # ===== Finite difference gradients (if provided) =====
        if fd_grads is not None and fd_sample_indices is not None:
            fd_group = f.create_group("finite_difference")
            fd_group.create_dataset("gradients", data=fd_grads)  # Shape: (Nvar, num_samples)
            fd_group.create_dataset("sample_indices", data=fd_sample_indices)  # Shape: (num_samples,)

            # Also split by variable for easier plotting
            fd_group.create_dataset("dJ_drho0_fd", data=fd_grads[0])
            fd_group.create_dataset("dJ_dmomx0_fd", data=fd_grads[1])
            if cfg.SPATIAL_DIMENSION >= 2:
                fd_group.create_dataset("dJ_dmomy0_fd", data=fd_grads[2])
            if cfg.SPATIAL_DIMENSION == 3:
                fd_group.create_dataset("dJ_dmomz0_fd", data=fd_grads[3])
            fd_group.create_dataset("dJ_dE0_fd", data=fd_grads[-1])

            # Statistics
            fd_group.attrs["num_samples"] = len(fd_sample_indices)
            fd_group.attrs["min"] = float(jnp.min(fd_grads))
            fd_group.attrs["max"] = float(jnp.max(fd_grads))
            fd_group.attrs["mean"] = float(jnp.mean(fd_grads))

        # ===== Observable info =====
        observable = f.create_group("observable")
        observable.attrs["name"] = observable_name
        observable.attrs["value"] = float(observable_value)

        # Store observation point if applicable
        if "density_at_point" in observable_name:
            if cfg.SPATIAL_DIMENSION == 1:
                obs_point_x = cfg.Nx // 2
                observable.attrs["observation_index_x"] = obs_point_x
                observable.attrs["observation_position_x"] = float(x_positions[obs_point_x])
            elif cfg.SPATIAL_DIMENSION == 2:
                obs_point_x = cfg.Nx // 2
                obs_point_y = cfg.Ny // 2
                observable.attrs["observation_index_x"] = obs_point_x
                observable.attrs["observation_index_y"] = obs_point_y
                observable.attrs["observation_position_x"] = float(x_positions[obs_point_x])
                observable.attrs["observation_position_y"] = float(y_positions[obs_point_y])
            elif cfg.SPATIAL_DIMENSION == 3:
                obs_point_x = cfg.Nx // 2
                obs_point_y = cfg.Ny // 2
                obs_point_z = cfg.Nz // 2
                observable.attrs["observation_index_x"] = obs_point_x
                observable.attrs["observation_index_y"] = obs_point_y
                observable.attrs["observation_index_z"] = obs_point_z
                observable.attrs["observation_position_x"] = float(x_positions[obs_point_x])
                observable.attrs["observation_position_y"] = float(y_positions[obs_point_y])
                observable.attrs["observation_position_z"] = float(z_positions[obs_point_z])

        # ===== Coordinates =====
        coords = f.create_group("coordinates")
        coords.create_dataset("x", data=x_positions)
        if cfg.SPATIAL_DIMENSION >= 2:
            coords.create_dataset("y", data=y_positions)
        if cfg.SPATIAL_DIMENSION == 3:
            coords.create_dataset("z", data=z_positions)

        # ===== Metadata =====
        metadata = f.create_group("metadata")
        metadata.attrs["test_problem"] = cfg.test_problem
        metadata.attrs["spatial_dimension"] = cfg.SPATIAL_DIMENSION
        metadata.attrs["Nx"] = cfg.Nx
        metadata.attrs["Ny"] = cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else 0
        metadata.attrs["Nz"] = cfg.Nz if cfg.SPATIAL_DIMENSION == 3 else 0
        metadata.attrs["Lx"] = cfg.Lx
        metadata.attrs["Ly"] = cfg.Ly if cfg.SPATIAL_DIMENSION >= 2 else 0.0
        metadata.attrs["Lz"] = cfg.Lz if cfg.SPATIAL_DIMENSION == 3 else 0.0
        metadata.attrs["dx"] = dx
        metadata.attrs["dy"] = cfg.Ly / cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else 0.0
        metadata.attrs["dz"] = cfg.Lz / cfg.Nz if cfg.SPATIAL_DIMENSION == 3 else 0.0
        metadata.attrs["t_start"] = cfg.t_start
        metadata.attrs["t_end"] = cfg.t_end
        metadata.attrs["gamma"] = cfg.gamma
        metadata.attrs["order"] = cfg.ORDER
        metadata.attrs["limiter"] = cfg.LIMITER
        metadata.attrs["time_solver"] = cfg.TIME_SOLVER
        metadata.attrs["boundary"] = cfg.BOUNDARY
        metadata.attrs["use_smooth_limiters"] = cfg.use_smooth_limiters
        metadata.attrs["smooth_limiter_sharpness"] = cfg.smooth_limiter_sharpness

    print(f"Autodiff results saved to {output_path}")
    return output_path
