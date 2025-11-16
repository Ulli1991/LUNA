from code_configuration.config import Config
from ic_generation.ic import initial_conditions
from time_integration.integrators import *
from utils.io import write_snapshot
from time_integration.cfl import compute_dt

import jax.numpy as jnp
import jax
import time
import os

def main():

    # --- Load configuration (pure Python object) ---
    cfg = Config()

    # --- Precompute constants to pass explicitly ---
    dx = cfg.Lx / cfg.Nx
    dy = cfg.Ly / cfg.Ny
    gamma = cfg.gamma
    order = cfg.ORDER
    limiter = cfg.LIMITER
    time_solver = cfg.TIME_SOLVER

    device_type = cfg.DEVICE
    if device_type == 0:
        print("Initialzing Run on CPU...")
        jax.config.update("jax_platform_name", "cpu")
    else: 
        print("Initialzing Run on GPU...")
        jax.config.update("jax_platform_name", "rocm")

    # --- Set output frequecny ---
    snapshot_times = jnp.linspace(cfg.t_start, cfg.t_end, cfg.n_snapshots)
    snapshot_idx = 0

    # --- Initialize fields ---
    U = initial_conditions(cfg)

    # --- Output directory ---
    os.makedirs(cfg.output_dir, exist_ok=True)

    # --- Time loop variables ---
    t = cfg.t_start
    step = 0

    print("Starting simulation…")
    # Start the timer for the total runtime
    start_time = time.time()  # Record the start time of the simulation

    # --- Time integration loop ---
    while t < cfg.t_end:

        # dt is computed with pure arrays/scalars only
        dt = compute_dt(U, dx, dy, gamma, cfg.CFL)
        if t + dt > cfg.t_end:
            dt = cfg.t_end - t

        # time_stepping
        U = time_step(U, dt, dx, dy, gamma, order, limiter, cfg.TIME_SOLVER)
        t += dt
        step += 1

        print(f"Step {step:5d}, t = {t:.5f}, dt = {dt:.5e}")

        # Save snapshot every N steps (cfg never touches JAX)
        if t >= snapshot_times[snapshot_idx]:
            print("Start saving snapshot")
            # Call the updated write_snapshot function to save the full state to HDF5
            write_snapshot(U, cfg, t, snapshot_idx)
            snapshot_idx += 1
            if snapshot_idx >= len(snapshot_times):
                snapshot_idx = len(snapshot_times) - 1
            print("Finished saving snapshot")
    
    # End the timer and calculate the total runtime
    end_time = time.time()  # Record the end time of the simulation
    total_time = end_time - start_time  # Calculate the total runtime

    print(f"Done! Took {total_time:.5f} seconds! Good Bye and Good Day!!!")

    if cfg.PLOT_AFTER_SIM:
        print("Generating density plots from all HDF5 snapshots...")
        from utils.plotting_hdf5 import plot_density_from_hdf5_all
        plot_density_from_hdf5_all(cfg.output_dir, cfg.base_name)


if __name__ == "__main__":
    main()

