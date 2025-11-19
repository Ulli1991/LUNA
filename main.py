from config import Config
from ic_generation.ic import initial_conditions
from time_integration.integrators import *
from time_integration.cfl import compute_dt
from io.write_snapshot import write_snapshot

import jax.numpy as jnp
import jax
import time
import os

def main():

    # --- Load configuration (pure Python object) ---
    cfg = Config()

    device_type = cfg.DEVICE
    if device_type == 0:
        print("Initializing Run on CPU...")
        jax.config.update("jax_platform_name", "cpu")
    else: 
        print("Initializing Run on GPU...")
        jax.config.update("jax_platform_name", "rocm")

    # --- Precompute constants to pass explicitly ---
    dx = cfg.Lx / cfg.Nx
    dy = cfg.Ly / cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else None
    dz = cfg.Lz / cfg.Nz if cfg.SPATIAL_DIMENSION == 3 else None

    gamma = cfg.gamma
    order = cfg.ORDER
    limiter = cfg.LIMITER
    time_solver = cfg.TIME_SOLVER
    boundary = cfg.BOUNDARY

    # --- Set output frequency ---
    # Compute number of snapshots based on time between snapshots and total simulation time
    n_snapshots = int(cfg.t_end / cfg.time_bet_snaps)  # Number of snapshots based on desired interval
    snapshot_times = jnp.linspace(cfg.t_start, cfg.t_end, n_snapshots)
    
    # If the final time t_end is not exactly included, add it
    if snapshot_times[-1] < cfg.t_end:
        snapshot_times = jnp.append(snapshot_times, cfg.t_end)

    snapshot_idx = 0
    
    # --- Output directory ---
    os.makedirs(cfg.output_dir, exist_ok=True)

    # --- Initialize fields ---
    U = initial_conditions(cfg)

    # --- Time loop variables ---
    t = cfg.t_start
    step = 0

    print("Starting simulation…")
    # Start the timer for the total runtime
    start_time = time.time()  # Record the start time of the simulation

    # --- Time integration loop ---
    while t < cfg.t_end:

        # Compute the current dt based on CFL and the time to the next snapshot
        t_snapshot = snapshot_times[snapshot_idx]
        dt = compute_dt(U, gamma, cfg.CFL, t, t_snapshot, dx, dy, dz)
        if t + dt > cfg.t_end:
            dt = cfg.t_end - t  # Ensure we do not overshoot the final time

        # Time stepping
        U = time_step(U, dt, gamma, order, limiter, time_solver, boundary, dx, dy, dz)
        t += dt
        step += 1

        print(f"Step {step:5d}, t = {t:.5f}, dt = {dt:.5e}")

        # Save snapshot if the current time matches the snapshot time
        if t >= t_snapshot:
            print(f"Start saving snapshot at t = {t:.5f}")
            # Call the updated write_snapshot function to save the full state to HDF5
            write_snapshot(U, cfg, t, snapshot_idx)
            snapshot_idx += 1

            # Ensure we don't overshoot the snapshot time
            if snapshot_idx < len(snapshot_times):
                t_snapshot = snapshot_times[snapshot_idx]
            else:
                t_snapshot = cfg.t_end  # Once all snapshots are saved, set t_snapshot to the end time
            
            print("Finished saving snapshot")

    # End the timer and calculate the total runtime
    end_time = time.time()  # Record the end time of the simulation
    total_time = end_time - start_time  # Calculate the total runtime

    print(f"Done! Took {total_time:.5f} seconds! Good Bye and Good Day!!!")

    # Plotting phase
    if cfg.PLOT_AFTER_SIM:
        if cfg.SPATIAL_DIMENSION == 1:
            print("Generating 1D fluid quantity plots...")
            from utils.plot_1d_fluid_quants import plot_1d_fluid_quantities
            plot_1d_fluid_quantities(cfg.output_dir, cfg.base_name)
        else:
            print("Generating density plots from all HDF5 snapshots...")
            from utils.plot_density2d import plot_density2d
            plot_density2d(cfg.output_dir, cfg.base_name)

if __name__ == "__main__":
    main()

