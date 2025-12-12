"""
Finalization and post-processing for distributed simulations.

This module handles:
- Final statistics and reporting
- Plotting (parallel where possible)
- Cleanup
"""
import time as pytime
import jax
from jax.experimental import multihost_utils
from utils.memory_utils import print_memory_usage


def finalize_simulation_distributed(U_sharded, t_final, step_final, start_time, cfg, grid, grid_shape):
    """
    Finalize simulation: print statistics and generate plots.

    Parameters
    ----------
    U_sharded : array
        Final sharded state
    t_final : float
        Final time
    step_final : int
        Final step
    start_time : float
        Start time (wall clock)
    cfg : Config
        Configuration object
    grid : GridParams
        Grid parameters
    grid_shape : tuple
        Process grid shape
    """
    # Cache process info
    is_main_process = (jax.process_index() == 0)
    process_id = jax.process_index()
    process_count = jax.process_count()

    if is_main_process:
        end_time = pytime.time()
        total_time = end_time - start_time

        print("\n" + "="*60)
        print("SIMULATION COMPLETED")
        print("="*60)
        print(f"✓ Total runtime: {total_time:.2f} seconds")
        print(f"✓ Total steps: {step_final}")
        print(f"✓ Final time: {t_final:.5f}")

        if step_final > 1:
            total_cell_updates = grid.total_cells * (step_final - 1)
            avg_cells_per_sec = total_cell_updates / total_time
            avg_mcells_per_sec = avg_cells_per_sec / 1e6
            print(f"✓ Average performance: {avg_mcells_per_sec:.2f} Mcells/s (global grid)")
            print(f"✓ Time per step: {total_time/(step_final-1)*1000:.2f} ms (excluding first step)")
        print()

        print(f"Number of devices: {len(jax.devices())}")
        print(f"Process grid: {grid_shape}")
        print()

        print_memory_usage("Final memory usage: ")
        print()

    # Plotting phase - ALL processes participate for parallel plotting!
    if cfg.PLOT_AFTER_SIM:
        if is_main_process:
            print("="*60)
            print("GENERATING PLOTS (PARALLEL)")
            print("="*60)

        if cfg.SPATIAL_DIMENSION == 1 or cfg.SPATIAL_DIMENSION == 3:
            if is_main_process:
                print("Generating 1D fluid quantity plots...")
                from utils.plot_1d_fluid_quants import plot_1d_fluid_quantities
                plot_1d_fluid_quantities(cfg.output_dir, cfg.base_name)
                print("✓ Plots generated successfully")
        elif cfg.SPATIAL_DIMENSION == 2:
            if is_main_process:
                print(f"All {process_count} processes generating density plots in parallel...")
            from utils.plot_density2d import plot_density2d
            plot_density2d(cfg.output_dir, cfg.base_name,
                          process_id=process_id,
                          num_processes=process_count)

        # Synchronize all processes after plotting completes
        multihost_utils.sync_global_devices("plotting_complete")

        if is_main_process:
            print("✓ All plots generated successfully")
            print()

    if is_main_process:
        print("="*60)
        print("Good Bye and Good Day!")
        print("="*60)
