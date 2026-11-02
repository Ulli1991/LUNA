"""The end of a gas run: the final statistics and, with cfg.PLOT_AFTER_SIM,
the plots. Called from Outputs.finish (luna/setup/outputs.py)."""
import os

import jax
import numpy as np
from jax.experimental import multihost_utils

from luna.utils.memory_utils import print_memory_usage


def finalize_simulation_distributed(state, cfg, grid, mesh, step_start=0, is_main=None):
    """Print the run's statistics and, when asked, generate the plots."""
    is_main_process = jax.process_index() == 0 if is_main is None else is_main
    process_id = jax.process_index()
    process_count = jax.process_count()
    t_final, step_final = state.t, state.step
    total_time = float(state.stats.get("wall_s", 0.0))
    grid_shape = tuple(int(v) for v in mesh.shape.values())

    if is_main_process:

        print("\n" + "=" * 60)
        print("SIMULATION COMPLETED")
        print("=" * 60)
        print(f"✓ Total runtime: {total_time:.2f} seconds")
        print(f"✓ Total steps: {step_final}")
        print(f"✓ Final time: {t_final:.5f}")

        # From the first perf line on (luna/setup/outputs.py): the first step's
        # compile is not in it; snapshots and diagnostics are.
        n_timed = int(state.stats.get("timed_steps", 0))
        timed_s = float(state.stats.get("timed_s", 0.0))
        if n_timed > 0 and timed_s > 0.0:
            rate = grid.total_cells * n_timed / timed_s
            print(f"✓ Performance after step {step_final - n_timed} (no compile, output "
                  f"included): {rate / 1e6:.4g} Mcells/s (global grid), "
                  f"{1e3 * timed_s / n_timed:.4g} ms/step over {n_timed} steps")
        print()

        print(f"Number of devices: {len(jax.devices())}")
        print(f"Process grid: {grid_shape}")
        print()

        print_memory_usage("Final memory usage: ")
        print()

    # Plotting phase - ALL processes participate for parallel plotting!
    if cfg.PLOT_AFTER_SIM:
        if is_main_process:
            print("=" * 60)
            print("GENERATING PLOTS (PARALLEL)")
            print("=" * 60)

        n_plots = 0
        if cfg.SPATIAL_DIMENSION == 1 or cfg.SPATIAL_DIMENSION == 3:
            if is_main_process:
                print("Generating 1D fluid quantity plots...")
                from luna.utils.plot_1d_fluid_quants import plot_1d_fluid_quantities

                n_plots = plot_1d_fluid_quantities(cfg.output_dir, cfg.base_name)
        elif cfg.SPATIAL_DIMENSION == 2:
            if is_main_process:
                print(f"All {process_count} processes generating density plots in parallel...")
            from luna.utils.plot_density2d import plot_density2d

            n_plots = plot_density2d(
                cfg.output_dir, cfg.base_name, process_id=process_id, num_processes=process_count
            )

        # every process waits here for the others' plots, and learns their count
        n_plots = int(np.sum(multihost_utils.process_allgather(np.int32(n_plots))))

        if is_main_process:
            if n_plots:
                print(f"✓ {n_plots} plot(s) written to {os.path.join(cfg.output_dir, 'png')}")
            else:
                print(f"✗ No plots generated: no snapshots under {cfg.output_dir}")
            print()

    if is_main_process:
        print("=" * 60)
        print("Good Bye and Good Day!")
        print("=" * 60)
