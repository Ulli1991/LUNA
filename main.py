"""
Multi-node distributed JAX hydro solver.

This is the main entry point for distributed multi-node simulations using
JAX's automatic sharding with NamedSharding.
"""
from config import Config
from utils.grid_params import GridParams
from utils.logo import print_LUNA_banner
from setup import (
    initialize_jax_distributed,
    create_distributed_mesh,
    initialize_simulation_distributed,
    create_solver_distributed,
    run_simulation_distributed,
    finalize_simulation_distributed
)


def main():
    """Main entry point for distributed simulation."""
    # === Initialize JAX distributed ===
    initialize_jax_distributed()

    # === Print logo (only process 0) ===
    import jax
    if jax.process_index() == 0:
        print_LUNA_banner()

    # === Load configuration ===
    cfg = Config()

    # === Create device mesh for domain decomposition ===
    mesh, grid_shape = create_distributed_mesh(cfg)

    # === Initialize or restart simulation ===
    U_sharded, t_start, step_start = initialize_simulation_distributed(cfg, mesh)

    # === Create grid parameters (immutable, pre-computed constants) ===
    grid = GridParams.from_config(cfg)

    # === Create solver (time integrator with integrated dt computation) ===
    time_step_fn, integrator_name, sharding = create_solver_distributed(cfg, grid, mesh)

    # === Run simulation ===
    U_sharded, t_final, step_final, start_time = run_simulation_distributed(
        U_sharded, t_start, step_start, cfg, grid,
        time_step_fn, sharding
    )

    # === Finalize (statistics, plotting, cleanup) ===
    finalize_simulation_distributed(
        U_sharded, t_final, step_final, start_time, cfg, grid, grid_shape
    )


if __name__ == "__main__":
    main()
