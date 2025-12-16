"""
Distributed simulation setup and initialization for multi-node JAX.

This module handles:
- JAX distributed initialization
- Device mesh creation
- Initial conditions or checkpoint restart with proper sharding
- Solver creation for distributed execution
"""
import os
import sys
from typing import Tuple, Callable
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax import Array

from file_io.read_checkpoint import read_checkpoint, find_latest_checkpoint
from file_io.checkpoint_utils import validate_checkpoint_compatibility
from hydro.physics_checks import validate_state
from ic_generation.ic import initial_conditions
from utils.memory_utils import print_memory_usage
from utils.grid_params import GridParams
from utils.distributed import (
    create_device_mesh,
    shard_array_to_devices,
    compute_local_slice,
    create_sharded_array_from_local
)
from time_integration.sharded_integrators import (
    create_rk2_sharded_step,
    create_muscl_hancock_sharded_step,
    create_rk4_sharded_step
)
from time_integration.sharded_integrators_modular import (
    create_muscl_hancock_modular_step
)
from time_integration.sharded_integrators_reference_style import (
    create_muscl_hancock_reference_style
)
from time_integration.cfl import create_compute_dt_sharded
from utils.halo_exchange_auto import pad_with_halo_auto
from hydro.divergence import flux_divergence_from_padded


def initialize_jax_distributed():
    """
    Initialize JAX for distributed multi-node execution.

    Uses SLURM environment variables for coordination.
    """
    print("="*60)
    print("MULTI-NODE SHARDING MODE (1 GPU per process)")
    print("="*60)

    try:
        jax.distributed.initialize(
            coordinator_address=os.environ.get('JAX_COORDINATOR_ADDRESS'),
            num_processes=int(os.environ.get('SLURM_NTASKS', 1)),
            process_id=int(os.environ.get('SLURM_PROCID', 0)),
        )
        print(f"✓ JAX distributed initialized")
        print(f"  Process: {jax.process_index()} / {jax.process_count()}")
    except (RuntimeError, ValueError) as e:
        print(f"  Running in single-process mode (use SLURM/MPI for multi-node)")
        print(f"  Error: {e}")

    # Configure JAX
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", "high")
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

    print(f"Platform: {jax.default_backend()}")
    print(f"Total devices: {len(jax.devices())}")
    print(f"Local devices: {len(jax.local_devices())}")

    # Show device topology
    for i, dev in enumerate(jax.devices()):
        print(f"  Device {i}: {dev.device_kind} (process {dev.process_index})")
    print()


def create_distributed_mesh(cfg) -> Tuple[Mesh, Tuple[int, ...]]:
    """
    Create device mesh for domain decomposition.

    Parameters
    ----------
    cfg : Config
        Configuration object

    Returns
    -------
    mesh : Mesh
        JAX mesh object
    grid_shape : tuple
        Process grid shape (e.g., (2, 4) for 2x4 decomposition)
    """
    # Cache process info
    is_main_process = (jax.process_index() == 0)

    n_devices = len(jax.devices())
    if n_devices < 1:
        print(f"ERROR: No devices available!")
        sys.exit(1)

    if is_main_process:
        print("="*60)
        print(f"DOMAIN DECOMPOSITION ({n_devices} devices)")
        print("="*60)

    mesh, grid_shape = create_device_mesh(
        n_devices,
        cfg.SPATIAL_DIMENSION,
        decomposition=cfg.DECOMPOSITION,
        Nx=cfg.Nx,
        Ny=cfg.Ny
    )

    if is_main_process:
        print(f"Decomposition strategy: {cfg.DECOMPOSITION}")
        print(f"Process grid: {grid_shape}")
        decomp_type = "slab (1D)" if grid_shape[-1] == 1 else "block (2D/3D)"
        print(f"Actual decomposition: {decomp_type}")
        print()

    return mesh, grid_shape


def initialize_simulation_distributed(cfg, mesh: Mesh) -> Tuple[Array, float, int]:
    """
    Initialize or restart simulation with proper sharding for distributed execution.

    Parameters
    ----------
    cfg : Config
        Configuration object
    mesh : Mesh
        JAX mesh for sharding

    Returns
    -------
    U_sharded : Array
        Sharded initial state (conserved variables)
    t_start : float
        Start time
    step_start : int
        Start step
    """
    # Cache process info
    is_main_process = (jax.process_index() == 0)
    process_id = jax.process_index()

    if is_main_process:
        print("="*60)

    if cfg.restart:
        # === RESTART FROM CHECKPOINT ===
        if is_main_process:
            print("RESTART FROM CHECKPOINT")
            print("="*60)

        # Find checkpoint file
        if cfg.restart_file is not None:
            checkpoint_file = cfg.restart_file
        else:
            checkpoint_file = find_latest_checkpoint(cfg.checkpoint_dir)
            if checkpoint_file is None:
                sys.exit(f"ERROR: No checkpoint found in {cfg.checkpoint_dir}")

        # Load checkpoint (all processes load it)
        checkpoint_data = read_checkpoint(checkpoint_file)

        # Validate compatibility
        validate_checkpoint_compatibility(checkpoint_data, cfg)

        # Extract state
        U_global = checkpoint_data["U"]
        t_start = checkpoint_data["time"]
        step_start = checkpoint_data["step"]

        if is_main_process:
            print(f"✓ Restarting from t = {t_start:.5f}, step = {step_start}")
            print()
            print("Sharding global array to all devices...")

        # FIX: Shard the global array (was missing in original code!)
        U_sharded = shard_array_to_devices(U_global, mesh, cfg.SPATIAL_DIMENSION)

    else:
        # === INITIAL CONDITIONS ===
        if is_main_process:
            print("INITIAL CONDITIONS")
            print("="*60)

        t_start = cfg.t_start
        step_start = 0

        # In multi-process mode, ALWAYS use local generation (saves memory)
        if jax.process_count() > 1:
            if is_main_process:
                print("Multi-process mode: using local IC generation")
                print("Each process generates only its local portion (memory efficient)")

            # Compute which slice this process should generate
            if cfg.SPATIAL_DIMENSION == 3:
                x_slice, y_slice, z_slice = compute_local_slice(
                    cfg.Nx, cfg.Ny, mesh, process_id, cfg.Nz
                )
            else:
                x_slice, y_slice, z_slice = compute_local_slice(
                    cfg.Nx, cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else 1,
                    mesh, process_id
                )

            # Generate only local portion
            if cfg.test_problem == "Kelvin-Helmholtz":
                from ic_generation.kh import kelvin_helmholtz
                U_local = kelvin_helmholtz(cfg, x_slice=x_slice, y_slice=y_slice)
            elif cfg.test_problem == "Sod-Shock":
                from ic_generation.sod import sod_shock
                U_local = sod_shock(cfg, x_slice=x_slice)
            elif cfg.test_problem == "Sod-Shock-3D":
                from ic_generation.sod3d import sod_shock_3d
                U_local = sod_shock_3d(cfg, x_slice=x_slice, y_slice=y_slice, z_slice=z_slice)
            else:
                if is_main_process:
                    print(f"\n{'='*60}")
                    print(f"ERROR: Local IC generation not implemented for '{cfg.test_problem}'")
                    print(f"{'='*60}")
                    print(f"Multi-process mode requires local IC generation.")
                    print(f"Currently supported: Kelvin-Helmholtz, Sod-Shock, Sod-Shock-3D")
                    print(f"\nOptions:")
                    print(f"  1. Use one of the supported test problems")
                    print(f"  2. Run in single-process mode (1 GPU)")
                    print(f"  3. Implement local generation for {cfg.test_problem}")
                    print(f"{'='*60}\n")
                sys.exit(1)

            if is_main_process:
                print(f"Local shape per process: varies by decomposition")

            # Create sharded array from local pieces
            # FIX: Use proper number of variables based on spatial dimension
            if cfg.SPATIAL_DIMENSION == 1:
                global_shape = (3, cfg.Nx)  # [rho, rho*u, E]
            elif cfg.SPATIAL_DIMENSION == 2:
                global_shape = (4, cfg.Nx, cfg.Ny)  # [rho, rho*ux, rho*uy, E]
            else:  # 3D
                global_shape = (5, cfg.Nx, cfg.Ny, cfg.Nz)  # [rho, rho*ux, rho*uy, rho*uz, E]

            U_sharded = create_sharded_array_from_local(U_local, global_shape, mesh, cfg.SPATIAL_DIMENSION)

            if is_main_process:
                print("✓ Sharded array created from local pieces (no validation for large arrays)")
                print()

        else:
            # Single-process mode: traditional global generation
            if is_main_process:
                print("Single-process mode: generating global initial conditions...")

            U_global = initial_conditions(cfg)

            if is_main_process:
                print(f"Global shape: {U_global.shape}")
                print("Validating initial conditions...")
                validate_state(U_global, cfg.gamma, step=0, time=cfg.t_start)
                print("✓ Initial conditions validated successfully")
                print()
                print("Sharding global array to all devices...")

            # Shard the global array to all devices
            U_sharded = shard_array_to_devices(U_global, mesh, cfg.SPATIAL_DIMENSION)

    if is_main_process:
        print(f"✓ Array sharded across {len(jax.devices())} devices")
        print(f"  Each process keeps only its local shard")
        print()
        print_memory_usage("After initialization: ")
        print()

    return U_sharded, t_start, step_start


def create_solver_distributed(cfg, grid: GridParams, mesh: Mesh) -> Tuple[Callable, str, NamedSharding]:
    """
    Create time stepping function with integrated dt computation for distributed execution.

    Parameters
    ----------
    cfg : Config
        Configuration object
    grid : GridParams
        Grid parameters
    mesh : Mesh
        JAX mesh for sharding

    Returns
    -------
    time_step_fn : Callable
        Combined time stepping function: (U, t, t_snapshot, t_end) -> (U_new, dt)
        Computes dt and performs time step in ONE function call (performance optimization)
    integrator_name : str
        Name of the integrator (e.g., "RK2", "MUSCL-Hancock", "RK4")
    sharding : NamedSharding
        Sharding specification for arrays
    """
    # Cache process info
    is_main_process = (jax.process_index() == 0)

    # Create NamedSharding for automatic sharding
    if cfg.SPATIAL_DIMENSION == 1:
        sharding = NamedSharding(mesh, P(None, 'x'))
    elif cfg.SPATIAL_DIMENSION == 2:
        sharding = NamedSharding(mesh, P(None, 'x', 'y'))
    else:  # 3D
        sharding = NamedSharding(mesh, P(None, 'x', 'y', 'z'))

    if is_main_process:
        print(f"Using automatic sharding with NamedSharding")
        print(f"Sharding spec: {sharding.spec}")
        print(f"Halo exchange: jnp.roll (standard JAX approach)")

    # Use standard JAX halo exchange
    def pad_halo_fn(U, boundary, spatial_dim):
        return pad_with_halo_auto(U, boundary, spatial_dim)

    # Select time integrator
    if cfg.TIME_SOLVER == 0:
        # RK2
        time_step_fn = create_rk2_sharded_step(
            pad_halo_fn, flux_divergence_from_padded,
            cfg.gamma, cfg.ORDER, cfg.LIMITER,
            grid.dx, grid.dy, grid.dz, grid.inv_dx, grid.inv_dy, grid.inv_dz,
            cfg.BOUNDARY, cfg.SPATIAL_DIMENSION
        )
        integrator_name = "RK2"
    elif cfg.TIME_SOLVER == 1:
        # MUSCL-Hancock - choose JIT strategy
        if cfg.JIT_STRATEGY == "reference":
            # Match reference code: main function NOT @jax.jit, many small @jax.jit helpers
            time_step_fn = create_muscl_hancock_reference_style(
                cfg.gamma, cfg.ORDER, cfg.LIMITER,
                grid.dx, grid.dy, grid.dz, grid.inv_dx, grid.inv_dy, grid.inv_dz,
                cfg.BOUNDARY, cfg.SPATIAL_DIMENSION, cfg.CFL
            )
            integrator_name = "MUSCL-Hancock (REFERENCE STYLE: many small @jax.jit)"
        elif cfg.JIT_STRATEGY == "modular":
            # Few large @jax.jit functions
            time_step_fn = create_muscl_hancock_modular_step(
                pad_halo_fn, flux_divergence_from_padded,
                cfg.gamma, cfg.ORDER, cfg.LIMITER,
                grid.dx, grid.dy, grid.dz, grid.inv_dx, grid.inv_dy, grid.inv_dz,
                cfg.BOUNDARY, cfg.SPATIAL_DIMENSION, cfg.CFL
            )
            integrator_name = "MUSCL-Hancock (MODULAR: few large @jax.jit)"
        else:  # "monolithic"
            # One big @jax.jit function (original)
            time_step_fn = create_muscl_hancock_sharded_step(
                pad_halo_fn, flux_divergence_from_padded,
                cfg.gamma, cfg.ORDER, cfg.LIMITER,
                grid.dx, grid.dy, grid.dz, grid.inv_dx, grid.inv_dy, grid.inv_dz,
                cfg.BOUNDARY, cfg.SPATIAL_DIMENSION, cfg.CFL
            )
            integrator_name = "MUSCL-Hancock (MONOLITHIC: one big @jax.jit)"
    elif cfg.TIME_SOLVER == 2:
        # RK4
        time_step_fn = create_rk4_sharded_step(
            pad_halo_fn, flux_divergence_from_padded,
            cfg.gamma, cfg.ORDER, cfg.LIMITER,
            grid.dx, grid.dy, grid.dz, grid.inv_dx, grid.inv_dy, grid.inv_dz,
            cfg.BOUNDARY, cfg.SPATIAL_DIMENSION
        )
        integrator_name = "RK4"
    else:
        if is_main_process:
            print(f"ERROR: TIME_SOLVER={cfg.TIME_SOLVER} not supported")
        sys.exit(1)

    if is_main_process:
        print(f"Using {integrator_name} time integration")
        print(f"Note: dt computation now integrated into time_step (1 call per timestep)")

    # No longer need separate compute_dt_fn - it's integrated into time_step_fn
    return time_step_fn, integrator_name, sharding
