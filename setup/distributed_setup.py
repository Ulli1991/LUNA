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
from typing import Callable, Tuple

import jax
from jax import Array
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from file_io.checkpoint_utils import validate_checkpoint_compatibility
from file_io.read_checkpoint import find_latest_checkpoint, read_checkpoint
from hydro.physics_checks import validate_state
from ic_generation.ic import initial_conditions
from time_integration.integrators import create_muscl_hancock_modular_step
from utils.distributed import (
    compute_local_slice,
    create_device_mesh,
    create_sharded_array_from_local,
    shard_array_to_devices,
)
from utils.grid_params import GridParams
from utils.halo_exchange_auto import pad_with_halo_auto
from utils.memory_utils import print_memory_usage


def initialize_jax_distributed(enable_x64: bool = False):
    """
    Initialize JAX for distributed multi-node execution.

    Uses SLURM environment variables for coordination.

    Parameters
    ----------
    enable_x64 : bool
        If True, enable float64. Default False (float32) for production
        performance; flip to True for convergence tests where the
        perturbation amplitude is near the float32 round-off floor.
    """
    print("=" * 60)
    print("MULTI-NODE SHARDING MODE (1 GPU per process)")
    print("=" * 60)

    try:
        jax.distributed.initialize(
            coordinator_address=os.environ.get("JAX_COORDINATOR_ADDRESS"),
            num_processes=int(os.environ.get("SLURM_NTASKS", 1)),
            process_id=int(os.environ.get("SLURM_PROCID", 0)),
        )
        print("✓ JAX distributed initialized")
        print(f"  Process: {jax.process_index()} / {jax.process_count()}")
    except (RuntimeError, ValueError) as e:
        print("  Running in single-process mode (use SLURM/MPI for multi-node)")
        print(f"  Error: {e}")

    # Configure JAX
    jax.config.update("jax_enable_x64", enable_x64)
    jax.config.update("jax_default_matmul_precision", "high")
    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.jax_cache"))
    if enable_x64:
        print("✓ float64 enabled (jax_enable_x64=True)")

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
    is_main_process = jax.process_index() == 0

    n_devices = len(jax.devices())
    if n_devices < 1:
        print("ERROR: No devices available!")
        sys.exit(1)

    if is_main_process:
        print("=" * 60)
        print(f"DOMAIN DECOMPOSITION ({n_devices} devices)")
        print("=" * 60)

    mesh, grid_shape = create_device_mesh(
        n_devices,
        cfg.SPATIAL_DIMENSION,
        decomposition=cfg.DECOMPOSITION,
        Nx=cfg.Nx,
        Ny=cfg.Ny,
        mesh_shape=getattr(cfg, "MESH_SHAPE", None),
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
    is_main_process = jax.process_index() == 0
    process_id = jax.process_index()

    if is_main_process:
        print("=" * 60)

    if cfg.restart:
        # === RESTART FROM CHECKPOINT ===
        if is_main_process:
            print("RESTART FROM CHECKPOINT")
            print("=" * 60)

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
            print("=" * 60)

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
                    cfg.Nx, cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else 1, mesh, process_id
                )

            # Generate only the local portion via the unified IC dispatcher.
            # All IC functions are sharded-aware and accept the relevant slice
            # kwargs for their dimensionality; the dispatcher routes them.
            try:
                U_local = initial_conditions(cfg, x_slice=x_slice, y_slice=y_slice, z_slice=z_slice)
            except ValueError as e:
                if is_main_process:
                    print(f"\n{'='*60}")
                    print(f"ERROR: {e}")
                    print(f"{'='*60}\n")
                sys.exit(1)

            if is_main_process:
                print("Local shape per process: varies by decomposition")

            # Create sharded array from local pieces
            # FIX: Use proper number of variables based on spatial dimension
            if cfg.SPATIAL_DIMENSION == 1:
                global_shape = (3, cfg.Nx)  # [rho, rho*u, E]
            elif cfg.SPATIAL_DIMENSION == 2:
                global_shape = (4, cfg.Nx, cfg.Ny)  # [rho, rho*ux, rho*uy, E]
            else:  # 3D
                global_shape = (5, cfg.Nx, cfg.Ny, cfg.Nz)  # [rho, rho*ux, rho*uy, rho*uz, E]

            U_sharded = create_sharded_array_from_local(
                U_local, global_shape, mesh, cfg.SPATIAL_DIMENSION
            )

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
        print("  Each process keeps only its local shard")
        print()
        print_memory_usage("After initialization: ")
        print()

    return U_sharded, t_start, step_start


def create_solver_distributed(
    cfg, grid: GridParams, mesh: Mesh
) -> Tuple[Callable, str, NamedSharding]:
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
    is_main_process = jax.process_index() == 0

    # Create NamedSharding for automatic sharding
    if cfg.SPATIAL_DIMENSION == 1:
        sharding = NamedSharding(mesh, P(None, "x"))
    elif cfg.SPATIAL_DIMENSION == 2:
        sharding = NamedSharding(mesh, P(None, "x", "y"))
    else:  # 3D
        sharding = NamedSharding(mesh, P(None, "x", "y", "z"))

    # Pick the halo exchange path: shard_map (explicit ppermute, opt-in via
    # cfg.USE_SHARDMAP) or the legacy auto-sharded jnp.pad. shard_map only
    # supports periodic/zero-grad 3D at the moment (phase 1) and falls back
    # to the auto path otherwise — so KH 2D, solar (reflective), etc. all
    # keep working until phase 2/4 of the shard_map plan.
    use_shardmap = bool(getattr(cfg, "USE_SHARDMAP", False))
    if use_shardmap and cfg.SPATIAL_DIMENSION == 3 and cfg.BOUNDARY in (0, 1):
        from utils.halo_exchange_shardmap import make_halo_3d_shardmap

        pad_halo_fn = make_halo_3d_shardmap(mesh)
        if is_main_process:
            print("Using shard_map with explicit per-shard ppermute halos")
            print(f"Sharding spec: {sharding.spec}")
    else:
        if use_shardmap and is_main_process:
            print(
                "shard_map requested but not eligible "
                f"(SPATIAL_DIMENSION={cfg.SPATIAL_DIMENSION}, BOUNDARY={cfg.BOUNDARY}) "
                "— falling back to auto-sharded jnp.pad."
            )
        use_shardmap = False
        if is_main_process:
            print("Using automatic sharding with NamedSharding")
            print(f"Sharding spec: {sharding.spec}")
            print("Halo exchange: jnp.roll (standard JAX approach)")

        def pad_halo_fn(U, boundary, spatial_dim):
            return pad_with_halo_auto(U, boundary, spatial_dim)

    # Select time integrator. Each factory returns (compute_dt_fn, step_fn):
    #   compute_dt_fn(U) -> dt_cfl   (pure CFL, runner does the clamping)
    #   step_fn(U, dt)   -> U_new
    # All three share the same modular pattern (Python orchestration of
    # small @jax.jit subfunctions) — the one that works multi-node.
    # MUSCL-Hancock is the only supported integrator. Optional well-balanced
    # gravity path uses the same scheme with an equilibrium-flux subtraction.
    if getattr(cfg, "use_well_balanced", False):
        from ic_generation.ic import equilibrium_state
        from time_integration.wb_integrator import create_wb_muscl_hancock_step
        from utils.distributed import compute_local_slice

        if jax.process_count() > 1:
            local_slices = compute_local_slice(cfg.Nx, cfg.Ny, mesh, jax.process_index(), cfg.Nz)
            U_eq_local = equilibrium_state(
                cfg,
                x_slice=local_slices[0],
                y_slice=local_slices[1],
                z_slice=local_slices[2],
            )
            if cfg.SPATIAL_DIMENSION == 2:
                u_eq_global_shape = (4, cfg.Nx, cfg.Ny)
            else:
                u_eq_global_shape = (5, cfg.Nx, cfg.Ny, cfg.Nz)
            U_eq = create_sharded_array_from_local(
                U_eq_local, u_eq_global_shape, mesh, cfg.SPATIAL_DIMENSION
            )
        else:
            U_eq_global = equilibrium_state(cfg)
            U_eq = jax.device_put(U_eq_global, sharding)
        compute_dt_fn, step_fn = create_wb_muscl_hancock_step(
            pad_halo_fn,
            cfg.gamma,
            cfg.ORDER,
            cfg.LIMITER,
            grid.dx,
            grid.dy,
            grid.dz,
            grid.inv_dx,
            grid.inv_dy,
            grid.inv_dz,
            cfg.BOUNDARY,
            cfg.SPATIAL_DIMENSION,
            cfg.CFL,
            U_eq=U_eq,
            gravity_z=cfg.gravity_z,
            sharding=sharding,
            riemann_solver=getattr(cfg, "riemann_solver", "hllc"),
        )
        integrator_name = (
            f"WB-MUSCL-Hancock "
            f"({getattr(cfg, 'riemann_solver', 'hllc').upper()}, "
            f"g_z={cfg.gravity_z})"
        )
    else:
        compute_dt_fn, step_fn = create_muscl_hancock_modular_step(
            pad_halo_fn,
            cfg.gamma,
            cfg.ORDER,
            cfg.LIMITER,
            grid.dx,
            grid.dy,
            grid.dz,
            grid.inv_dx,
            grid.inv_dy,
            grid.inv_dz,
            cfg.BOUNDARY,
            cfg.SPATIAL_DIMENSION,
            cfg.CFL,
            sharding=sharding,
            riemann_solver=getattr(cfg, "riemann_solver", "hllc"),
            use_shardmap=use_shardmap,
            mesh=mesh if use_shardmap else None,
        )
        integrator_name = (
            f"MUSCL-Hancock ({getattr(cfg, 'riemann_solver', 'hllc').upper()}"
            + (", shard_map" if use_shardmap else "")
            + ")"
        )

    if is_main_process:
        print(f"Using {integrator_name} time integration")
        print("Runner controls dt clamping (snapshot times, t_end)")

    return compute_dt_fn, step_fn, integrator_name, sharding
