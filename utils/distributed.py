"""
Distributed computing utilities for JAX multi-node execution.

Functions for creating device meshes, sharding arrays, and gathering results.
"""

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def compute_local_slice(Nx, Ny, mesh, process_index, Nz=1):
    """Compute this process's slice of the global (Nx, Ny, Nz) grid.

    Returns ((x0, x1), (y0, y1), (z0, z1)) start/end index pairs.
    """
    # Get grid shape as integers from mesh
    # mesh.shape is a dict with axis names, mesh.devices.shape gives actual dimensions
    grid_shape = mesh.devices.shape

    if len(grid_shape) == 1:  # 1D decomposition
        nx_proc = grid_shape[0]
        # Determine x-slice for this process
        x_per_proc = Nx // nx_proc
        x_start = process_index * x_per_proc
        x_end = (process_index + 1) * x_per_proc if process_index < nx_proc - 1 else Nx
        return (x_start, x_end), (0, Ny), (0, Nz)

    elif len(grid_shape) == 2:  # 2D decomposition
        nx_proc, ny_proc = grid_shape
        # Convert linear process_index to 2D grid coordinates
        proc_x = process_index // ny_proc
        proc_y = process_index % ny_proc

        # Determine x-slice
        x_per_proc = Nx // nx_proc
        x_start = proc_x * x_per_proc
        x_end = (proc_x + 1) * x_per_proc if proc_x < nx_proc - 1 else Nx

        # Determine y-slice
        y_per_proc = Ny // ny_proc
        y_start = proc_y * y_per_proc
        y_end = (proc_y + 1) * y_per_proc if proc_y < ny_proc - 1 else Ny

        return (x_start, x_end), (y_start, y_end), (0, Nz)

    else:  # 3D decomposition
        nx_proc, ny_proc, nz_proc = grid_shape
        # Convert linear process_index to 3D grid coordinates
        proc_x = process_index // (ny_proc * nz_proc)
        proc_y = (process_index % (ny_proc * nz_proc)) // nz_proc
        proc_z = process_index % nz_proc

        # Determine x-slice
        x_per_proc = Nx // nx_proc
        x_start = proc_x * x_per_proc
        x_end = (proc_x + 1) * x_per_proc if proc_x < nx_proc - 1 else Nx

        # Determine y-slice
        y_per_proc = Ny // ny_proc
        y_start = proc_y * y_per_proc
        y_end = (proc_y + 1) * y_per_proc if proc_y < ny_proc - 1 else Ny

        # Determine z-slice
        z_per_proc = Nz // nz_proc
        z_start = proc_z * z_per_proc
        z_end = (proc_z + 1) * z_per_proc if proc_z < nz_proc - 1 else Nz

        return (x_start, x_end), (y_start, y_end), (z_start, z_end)


def create_device_mesh(
    n_devices, spatial_dimension, decomposition="auto", Nx=None, Ny=None, mesh_shape=None
):
    """Create a JAX device mesh; returns (mesh, grid_shape).

    decomposition: "auto" (heuristic), "slab" (1D), or "block" (2D/3D).
    mesh_shape, if given, bypasses the heuristic: explicit per-axis process
    grid whose length must match spatial_dimension and product n_devices.
    """
    # Explicit mesh shape (AthenaK-style): user picks the per-axis decomposition.
    if mesh_shape is not None:
        if len(mesh_shape) != spatial_dimension:
            raise ValueError(
                f"mesh_shape {mesh_shape} length must match SPATIAL_DIMENSION={spatial_dimension}"
            )
        prod = 1
        for d in mesh_shape:
            prod *= d
        if prod != n_devices:
            raise ValueError(
                f"mesh_shape {mesh_shape} product ({prod}) must equal n_devices ({n_devices})"
            )
        axis_names = ("x",)[:spatial_dimension] if spatial_dimension == 1 else (
            ("x", "y") if spatial_dimension == 2 else ("x", "y", "z")
        )
        devices = mesh_utils.create_device_mesh(tuple(mesh_shape))
        mesh = Mesh(devices, axis_names)
        return mesh, tuple(mesh_shape)

    # Auto-select decomposition strategy based on problem size and GPU count
    if decomposition == "auto":
        if spatial_dimension == 1:
            decomposition = "slab"  # Only option for 1D
        elif spatial_dimension == 2:
            # Heuristic: Use slab for small-medium GPU counts, block for large
            # Also consider problem size: use slab if N/P > 512 per dimension
            if n_devices <= 16:
                # Small GPU count: slab is better (minimizes latency)
                decomposition = "slab"
            elif n_devices >= 64:
                # Large GPU count: must use block decomposition
                decomposition = "block"
            elif Nx is not None and Ny is not None:
                # Medium GPU count: check if slab keeps enough cells per GPU
                cells_per_gpu_slab = (Nx // n_devices) * Ny
                cells_per_gpu_block = (Nx // int(jnp.sqrt(n_devices))) * (
                    Ny // int(jnp.sqrt(n_devices))
                )
                # Use slab if it gives > 1M cells per GPU or > 512 cells per dimension
                if (Nx // n_devices) >= 512 or cells_per_gpu_slab >= 1_000_000:
                    decomposition = "slab"
                else:
                    decomposition = "block"
            else:
                # Default to slab for medium counts
                decomposition = "slab"
        else:  # 3D
            # For 3D: prefer slab or 2D decomposition unless GPU count is very large
            if n_devices <= 16:
                decomposition = "slab"  # 1D pencil
            else:
                decomposition = "block"  # 2D or 3D

    # Create mesh based on selected strategy
    if spatial_dimension == 1:
        # 1D problem: only 1D decomposition possible
        devices = mesh_utils.create_device_mesh((n_devices,))
        mesh = Mesh(devices, ("x",))
        grid_shape = (n_devices,)

    elif spatial_dimension == 2:
        if decomposition == "slab":
            # 1D slab decomposition: all devices along one axis
            # This minimizes communication messages (2 neighbors per GPU)
            devices = mesh_utils.create_device_mesh((n_devices, 1))
            mesh = Mesh(devices, ("x", "y"))
            grid_shape = (n_devices, 1)
        else:  # "block"
            # 2D block decomposition: spread devices in 2D grid
            # This minimizes surface-to-volume ratio for large problems
            nx_proc = int(jnp.sqrt(n_devices))
            while n_devices % nx_proc != 0:
                nx_proc -= 1
            ny_proc = n_devices // nx_proc

            devices = mesh_utils.create_device_mesh((nx_proc, ny_proc))
            mesh = Mesh(devices, ("x", "y"))
            grid_shape = (nx_proc, ny_proc)

    elif spatial_dimension == 3:
        if decomposition == "slab":
            # 1D pencil decomposition for 3D
            devices = mesh_utils.create_device_mesh((n_devices, 1, 1))
            mesh = Mesh(devices, ("x", "y", "z"))
            grid_shape = (n_devices, 1, 1)
        else:  # "block"
            # 2D slab or 3D block decomposition
            # For moderate P, use 2D (nx × ny × 1)
            # For large P, use 3D (nx × ny × nz)
            if n_devices < 64:
                # 2D slab decomposition (better than 3D block for moderate P)
                nx_proc = int(jnp.sqrt(n_devices))
                while n_devices % nx_proc != 0:
                    nx_proc -= 1
                ny_proc = n_devices // nx_proc
                devices = mesh_utils.create_device_mesh((nx_proc, ny_proc, 1))
                mesh = Mesh(devices, ("x", "y", "z"))
                grid_shape = (nx_proc, ny_proc, 1)
            else:
                # 3D block decomposition (only for very large P)
                nz_proc = int(n_devices ** (1 / 3))
                while n_devices % (nz_proc * nz_proc * nz_proc) != 0 and nz_proc > 1:
                    nz_proc -= 1
                remaining = n_devices // nz_proc
                nx_proc = int(jnp.sqrt(remaining))
                while remaining % nx_proc != 0:
                    nx_proc -= 1
                ny_proc = remaining // nx_proc

                devices = mesh_utils.create_device_mesh((nx_proc, ny_proc, nz_proc))
                mesh = Mesh(devices, ("x", "y", "z"))
                grid_shape = (nx_proc, ny_proc, nz_proc)
    else:
        raise NotImplementedError(f"Spatial dimension {spatial_dimension} not implemented")

    return mesh, grid_shape


def create_sharded_array_from_local(U_local, global_shape, mesh, spatial_dimension):
    """Build a globally sharded array from each process's local block,
    avoiding materializing the full global array."""
    # Create sharding spec
    if spatial_dimension == 1:
        sharding = NamedSharding(mesh, P(None, "x"))
    elif spatial_dimension == 2:
        sharding = NamedSharding(mesh, P(None, "x", "y"))
    else:
        sharding = NamedSharding(mesh, P(None, "x", "y", "z"))

    # Put local array on local device
    local_device = jax.local_devices()[0]
    U_local_on_device = jax.device_put(U_local, local_device)

    # Create global sharded array from local pieces
    # This uses JAX's make_array_from_single_device_arrays
    U_sharded = jax.make_array_from_single_device_arrays(
        shape=global_shape, sharding=sharding, arrays=[U_local_on_device]
    )

    return U_sharded


def shard_array_to_devices(U_global, mesh, spatial_dimension):
    """Shard a global (Nvar, Nx, Ny, Nz) array across devices per the mesh."""
    if spatial_dimension == 1:
        sharding = NamedSharding(mesh, P(None, "x"))
    elif spatial_dimension == 2:
        sharding = NamedSharding(mesh, P(None, "x", "y"))
    else:
        sharding = NamedSharding(mesh, P(None, "x", "y", "z"))

    # Shard the array
    U_sharded = jax.device_put(U_global, sharding)

    return U_sharded


def gather_array_from_devices(U_sharded):
    """Gather a sharded array into a global array replicated on all processes.

    Collective operation — MUST be called on all processes.
    """
    # Use process_allgather to properly gather data across multiple processes
    # This is a collective operation - all processes must participate
    U_global = process_allgather(U_sharded)
    return U_global
