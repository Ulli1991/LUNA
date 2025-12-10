"""
Distributed computing utilities for JAX multi-node execution.

Functions for creating device meshes, sharding arrays, and gathering results.
"""
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.multihost_utils import process_allgather


def compute_local_slice(Nx, Ny, mesh, process_index):
    """
    Compute which slice of the global array this process should generate.

    Parameters
    ----------
    Nx, Ny : int
        Global grid dimensions
    mesh : Mesh
        JAX mesh object
    process_index : int
        Current process index

    Returns
    -------
    x_slice, y_slice : tuple of (start, end)
        Slice indices for this process
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
        return (x_start, x_end), (0, Ny)

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

        return (x_start, x_end), (y_start, y_end)

    else:
        raise NotImplementedError("3D not yet implemented")


def create_device_mesh(n_devices, spatial_dimension):
    """
    Create optimal device mesh for the given number of devices.

    Parameters
    ----------
    n_devices : int
        Total number of devices
    spatial_dimension : int
        1 or 2

    Returns
    -------
    mesh : Mesh
        JAX mesh object
    grid_shape : tuple
        Process grid shape
    """
    if spatial_dimension == 1:
        # 1D decomposition along x
        devices = mesh_utils.create_device_mesh((n_devices,))
        mesh = Mesh(devices, ('x',))
        grid_shape = (n_devices,)
    elif spatial_dimension == 2:
        # 2D decomposition
        # Find factors of n_devices that are close to square
        nx_proc = int(jnp.sqrt(n_devices))
        while n_devices % nx_proc != 0:
            nx_proc -= 1
        ny_proc = n_devices // nx_proc

        devices = mesh_utils.create_device_mesh((nx_proc, ny_proc))
        mesh = Mesh(devices, ('x', 'y'))
        grid_shape = (nx_proc, ny_proc)
    else:
        raise NotImplementedError("3D not yet implemented")

    return mesh, grid_shape


def create_sharded_array_from_local(U_local, global_shape, mesh, spatial_dimension):
    """
    Create a sharded JAX array from local arrays (one per process).
    Avoids the expensive global array creation and verification.

    Parameters
    ----------
    U_local : array
        Local portion of the array for this process
    global_shape : tuple
        Shape of the full global array
    mesh : Mesh
        JAX mesh object
    spatial_dimension : int
        1 or 2

    Returns
    -------
    U_sharded : array
        Globally sharded array
    """
    # Create sharding spec
    if spatial_dimension == 1:
        sharding = NamedSharding(mesh, P(None, 'x'))
    elif spatial_dimension == 2:
        sharding = NamedSharding(mesh, P(None, 'x', 'y'))
    else:
        raise NotImplementedError("3D not yet implemented")

    # Put local array on local device
    local_device = jax.local_devices()[0]
    U_local_on_device = jax.device_put(U_local, local_device)

    # Create global sharded array from local pieces
    # This uses JAX's make_array_from_single_device_arrays
    U_sharded = jax.make_array_from_single_device_arrays(
        shape=global_shape,
        sharding=sharding,
        arrays=[U_local_on_device]
    )

    return U_sharded


def shard_array_to_devices(U_global, mesh, spatial_dimension):
    """
    Shard a global array across devices according to the mesh.

    Parameters
    ----------
    U_global : array (Nvar, Nx, Ny, Nz)
        Global array
    mesh : Mesh
        Device mesh
    spatial_dimension : int
        Spatial dimension

    Returns
    -------
    U_sharded : array
        Sharded array distributed across devices
    """
    if spatial_dimension == 1:
        # Shard along x dimension
        # Array shape: (Nvar, Nx)
        # Partition: (None, 'x')
        sharding = NamedSharding(mesh, P(None, 'x'))
    elif spatial_dimension == 2:
        # Shard along x and y dimensions
        # Array shape: (Nvar, Nx, Ny)
        # Partition: (None, 'x', 'y')
        sharding = NamedSharding(mesh, P(None, 'x', 'y'))
    else:
        raise NotImplementedError("3D not yet implemented")

    # Shard the array
    U_sharded = jax.device_put(U_global, sharding)

    return U_sharded


def gather_array_from_devices(U_sharded):
    """
    Gather a sharded array back to a global array across all processes.

    All processes must call this function (collective operation).
    The result will be available on all processes.

    Parameters
    ----------
    U_sharded : sharded array
        Distributed array

    Returns
    -------
    U_global : array
        Global array (replicated on all processes)
    """
    # Use process_allgather to properly gather data across multiple processes
    # This is a collective operation - all processes must participate
    U_global = process_allgather(U_sharded)
    return U_global
