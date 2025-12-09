"""
Distributed computing utilities for JAX multi-node execution.

Functions for creating device meshes, sharding arrays, and gathering results.
"""
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.multihost_utils import process_allgather


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
