"""Device meshes, sharding and local-slice helpers for multi-device runs."""

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def compute_local_slice(Nx, Ny, mesh, process_index, Nz=1):
    """Return this process's ((x0, x1), (y0, y1), (z0, z1)) slice of the global grid."""
    grid_shape = mesh.devices.shape

    if len(grid_shape) == 1:
        nx_proc = grid_shape[0]
        x_per_proc = Nx // nx_proc
        x_start = process_index * x_per_proc
        x_end = (process_index + 1) * x_per_proc if process_index < nx_proc - 1 else Nx
        return (x_start, x_end), (0, Ny), (0, Nz)

    elif len(grid_shape) == 2:
        nx_proc, ny_proc = grid_shape
        proc_x = process_index // ny_proc
        proc_y = process_index % ny_proc

        x_per_proc = Nx // nx_proc
        x_start = proc_x * x_per_proc
        x_end = (proc_x + 1) * x_per_proc if proc_x < nx_proc - 1 else Nx

        y_per_proc = Ny // ny_proc
        y_start = proc_y * y_per_proc
        y_end = (proc_y + 1) * y_per_proc if proc_y < ny_proc - 1 else Ny

        return (x_start, x_end), (y_start, y_end), (0, Nz)

    else:
        nx_proc, ny_proc, nz_proc = grid_shape
        proc_x = process_index // (ny_proc * nz_proc)
        proc_y = (process_index % (ny_proc * nz_proc)) // nz_proc
        proc_z = process_index % nz_proc

        x_per_proc = Nx // nx_proc
        x_start = proc_x * x_per_proc
        x_end = (proc_x + 1) * x_per_proc if proc_x < nx_proc - 1 else Nx

        y_per_proc = Ny // ny_proc
        y_start = proc_y * y_per_proc
        y_end = (proc_y + 1) * y_per_proc if proc_y < ny_proc - 1 else Ny

        z_per_proc = Nz // nz_proc
        z_start = proc_z * z_per_proc
        z_end = (proc_z + 1) * z_per_proc if proc_z < nz_proc - 1 else Nz

        return (x_start, x_end), (y_start, y_end), (z_start, z_end)


def create_device_mesh(
    n_devices, spatial_dimension, decomposition="auto", Nx=None, Ny=None, mesh_shape=None
):
    """Create a device mesh; return (mesh, grid_shape).

    decomposition: "auto", "slab" or "block"; an explicit mesh_shape overrides it.
    """
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

    if decomposition == "auto":
        if spatial_dimension == 1:
            decomposition = "slab"
        elif spatial_dimension == 2:
            if n_devices <= 16:
                decomposition = "slab"
            elif n_devices >= 64:
                decomposition = "block"
            elif Nx is not None and Ny is not None:
                cells_per_gpu_slab = (Nx // n_devices) * Ny
                _cells_per_gpu_block = (Nx // int(jnp.sqrt(n_devices))) * (
                    Ny // int(jnp.sqrt(n_devices))
                )
                if (Nx // n_devices) >= 512 or cells_per_gpu_slab >= 1_000_000:
                    decomposition = "slab"
                else:
                    decomposition = "block"
            else:
                decomposition = "slab"
        else:
            if n_devices <= 16:
                decomposition = "slab"
            else:
                decomposition = "block"

    if spatial_dimension == 1:
        devices = mesh_utils.create_device_mesh((n_devices,))
        mesh = Mesh(devices, ("x",))
        grid_shape = (n_devices,)

    elif spatial_dimension == 2:
        if decomposition == "slab":
            devices = mesh_utils.create_device_mesh((n_devices, 1))
            mesh = Mesh(devices, ("x", "y"))
            grid_shape = (n_devices, 1)
        else:
            nx_proc = int(jnp.sqrt(n_devices))
            while n_devices % nx_proc != 0:
                nx_proc -= 1
            ny_proc = n_devices // nx_proc

            devices = mesh_utils.create_device_mesh((nx_proc, ny_proc))
            mesh = Mesh(devices, ("x", "y"))
            grid_shape = (nx_proc, ny_proc)

    elif spatial_dimension == 3:
        if decomposition == "slab":
            devices = mesh_utils.create_device_mesh((n_devices, 1, 1))
            mesh = Mesh(devices, ("x", "y", "z"))
            grid_shape = (n_devices, 1, 1)
        else:
            if n_devices < 64:
                nx_proc = int(jnp.sqrt(n_devices))
                while n_devices % nx_proc != 0:
                    nx_proc -= 1
                ny_proc = n_devices // nx_proc
                devices = mesh_utils.create_device_mesh((nx_proc, ny_proc, 1))
                mesh = Mesh(devices, ("x", "y", "z"))
                grid_shape = (nx_proc, ny_proc, 1)
            else:
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
    """Build a globally sharded array from this process's local block, without a global array."""
    if spatial_dimension == 1:
        sharding = NamedSharding(mesh, P(None, "x"))
    elif spatial_dimension == 2:
        sharding = NamedSharding(mesh, P(None, "x", "y"))
    else:
        sharding = NamedSharding(mesh, P(None, "x", "y", "z"))

    local_device = jax.local_devices()[0]
    U_local_on_device = jax.device_put(U_local, local_device)

    U_sharded = jax.make_array_from_single_device_arrays(
        shape=global_shape, sharding=sharding, arrays=[U_local_on_device]
    )

    return U_sharded


def shard_array_to_devices(U_global, mesh, spatial_dimension):
    """Shard a global (Nvar, Nx[, Ny[, Nz]]) array across the mesh."""
    if spatial_dimension == 1:
        sharding = NamedSharding(mesh, P(None, "x"))
    elif spatial_dimension == 2:
        sharding = NamedSharding(mesh, P(None, "x", "y"))
    else:
        sharding = NamedSharding(mesh, P(None, "x", "y", "z"))

    U_sharded = jax.device_put(U_global, sharding)

    return U_sharded
