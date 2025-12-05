"""
Domain decomposition utilities for multi-GPU parallelism.

This module handles splitting the global domain into local subdomains,
one per GPU/process.
"""
import jax.numpy as jnp


def compute_decomposition(nx_global, ny_global, n_devices, spatial_dim):
    """
    Compute domain decomposition for n_devices.

    Parameters
    ----------
    nx_global, ny_global : int
        Global grid dimensions
    n_devices : int
        Number of devices (GPUs/processes)
    spatial_dim : int
        1, 2, or 3

    Returns
    -------
    grid_shape : tuple
        Shape of process grid, e.g., (2, 2) for 4 devices in 2D
    """
    if spatial_dim == 1:
        return (n_devices,)
    elif spatial_dim == 2:
        # Try to make as square as possible
        nx_proc = int(n_devices ** 0.5)
        while n_devices % nx_proc != 0:
            nx_proc -= 1
        ny_proc = n_devices // nx_proc
        return (nx_proc, ny_proc)
    elif spatial_dim == 3:
        # Simple 3D decomposition
        nx_proc = int(round(n_devices ** (1/3)))
        while n_devices % nx_proc != 0:
            nx_proc -= 1
        ny_proc = int((n_devices // nx_proc) ** 0.5)
        while (n_devices // nx_proc) % ny_proc != 0:
            ny_proc -= 1
        nz_proc = n_devices // (nx_proc * ny_proc)
        return (nx_proc, ny_proc, nz_proc)


def get_local_domain_info(process_id, nx_global, ny_global, nz_global,
                           grid_shape, spatial_dim):
    """
    Get local domain information for a given process.

    Parameters
    ----------
    process_id : int
        Process ID (0 to n_devices-1)
    nx_global, ny_global, nz_global : int
        Global grid dimensions
    grid_shape : tuple
        Process grid shape from compute_decomposition
    spatial_dim : int
        1, 2, or 3

    Returns
    -------
    dict with keys:
        - 'local_shape': (nx_local, ny_local, nz_local) or subset
        - 'global_start': (ix_start, iy_start, iz_start) or subset
        - 'global_end': (ix_end, iy_end, iz_end) or subset
        - 'proc_coords': (px, py, pz) or subset - this process's coordinates in grid
        - 'neighbors': dict with keys 'left', 'right', 'bottom', 'top', 'back', 'front'
                       values are neighbor process IDs or None if boundary
    """
    if spatial_dim == 1:
        nx_proc = grid_shape[0]
        px = process_id

        nx_local = nx_global // nx_proc
        ix_start = px * nx_local
        ix_end = (px + 1) * nx_local

        neighbors = {
            'left': px - 1 if px > 0 else None,
            'right': px + 1 if px < nx_proc - 1 else None,
        }

        return {
            'local_shape': (nx_local,),
            'global_start': (ix_start,),
            'global_end': (ix_end,),
            'proc_coords': (px,),
            'neighbors': neighbors
        }

    elif spatial_dim == 2:
        nx_proc, ny_proc = grid_shape
        px = process_id // ny_proc
        py = process_id % ny_proc

        nx_local = nx_global // nx_proc
        ny_local = ny_global // ny_proc

        ix_start = px * nx_local
        ix_end = (px + 1) * nx_local
        iy_start = py * ny_local
        iy_end = (py + 1) * ny_local

        neighbors = {
            'left': process_id - ny_proc if px > 0 else None,
            'right': process_id + ny_proc if px < nx_proc - 1 else None,
            'bottom': process_id - 1 if py > 0 else None,
            'top': process_id + 1 if py < ny_proc - 1 else None,
        }

        return {
            'local_shape': (nx_local, ny_local),
            'global_start': (ix_start, iy_start),
            'global_end': (ix_end, iy_end),
            'proc_coords': (px, py),
            'neighbors': neighbors
        }

    elif spatial_dim == 3:
        nx_proc, ny_proc, nz_proc = grid_shape
        # Map 1D process_id to 3D coordinates
        px = process_id // (ny_proc * nz_proc)
        py = (process_id // nz_proc) % ny_proc
        pz = process_id % nz_proc

        nx_local = nx_global // nx_proc
        ny_local = ny_global // ny_proc
        nz_local = nz_global // nz_proc

        ix_start = px * nx_local
        ix_end = (px + 1) * nx_local
        iy_start = py * ny_local
        iy_end = (py + 1) * ny_local
        iz_start = pz * nz_local
        iz_end = (pz + 1) * nz_local

        neighbors = {
            'left': process_id - (ny_proc * nz_proc) if px > 0 else None,
            'right': process_id + (ny_proc * nz_proc) if px < nx_proc - 1 else None,
            'bottom': process_id - nz_proc if py > 0 else None,
            'top': process_id + nz_proc if py < ny_proc - 1 else None,
            'back': process_id - 1 if pz > 0 else None,
            'front': process_id + 1 if pz < nz_proc - 1 else None,
        }

        return {
            'local_shape': (nx_local, ny_local, nz_local),
            'global_start': (ix_start, iy_start, iz_start),
            'global_end': (ix_end, iy_end, iz_end),
            'proc_coords': (px, py, pz),
            'neighbors': neighbors
        }


def extract_local_subdomain(U_global, domain_info, spatial_dim):
    """
    Extract local subdomain from global array.

    Parameters
    ----------
    U_global : array
        Global state array (Nvar, Nx) or (Nvar, Nx, Ny) or (Nvar, Nx, Ny, Nz)
    domain_info : dict
        From get_local_domain_info
    spatial_dim : int
        1, 2, or 3

    Returns
    -------
    U_local : array
        Local subdomain (Nvar, nx_local, ...)
    """
    if spatial_dim == 1:
        ix_start, = domain_info['global_start']
        ix_end, = domain_info['global_end']
        return U_global[:, ix_start:ix_end]

    elif spatial_dim == 2:
        ix_start, iy_start = domain_info['global_start']
        ix_end, iy_end = domain_info['global_end']
        return U_global[:, ix_start:ix_end, iy_start:iy_end]

    elif spatial_dim == 3:
        ix_start, iy_start, iz_start = domain_info['global_start']
        ix_end, iy_end, iz_end = domain_info['global_end']
        return U_global[:, ix_start:ix_end, iy_start:iy_end, iz_start:iz_end]


def reconstruct_global_array(U_locals, domain_infos, nx_global, ny_global, nz_global, spatial_dim):
    """
    Reconstruct global array from local subdomains.

    Parameters
    ----------
    U_locals : array
        Stacked local arrays with shape (n_devices, Nvar, nx_local, ...)
    domain_infos : list of dict
        Domain info for each device from get_local_domain_info
    nx_global, ny_global, nz_global : int
        Global grid dimensions
    spatial_dim : int
        1, 2, or 3

    Returns
    -------
    U_global : array
        Reconstructed global array (Nvar, Nx) or (Nvar, Nx, Ny) or (Nvar, Nx, Ny, Nz)
    """
    n_devices = U_locals.shape[0]
    nvar = U_locals.shape[1]

    # Create empty global array
    if spatial_dim == 1:
        U_global = jnp.zeros((nvar, nx_global))
        for device_id in range(n_devices):
            ix_start, = domain_infos[device_id]['global_start']
            ix_end, = domain_infos[device_id]['global_end']
            U_global = U_global.at[:, ix_start:ix_end].set(U_locals[device_id])

    elif spatial_dim == 2:
        U_global = jnp.zeros((nvar, nx_global, ny_global))
        for device_id in range(n_devices):
            ix_start, iy_start = domain_infos[device_id]['global_start']
            ix_end, iy_end = domain_infos[device_id]['global_end']
            U_global = U_global.at[:, ix_start:ix_end, iy_start:iy_end].set(U_locals[device_id])

    elif spatial_dim == 3:
        U_global = jnp.zeros((nvar, nx_global, ny_global, nz_global))
        for device_id in range(n_devices):
            ix_start, iy_start, iz_start = domain_infos[device_id]['global_start']
            ix_end, iy_end, iz_end = domain_infos[device_id]['global_end']
            U_global = U_global.at[:, ix_start:ix_end, iy_start:iy_end, iz_start:iz_end].set(U_locals[device_id])

    return U_global
