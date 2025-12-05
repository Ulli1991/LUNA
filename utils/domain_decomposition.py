"""
Domain decomposition utilities for multi-process JAX SPMD parallelization.

This module handles splitting the global domain across multiple processes
and determining local subdomain properties, neighbors, and boundary types.
"""
import jax
import jax.numpy as jnp


def decompose_process_grid(n_processes, spatial_dim):
    """
    Decompose n_processes into a grid for domain decomposition.

    For 2D: tries to create as square a grid as possible
    For 3D: tries to balance all three dimensions

    Parameters
    ----------
    n_processes : int
        Total number of processes
    spatial_dim : int
        1, 2, or 3

    Returns
    -------
    grid_shape : tuple
        (nx_proc,) for 1D, (nx_proc, ny_proc) for 2D, (nx_proc, ny_proc, nz_proc) for 3D
    """
    if spatial_dim == 1:
        return (n_processes,)

    elif spatial_dim == 2:
        # Find factors that make grid as square as possible
        nx = int(n_processes ** 0.5)
        while n_processes % nx != 0:
            nx -= 1
        ny = n_processes // nx
        return (nx, ny)

    elif spatial_dim == 3:
        # Factorize into nx * ny * nz
        nx = int(round(n_processes ** (1/3)))
        while n_processes % nx != 0:
            nx -= 1
        ny = int((n_processes // nx) ** 0.5)
        while (n_processes // nx) % ny != 0:
            ny -= 1
        nz = n_processes // (nx * ny)
        return (nx, ny, nz)

    else:
        raise ValueError(f"spatial_dim must be 1, 2, or 3, got {spatial_dim}")


def get_process_coordinates(process_id, grid_shape, spatial_dim):
    """
    Convert process ID to coordinates in the process grid.

    Parameters
    ----------
    process_id : int
        This process's rank
    grid_shape : tuple
        From decompose_process_grid()
    spatial_dim : int
        1, 2, or 3

    Returns
    -------
    coords : tuple
        (ix,) for 1D, (ix, iy) for 2D, (ix, iy, iz) for 3D
    """
    if spatial_dim == 1:
        return (process_id,)

    elif spatial_dim == 2:
        nx_proc, ny_proc = grid_shape
        ix = process_id // ny_proc
        iy = process_id % ny_proc
        return (ix, iy)

    elif spatial_dim == 3:
        nx_proc, ny_proc, nz_proc = grid_shape
        # Row-major ordering: process_id = ix * (ny*nz) + iy * nz + iz
        ix = process_id // (ny_proc * nz_proc)
        remainder = process_id % (ny_proc * nz_proc)
        iy = remainder // nz_proc
        iz = remainder % nz_proc
        return (ix, iy, iz)


def get_local_domain_indices(process_coords, grid_shape, global_shape, spatial_dim):
    """
    Get the global index range for this process's local subdomain.

    Parameters
    ----------
    process_coords : tuple
        From get_process_coordinates()
    grid_shape : tuple
        From decompose_process_grid()
    global_shape : tuple
        (Nx,) for 1D, (Nx, Ny) for 2D, (Nx, Ny, Nz) for 3D
    spatial_dim : int
        1, 2, or 3

    Returns
    -------
    local_indices : tuple of tuples
        ((i_start, i_end),) for 1D
        ((i_start, i_end), (j_start, j_end)) for 2D
        ((i_start, i_end), (j_start, j_end), (k_start, k_end)) for 3D

    local_shape : tuple
        (Nx_local,) for 1D, (Nx_local, Ny_local) for 2D, etc.
    """
    if spatial_dim == 1:
        ix, = process_coords
        nx_proc, = grid_shape
        Nx, = global_shape

        Nx_local = Nx // nx_proc
        i_start = ix * Nx_local
        i_end = (ix + 1) * Nx_local if ix < nx_proc - 1 else Nx

        return ((i_start, i_end),), (i_end - i_start,)

    elif spatial_dim == 2:
        ix, iy = process_coords
        nx_proc, ny_proc = grid_shape
        Nx, Ny = global_shape

        Nx_local = Nx // nx_proc
        Ny_local = Ny // ny_proc

        i_start = ix * Nx_local
        i_end = (ix + 1) * Nx_local if ix < nx_proc - 1 else Nx

        j_start = iy * Ny_local
        j_end = (iy + 1) * Ny_local if iy < ny_proc - 1 else Ny

        return ((i_start, i_end), (j_start, j_end)), (i_end - i_start, j_end - j_start)

    elif spatial_dim == 3:
        ix, iy, iz = process_coords
        nx_proc, ny_proc, nz_proc = grid_shape
        Nx, Ny, Nz = global_shape

        Nx_local = Nx // nx_proc
        Ny_local = Ny // ny_proc
        Nz_local = Nz // nz_proc

        i_start = ix * Nx_local
        i_end = (ix + 1) * Nx_local if ix < nx_proc - 1 else Nx

        j_start = iy * Ny_local
        j_end = (iy + 1) * Ny_local if iy < ny_proc - 1 else Ny

        k_start = iz * Nz_local
        k_end = (iz + 1) * Nz_local if iz < nz_proc - 1 else Nz

        return ((i_start, i_end), (j_start, j_end), (k_start, k_end)), \
               (i_end - i_start, j_end - j_start, k_end - k_start)


def get_neighbor_ids(process_coords, grid_shape, spatial_dim):
    """
    Get the process IDs of neighboring processes.

    Returns -1 for neighbors that don't exist (physical boundaries).

    Parameters
    ----------
    process_coords : tuple
        From get_process_coordinates()
    grid_shape : tuple
        From decompose_process_grid()
    spatial_dim : int
        1, 2, or 3

    Returns
    -------
    neighbors : dict
        Keys: 'left', 'right' for 1D
              'left', 'right', 'bottom', 'top' for 2D
              'left', 'right', 'bottom', 'top', 'back', 'front' for 3D
        Values: process_id or -1 if no neighbor (physical boundary)
    """
    neighbors = {}

    if spatial_dim == 1:
        ix, = process_coords
        nx_proc, = grid_shape

        neighbors['left'] = ix - 1 if ix > 0 else -1
        neighbors['right'] = ix + 1 if ix < nx_proc - 1 else -1

    elif spatial_dim == 2:
        ix, iy = process_coords
        nx_proc, ny_proc = grid_shape

        # Convert coordinates to process ID
        def coords_to_id(i, j):
            if 0 <= i < nx_proc and 0 <= j < ny_proc:
                return i * ny_proc + j
            return -1

        neighbors['left'] = coords_to_id(ix - 1, iy)
        neighbors['right'] = coords_to_id(ix + 1, iy)
        neighbors['bottom'] = coords_to_id(ix, iy - 1)
        neighbors['top'] = coords_to_id(ix, iy + 1)

    elif spatial_dim == 3:
        ix, iy, iz = process_coords
        nx_proc, ny_proc, nz_proc = grid_shape

        def coords_to_id(i, j, k):
            if 0 <= i < nx_proc and 0 <= j < ny_proc and 0 <= k < nz_proc:
                return i * (ny_proc * nz_proc) + j * nz_proc + k
            return -1

        neighbors['left'] = coords_to_id(ix - 1, iy, iz)
        neighbors['right'] = coords_to_id(ix + 1, iy, iz)
        neighbors['bottom'] = coords_to_id(ix, iy - 1, iz)
        neighbors['top'] = coords_to_id(ix, iy + 1, iz)
        neighbors['back'] = coords_to_id(ix, iy, iz - 1)
        neighbors['front'] = coords_to_id(ix, iy, iz + 1)

    return neighbors


class LocalDomain:
    """
    Container for all information about a process's local subdomain.

    Attributes
    ----------
    process_id : int
        This process's rank
    n_processes : int
        Total number of processes
    spatial_dim : int
        1, 2, or 3
    global_shape : tuple
        Shape of the global domain
    grid_shape : tuple
        Shape of the process grid
    process_coords : tuple
        This process's coordinates in the grid
    local_indices : tuple of tuples
        Global index range for this subdomain
    local_shape : tuple
        Shape of the local subdomain
    neighbors : dict
        Neighbor process IDs (-1 for physical boundaries)
    """

    def __init__(self, process_id, n_processes, spatial_dim, global_shape):
        """
        Initialize local domain information.

        Parameters
        ----------
        process_id : int
            This process's rank (from jax.process_index())
        n_processes : int
            Total number of processes (from jax.process_count())
        spatial_dim : int
            1, 2, or 3
        global_shape : tuple
            (Nx,) for 1D, (Nx, Ny) for 2D, (Nx, Ny, Nz) for 3D
        """
        self.process_id = process_id
        self.n_processes = n_processes
        self.spatial_dim = spatial_dim
        self.global_shape = global_shape

        # Compute decomposition
        self.grid_shape = decompose_process_grid(n_processes, spatial_dim)
        self.process_coords = get_process_coordinates(process_id, self.grid_shape, spatial_dim)
        self.local_indices, self.local_shape = get_local_domain_indices(
            self.process_coords, self.grid_shape, global_shape, spatial_dim
        )
        self.neighbors = get_neighbor_ids(self.process_coords, self.grid_shape, spatial_dim)

    def is_physical_boundary(self, direction):
        """
        Check if a given direction is a physical (global) boundary.

        Parameters
        ----------
        direction : str
            'left', 'right', 'bottom', 'top', 'back', 'front'

        Returns
        -------
        bool
            True if this direction is a physical boundary (no neighbor)
        """
        return self.neighbors[direction] == -1

    def get_global_coordinates(self, i_local, j_local=None, k_local=None):
        """
        Convert local cell indices to global coordinates.

        Parameters
        ----------
        i_local : int or array
            Local x-index
        j_local : int or array, optional
            Local y-index (for 2D/3D)
        k_local : int or array, optional
            Local z-index (for 3D)

        Returns
        -------
        x : float or array
            Global x-coordinate
        y : float or array, optional
            Global y-coordinate (for 2D/3D)
        z : float or array, optional
            Global z-coordinate (for 3D)
        """
        if self.spatial_dim == 1:
            i_start, = self.local_indices[0]
            i_global = i_start + i_local
            return i_global

        elif self.spatial_dim == 2:
            (i_start, _), (j_start, _) = self.local_indices
            i_global = i_start + i_local
            j_global = j_start + j_local
            return i_global, j_global

        elif self.spatial_dim == 3:
            (i_start, _), (j_start, _), (k_start, _) = self.local_indices
            i_global = i_start + i_local
            j_global = j_start + j_local
            k_global = k_start + k_local
            return i_global, j_global, k_global

    def __repr__(self):
        return (f"LocalDomain(process {self.process_id}/{self.n_processes}, "
                f"coords={self.process_coords}, local_shape={self.local_shape}, "
                f"global_indices={self.local_indices})")
