"""
Domain decomposition strategies.

For fixed grid: Regular Cartesian decomposition
For AMR: Octree decomposition (future)
"""

from typing import List, Tuple
import numpy as np
from grid.patch import GridPatch


def compute_optimal_decomposition(n_gpus: int, ndim: int) -> Tuple[int, ...]:
    """
    Compute optimal Cartesian decomposition for given number of GPUs.

    Tries to create decomposition that:
    - Is as cubic/square as possible (minimizes surface area / volume)
    - Uses all GPUs (product equals n_gpus)

    Parameters
    ----------
    n_gpus : int
        Number of GPUs/processes
    ndim : int
        Spatial dimensionality (1, 2, or 3)

    Returns
    -------
    Tuple[int, ...]
        Decomposition (nx,) or (nx, ny) or (nx, ny, nz)

    Examples
    --------
    >>> compute_optimal_decomposition(8, 3)
    (2, 2, 2)
    >>> compute_optimal_decomposition(12, 3)
    (3, 2, 2)
    >>> compute_optimal_decomposition(4, 2)
    (2, 2)
    """
    if ndim == 1:
        return (n_gpus,)

    # For 2D and 3D, factorize n_gpus
    factors = factorize(n_gpus)

    if ndim == 2:
        # Try to make it as square as possible
        nx = int(np.sqrt(n_gpus))
        while n_gpus % nx != 0:
            nx -= 1
        ny = n_gpus // nx

        # Prefer larger dimension first
        return (max(nx, ny), min(nx, ny))

    elif ndim == 3:
        # Try to make it as cubic as possible
        # Start from cube root and work down
        nx = int(round(n_gpus ** (1/3)))

        while n_gpus % nx != 0:
            nx -= 1

        remaining = n_gpus // nx

        # Factor remaining into ny, nz
        ny = int(np.sqrt(remaining))
        while remaining % ny != 0:
            ny -= 1
        nz = remaining // ny

        # Sort to have larger dimensions first
        dims = sorted([nx, ny, nz], reverse=True)
        return tuple(dims)

    else:
        raise ValueError(f"Invalid dimensionality: {ndim}")


def factorize(n: int) -> List[int]:
    """
    Factorize integer into prime factors.

    Parameters
    ----------
    n : int
        Number to factorize

    Returns
    -------
    List[int]
        List of prime factors
    """
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def create_uniform_decomposition(
    global_shape: Tuple[int, ...],
    global_extent: Tuple[float, ...],
    n_gpus: int,
    decomposition: Tuple[int, ...] = None,
    boundary_type: int = 0,
) -> List[GridPatch]:
    """
    Create uniform Cartesian decomposition of domain.

    Parameters
    ----------
    global_shape : Tuple[int, ...]
        Global grid dimensions (Nx,) or (Nx, Ny) or (Nx, Ny, Nz)
    global_extent : Tuple[float, ...]
        Physical domain size (Lx,) or (Lx, Ly) or (Lx, Ly, Lz)
    n_gpus : int
        Number of GPUs/processes
    decomposition : Tuple[int, ...], optional
        Explicit decomposition (nx,) or (nx, ny) or (nx, ny, nz)
        If None, will compute optimal decomposition
    boundary_type : int
        Boundary condition type for domain boundaries
        0=periodic, 1=zero-gradient, 2=reflective

    Returns
    -------
    List[GridPatch]
        List of patches (one per GPU)

    Raises
    ------
    ValueError
        If decomposition doesn't match n_gpus or dimensions inconsistent

    Examples
    --------
    >>> # 8 GPUs, 2048^3 grid
    >>> patches = create_uniform_decomposition(
    ...     global_shape=(2048, 2048, 2048),
    ...     global_extent=(1.0, 1.0, 1.0),
    ...     n_gpus=8
    ... )
    >>> len(patches)
    8
    >>> patches[0].shape
    (1024, 1024, 1024)
    """
    ndim = len(global_shape)

    # Validate inputs
    if len(global_extent) != ndim:
        raise ValueError(
            f"global_shape and global_extent dimensionality mismatch: "
            f"{ndim} vs {len(global_extent)}"
        )

    # Compute or validate decomposition
    if decomposition is None:
        decomposition = compute_optimal_decomposition(n_gpus, ndim)
    else:
        if len(decomposition) != ndim:
            raise ValueError(
                f"decomposition dimensionality {len(decomposition)} doesn't match "
                f"domain dimensionality {ndim}"
            )
        product = 1
        for n in decomposition:
            product *= n
        if product != n_gpus:
            raise ValueError(
                f"decomposition {decomposition} doesn't multiply to {n_gpus}"
            )

    # Compute grid spacing
    dx = global_extent[0] / global_shape[0]
    dy = global_extent[1] / global_shape[1] if ndim >= 2 else None
    dz = global_extent[2] / global_shape[2] if ndim == 3 else None

    # Create patches
    patches = []

    # Compute patch sizes (try to make them as equal as possible)
    patch_sizes = []
    for i in range(ndim):
        sizes = distribute_cells(global_shape[i], decomposition[i])
        patch_sizes.append(sizes)

    # Create patches in order
    rank = 0

    if ndim == 1:
        nx = decomposition[0]
        ix_offset = 0

        for ix in range(nx):
            # Patch shape
            patch_nx = patch_sizes[0][ix]
            shape = (patch_nx,)

            # Global offset
            global_offset = (ix_offset,)

            # Boundary types
            boundary_types = {}
            if ix == 0:
                boundary_types['left'] = boundary_type
            if ix == nx - 1:
                boundary_types['right'] = boundary_type

            # Create patch
            patch = GridPatch(
                rank=rank,
                shape=shape,
                dx=dx,
                dy=None,
                dz=None,
                global_offset=global_offset,
                level=0,
                boundary_types=boundary_types,
            )
            patches.append(patch)

            ix_offset += patch_nx
            rank += 1

    elif ndim == 2:
        nx, ny = decomposition
        iy_offset = 0

        for iy in range(ny):
            ix_offset = 0

            for ix in range(nx):
                # Patch shape
                patch_nx = patch_sizes[0][ix]
                patch_ny = patch_sizes[1][iy]
                shape = (patch_nx, patch_ny)

                # Global offset
                global_offset = (ix_offset, iy_offset)

                # Boundary types
                boundary_types = {}
                if ix == 0:
                    boundary_types['left'] = boundary_type
                if ix == nx - 1:
                    boundary_types['right'] = boundary_type
                if iy == 0:
                    boundary_types['bottom'] = boundary_type
                if iy == ny - 1:
                    boundary_types['top'] = boundary_type

                # Create patch
                patch = GridPatch(
                    rank=rank,
                    shape=shape,
                    dx=dx,
                    dy=dy,
                    dz=None,
                    global_offset=global_offset,
                    level=0,
                    boundary_types=boundary_types,
                )
                patches.append(patch)

                ix_offset += patch_nx
                rank += 1

            iy_offset += patch_ny

    elif ndim == 3:
        nx, ny, nz = decomposition
        iz_offset = 0

        for iz in range(nz):
            iy_offset = 0

            for iy in range(ny):
                ix_offset = 0

                for ix in range(nx):
                    # Patch shape
                    patch_nx = patch_sizes[0][ix]
                    patch_ny = patch_sizes[1][iy]
                    patch_nz = patch_sizes[2][iz]
                    shape = (patch_nx, patch_ny, patch_nz)

                    # Global offset
                    global_offset = (ix_offset, iy_offset, iz_offset)

                    # Boundary types
                    boundary_types = {}
                    if ix == 0:
                        boundary_types['left'] = boundary_type
                    if ix == nx - 1:
                        boundary_types['right'] = boundary_type
                    if iy == 0:
                        boundary_types['bottom'] = boundary_type
                    if iy == ny - 1:
                        boundary_types['top'] = boundary_type
                    if iz == 0:
                        boundary_types['back'] = boundary_type
                    if iz == nz - 1:
                        boundary_types['front'] = boundary_type

                    # Create patch
                    patch = GridPatch(
                        rank=rank,
                        shape=shape,
                        dx=dx,
                        dy=dy,
                        dz=dz,
                        global_offset=global_offset,
                        level=0,
                        boundary_types=boundary_types,
                    )
                    patches.append(patch)

                    ix_offset += patch_nx
                    rank += 1

                iy_offset += patch_ny

            iz_offset += patch_nz

    else:
        raise ValueError(f"Invalid dimensionality: {ndim}")

    return patches


def distribute_cells(n_cells: int, n_patches: int) -> List[int]:
    """
    Distribute cells as evenly as possible among patches.

    Parameters
    ----------
    n_cells : int
        Total number of cells
    n_patches : int
        Number of patches

    Returns
    -------
    List[int]
        Number of cells for each patch

    Examples
    --------
    >>> distribute_cells(100, 3)
    [34, 33, 33]
    >>> distribute_cells(2048, 8)
    [256, 256, 256, 256, 256, 256, 256, 256]
    """
    base_size = n_cells // n_patches
    remainder = n_cells % n_patches

    sizes = []
    for i in range(n_patches):
        # First 'remainder' patches get an extra cell
        size = base_size + (1 if i < remainder else 0)
        sizes.append(size)

    return sizes
