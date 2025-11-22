"""
MPI gather utilities for reconstructing global arrays from distributed patches.

This module provides functions to gather data from all MPI ranks and
reconstruct the global domain for output.
"""

import numpy as np
import jax.numpy as jnp


def reconstruct_global_array(gathered_fields, patches, global_shape, ndim):
    """
    Reconstruct global array from gathered patch data.

    Parameters
    ----------
    gathered_fields : list of lists
        For each field: list of patch data [field_idx][patch_idx]
    patches : list of GridPatch
        All patches (for metadata: shape, offset)
    global_shape : tuple
        Global grid shape (Nx,) or (Nx, Ny) or (Nx, Ny, Nz)
    ndim : int
        Spatial dimensionality (1, 2, or 3)

    Returns
    -------
    U_global : jnp.ndarray
        Reconstructed global array with shape (n_fields, Nx, Ny, Nz)
    """
    n_fields = len(gathered_fields)
    n_patches = len(patches)

    # Allocate global array
    if ndim == 1:
        U_global = np.zeros((n_fields, global_shape[0]), dtype=np.float32)
    elif ndim == 2:
        U_global = np.zeros((n_fields, global_shape[0], global_shape[1]), dtype=np.float32)
    else:
        U_global = np.zeros((n_fields, global_shape[0], global_shape[1], global_shape[2]), dtype=np.float32)

    # Place each patch's data into the global array
    for patch_idx in range(n_patches):
        patch = patches[patch_idx]
        offset = patch.global_offset
        shape = patch.shape

        for field_idx in range(n_fields):
            patch_data = np.asarray(gathered_fields[field_idx][patch_idx])

            if ndim == 1:
                ix_start = offset[0]
                ix_end = offset[0] + shape[0]
                U_global[field_idx, ix_start:ix_end] = patch_data
            elif ndim == 2:
                ix_start, iy_start = offset
                ix_end = ix_start + shape[0]
                iy_end = iy_start + shape[1]
                U_global[field_idx, ix_start:ix_end, iy_start:iy_end] = patch_data
            else:
                ix_start, iy_start, iz_start = offset
                ix_end = ix_start + shape[0]
                iy_end = iy_start + shape[1]
                iz_end = iz_start + shape[2]
                U_global[field_idx, ix_start:ix_end, iy_start:iy_end, iz_start:iz_end] = patch_data

    return jnp.array(U_global)
