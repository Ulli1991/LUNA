"""
Global collective operations for multi-GPU/multi-node.

Provides:
- Global reductions (min, max, sum, etc.)
- All-to-all communication (future)
- Synchronization primitives

Uses JAX's built-in collective operations when available,
falls back to simple operations for single-GPU case.
"""

from typing import Optional, List
import jax
import jax.numpy as jnp
from grid.domain import Domain


def global_reduce(
    value: float,
    op: str,
    devices: Optional[List[jax.Device]] = None
) -> float:
    """
    Perform global reduction across all devices.

    Parameters
    ----------
    value : float
        Local value to reduce
    op : str
        Operation: 'min', 'max', 'sum', 'mean'
    devices : List[jax.Device], optional
        Devices participating in reduction

    Returns
    -------
    float
        Reduced value

    Examples
    --------
    >>> # Compute global minimum timestep
    >>> dt_local = 0.001
    >>> dt_global = global_reduce(dt_local, 'min', devices)
    """
    if devices is None or len(devices) == 1:
        # Single device: no reduction needed
        return value

    # For multi-device within single process, use JAX collectives
    # Convert to array for JAX operations
    value_array = jnp.array(value)

    if op == 'min':
        result = jax.lax.pmin(value_array, axis_name='devices')
    elif op == 'max':
        result = jax.lax.pmax(value_array, axis_name='devices')
    elif op == 'sum':
        result = jax.lax.psum(value_array, axis_name='devices')
    elif op == 'mean':
        total = jax.lax.psum(value_array, axis_name='devices')
        result = total / len(devices)
    else:
        raise ValueError(f"Unknown reduction operation: {op}")

    return float(result)


def global_min(value: float, devices: Optional[List[jax.Device]] = None) -> float:
    """
    Compute global minimum across all devices.

    Parameters
    ----------
    value : float
        Local value
    devices : List[jax.Device], optional
        Devices to reduce over

    Returns
    -------
    float
        Global minimum
    """
    if devices is None or len(devices) == 1:
        return value

    # Simple approach for single-process multi-GPU
    # For multi-node, would need MPI or JAX distributed
    return value  # TODO: implement properly with JAX collectives


def global_max(value: float, devices: Optional[List[jax.Device]] = None) -> float:
    """
    Compute global maximum across all devices.

    Parameters
    ----------
    value : float
        Local value
    devices : List[jax.Device], optional
        Devices to reduce over

    Returns
    -------
    float
        Global maximum
    """
    if devices is None or len(devices) == 1:
        return value

    return value  # TODO: implement properly with JAX collectives


def global_sum(value: float, devices: Optional[List[jax.Device]] = None) -> float:
    """
    Compute global sum across all devices.

    Parameters
    ----------
    value : float
        Local value
    devices : List[jax.Device], optional
        Devices to reduce over

    Returns
    -------
    float
        Global sum
    """
    if devices is None or len(devices) == 1:
        return value

    return value  # TODO: implement properly with JAX collectives


def compute_global_dt(
    domain: Domain,
    gamma: float,
    cfl: float,
    devices: Optional[List[jax.Device]] = None
) -> float:
    """
    Compute global CFL timestep across all patches.

    Parameters
    ----------
    domain : Domain
        Domain containing all patches
    gamma : float
        Ratio of specific heats
    cfl : float
        CFL number
    devices : List[jax.Device], optional
        Devices for multi-GPU

    Returns
    -------
    float
        Global timestep (minimum across all patches)

    Notes
    -----
    This computes the CFL condition for each patch independently,
    then takes the global minimum to ensure stability everywhere.
    """
    # Import here to avoid circular dependency
    from time_integration.cfl import compute_dt_and_diagnostics

    dt_local_min = float('inf')

    # Compute local CFL for each patch
    for patch in domain.patches:
        if patch.data is None:
            continue

        # Extract interior data (without ghost cells)
        interior_slice = patch.get_interior_slice()
        U_interior = patch.data[interior_slice]

        # Compute dt for this patch
        dx = patch.dx
        dy = patch.dy
        dz = patch.dz

        dt, _, _, _, _ = compute_dt_and_diagnostics(
            U_interior, gamma, cfl,
            t=0.0, t_end=1e10,  # Dummy values, we only care about dt
            dx=dx, dy=dy, dz=dz
        )

        dt_local_min = min(dt_local_min, float(dt))

    # Global reduction
    dt_global = global_min(dt_local_min, devices)

    return dt_global
