"""
Collective operations for sharded arrays.

These functions perform global reductions across all devices in a mesh.
"""
import jax
import jax.numpy as jnp


def global_min(value, mesh_axes):
    """
    Compute global minimum across all devices.

    Parameters
    ----------
    value : scalar or array
        Local value to reduce
    mesh_axes : tuple of str
        Mesh axes to reduce over (e.g., ('x', 'y') for 2D decomposition)

    Returns
    -------
    global_min : scalar
        Minimum value across all devices
    """
    result = value
    for axis in mesh_axes:
        result = jax.lax.pmin(result, axis)
    return result


def global_max(value, mesh_axes):
    """
    Compute global maximum across all devices.

    Parameters
    ----------
    value : scalar or array
        Local value to reduce
    mesh_axes : tuple of str
        Mesh axes to reduce over

    Returns
    -------
    global_max : scalar
        Maximum value across all devices
    """
    result = value
    for axis in mesh_axes:
        result = jax.lax.pmax(result, axis)
    return result


def global_any(value, mesh_axes):
    """
    Compute global logical OR across all devices.

    Parameters
    ----------
    value : bool
        Local boolean value
    mesh_axes : tuple of str
        Mesh axes to reduce over

    Returns
    -------
    global_any : bool
        True if any device has True
    """
    result = value
    for axis in mesh_axes:
        # Convert to int, sum, then check if > 0
        result = jax.lax.psum(result.astype(jnp.int32), axis) > 0
    return result


def global_sum(value, mesh_axes):
    """
    Compute global sum across all devices.

    Parameters
    ----------
    value : scalar or array
        Local value to sum
    mesh_axes : tuple of str
        Mesh axes to reduce over

    Returns
    -------
    global_sum : scalar or array
        Sum across all devices
    """
    result = value
    for axis in mesh_axes:
        result = jax.lax.psum(result, axis)
    return result
