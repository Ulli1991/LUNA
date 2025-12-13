# physics_checks.py
"""
Physics validation functions to detect unphysical states.
GPU-resident checks with minimal CPU synchronization overhead.
"""

import jax.numpy as jnp
from jax import jit


def compute_physics_diagnostics(U, gamma):
    """
    Compute physics diagnostics on GPU (JIT-compiled).
    Returns scalar values for validation with minimal CPU transfer overhead.

    Parameters
    ----------
    U : array
        Conservative state, shape (Nvar, Nx[, Ny[, Nz]])
    gamma : float
        Ratio of specific heats

    Returns
    -------
    min_rho : float
        Minimum density in domain
    min_p : float
        Minimum pressure in domain
    has_nan : bool
        True if any NaN detected
    has_inf : bool
        True if any Inf detected
    """
    # Check for NaN/Inf in conservative state
    has_nan = jnp.any(jnp.isnan(U))
    has_inf = jnp.any(jnp.isinf(U))

    # Extract density
    rho = U[0]

    # Compute pressure based on dimensionality
    Nvar = U.shape[0]

    # Compute velocities and kinetic energy
    vx = U[1] / rho
    vy = jnp.where(Nvar >= 4, U[2] / rho, 0.0)
    vz = jnp.where(Nvar >= 5, U[3] / rho, 0.0)

    kinetic = 0.5 * rho * (vx**2 + vy**2 + vz**2)

    # Pressure computation
    p = (gamma - 1.0) * (U[-1] - kinetic)

    # Compute minimum values (GPU reduction)
    min_rho = jnp.min(rho)
    min_p = jnp.min(p)

    return min_rho, min_p, has_nan, has_inf


def validate_from_diagnostics(min_rho, min_p, has_nan, has_inf, step=None, time=None):
    """
    Validate physics diagnostics (use when diagnostics are pre-computed).
    This function only performs the validation checks without recomputing.

    Parameters
    ----------
    min_rho : float
        Minimum density in domain
    min_p : float
        Minimum pressure in domain
    has_nan : bool
        True if any NaN detected
    has_inf : bool
        True if any Inf detected
    step : int, optional
        Current timestep number (for error message)
    time : float, optional
        Current simulation time (for error message)

    Raises
    ------
    RuntimeError
        If unphysical states are detected
    """
    if has_nan:
        msg = "NaN detected in conservative state"
        if step is not None:
            msg += f" at step {step}"
        if time is not None:
            msg += f", t={time:.6e}"
        raise RuntimeError(msg)

    if has_inf:
        msg = "Inf detected in conservative state"
        if step is not None:
            msg += f" at step {step}"
        if time is not None:
            msg += f", t={time:.6e}"
        raise RuntimeError(msg)

    if min_rho <= 0.0:
        msg = f"Negative or zero density detected: min(rho) = {float(min_rho):.6e}"
        if step is not None:
            msg += f" at step {step}"
        if time is not None:
            msg += f", t={time:.6e}"
        raise RuntimeError(msg)

    if min_p <= 0.0:
        msg = f"Negative or zero pressure detected: min(p) = {float(min_p):.6e}"
        if step is not None:
            msg += f" at step {step}"
        if time is not None:
            msg += f", t={time:.6e}"
        raise RuntimeError(msg)


def validate_state(U, gamma, step=None, time=None):
    """
    Validate conservative state for physical correctness.
    Uses GPU-resident computation with minimal data transfer.
    Only transfers 4 scalar values from GPU to CPU for checking.

    Parameters
    ----------
    U : array
        Conservative state, shape (Nvar, Nx[, Ny[, Nz]])
    gamma : float
        Ratio of specific heats
    step : int, optional
        Current timestep number (for error message)
    time : float, optional
        Current simulation time (for error message)

    Raises
    ------
    RuntimeError
        If unphysical states are detected
    """
    # Compute diagnostics on GPU (JIT-compiled, efficient)
    min_rho, min_p, has_nan, has_inf = compute_physics_diagnostics(U, gamma)

    # Validate using the diagnostics
    validate_from_diagnostics(min_rho, min_p, has_nan, has_inf, step, time)
