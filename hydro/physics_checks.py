# physics_checks.py
"""
Physics validation functions to detect unphysical states.
GPU-resident checks with minimal CPU synchronization overhead.
"""

import jax.numpy as jnp


def compute_physics_diagnostics(U, gamma):
    """GPU-resident diagnostics of a conservative state: returns scalars
    (min_rho, min_p, has_nan, has_inf)."""
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
    """Raise RuntimeError on NaN/Inf or non-positive min density/pressure;
    ``step``/``time`` only decorate the error message."""
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
    """Validate a conservative state (GPU-resident; transfers only 4 scalars);
    raises RuntimeError on unphysical values."""
    # Compute diagnostics on GPU (JIT-compiled, efficient)
    min_rho, min_p, has_nan, has_inf = compute_physics_diagnostics(U, gamma)

    # Validate using the diagnostics
    validate_from_diagnostics(min_rho, min_p, has_nan, has_inf, step, time)
