"""Checks for unphysical hydro states (NaN/Inf, non-positive density or pressure)."""
import jax.numpy as jnp

def compute_physics_diagnostics(U, gamma):
    """Return (min_rho, min_p, has_nan, has_inf) of a conservative state."""
    has_nan = jnp.any(jnp.isnan(U))
    has_inf = jnp.any(jnp.isinf(U))

    rho = U[0]
    Nvar = U.shape[0]

    vx = U[1] / rho
    vy = U[2] / rho if Nvar >= 4 else 0.0
    vz = U[3] / rho if Nvar >= 5 else 0.0

    kinetic = 0.5 * rho * (vx**2 + vy**2 + vz**2)

    p = (gamma - 1.0) * (U[-1] - kinetic)

    min_rho = jnp.min(rho)
    min_p = jnp.min(p)

    return min_rho, min_p, has_nan, has_inf

def validate_from_diagnostics(min_rho, min_p, has_nan, has_inf, step=None, time=None):
    """Raise RuntimeError on NaN/Inf or non-positive density/pressure."""
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
    """Raise RuntimeError if the conservative state is unphysical."""
    min_rho, min_p, has_nan, has_inf = compute_physics_diagnostics(U, gamma)
    validate_from_diagnostics(min_rho, min_p, has_nan, has_inf, step, time)
