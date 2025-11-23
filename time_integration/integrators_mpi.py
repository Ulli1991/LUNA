"""
MPI-specific time integrators for multi-GPU execution.

These integrators handle the case where ghost cells come from MPI halo exchange
rather than local padding. Ghost cells are held fixed during RK sub-steps,
which is the standard approach for MPI hydrodynamics codes.
"""

import jax
import jax.numpy as jnp
from functools import partial


# =============================================================================
# FAST MPI TIME INTEGRATORS (concatenation-based)
# =============================================================================
# These use the fast flux function that takes (U_interior, left_ghost, right_ghost)

@partial(jax.jit, static_argnames=['flux_fn'])
def mpi_rk4_step_fast(U_interior, left_ghost, right_ghost, dt, flux_fn):
    """
    RK4 time step using fast concatenation-based flux function.

    Ghost cells are passed as separate arrays and concatenated in flux_fn.
    This avoids expensive .at[].set() operations.
    """
    k1 = flux_fn(U_interior, left_ghost, right_ghost)
    k2 = flux_fn(U_interior + 0.5 * dt * k1, left_ghost, right_ghost)
    k3 = flux_fn(U_interior + 0.5 * dt * k2, left_ghost, right_ghost)
    k4 = flux_fn(U_interior + dt * k3, left_ghost, right_ghost)
    return U_interior + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


@partial(jax.jit, static_argnames=['flux_fn'])
def mpi_rk2_step_fast(U_interior, left_ghost, right_ghost, dt, flux_fn):
    """RK2 time step using fast concatenation-based flux function."""
    k1 = flux_fn(U_interior, left_ghost, right_ghost)
    k2 = flux_fn(U_interior + 0.5 * dt * k1, left_ghost, right_ghost)
    return U_interior + dt * k2


@partial(jax.jit, static_argnames=['flux_fn'])
def mpi_muscl_hancock_step_fast(U_interior, left_ghost, right_ghost, dt, flux_fn):
    """MUSCL-Hancock time step using fast concatenation-based flux function."""
    k1 = flux_fn(U_interior, left_ghost, right_ghost)
    k2 = flux_fn(U_interior + dt * k1, left_ghost, right_ghost)
    return U_interior + dt * 0.5 * (k1 + k2)


# =============================================================================
# ORIGINAL MPI TIME INTEGRATORS (for compatibility)
# =============================================================================

@partial(jax.jit, static_argnames=['flux_fn'])
def mpi_rk2_step(U_interior, U_with_ghost, dt, flux_fn):
    """
    RK2 time step for MPI mode.

    U_interior: interior data (no ghost cells)
    U_with_ghost: data with 2 ghost cells on each side (for flux computation)
    dt: timestep
    flux_fn: function(U_interior, U_with_ghost) -> dU

    Ghost cells are held fixed during RK sub-steps (standard MPI hydro approach).
    """
    k1 = flux_fn(U_interior, U_with_ghost)
    k2 = flux_fn(U_interior + 0.5 * dt * k1, U_with_ghost)
    return U_interior + dt * k2


@partial(jax.jit, static_argnames=['flux_fn'])
def mpi_rk4_step(U_interior, U_with_ghost, dt, flux_fn):
    """RK4 time step for MPI mode."""
    k1 = flux_fn(U_interior, U_with_ghost)
    k2 = flux_fn(U_interior + 0.5 * dt * k1, U_with_ghost)
    k3 = flux_fn(U_interior + 0.5 * dt * k2, U_with_ghost)
    k4 = flux_fn(U_interior + dt * k3, U_with_ghost)
    return U_interior + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


@partial(jax.jit, static_argnames=['flux_fn'])
def mpi_muscl_hancock_step(U_interior, U_with_ghost, dt, flux_fn):
    """MUSCL-Hancock (predictor-corrector) time step for MPI mode."""
    k1 = flux_fn(U_interior, U_with_ghost)
    k2 = flux_fn(U_interior + dt * k1, U_with_ghost)
    return U_interior + dt * 0.5 * (k1 + k2)
