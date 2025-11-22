"""
MPI-specific time integrators for multi-GPU execution.

These integrators handle the case where ghost cells come from MPI halo exchange
rather than local padding. Two versions are provided:

1. Standard version: Ghost cells held fixed during RK sub-steps (faster)
2. Substep version: Ghost cells updated between RK sub-steps (more accurate
   for shocks at boundaries)
"""

import jax
import jax.numpy as jnp
from functools import partial

from parallel.halo_exchange_mpi import update_ghosts_mpi


# =============================================================================
# STANDARD MPI TIME INTEGRATORS
# =============================================================================
# Ghost cells are held fixed during RK sub-steps. This is the standard approach
# for MPI hydrodynamics codes and is faster due to fewer communication calls.

@partial(jax.jit, static_argnames=['flux_fn'])
def mpi_rk2_step(U_interior, U_with_ghost, dt, flux_fn):
    """
    RK2 time step for MPI mode (standard - ghost cells fixed during sub-steps).

    U_interior: interior data (no ghost cells)
    U_with_ghost: data with 2 ghost cells on each side (for flux computation)
    dt: timestep
    flux_fn: function(U_interior, U_with_ghost) -> dU

    Ghost cells are held fixed during RK sub-steps (standard MPI hydro approach).
    For improved accuracy at boundaries with shocks, use mpi_rk2_step_substep instead.
    """
    k1 = flux_fn(U_interior, U_with_ghost)
    k2 = flux_fn(U_interior + 0.5 * dt * k1, U_with_ghost)
    return U_interior + dt * k2


@partial(jax.jit, static_argnames=['flux_fn'])
def mpi_rk4_step(U_interior, U_with_ghost, dt, flux_fn):
    """RK4 time step for MPI mode (standard - ghost cells fixed during sub-steps)."""
    k1 = flux_fn(U_interior, U_with_ghost)
    k2 = flux_fn(U_interior + 0.5 * dt * k1, U_with_ghost)
    k3 = flux_fn(U_interior + 0.5 * dt * k2, U_with_ghost)
    k4 = flux_fn(U_interior + dt * k3, U_with_ghost)
    return U_interior + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


@partial(jax.jit, static_argnames=['flux_fn'])
def mpi_muscl_hancock_step(U_interior, U_with_ghost, dt, flux_fn):
    """MUSCL-Hancock time step for MPI mode (standard - ghost cells fixed during sub-steps)."""
    k1 = flux_fn(U_interior, U_with_ghost)
    k2 = flux_fn(U_interior + dt * k1, U_with_ghost)
    return U_interior + dt * 0.5 * (k1 + k2)


# =============================================================================
# MPI TIME INTEGRATORS WITH SUB-STEP HALO EXCHANGE
# =============================================================================
# These versions update ghost cells between RK sub-steps, providing better
# accuracy for problems with shocks at domain boundaries. The tradeoff is
# additional communication overhead.

@partial(jax.jit, static_argnames=['flux_fn'])
def _compute_k_stage(U_interior, U_with_ghost, flux_fn):
    """JIT-compiled single RK stage evaluation."""
    return flux_fn(U_interior, U_with_ghost)


def mpi_rk2_step_substep(patch, dt, flux_fn, boundary_type, comm, interior_slice, slice_2g):
    """
    RK2 time step with sub-step halo exchange (more accurate at boundaries).

    This version updates ghost cells between RK stages, providing better accuracy
    for problems with shocks at domain/subdomain boundaries.

    Parameters
    ----------
    patch : GridPatch
        The local patch (will be modified in place)
    dt : float
        Timestep
    flux_fn : callable
        Flux function
    boundary_type : int
        Boundary condition type
    comm : MPI.Comm
        MPI communicator
    interior_slice : tuple
        Slice for interior cells
    slice_2g : tuple
        Slice for data with ghost cells

    Returns
    -------
    U_interior : jnp.ndarray
        Updated interior data
    """
    # Stage 1: k1 = f(U^n)
    U_interior = patch.data[interior_slice]
    U_with_ghost = patch.data[slice_2g]
    k1 = _compute_k_stage(U_interior, U_with_ghost, flux_fn)

    # Update to intermediate state: U* = U^n + 0.5*dt*k1
    U_star = U_interior + 0.5 * dt * k1
    patch.data = patch.data.at[interior_slice].set(U_star)

    # Sub-step halo exchange: update ghost cells for intermediate state
    update_ghosts_mpi(patch, boundary_type, comm)

    # Stage 2: k2 = f(U*)
    U_star = patch.data[interior_slice]  # Re-extract (data object may have changed)
    U_with_ghost = patch.data[slice_2g]  # Fresh ghost cells!
    k2 = _compute_k_stage(U_star, U_with_ghost, flux_fn)

    # Final update: U^{n+1} = U^n + dt*k2
    # Note: we need original U^n, but we overwrote it. Recompute from U_star.
    U_n = U_star - 0.5 * dt * k1  # Recover U^n
    U_new = U_n + dt * k2

    return U_new


def mpi_rk4_step_substep(patch, dt, flux_fn, boundary_type, comm, interior_slice, slice_2g):
    """
    RK4 time step with sub-step halo exchange (more accurate at boundaries).

    This version updates ghost cells between RK stages, providing better accuracy
    for problems with shocks at domain/subdomain boundaries.
    """
    # Store original U^n
    U_n = patch.data[interior_slice].copy()

    # Stage 1: k1 = f(U^n)
    U_with_ghost = patch.data[slice_2g]
    k1 = _compute_k_stage(U_n, U_with_ghost, flux_fn)

    # Update to U^n + 0.5*dt*k1
    U_stage = U_n + 0.5 * dt * k1
    patch.data = patch.data.at[interior_slice].set(U_stage)
    update_ghosts_mpi(patch, boundary_type, comm)

    # Stage 2: k2 = f(U^n + 0.5*dt*k1)
    U_with_ghost = patch.data[slice_2g]
    k2 = _compute_k_stage(U_stage, U_with_ghost, flux_fn)

    # Update to U^n + 0.5*dt*k2
    U_stage = U_n + 0.5 * dt * k2
    patch.data = patch.data.at[interior_slice].set(U_stage)
    update_ghosts_mpi(patch, boundary_type, comm)

    # Stage 3: k3 = f(U^n + 0.5*dt*k2)
    U_with_ghost = patch.data[slice_2g]
    k3 = _compute_k_stage(U_stage, U_with_ghost, flux_fn)

    # Update to U^n + dt*k3
    U_stage = U_n + dt * k3
    patch.data = patch.data.at[interior_slice].set(U_stage)
    update_ghosts_mpi(patch, boundary_type, comm)

    # Stage 4: k4 = f(U^n + dt*k3)
    U_with_ghost = patch.data[slice_2g]
    k4 = _compute_k_stage(U_stage, U_with_ghost, flux_fn)

    # Final update: U^{n+1} = U^n + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    U_new = U_n + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return U_new


def mpi_muscl_hancock_step_substep(patch, dt, flux_fn, boundary_type, comm, interior_slice, slice_2g):
    """
    MUSCL-Hancock (predictor-corrector) with sub-step halo exchange.
    """
    # Store original U^n
    U_n = patch.data[interior_slice].copy()

    # Predictor: k1 = f(U^n)
    U_with_ghost = patch.data[slice_2g]
    k1 = _compute_k_stage(U_n, U_with_ghost, flux_fn)

    # Update to U^n + dt*k1 (full step predictor)
    U_pred = U_n + dt * k1
    patch.data = patch.data.at[interior_slice].set(U_pred)
    update_ghosts_mpi(patch, boundary_type, comm)

    # Corrector: k2 = f(U^n + dt*k1)
    U_with_ghost = patch.data[slice_2g]
    k2 = _compute_k_stage(U_pred, U_with_ghost, flux_fn)

    # Final update: U^{n+1} = U^n + 0.5*dt*(k1 + k2)
    U_new = U_n + dt * 0.5 * (k1 + k2)

    return U_new
