import jax.numpy as jnp
from jax import jit, pmap
from functools import partial

@jit
def compute_dt(U, gamma, CFL, t, t_snapshot, dx, dy=None, dz=None):
    """
    Compute timestep based on CFL condition and ensure exact snapshot time alignment.

    Parameters
    ----------
    U : array
        Conservative variables, shape (Nvar, Nx[, Ny[, Nz]])
        Order: [rho, rho*vx, rho*vy, rho*vz, E]
    dx, dy, dz : float
        Grid spacings (dy and dz optional for lower dimensions)
    gamma : float
        Adiabatic index
    CFL : float
        CFL number
    t : float
        Current simulation time
    t_snapshot : float
        Next snapshot time

    Returns
    -------
    dt : float
        Computed timestep, possibly adjusted to ensure snapshot time alignment
    """
    # Cache reciprocal (used for velocities and sound speed)
    rho = U[0]
    inv_rho = 1.0 / rho

    # Compute velocities (using cached reciprocal)
    vx = U[1] * inv_rho
    vy = U[2] * inv_rho if U.shape[0] >= 4 else 0.0
    vz = U[3] * inv_rho if U.shape[0] >= 5 else 0.0

    # Velocity squared (computed once and reused)
    v2 = vx**2 + vy**2 + vz**2

    # Kinetic energy
    kinetic = 0.5 * rho * v2
    # Pressure
    p = (gamma - 1.0) * (U[-1] - kinetic)
    # Sound speed (using cached reciprocal)
    c = jnp.sqrt(gamma * p * inv_rho)

    # Total velocity magnitude + sound speed
    smax = jnp.sqrt(v2) + c

    # Minimum grid spacing in available directions
    spacing = [dx]
    if dy is not None:
        spacing.append(dy)
    if dz is not None:
        spacing.append(dz)
    dx_min = jnp.min(jnp.array(spacing))

    # Standard CFL timestep
    dt_cfl = CFL * dx_min / jnp.max(smax)

    # Compute the time difference to the next snapshot
    time_to_snapshot = t_snapshot - t

    # If the time difference is smaller than the Courant timestep, adjust dt
    dt = jnp.where(time_to_snapshot < dt_cfl, time_to_snapshot, dt_cfl)

    return dt


@jit
def compute_dt_and_diagnostics(U, gamma, CFL, t, t_snapshot, dx, dy=None, dz=None):
    """
    Fused function: compute timestep AND physics diagnostics in one pass.
    Eliminates duplicate computation of velocities, kinetic energy, and pressure.

    This combines the functionality of compute_dt() and compute_physics_diagnostics()
    to avoid redundant calculations that were happening every timestep.

    Parameters
    ----------
    U : array
        Conservative variables, shape (Nvar, Nx[, Ny[, Nz]])
    gamma : float
        Adiabatic index
    CFL : float
        CFL number
    t : float
        Current simulation time
    t_snapshot : float
        Next snapshot time
    dx, dy, dz : float
        Grid spacings (dy and dz optional for lower dimensions)

    Returns
    -------
    dt : float
        Computed timestep
    min_rho : float
        Minimum density in domain
    min_p : float
        Minimum pressure in domain
    has_nan : bool
        True if any NaN detected
    has_inf : bool
        True if any Inf detected
    """
    # Check for NaN/Inf first (cheap operation)
    has_nan = jnp.any(jnp.isnan(U))
    has_inf = jnp.any(jnp.isinf(U))

    # Cache reciprocal (used for velocities and sound speed)
    rho = U[0]
    inv_rho = 1.0 / rho

    # Compute velocities (using cached reciprocal)
    vx = U[1] * inv_rho
    vy = U[2] * inv_rho if U.shape[0] >= 4 else 0.0
    vz = U[3] * inv_rho if U.shape[0] >= 5 else 0.0

    # Velocity squared (computed once, used multiple times)
    v2 = vx**2 + vy**2 + vz**2

    # Kinetic energy (computed once, used for both pressure and dt)
    kinetic = 0.5 * rho * v2

    # Pressure (computed once, used for both sound speed and diagnostics)
    p = (gamma - 1.0) * (U[-1] - kinetic)

    # Sound speed (using cached reciprocal)
    c = jnp.sqrt(gamma * p * inv_rho)

    # Total velocity magnitude + sound speed (for dt)
    smax = jnp.sqrt(v2) + c

    # Minimum grid spacing
    spacing = [dx]
    if dy is not None:
        spacing.append(dy)
    if dz is not None:
        spacing.append(dz)
    dx_min = jnp.min(jnp.array(spacing))

    # Standard CFL timestep
    dt_cfl = CFL * dx_min / jnp.max(smax)

    # Adjust for snapshot time
    # If time_to_snapshot is zero or negative (already at/past snapshot), use CFL dt
    time_to_snapshot = t_snapshot - t
    dt = jnp.where(time_to_snapshot > 1e-14,
                   jnp.minimum(time_to_snapshot, dt_cfl),
                   dt_cfl)

    # Physics diagnostics (for validation)
    min_rho = jnp.min(rho)
    min_p = jnp.min(p)

    return dt, min_rho, min_p, has_nan, has_inf


def create_compute_local_dt_pmap(gamma, CFL, dx, dy, dz):
    """
    Create a pmapped function to compute dt for local subdomains on each GPU.

    Parameters
    ----------
    gamma : float
        Adiabatic index
    CFL : float
        CFL number
    dx, dy, dz : float
        Grid spacing

    Returns
    -------
    compute_local_dt : callable
        Pmapped function with signature (U_local, t, t_snapshot) -> (dt, min_rho, min_p, has_nan, has_inf)
        where U_local has shape (n_devices, ...) and t, t_snapshot are arrays of shape (n_devices,)
    """
    @partial(pmap, in_axes=(0, 0, 0), axis_name='batch')
    def compute_local_dt(U_local_single, t, t_snapshot):
        """Compute dt for local subdomain."""
        dt_local, min_rho, min_p, has_nan, has_inf = compute_dt_and_diagnostics(
            U_local_single, gamma, CFL, t, t_snapshot, dx, dy, dz
        )
        return dt_local, min_rho, min_p, has_nan, has_inf

    return compute_local_dt


def create_compute_dt_sharded(gamma, CFL, dx, dy, dz, enable_diagnostics=True):
    """
    Create a JIT-compiled function to compute dt for sharded arrays with automatic sharding.

    With automatic sharding, JAX handles the communication automatically, so we just
    compute on the sharded array and do global reductions (min/any).

    Parameters
    ----------
    gamma : float
        Adiabatic index
    CFL : float
        CFL number
    dx, dy, dz : float
        Grid spacing
    enable_diagnostics : bool
        If True, compute full diagnostics (min_rho, min_p, NaN/Inf checks)
        If False, only compute dt for maximum performance

    Returns
    -------
    compute_dt_fn : callable
        JIT-compiled function with signature (U, t, t_snapshot) -> (dt, min_rho, min_p, has_nan, has_inf)
        For sharded arrays, this automatically handles global reductions across all devices
    """
    if enable_diagnostics:
        @jit
        def compute_dt_sharded(U, t_val, t_snapshot_val):
            """Compute dt and diagnostics with automatic sharding."""
            # Compute on sharded array (JAX handles communication automatically)
            dt_local, min_rho_local, min_p_local, has_nan_local, has_inf_local = \
                compute_dt_and_diagnostics(U, gamma, CFL, t_val, t_snapshot_val, dx, dy, dz)

            # Global reductions across all shards
            dt = jnp.min(dt_local)
            min_rho = jnp.min(min_rho_local)
            min_p = jnp.min(min_p_local)
            has_nan = jnp.any(has_nan_local)
            has_inf = jnp.any(has_inf_local)

            return dt, min_rho, min_p, has_nan, has_inf
    else:
        # Fast path: dt only, no diagnostics (10-20% speedup)
        @jit
        def compute_dt_sharded(U, t_val, t_snapshot_val):
            """Compute dt only (no diagnostics) - maximum performance."""
            dt_local = compute_dt(U, gamma, CFL, t_val, t_snapshot_val, dx, dy, dz)
            dt = jnp.min(dt_local)
            # Return dummy values for diagnostics for consistent interface
            return dt, 0.0, 0.0, False, False

    return compute_dt_sharded

