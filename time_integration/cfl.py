import jax.numpy as jnp

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
    dt : float32
        Computed timestep, possibly adjusted to ensure snapshot time alignment
    """
    # Ensure all variables are float32
    U = U.astype(jnp.float32)
    dx = jnp.float32(dx)
    if dy is not None:
        dy = jnp.float32(dy)
    if dz is not None:
        dz = jnp.float32(dz)
    t = jnp.float32(t)
    t_snapshot = jnp.float32(t_snapshot)
    gamma = jnp.float32(gamma)
    CFL = jnp.float32(CFL)

    rho = U[0]
    vx = U[1] / rho
    vy = U[2] / rho if U.shape[0] >= 4 else jnp.float32(0.0)
    vz = U[3] / rho if U.shape[0] >= 5 else jnp.float32(0.0)

    # Kinetic energy
    kinetic = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    # Pressure
    p = (gamma - 1.0) * (U[-1] - kinetic)
    # Sound speed
    c = jnp.sqrt(gamma * p / rho)

    # Total velocity magnitude + sound speed
    smax = jnp.sqrt(vx**2 + vy**2 + vz**2) + c

    # Minimum grid spacing in available directions
    spacing = [dx]
    if dy is not None:
        spacing.append(dy)
    if dz is not None:
        spacing.append(dz)
    dx_min = jnp.min(jnp.array(spacing, dtype=jnp.float32))

    # Standard CFL timestep
    dt_cfl = CFL * dx_min / jnp.max(smax)

    # Compute the time difference to the next snapshot
    time_to_snapshot = t_snapshot - t

    # If the time difference is smaller than the Courant timestep, adjust dt
    dt = jnp.where(time_to_snapshot < dt_cfl, time_to_snapshot, dt_cfl)

    return dt.astype(jnp.float32)

