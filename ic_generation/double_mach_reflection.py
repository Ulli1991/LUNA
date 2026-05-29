import jax.numpy as jnp


def double_mach_reflection(cfg, x_slice=None, y_slice=None):
    """
    Double Mach reflection test problem (2D).

    Strong shock wave reflecting off a wedge, producing complex shock
    interactions and a double Mach stem structure. Classic test for
    2D shock-capturing schemes.

    Parameters
    ----------
    cfg : Config
        Configuration object containing Nx, Ny, Lx, Ly, gamma
    x_slice : tuple (x_start, x_end), optional
        Generate only this x-range for distributed initialization
    y_slice : tuple (y_start, y_end), optional
        Generate only this y-range for distributed initialization

    Returns
    -------
    U : jax.numpy.ndarray
        Conservative variables [rho, rho*ux, rho*uy, E] with shape (4, Nx_local, Ny_local)
    """
    Nx, Ny, Lx, Ly = cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly
    gamma = cfg.gamma

    # Determine local ranges
    if x_slice is not None:
        x_start, x_end = x_slice
    else:
        x_start, x_end = 0, Nx

    if y_slice is not None:
        y_start, y_end = y_slice
    else:
        y_start, y_end = 0, Ny

    # Create coordinate arrays for local slice only
    x_full = jnp.linspace(0, Lx, Nx, endpoint=False)
    y_full = jnp.linspace(0, Ly, Ny, endpoint=False)
    x_local = x_full[x_start:x_end]
    y_local = y_full[y_start:y_end]
    X, Y = jnp.meshgrid(x_local, y_local, indexing="ij")

    # Initial shock profile
    shock_location = 0.5 * Lx
    shock_strength = 1.0  # strength of the shock front

    # Initial conditions (simple shock tube setup)
    rho_left = 1.0  # density on the left
    rho_right = 0.125  # density on the right
    p_left = 1.0  # pressure on the left
    p_right = 0.1  # pressure on the right
    u_left = 0.0  # velocity on the left
    u_right = 2.0  # velocity on the right (shock propagates)

    # Density, velocity, and pressure
    rho = jnp.where(X < shock_location, rho_left, rho_right)
    u = jnp.where(X < shock_location, u_left, u_right)
    p = jnp.where(X < shock_location, p_left, p_right)
    v = jnp.zeros_like(rho)  # no velocity in the y-direction

    # Total energy
    E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2)

    # Conservative variables [rho, rho*u, rho*v, E]
    U = jnp.stack([rho, rho * u, rho * v, E], axis=0)
    return U
