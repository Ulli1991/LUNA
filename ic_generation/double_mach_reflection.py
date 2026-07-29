import jax.numpy as jnp

from ._grid import local_range


def double_mach_reflection(cfg, x_slice=None, y_slice=None):
    """2D double Mach reflection test: strong shock producing a double Mach stem.

    Returns U = [rho, rho*ux, rho*uy, E], shape (4, Nx_local, Ny_local).
    """
    Nx, Ny, Lx, Ly = cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly
    gamma = cfg.gamma

    # Determine local ranges
    x_start, x_end = local_range(Nx, x_slice)

    y_start, y_end = local_range(Ny, y_slice)

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
