import jax.numpy as jnp

from ._grid import local_range


def woodward_colella(cfg, x_slice=None):
    """1D Woodward-Colella interacting blast waves: rho=1, u=0, p = 1000 / 0.01 / 100
    in x < 0.1Lx / middle / x >= 0.9Lx.
    Returns U = [rho, rho*u, E], shape (3, Nx_local).
    """
    Nx, Lx = cfg.Nx, cfg.Lx
    gamma = cfg.gamma

    # Determine local range
    x_start, x_end = local_range(Nx, x_slice)

    # Create coordinate array for local slice only
    x_full = jnp.linspace(0, Lx, Nx, endpoint=False)
    x = x_full[x_start:x_end]

    # Region interfaces
    x1 = 0.1 * Lx
    x2 = 0.9 * Lx

    # Masks
    left = x < x1
    middle = (x >= x1) & (x < x2)
    right = x >= x2

    # Density (same in all regions)
    rho = jnp.ones_like(x)

    # Velocity
    u = jnp.zeros_like(x)

    # Pressure
    p = jnp.where(left, 1000.0, jnp.where(middle, 0.01, 100.0))

    # Total energy
    E = p / (gamma - 1.0) + 0.5 * rho * (u**2)

    # Conservative vars [rho, rho*u, E]
    U = jnp.stack([rho, rho * u, E], axis=0)
    return U
