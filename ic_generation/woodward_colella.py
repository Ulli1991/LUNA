import jax.numpy as jnp

def woodward_colella(cfg):
    """
    Woodward-Colella blast wave interaction test problem (1D).

    Two strong blast waves propagating toward each other in a tube.
    Tests the scheme's ability to handle strong shocks and extreme
    pressure ratios (p_max/p_min = 100,000).

    Regions:
    - Left (x < 0.1): p = 1000
    - Middle (0.1 <= x < 0.9): p = 0.01
    - Right (x >= 0.9): p = 100

    Parameters
    ----------
    cfg : Config
        Configuration object containing Nx, Lx, gamma

    Returns
    -------
    U : jax.numpy.ndarray
        Conservative variables [rho, rho*u, E] with shape (3, Nx)
    """
    Nx, Lx = cfg.Nx, cfg.Lx
    gamma = cfg.gamma

    # 1D coordinate
    x = jnp.linspace(0, Lx, Nx, endpoint=False)

    # Region interfaces
    x1 = 0.1 * Lx
    x2 = 0.9 * Lx

    # Masks
    left   = (x < x1)
    middle = (x >= x1) & (x < x2)
    right  = (x >= x2)

    # Density (same in all regions)
    rho = jnp.ones_like(x)

    # Velocity
    u = jnp.zeros_like(x)

    # Pressure
    p = (
        jnp.where(left,   1000.0,
        jnp.where(middle,   0.01,
                              100.0))
    )

    # Total energy
    E = p / (gamma - 1.0) + 0.5 * rho * (u**2)

    # Conservative vars [rho, rho*u, E]
    U = jnp.stack([rho, rho*u, E], axis=0)
    return U

