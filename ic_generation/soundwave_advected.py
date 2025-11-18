def soundwave_advected(cfg):
    Nx, Lx = cfg.Nx, cfg.Lx
    gamma = cfg.gamma

    x = jnp.linspace(0, Lx, Nx, endpoint=False)

    # Base state
    rho0    = 1.0
    p0      = 1.0
    U0      = 1.0
    epsilon = 1e-3 
    m       = 1

    k = 2.0 * jnp.pi * m / Lx

    # Perturbed fields
    rho = rho0 * (1.0 + epsilon * jnp.sin(k * x))
    u   = U0  + epsilon * jnp.sin(k * x)
    p   = p0 * (1.0 + epsilon * jnp.sin(k * x))

    E = p/(gamma-1) + 0.5 * rho * u**2
    return jnp.stack([rho, rho*u, E], axis=0)

