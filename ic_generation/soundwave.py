def soundwave(cfg):
    Nx, Lx = cfg.Nx, cfg.Lx
    gamma = cfg.gamma

    x = jnp.linspace(0, Lx, Nx, endpoint=False)

    rho0 =    1.0
    p0   =    1.0
    u0   =    0.0
    epsilon = 1e-3
    m =       1

    k = 2.0 * jnp.pi * m / Lx

    # Small-amplitude density perturbation
    rho = rho0 * (1.0 + epsilon * jnp.sin(k * x))
    u   = jnp.zeros_like(x)
    p   = p0 * (1.0 + epsilon * jnp.sin(k * x))

    E = p/(gamma-1) + 0.5 * rho * u**2
    return jnp.stack([rho, rho*u, E], axis=0)

