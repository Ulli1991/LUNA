def shu_osher(cfg):
    Nx, Lx = cfg.Nx, cfg.Lx
    gamma = cfg.gamma

    x = jnp.linspace(0, Lx, Nx, endpoint=False)

    # left state
    rho_L = 3.857143
    u_L   = 2.629369
    p_L   = 10.33333

    # right oscillatory region
    rho_R = 1.0 + 0.2 * jnp.sin(5.0 * x)
    u_R   = 0.0
    p_R   = 1.0

    left = x < -4.0  # assumed domain [-5,5]

    rho = jnp.where(left, rho_L, rho_R)
    u   = jnp.where(left, u_L,   u_R)
    p   = jnp.where(left, p_L,   p_R)

    E = p/(gamma-1) + 0.5 * rho * u**2

    return jnp.stack([rho, rho*u, E], axis=0)

