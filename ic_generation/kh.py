import jax.numpy as jnp

def kelvin_helmholtz(cfg):
    Nx, Ny = cfg.Nx, cfg.Ny
    Lx, Ly = cfg.Lx, cfg.Ly
    gamma = cfg.gamma

    x = jnp.linspace(0, Lx, Nx, endpoint=False)
    y = jnp.linspace(0, Ly, Ny, endpoint=False)
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    # KH parameters
    U0 = 1.0
    a = 0.05
    A = 0.01
    sigma = 0.2
    Y1, Y2 = 0.5, 1.5
    rho_low, rho_high = 1.0, 2.0
    delta = (rho_high - rho_low) / rho_low

    rho = 1.0 + 0.5 * delta * (jnp.tanh((Y - Y1)/a) - jnp.tanh((Y - Y2)/a))
    ux  = -U0 * ((jnp.tanh((Y - Y1)/a) - jnp.tanh((Y - Y2)/a)) - 1.0)
    uy  = A * jnp.sin(2*jnp.pi*X) * (
            jnp.exp(-((Y - Y1)**2)/sigma**2) +
            jnp.exp(-((Y - Y2)**2)/sigma**2))

    p = jnp.ones((Nx, Ny)) * 10.0
    E = p / (gamma - 1.0) + 0.5 * rho * (ux**2 + uy**2)

    U = jnp.stack([rho, rho*ux, rho*uy, E], axis=0)

    return U
