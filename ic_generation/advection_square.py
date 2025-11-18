import jax.numpy as jnp

def advection_square(cfg):
    Nx, Ny, Lx, Ly = cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly
    gamma = cfg.gamma

    # 2D coordinates
    x = jnp.linspace(0, Lx, Nx, endpoint=False)
    y = jnp.linspace(0, Ly, Ny, endpoint=False)
    X, Y = jnp.meshgrid(x, y)

    # Square in the middle of the domain
    square = (X > 0.3 * Lx) & (X < 0.7 * Lx) & (Y > 0.3 * Ly) & (Y < 0.7 * Ly)

    # Initial conditions: density = 1, velocity = 1, pressure = 1 (uniform)
    rho = square.astype(jnp.float32) + 1.0  # square with value 1 inside
    u = jnp.zeros_like(rho)  # zero velocity
    v = jnp.zeros_like(rho)  # zero velocity
    p = jnp.ones_like(rho)  # constant pressure

    # Total energy
    E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2)

    # Conservative variables [rho, rho*u, rho*v, E]
    U = jnp.stack([rho, rho*u, rho*v, E], axis=0)
    return U

