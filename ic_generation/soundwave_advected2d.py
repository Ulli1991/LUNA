import jax.numpy as jnp

def soundwave_advected2d(cfg):
    Nx, Ny, Lx, Ly = cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly
    gamma = cfg.gamma

    # 2D coordinates
    x = jnp.linspace(0, Lx, Nx, endpoint=False)
    y = jnp.linspace(0, Ly, Ny, endpoint=False)
    X, Y = jnp.meshgrid(x, y)

    # Sine wave advection
    wavelength = 0.2 * Lx  # wavelength of sine wave
    k = 2 * jnp.pi / wavelength
    wave = jnp.sin(k * X) * jnp.sin(k * Y)

    # Initial conditions
    rho = 1.0 + 0.5 * wave  # density perturbation
    u = jnp.ones_like(rho)  # velocity in x-direction
    v = jnp.ones_like(rho)  # velocity in y-direction
    p = jnp.ones_like(rho)  # constant pressure

    # Total energy
    E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2)

    # Conservative variables [rho, rho*u, rho*v, E]
    U = jnp.stack([rho, rho*u, rho*v, E], axis=0)
    return U

