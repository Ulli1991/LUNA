import jax.numpy as jnp

def double_mach_reflection(cfg):
    Nx, Ny, Lx, Ly = cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly
    gamma = cfg.gamma

    # 2D coordinates
    x = jnp.linspace(0, Lx, Nx, endpoint=False)
    y = jnp.linspace(0, Ly, Ny, endpoint=False)
    X, Y = jnp.meshgrid(x, y, indexing='ij')

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
    U = jnp.stack([rho, rho*u, rho*v, E], axis=0)
    return U

