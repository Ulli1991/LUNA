import jax.numpy as jnp

def sod_shock(cfg):
    Nx, Lx = cfg.Nx, cfg.Lx
    gamma = cfg.gamma

    # 1D coordinate
    x = jnp.linspace(0, Lx, Nx, endpoint=False)

    # Interface
    x0 = 0.5 * Lx

    # Left and right states
    rho_L, rho_R = 1.0, 0.125
    p_L,   p_R   = 1.0, 0.1
    u_L,   u_R   = 0.0, 0.0

    left  = x < x0

    rho = jnp.where(left, rho_L, rho_R)
    u   = jnp.where(left, u_L,   u_R)

    p = jnp.where(left, p_L, p_R)

    # Total energy
    E = p / (gamma - 1.0) + 0.5 * rho * (u**2)

    # Conservative vars [rho, rho*u, rho*v, E]
    U = jnp.stack([rho, rho*u, E], axis=0)
    return U
