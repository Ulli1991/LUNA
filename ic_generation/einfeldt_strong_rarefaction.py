import jax.numpy as jnp

def strong_rarefaction(cfg):
    """
    Einfeldt strong rarefaction.
    Left:  (ρ = 1, m = −2, e = 3)
    Right: (ρ = 1, m = +2, e = 3)
    """
    Nx, Lx = cfg.Nx, cfg.Lx
    gamma = cfg.gamma

    x = jnp.linspace(0, Lx, Nx, endpoint=False)
    x0 = 0.5 * Lx
    left = x < x0

    rho_L, rho_R = 1.0, 1.0
    m_L, m_R     = -2.0, 2.0
    e_L, e_R     = 3.0, 3.0

    u_L, u_R = m_L / rho_L, m_R / rho_R
    E_L, E_R = e_L, e_R

    rho = jnp.where(left, rho_L, rho_R)
    u   = jnp.where(left, u_L,   u_R)
    E   = jnp.where(left, E_L,   E_R)

    p = (gamma - 1.0) * (E - 0.5 * rho * u**2)
    U = jnp.stack([rho, rho*u, E], axis=0)
    return U

