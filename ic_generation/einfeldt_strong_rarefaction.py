import jax.numpy as jnp


def einfeldt_strong_rarefaction(cfg, x_slice=None):
    """
    Einfeldt strong rarefaction test problem (1D/2D).

    Two states with strong opposing velocities creating a rarefaction wave.
    From Athena test suite (1-2-0-3).
    Left: (ρ=1, m=-2, n=0, e=3)
    Right: (ρ=1, m=+2, n=0, e=3)

    Parameters
    ----------
    cfg : Config
        Configuration object containing Nx, Lx, gamma
    x_slice : tuple (x_start, x_end), optional
        Generate only this x-range for distributed initialization

    Returns
    -------
    U : jax.numpy.ndarray
        Conservative variables [rho, rho*u, rho*v, E] with shape (4, Nx_local)
    """
    Nx, Lx = cfg.Nx, cfg.Lx
    gamma = cfg.gamma

    # Determine local range
    if x_slice is not None:
        x_start, x_end = x_slice
    else:
        x_start, x_end = 0, Nx

    # Create coordinate array for local slice only
    x_full = jnp.linspace(0, Lx, Nx, endpoint=False)
    x = x_full[x_start:x_end]
    x0 = 0.5 * Lx
    left = x < x0

    rho_L, rho_R = 1.0, 1.0
    m_L, m_R = -2.0, 2.0
    n_L, n_R = 0.0, 0.0
    e_L, e_R = 3.0, 3.0

    u_L, u_R = m_L / rho_L, m_R / rho_R
    v_L, v_R = n_L / rho_L, n_R / rho_R
    E_L, E_R = e_L, e_R

    rho = jnp.where(left, rho_L, rho_R)
    u = jnp.where(left, u_L, u_R)
    v = jnp.where(left, v_L, v_R)
    E = jnp.where(left, E_L, E_R)

    p = (gamma - 1.0) * (E - 0.5 * rho * (u**2 + v**2))
    # 2D conservative variables: [rho, rho*u, rho*v, E]
    U = jnp.stack([rho, rho * u, rho * v, E], axis=0)
    return U
