import jax.numpy as jnp


def sod_shock(cfg, x_slice=None):
    """1D Sod shock tube: discontinuity at 0.5*Lx, L=(1, 0, 1), R=(0.125, 0, 0.1).
    Returns U = [rho, rho*u, E], shape (3, Nx_local).
    """
    Nx, Lx = cfg.Nx, cfg.Lx
    gamma = cfg.gamma

    # Determine local range
    if x_slice is not None:
        x_start, x_end = x_slice
        Nx_local = x_end - x_start
    else:
        x_start, x_end = 0, Nx
        Nx_local = Nx

    # Create coordinate array for local slice only
    x_full = jnp.linspace(0, Lx, Nx, endpoint=False)
    x = x_full[x_start:x_end]

    # Interface
    x0 = 0.5 * Lx

    # Left and right states
    rho_L, rho_R = 1.0, 0.125
    p_L, p_R = 1.0, 0.1
    u_L, u_R = 0.0, 0.0

    left = x < x0

    rho = jnp.where(left, rho_L, rho_R)
    u = jnp.where(left, u_L, u_R)

    p = jnp.where(left, p_L, p_R)

    # Total energy
    E = p / (gamma - 1.0) + 0.5 * rho * (u**2)

    # Conservative vars [rho, rho*u, rho*v, E]
    U = jnp.stack([rho, rho * u, E], axis=0)
    return U
