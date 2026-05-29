import jax.numpy as jnp


def sod_shock(cfg, x_slice=None):
    """
    Sod shock tube test problem (1D).

    Classic Riemann problem with a discontinuity at x=0.5*Lx.
    Left state: rho=1.0, p=1.0, u=0.0
    Right state: rho=0.125, p=0.1, u=0.0

    Generates a shock wave, contact discontinuity, and rarefaction wave.

    Parameters
    ----------
    cfg : Config
        Configuration object containing Nx, Lx, gamma
    x_slice : tuple (x_start, x_end), optional
        Generate only this x-range for distributed initialization

    Returns
    -------
    U : jax.numpy.ndarray
        Conservative variables [rho, rho*u, E] with shape (3, Nx_local)
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
