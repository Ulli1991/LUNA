import jax.numpy as jnp


def advection_square(cfg, x_slice=None, y_slice=None):
    """
    Square advection test problem (2D).

    Square density region advected diagonally with uniform velocity (u=v=1).
    Tests the scheme's ability to preserve sharp discontinuities during advection.

    Parameters
    ----------
    cfg : Config
        Configuration object containing Nx, Ny, Lx, Ly, gamma
    x_slice : tuple (x_start, x_end), optional
        Generate only this x-range for distributed initialization
    y_slice : tuple (y_start, y_end), optional
        Generate only this y-range for distributed initialization

    Returns
    -------
    U : jax.numpy.ndarray
        Conservative variables [rho, rho*ux, rho*uy, E] with shape (4, Nx_local, Ny_local)
    """
    Nx, Ny, Lx, Ly = cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly
    gamma = cfg.gamma

    # Determine local ranges
    if x_slice is not None:
        x_start, x_end = x_slice
    else:
        x_start, x_end = 0, Nx

    if y_slice is not None:
        y_start, y_end = y_slice
    else:
        y_start, y_end = 0, Ny

    dx = Lx / Nx
    dy = Ly / Ny
    x_full = (jnp.arange(Nx) + 0.5) * dx
    y_full = (jnp.arange(Ny) + 0.5) * dy
    x_local = x_full[x_start:x_end]
    y_local = y_full[y_start:y_end]
    X, Y = jnp.meshgrid(x_local, y_local, indexing="ij")

    # Square in the middle of the domain
    square = (X > 0.25 * Lx) & (X < 0.75 * Lx) & (Y > 0.25 * Ly) & (Y < 0.75 * Ly)

    rho = square.astype(jnp.float64) + 1.0  # square with value 1 inside
    u = jnp.ones_like(rho)  # vx = 1
    v = jnp.ones_like(rho)  # vy = 1
    p = jnp.ones_like(rho)  # constant pressure

    # Total energy
    E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2)

    # Conservative variables [rho, rho*u, rho*v, E]
    U = jnp.stack([rho, rho * u, rho * v, E], axis=0)
    return U
