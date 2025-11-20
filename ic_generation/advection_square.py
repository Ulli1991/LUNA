import jax.numpy as jnp

def advection_square(cfg):
    """
    Square advection test problem (2D).

    Square density region advected diagonally with uniform velocity (u=v=1).
    Tests the scheme's ability to preserve sharp discontinuities during advection.

    Parameters
    ----------
    cfg : Config
        Configuration object containing Nx, Ny, Lx, Ly, gamma

    Returns
    -------
    U : jax.numpy.ndarray
        Conservative variables [rho, rho*ux, rho*uy, E] with shape (4, Nx, Ny)
    """
    Nx, Ny, Lx, Ly = cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly
    gamma = cfg.gamma

    # 2D coordinates
    x = jnp.linspace(0, Lx, Nx, endpoint=False)
    y = jnp.linspace(0, Ly, Ny, endpoint=False)
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    # Square in the middle of the domain
    square = (X > 0.25 * Lx) & (X < 0.75 * Lx) & (Y > 0.25 * Ly) & (Y < 0.75 * Ly)

    # Initial conditions: density = 1, velocity = 1, pressure = 1 (uniform)
    rho = square.astype(jnp.float32) + 1.0  # square with value 1 inside
    u = jnp.ones_like(rho)  # vx = 1
    v = jnp.ones_like(rho)  # vy = 1
    p = jnp.ones_like(rho)  # constant pressure

    # Total energy
    E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2)

    # Conservative variables [rho, rho*u, rho*v, E]
    U = jnp.stack([rho, rho*u, rho*v, E], axis=0)
    return U

