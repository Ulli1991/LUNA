"""2D square density advected diagonally."""
import jax.numpy as jnp

from ._grid import local_range


def advection_square(cfg, x_slice=None, y_slice=None):
    """U = [rho, rho*u, rho*v, E] for a square of rho = 2 moving at u = v = 1."""
    Nx, Ny, Lx, Ly = cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly
    gamma = cfg.gamma

    x_start, x_end = local_range(Nx, x_slice)

    y_start, y_end = local_range(Ny, y_slice)

    dx = Lx / Nx
    dy = Ly / Ny
    x_full = (jnp.arange(Nx) + 0.5) * dx
    y_full = (jnp.arange(Ny) + 0.5) * dy
    x_local = x_full[x_start:x_end]
    y_local = y_full[y_start:y_end]
    X, Y = jnp.meshgrid(x_local, y_local, indexing="ij")

    square = (X > 0.25 * Lx) & (X < 0.75 * Lx) & (Y > 0.25 * Ly) & (Y < 0.75 * Ly)

    rho = square.astype(jnp.float64) + 1.0
    u = jnp.ones_like(rho)
    v = jnp.ones_like(rho)
    p = jnp.ones_like(rho)

    E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2)

    U = jnp.stack([rho, rho * u, rho * v, E], axis=0)
    return U
