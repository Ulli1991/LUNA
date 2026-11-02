"""1D Sod shock tube."""
import jax.numpy as jnp

from ._grid import local_range


def sod_shock(cfg, x_slice=None):
    """U = [rho, rho*u, E] with the jump at Lx/2, L=(1, 0, 1), R=(0.125, 0, 0.1)."""
    Nx, Lx = cfg.Nx, cfg.Lx
    gamma = cfg.gamma

    x_start, x_end = local_range(Nx, x_slice)

    x_full = (jnp.arange(Nx) + 0.5) * (Lx / Nx)
    x = x_full[x_start:x_end]

    x0 = 0.5 * Lx

    rho_L, rho_R = 1.0, 0.125
    p_L, p_R = 1.0, 0.1
    u_L, u_R = 0.0, 0.0

    left = x < x0

    rho = jnp.where(left, rho_L, rho_R)
    u = jnp.where(left, u_L, u_R)

    p = jnp.where(left, p_L, p_R)

    E = p / (gamma - 1.0) + 0.5 * rho * (u**2)

    U = jnp.stack([rho, rho * u, E], axis=0)
    return U
