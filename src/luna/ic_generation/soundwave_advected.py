"""1D sound wave advected by a uniform flow."""
import jax.numpy as jnp

from ._grid import local_range


def soundwave_advected(cfg, x_slice=None):
    """U = [rho, rho*u, E] for a small perturbation on a flow at U0 = 1."""
    Nx, Lx = cfg.Nx, cfg.Lx
    gamma = cfg.gamma

    x_start, x_end = local_range(Nx, x_slice)

    x_full = (jnp.arange(Nx) + 0.5) * (Lx / Nx)
    x = x_full[x_start:x_end]

    rho0 = 1.0
    p0 = 1.0
    U0 = 1.0
    epsilon = 1e-6
    m = 1

    k = 2.0 * jnp.pi * m / Lx

    rho = rho0 * (1.0 + epsilon * jnp.sin(k * x))
    u = U0 + epsilon * jnp.sin(k * x)
    p = p0 * (1.0 + epsilon * jnp.sin(k * x))

    E = p / (gamma - 1) + 0.5 * rho * u**2
    return jnp.stack([rho, rho * u, E], axis=0)
