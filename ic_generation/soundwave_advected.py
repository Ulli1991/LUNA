import jax.numpy as jnp

from ._grid import local_range


def soundwave_advected(cfg, x_slice=None):
    """1D advected sound wave: small acoustic perturbation on a uniform flow (U0=1).
    Returns U = [rho, rho*u, E], shape (3, Nx_local).
    """
    Nx, Lx = cfg.Nx, cfg.Lx
    gamma = cfg.gamma

    # Determine local range
    x_start, x_end = local_range(Nx, x_slice)

    # Create coordinate array for local slice only
    x_full = jnp.linspace(0, Lx, Nx, endpoint=False)
    x = x_full[x_start:x_end]

    # Base state
    rho0 = 1.0
    p0 = 1.0
    U0 = 1.0
    epsilon = 1e-6
    m = 1

    k = 2.0 * jnp.pi * m / Lx

    # Perturbed fields
    rho = rho0 * (1.0 + epsilon * jnp.sin(k * x))
    u = U0 + epsilon * jnp.sin(k * x)
    p = p0 * (1.0 + epsilon * jnp.sin(k * x))

    E = p / (gamma - 1) + 0.5 * rho * u**2
    return jnp.stack([rho, rho * u, E], axis=0)
