import jax.numpy as jnp


def soundwave_advected(cfg, x_slice=None):
    """
    Advected sound wave test problem (1D).

    Small-amplitude acoustic wave superimposed on a uniform flow (U0 = 1.0).
    Tests the scheme's ability to maintain wave structure while advecting.

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
    else:
        x_start, x_end = 0, Nx

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
