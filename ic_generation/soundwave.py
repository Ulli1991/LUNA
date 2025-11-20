import jax.numpy as jnp

def soundwave(cfg):
    """
    Linear sound wave test problem (1D).

    Small-amplitude sinusoidal perturbation in density and pressure
    that propagates as an acoustic wave. Used to test accuracy and
    numerical dissipation of the scheme.

    Parameters
    ----------
    cfg : Config
        Configuration object containing Nx, Lx, gamma

    Returns
    -------
    U : jax.numpy.ndarray
        Conservative variables [rho, rho*u, E] with shape (3, Nx)
    """
    Nx, Lx = cfg.Nx, cfg.Lx
    gamma = cfg.gamma

    x = jnp.linspace(0, Lx, Nx, endpoint=False)

    rho0 =    1.0
    p0   =    1.0
    u0   =    0.0
    epsilon = 1e-6
    m =       1

    k = 2.0 * jnp.pi * m / Lx

    # Small-amplitude density perturbation
    rho = rho0 * (1.0 + epsilon * jnp.sin(k * x))
    u   = jnp.zeros_like(x)
    p   = p0 * (1.0 + epsilon * jnp.sin(k * x))

    E = p/(gamma-1) + 0.5 * rho * u**2
    return jnp.stack([rho, rho*u, E], axis=0)

