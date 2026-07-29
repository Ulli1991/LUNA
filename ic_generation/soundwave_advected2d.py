import jax.numpy as jnp

from ._grid import local_range


def soundwave_advected2d(cfg, x_slice=None, y_slice=None):
    """2D advected sound wave: sinusoidal density perturbation advected diagonally
    (u=v=1). Returns U = [rho, rho*ux, rho*uy, E], shape (4, Nx_local, Ny_local).
    """
    Nx, Ny, Lx, Ly = cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly
    gamma = cfg.gamma

    # Determine local ranges
    x_start, x_end = local_range(Nx, x_slice)

    y_start, y_end = local_range(Ny, y_slice)

    # Create coordinate arrays for local slice only
    x_full = jnp.linspace(0, Lx, Nx, endpoint=False)
    y_full = jnp.linspace(0, Ly, Ny, endpoint=False)
    x_local = x_full[x_start:x_end]
    y_local = y_full[y_start:y_end]
    X, Y = jnp.meshgrid(x_local, y_local, indexing="ij")

    # Sine wave advection
    wavelength = 0.2 * Lx  # wavelength of sine wave
    k = 2 * jnp.pi / wavelength
    wave = jnp.sin(k * X) * jnp.sin(k * Y)

    # Initial conditions
    rho = 1.0 + 0.5 * wave  # density perturbation
    u = jnp.ones_like(rho)  # velocity in x-direction
    v = jnp.ones_like(rho)  # velocity in y-direction
    p = jnp.ones_like(rho)  # constant pressure

    # Total energy
    E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2)

    # Conservative variables [rho, rho*u, rho*v, E]
    U = jnp.stack([rho, rho * u, rho * v, E], axis=0)
    return U
