import jax.numpy as jnp


def soundwave(cfg, x_slice=None):
    """
    Linear sound wave test problem (1D) — single right-going acoustic eigenmode.

    A small-amplitude perturbation aligned with the right-going acoustic
    eigenvector of the linearized Euler equations:
        δρ = A sin(kx)
        δu = A · cs / ρ₀ · sin(kx)
        δp = A · cs² · sin(kx)
    where cs² = γ · p₀ / ρ₀. The wave propagates at cs and returns exactly
    to the IC after one period t = Lx / cs.

    Single-eigenmode (not u=0 superposition) so the test matches AthenaK's
    linear_wave problem generator one-for-one. Default background:
    ρ₀ = 1, p₀ = 0.6, γ = 5/3 → cs² = 1, period = Lx.

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

    if x_slice is not None:
        x_start, x_end = x_slice
    else:
        x_start, x_end = 0, Nx

    # Cell-centered coordinates (finite-volume convention)
    dx = Lx / Nx
    x_full = (jnp.arange(Nx) + 0.5) * dx
    x = x_full[x_start:x_end]

    rho0 = 1.0
    p0   = 0.6      # with γ=5/3 → cs² = 1, period = Lx
    amp  = 1.0e-4   # eigenmode amplitude A
    m    = 1

    cs = jnp.sqrt(gamma * p0 / rho0)
    k = 2.0 * jnp.pi * m / Lx
    sin_kx = jnp.sin(k * x)

    single_mode = getattr(cfg, "soundwave_single_mode", True)
    if single_mode:
        # Right-going acoustic eigenmode (matches AthenaK linear_wave).
        rho = rho0 + amp * sin_kx
        u   = amp * cs / rho0 * sin_kx
        p   = p0 + amp * cs * cs * sin_kx
    else:
        # u=0 superposition of left- and right-going acoustic waves of equal
        # amplitude — standing-wave physics, the OLD soundwave setup.
        rho = rho0 + amp * sin_kx
        u   = jnp.zeros_like(x)
        p   = p0 + amp * sin_kx

    E = p / (gamma - 1.0) + 0.5 * rho * u * u
    return jnp.stack([rho, rho * u, E], axis=0)
