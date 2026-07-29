import jax.numpy as jnp


def soundwave(cfg, x_slice=None):
    """1D linear sound wave: single right-going acoustic eigenmode (matches
    AthenaK linear_wave; cfg.soundwave_single_mode=False gives the old u=0
    standing-wave superposition). Background ρ₀=1, p₀=0.6, γ=5/3 → cs=1,
    period=Lx. Returns U = [rho, rho*u, E], shape (3, Nx_local).
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
    p0 = 0.6  # with γ=5/3 → cs² = 1, period = Lx
    amp = 1.0e-4  # eigenmode amplitude A
    m = 1

    cs = jnp.sqrt(gamma * p0 / rho0)
    k = 2.0 * jnp.pi * m / Lx
    sin_kx = jnp.sin(k * x)

    single_mode = getattr(cfg, "soundwave_single_mode", True)
    if single_mode:
        # Right-going acoustic eigenmode (matches AthenaK linear_wave).
        rho = rho0 + amp * sin_kx
        u = amp * cs / rho0 * sin_kx
        p = p0 + amp * cs * cs * sin_kx
    else:
        # u=0 superposition of left- and right-going acoustic waves of equal
        # amplitude — standing-wave physics, the OLD soundwave setup.
        rho = rho0 + amp * sin_kx
        u = jnp.zeros_like(x)
        p = p0 + amp * sin_kx

    E = p / (gamma - 1.0) + 0.5 * rho * u * u
    return jnp.stack([rho, rho * u, E], axis=0)
