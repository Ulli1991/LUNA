"""2D Kelvin-Helmholtz: two periodic shear layers."""
from dataclasses import dataclass

import jax.numpy as jnp

from ._grid import local_range


@dataclass(frozen=True)
class Params:
    """Kelvin-Helmholtz's parameters, cfg.problem."""
    noise_amp: float = 0.0


def kelvin_helmholtz(cfg, x_slice=None, y_slice=None):
    """U for KH seeded with one sin mode, or white noise if cfg.problem.noise_amp > 0."""
    Nx, Ny = cfg.Nx, cfg.Ny
    Lx, Ly = cfg.Lx, cfg.Ly
    gamma = cfg.gamma

    x_start, x_end = local_range(Nx, x_slice)
    Nx_local = x_end - x_start

    y_start, y_end = local_range(Ny, y_slice)
    Ny_local = y_end - y_start

    x_full = (jnp.arange(Nx) + 0.5) * (Lx / Nx)
    y_full = (jnp.arange(Ny) + 0.5) * (Ly / Ny)
    x_local = x_full[x_start:x_end]
    y_local = y_full[y_start:y_end]
    X, Y = jnp.meshgrid(x_local, y_local, indexing="ij")

    # fractions of Ly so both layers are inside the box and the flow is periodic
    U0 = 1.0
    a = 0.025 * Ly
    A = 0.01
    sigma = 0.1 * Ly
    Y1, Y2 = 0.25 * Ly, 0.75 * Ly
    lam = 0.5 * Ly                   # seeded wavelength
    rho_low, rho_high = 1.0, 2.0
    delta = (rho_high - rho_low) / rho_low

    rho = 1.0 + 0.5 * delta * (jnp.tanh((Y - Y1) / a) - jnp.tanh((Y - Y2) / a))
    ux = -U0 * ((jnp.tanh((Y - Y1) / a) - jnp.tanh((Y - Y2) / a)) - 1.0)
    envelope = jnp.exp(-((Y - Y1) ** 2) / sigma**2) + jnp.exp(-((Y - Y2) ** 2) / sigma**2)

    # noise drawn on the full grid from a fixed seed, then sliced: shards and resolutions agree
    noise_amp = float(cfg.problem.noise_amp)
    if noise_amp > 0.0:
        import numpy as _np
        _rng = _np.random.default_rng(1234)
        nloc = jnp.asarray(_rng.standard_normal((Nx, Ny))[x_start:x_end, y_start:y_end])
        uy = noise_amp * nloc * envelope
    else:
        uy = A * jnp.sin(2 * jnp.pi * X / lam) * envelope

    p = jnp.ones((Nx_local, Ny_local)) * 10.0
    E = p / (gamma - 1.0) + 0.5 * rho * (ux**2 + uy**2)

    U = jnp.stack([rho, rho * ux, rho * uy, E], axis=0)

    return U
