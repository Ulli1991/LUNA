"""1D linear sound wave."""
from dataclasses import dataclass

import jax.numpy as jnp

from ._grid import local_range


@dataclass(frozen=True)
class Params:
    """Soundwave's parameters, cfg.problem."""
    single_mode: bool = True


def soundwave(cfg, x_slice=None):
    """U = [rho, rho*u, E]: a right-going eigenmode (AthenaK linear_wave), or u=0 if not single_mode."""
    Nx, Lx = cfg.Nx, cfg.Lx
    gamma = cfg.gamma

    x_start, x_end = local_range(Nx, x_slice)

    dx = Lx / Nx
    x_full = (jnp.arange(Nx) + 0.5) * dx
    x = x_full[x_start:x_end]

    rho0 = 1.0
    p0 = 0.6  # with γ=5/3 → cs² = 1, period = Lx
    amp = 1.0e-4
    m = 1

    cs = jnp.sqrt(gamma * p0 / rho0)
    k = 2.0 * jnp.pi * m / Lx
    sin_kx = jnp.sin(k * x)

    single_mode = cfg.problem.single_mode
    if single_mode:
        rho = rho0 + amp * sin_kx
        u = amp * cs / rho0 * sin_kx
        p = p0 + amp * cs * cs * sin_kx
    else:
        # equal left- and right-going waves (standing wave)
        rho = rho0 + amp * sin_kx
        u = jnp.zeros_like(x)
        p = p0 + amp * sin_kx

    E = p / (gamma - 1.0) + 0.5 * rho * u * u
    return jnp.stack([rho, rho * u, E], axis=0)
