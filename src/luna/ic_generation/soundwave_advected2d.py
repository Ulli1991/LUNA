"""2D entropy wave (a density pattern at constant pressure) advected diagonally at u = v = 1.

Not a sound wave despite the name: rho = 1 + amp sin(k x) sin(k y), p = 1, u = v = 1, periodic, so
the exact solution is rho(x - t, y - t), back to the initial state after t = Lx (for Lx = Ly).
"""
from dataclasses import dataclass

import jax.numpy as jnp

from ._grid import local_range


@dataclass(frozen=True)
class Params:
    """Soundwave-Advected-2D's parameters, cfg.problem."""
    waves: int = 5          # wavelengths per box side
    amp: float = 0.5


def soundwave_advected2d(cfg, x_slice=None, y_slice=None):
    """U = [rho, rho u, rho v, E] with rho as exact cell averages."""
    pr = cfg.problem
    Nx, Ny, Lx, Ly = cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly
    k = 2.0 * jnp.pi * pr.waves / Lx
    x_start, x_end = local_range(Nx, x_slice)
    y_start, y_end = local_range(Ny, y_slice)
    hx, hy = Lx / Nx, Ly / Ny
    x = (jnp.arange(x_start, x_end) + 0.5) * hx
    y = (jnp.arange(y_start, y_end) + 0.5) * hy
    sx = jnp.sin(k * x) * jnp.sin(0.5 * k * hx) / (0.5 * k * hx)   # cell average of sin(k x)
    sy = jnp.sin(k * y) * jnp.sin(0.5 * k * hy) / (0.5 * k * hy)
    rho = 1.0 + pr.amp * sx[:, None] * sy[None, :]
    one = jnp.ones_like(rho)
    E = one / (cfg.gamma - 1.0) + 0.5 * rho * 2.0
    return jnp.stack([rho, rho, rho, E], axis=0)
