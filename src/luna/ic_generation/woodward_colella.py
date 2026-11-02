"""1D Woodward-Colella interacting blast waves.

Woodward & Colella 1984: gamma = 1.4, rho = 1, u = 0, p = 1000 | 0.01 | 100 split at 0.1 and 0.9
of [0, 1], reflecting walls (BOUNDARY = 2), t = 0.038. No exact solution: references are converged
fine-grid runs (tools/hydro_reference.py).
"""
import jax.numpy as jnp

from ._grid import local_range

PRESSURES = (1000.0, 0.01, 100.0)


def woodward_colella(cfg, x_slice=None):
    """U = [rho, rho u, E] at rest, the pressure as exact cell averages (the two jump cells mixed)."""
    if abs(cfg.gamma - 1.4) > 1e-12:
        raise ValueError(f"Woodward-Colella is defined with gamma = 1.4; got gamma={cfg.gamma}")
    Nx, Lx = cfg.Nx, cfg.Lx
    dx = Lx / Nx
    x_start, x_end = local_range(Nx, x_slice)
    a = jnp.arange(x_start, x_end) * dx                         # left cell edges
    f1 = jnp.clip((0.1 * Lx - a) / dx, 0.0, 1.0)                # fraction left of 0.1 Lx
    f2 = jnp.clip((0.9 * Lx - a) / dx, 0.0, 1.0)                # fraction left of 0.9 Lx
    p_left, p_mid, p_right = PRESSURES
    p = f1 * p_left + (f2 - f1) * p_mid + (1.0 - f2) * p_right
    rho = jnp.ones_like(p)
    return jnp.stack([rho, jnp.zeros_like(p), p / (cfg.gamma - 1.0)], axis=0)
