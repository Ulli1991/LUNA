"""Hui / Liska-Wendroff implosion (LW 2003 sec. 4.7): the quadrant (0, L)^2 of the diamond x + y < L/2.

Inside rho 0.125, p 0.14; outside rho 1, p 1; at rest. Exact cell averages of rho and E: the line x + y = L/2
cuts cell (i, j) at a = N/2 - (i + j) cells above its lower-left corner (an exact half-integer), so the
area inside is a^2/2 (a <= 1) or 1 - (2 - a)^2/2 (a <= 2), symmetric in i <-> j and restriction-consistent.
"""
import jax.numpy as jnp

from ._grid import local_range

RHO_IN, P_IN, RHO_OUT, P_OUT = 0.125, 0.14, 1.0, 1.0


def inside_fraction(n, i, j):
    """Area fraction of cells (i, j) (integer arrays) inside x + y < L/2 on an n x n grid."""
    a = jnp.clip(0.5 * n - (i + j), 0.0, 2.0)
    return jnp.where(a <= 1.0, 0.5 * a * a, 1.0 - 0.5 * (2.0 - a) ** 2)


def implosion(cfg, x_slice=None, y_slice=None):
    """U of the implosion quadrant on the local shard; refuses a non-square grid or box."""
    if cfg.Nx != cfg.Ny or cfg.Lx != cfg.Ly:
        raise ValueError(f"LW-Implosion needs Nx == Ny and Lx == Ly; got {cfg.Nx}x{cfg.Ny}, {cfg.Lx}x{cfg.Ly}")
    x0, x1 = local_range(cfg.Nx, x_slice)
    y0, y1 = local_range(cfg.Ny, y_slice)
    i, j = jnp.meshgrid(jnp.arange(x0, x1), jnp.arange(y0, y1), indexing="ij")
    f = inside_fraction(cfg.Nx, i, j)
    rho = f * RHO_IN + (1.0 - f) * RHO_OUT
    E = (f * P_IN + (1.0 - f) * P_OUT) / (cfg.gamma - 1.0)
    z = jnp.zeros_like(rho)
    return jnp.stack([rho, z, z, E], axis=0)
