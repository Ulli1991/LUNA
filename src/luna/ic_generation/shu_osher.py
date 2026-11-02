"""1D Shu-Osher shock / entropy-wave interaction, and the exact cell averages it shares with
Titarev-Toro.

Shu & Osher 1989: gamma = 1.4, a Mach 3 shock, (rho, u, p) = (3.857143, 2.629369, 10.33333) for
x < -4, (1 + 0.2 sin(5 x), 0, 1) beyond; [-5, 5], t = 1.8. No exact solution: references are
converged fine-grid runs (tools/hydro_reference.py).
"""
from dataclasses import dataclass

import jax.numpy as jnp

from ._grid import local_range


@dataclass(frozen=True)
class Params:
    """Shu-Osher's parameters, cfg.problem."""
    amp: float = 0.2                 # density-wave amplitude ahead of the shock
    k: float = 5.0                   # rho = 1 + amp sin(k x)
    x0: float = -4.0                 # shock position, on [-Lx/2, Lx/2]


LEFT = (3.857143, 2.629369, 10.33333)   # rho, u, p behind the Mach 3 shock


def shock_into_wave(cfg, x_slice, left, x0, amp, k, name):
    """U = [rho, rho u, E]: `left` behind a shock at x0, (1 + amp sin(k x), 0, 1) ahead, as exact
    cell averages on [-Lx/2, Lx/2] (the shock cell mixed)."""
    if abs(cfg.gamma - 1.4) > 1e-12:
        raise ValueError(f"{name} is defined with gamma = 1.4; got gamma={cfg.gamma}")
    Nx, dx = cfg.Nx, cfg.Lx / cfg.Nx
    x_start, x_end = local_range(Nx, x_slice)
    a = -0.5 * cfg.Lx + jnp.arange(x_start, x_end) * dx       # left cell edges
    b = a + dx
    if not -0.5 * cfg.Lx < x0 < 0.5 * cfg.Lx:
        raise ValueError(f"{name}: shock x0={x0} outside [-Lx/2, Lx/2] = [{-0.5 * cfg.Lx}, {0.5 * cfg.Lx}]")
    f = jnp.clip((x0 - a) / dx, 0.0, 1.0)                       # fraction of each cell behind the shock
    lo = jnp.maximum(a, x0)                                     # the part of each cell ahead of the shock
    wave = jnp.where(b > lo, (jnp.cos(k * lo) - jnp.cos(k * b)) / k, 0.0)
    rho_r = (1.0 - f) + amp * wave / dx                         # cell average of 1 + amp sin(k x) there
    rho_l, u_l, p_l = left
    g1 = cfg.gamma - 1.0
    rho = f * rho_l + rho_r
    mom = f * rho_l * u_l
    E = f * (p_l / g1 + 0.5 * rho_l * u_l * u_l) + (1.0 - f) * 1.0 / g1
    return jnp.stack([rho, mom, E], axis=0)


def shu_osher(cfg, x_slice=None):
    """U = [rho, rho u, E] on [-Lx/2, Lx/2] as exact cell averages."""
    pr = cfg.problem
    return shock_into_wave(cfg, x_slice, LEFT, pr.x0, pr.amp, pr.k, "Shu-Osher")
