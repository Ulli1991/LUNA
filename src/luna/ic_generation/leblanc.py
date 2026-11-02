"""1D LeBlanc shock tube, as defined in the LANL Enhanced Verification Test Suite.

Kamm et al. 2008, LA-14379 (p. 42): gamma = 5/3, left (rho, u, p) = (1, 0, (2/3) 1e-1), right
(1e-3, 0, (2/3) 1e-10), i.e. specific internal energy 0.1 | 1e-7; 0 <= x <= 1, interface 0.3,
t_f = 0.5. A strong shock and a contact to the right, a strong transonic rarefaction to the left.
Other papers use other states under the same name ([0, 9], x0 = 3, t = 6 with p_R = 1e-7, rho E_R
= 1e-9 or e_R = 1e-10); they are Params away. Exact solution: luna.hydro.exact_riemann.
"""
from dataclasses import dataclass

import jax.numpy as jnp

from ._grid import local_range


@dataclass(frozen=True)
class Params:
    """LeBlanc's parameters, cfg.problem."""
    rho_l: float = 1.0
    p_l: float = (2.0 / 3.0) * 1e-1
    rho_r: float = 1e-3
    p_r: float = (2.0 / 3.0) * 1e-10
    x0: float = 0.3


def leblanc(cfg, x_slice=None):
    """U = [rho, rho u, E] at rest, the exact cell average of the jump at x0 (one mixed cell)."""
    if abs(cfg.gamma - 5.0 / 3.0) > 1e-12:
        raise ValueError(f"LeBlanc is defined with gamma = 5/3; got gamma={cfg.gamma}")
    pr = cfg.problem
    if not 0.0 < pr.x0 < cfg.Lx:
        raise ValueError(f"LeBlanc: interface x0={pr.x0} outside [0, Lx={cfg.Lx}]")
    Nx, dx = cfg.Nx, cfg.Lx / cfg.Nx
    x_start, x_end = local_range(Nx, x_slice)
    left_edge = jnp.arange(x_start, x_end) * dx
    f = jnp.clip((pr.x0 - left_edge) / dx, 0.0, 1.0)            # fraction of each cell left of x0
    g1 = cfg.gamma - 1.0
    rho = f * pr.rho_l + (1.0 - f) * pr.rho_r
    E = f * pr.p_l / g1 + (1.0 - f) * pr.p_r / g1
    return jnp.stack([rho, jnp.zeros_like(rho), E], axis=0)
