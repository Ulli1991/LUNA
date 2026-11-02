"""1D Titarev-Toro shock / high-frequency entropy-wave interaction.

Titarev & Toro 2004 (JCP 201, 238): gamma = 1.4, (rho, u, p) = (1.515695, 0.523346, 1.805) for
x < -4.5 (the post-shock state of a Mach 1.3 shock into (1, 0, 1)), (1 + 0.1 sin(20 pi x), 0, 1)
beyond; [-5, 5], t = 5. No exact solution: references are converged fine-grid runs
(luna-reference-solutions luna/titarev_toro/, tools/hydro_reference.py).
"""
from dataclasses import dataclass

import jax.numpy as jnp

from .shu_osher import shock_into_wave


@dataclass(frozen=True)
class Params:
    """Titarev-Toro's parameters, cfg.problem."""
    amp: float = 0.1                 # density-wave amplitude ahead of the shock
    wavenumber: float = 20.0         # rho = 1 + amp sin(wavenumber pi x)
    x0: float = -4.5                 # shock position, on [-Lx/2, Lx/2]


LEFT = (1.515695, 0.523346, 1.805)   # rho, u, p behind the shock


def titarev_toro(cfg, x_slice=None):
    """U = [rho, rho u, E] as exact cell averages on [-Lx/2, Lx/2] (the shock cell mixed)."""
    pr = cfg.problem
    return shock_into_wave(cfg, x_slice, LEFT, pr.x0, pr.amp, pr.wavenumber * jnp.pi, "Titarev-Toro")
