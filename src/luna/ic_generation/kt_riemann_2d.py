"""Kurganov & Tadmor 2002 (Numer. Meth. PDE 18, 584) 2D Riemann problems, configurations 3 and 6.

Four constant states in the quadrants of (0, 1)^2 split at x = y = 1/2 (KT02 eq. 3.2), gamma = 1.4, end time
0.3, data as printed (KT02 pp. 594, 596; Liska & Wendroff 2003 Table 4.3), quadrant 1 = NE, 2 = NW, 3 = SW,
4 = SE. Config 3: four backward shocks; config 6: four slip lines (Schulz-Rinne et al. 1993 Configuration B).
Run with BOUNDARY=1 (zero-gradient): the walls see supersonic inflow (diode BOUNDARY=4 is wrong there).
No exact 2D solution; the strips along the walls and the wall fluxes are exact 1D (tools/kt_riemann.py).
"""
from dataclasses import dataclass

import jax.numpy as jnp

from ._grid import local_range

GAMMA = 1.4
T_END = 0.3
# {config: {quadrant: (rho, u, v, p)}}, all printed digits
STATES = {
    3: {1: (1.5, 0.0, 0.0, 1.5), 2: (0.5323, 1.206, 0.0, 0.3),
        3: (0.138, 1.206, 1.206, 0.029), 4: (0.5323, 0.0, 1.206, 0.3)},
    6: {1: (1.0, 0.75, -0.5, 1.0), 2: (2.0, 0.75, 0.5, 1.0),
        3: (1.0, -0.75, 0.5, 1.0), 4: (3.0, -0.75, -0.5, 1.0)},
}


@dataclass(frozen=True)
class Params:
    """KT-Riemann-2D's parameters, cfg.problem."""
    config: int = 3                  # 3 | 6
    xc: float = 0.5                  # quadrant split, on a cell face
    yc: float = 0.5


def _face(c, n, length, name):
    """Index of the face at c; refuses a split inside a cell."""
    k = c * n / length
    if abs(k - round(k)) > 1e-9 or not 0 < round(k) < n:
        raise ValueError(f"KT-Riemann-2D: {name}={c} is not an interior cell face of {n} cells on [0, {length}]"
                         " (the jumps must lie on faces: even N for the split at 1/2)")
    return int(round(k))


def quadrants(cfg, states, xc, yc, x_slice=None, y_slice=None):
    """U = [rho, rho u, rho v, E] of four quadrant states {1: NE, 2: NW, 3: SW, 4: SE} split on faces xc, yc."""
    i0, j0 = _face(xc, cfg.Nx, cfg.Lx, "xc"), _face(yc, cfg.Ny, cfg.Ly, "yc")
    xs, xe = local_range(cfg.Nx, x_slice)
    ys, ye = local_range(cfg.Ny, y_slice)
    east = (jnp.arange(xs, xe) >= i0)[:, None]
    north = (jnp.arange(ys, ye) >= j0)[None, :]

    def pick(k):
        q = {n: s[k] for n, s in states.items()}
        return jnp.where(north, jnp.where(east, q[1], q[2]), jnp.where(east, q[4], q[3]))

    rho, u, v, p = (pick(k) for k in range(4))
    E = p / (cfg.gamma - 1.0) + 0.5 * rho * (u * u + v * v)
    return jnp.stack([rho, rho * u, rho * v, E], axis=0)


def kt_riemann_2d(cfg, x_slice=None, y_slice=None):
    """U of KT configuration cfg.problem.config (3 or 6) on the local shard."""
    pr = cfg.problem
    if pr.config not in STATES:
        raise ValueError(f"KT-Riemann-2D: config={pr.config}; implemented: {sorted(STATES)}")
    if abs(cfg.gamma - GAMMA) > 1e-12:
        raise ValueError(f"KT-Riemann-2D is defined with gamma = 1.4; got gamma={cfg.gamma}")
    return quadrants(cfg, STATES[pr.config], pr.xc, pr.yc, x_slice, y_slice)
