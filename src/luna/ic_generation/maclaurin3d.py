"""Maclaurin spheroid: uniform rotating oblate ellipsoid at Ω²(e), Chandrasekhar 1969 eq. 32a.

It should keep its shape.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ._grid import local_range


@dataclass(frozen=True)
class Params:
    """Maclaurin-Spheroid-3D's parameters, cfg.problem."""
    e: float = 0.5
    a: float = 1.0
    rho: float = 1.0
    rho_ambient: float = 0.0


def _maclaurin_omega2(e, G, rho):
    """Equilibrium Ω² for eccentricity e."""
    if e < 1e-6:
        return 0.0
    e2 = e * e
    s = np.sqrt(1.0 - e2)
    term1 = (2.0 * s / e ** 3) * (3.0 - 2.0 * e2) * np.arcsin(e)
    term2 = 6.0 * (1.0 - e2) / e2
    return np.pi * G * rho * (term1 - term2)


def maclaurin_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """U for the spheroid from cfg.problem."""
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    gamma = cfg.gamma

    e = float(cfg.problem.e)
    a = float(cfg.problem.a)
    c = a * np.sqrt(1.0 - e * e)
    rho_inside = float(cfg.problem.rho)
    rho_amb_cfg = float(cfg.problem.rho_ambient)
    G = float(cfg.GRAVITATIONAL_G)
    rho_amb = rho_amb_cfg if rho_amb_cfg > 0.0 else 1.0e-3 * rho_inside

    omega2 = _maclaurin_omega2(e, G, rho_inside)
    omega = np.sqrt(max(omega2, 0.0))

    # P_c = π G ρ² a² with f(e) = 1: a practical choice, not the exact HSE value
    P_c = np.pi * G * rho_inside * rho_inside * a * a

    dx = Lx / Nx; dy = Ly / Ny; dz = Lz / Nz
    cx, cy, cz = 0.5 * Lx, 0.5 * Ly, 0.5 * Lz
    xs, xe = local_range(Nx, x_slice)
    ys, ye = local_range(Ny, y_slice)
    zs, ze = local_range(Nz, z_slice)

    x_loc = (np.arange(xs, xe, dtype=np.float64) + 0.5) * dx - cx
    y_loc = (np.arange(ys, ye, dtype=np.float64) + 0.5) * dy - cy
    z_loc = (np.arange(zs, ze, dtype=np.float64) + 0.5) * dz - cz
    X, Y, Z = np.meshgrid(x_loc, y_loc, z_loc, indexing="ij")

    # surface at f = 1
    f = (X * X + Y * Y) / (a * a) + Z * Z / (c * c)

    # smoothstep over 3 cells at the surface: a sharp 1000x ρ jump drives spurious motion
    w_cells = 3.0
    w = w_cells * dx / a
    sqrt_f = np.sqrt(f)
    t_blend = np.clip((sqrt_f - (1.0 - w)) / (2.0 * w), 0.0, 1.0)
    smooth = 1.0 - t_blend * t_blend * (3.0 - 2.0 * t_blend)

    P_inside = P_c * np.maximum(1.0 - f, 0.0)
    P_amb = P_c * 1e-3
    rho = rho_inside * smooth + rho_amb * (1.0 - smooth)
    P = P_inside * smooth + P_amb * (1.0 - smooth)

    # rotation tapered by the same smoothstep so the ambient is at rest
    vx = -omega * Y * smooth
    vy = omega * X * smooth
    vz = np.zeros_like(rho)

    rho = jnp.asarray(rho); P = jnp.asarray(P)
    vx = jnp.asarray(vx); vy = jnp.asarray(vy); vz = jnp.asarray(vz)
    E = P / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz)

    return jnp.stack([rho, rho * vx, rho * vy, rho * vz, E])
