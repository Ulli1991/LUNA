"""Evrard 1988 cold-cloud collapse: ρ = M/(2πR²r) inside R, specific internal energy u.

Periodic boxes need L >= 4R so the images do not interfere.
"""

from dataclasses import dataclass

import jax.numpy as jnp

from ._grid import local_range


@dataclass(frozen=True)
class Params:
    """Evrard-Collapse-3D's parameters, cfg.problem."""
    M: float = 1.0
    R: float = 1.0
    u: float = 0.05
    rho_ambient: float = 0.0


def evrard_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """U for the Evrard cloud from the cfg.problem."""
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    gamma = cfg.gamma

    M = cfg.problem.M
    R = cfg.problem.R
    u_inside = cfg.problem.u
    rho_amb_cfg = cfg.problem.rho_ambient
    if rho_amb_cfg > 0.0:
        rho_amb = rho_amb_cfg
    else:
        rho_amb = 1.0e-3 * (3.0 * M / (4.0 * jnp.pi * R ** 3))

    dx = Lx / Nx; dy = Ly / Ny; dz = Lz / Nz
    cx, cy, cz = 0.5 * Lx, 0.5 * Ly, 0.5 * Lz

    xs, xe = local_range(Nx, x_slice)
    ys, ye = local_range(Ny, y_slice)
    zs, ze = local_range(Nz, z_slice)

    x = (jnp.arange(xs, xe) + 0.5) * dx - cx
    y = (jnp.arange(ys, ye) + 0.5) * dy - cy
    z = (jnp.arange(zs, ze) + 0.5) * dz - cz
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    r = jnp.sqrt(X * X + Y * Y + Z * Z)

    # r floored at dx/2 for the central cell
    rho_inside = M / (2.0 * jnp.pi * R * R) / jnp.maximum(r, 0.5 * dx)
    rho = jnp.where(r <= R, rho_inside, rho_amb)

    # the ambient has the same u as the cloud
    P = (gamma - 1.0) * rho * u_inside

    vx = jnp.zeros_like(rho)
    vy = jnp.zeros_like(rho)
    vz = jnp.zeros_like(rho)
    E = P / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz)

    return jnp.stack([rho, rho * vx, rho * vy, rho * vz, E])
