"""Agertz et al. (2007) blob test: a cloud at rest in a Mach 2.7 wind, periodic box {10, 10, 40} R, wind along z."""
from dataclasses import dataclass

import jax.numpy as jnp

from ._grid import local_range


@dataclass(frozen=True)
class Params:
    """Agertz-Blob-3D's parameters, cfg.problem (sec. 2, 3, 4.1 of the paper; code units R = rho_ext = p = 1)."""
    chi: float = 10.0         # cloud / ambient density, pressure equilibrium
    mach: float = 2.7         # wind speed over the ambient sound speed
    radius: float = 1.0       # cloud radius
    centre: tuple = (5.0, 5.0, 5.0)
    subsample: int = 4        # per axis: the cloud's volume fraction of each cell from subsample^3 points


def wind_speed(cfg):
    """The ambient flow speed M c_ext, c_ext = sqrt(gamma p / rho_ext)."""
    return cfg.problem.mach * (cfg.gamma ** 0.5)


def t_cc(cfg):
    """Cloud crushing time 2 R chi^1/2 / v (eq. 6); tau_KH = 1.6 t_cc."""
    return 2.0 * cfg.problem.radius * cfg.problem.chi ** 0.5 / wind_speed(cfg)


def agertz_blob(cfg, x_slice=None, y_slice=None, z_slice=None):
    """U: rho_ext = 1, p = 1 everywhere, the cloud chi denser at rest, the ambient gas moving at +z; cells cut by the
    cloud's surface get its volume fraction f (rho, momentum mixed by volume, the same p)."""
    pr = cfg.problem
    dx, dy, dz = cfg.Lx / cfg.Nx, cfg.Ly / cfg.Ny, cfg.Lz / cfg.Nz
    ranges = [local_range(n, s) for n, s in ((cfg.Nx, x_slice), (cfg.Ny, y_slice), (cfg.Nz, z_slice))]
    s = int(pr.subsample)
    off = (jnp.arange(s) + 0.5) / s
    f = None
    for a in off:
        for b in off:
            for c in off:
                X, Y, Z = jnp.meshgrid(*[(jnp.arange(*r) + o) * d - x0 for r, o, d, x0 in
                                         zip(ranges, (a, b, c), (dx, dy, dz), pr.centre)], indexing="ij")
                inside = (X * X + Y * Y + Z * Z < pr.radius ** 2).astype(X.dtype)
                f = inside if f is None else f + inside
    f = f / s ** 3
    rho = 1.0 + (pr.chi - 1.0) * f
    mz = (1.0 - f) * wind_speed(cfg)
    zero = jnp.zeros_like(rho)
    E = 1.0 / (cfg.gamma - 1.0) + 0.5 * mz * mz / rho
    return jnp.stack([rho, zero, zero, mz, E], axis=0)


def cloud_mass(U, cfg):
    """Mass of the gas with T < 0.9 T_ext and rho > 0.64 rho_cl (sec. 5), T = p / rho in units of T_ext."""
    rho = U[0]
    p = (cfg.gamma - 1.0) * (U[-1] - 0.5 * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / rho)
    cloud = (p / rho < 0.9) & (rho > 0.64 * cfg.problem.chi)
    return jnp.sum(jnp.where(cloud, rho, 0.0)) * (cfg.Lx / cfg.Nx) * (cfg.Ly / cfg.Ny) * (cfg.Lz / cfg.Nz)
