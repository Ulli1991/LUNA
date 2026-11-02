"""Zel'dovich pancake: one displaced mode along x, in super-comoving variables at a_init.

Needs the comoving runner; cold gas follows the exact solution until a = cfg.problem.a_caustic.
"""
from dataclasses import dataclass

import jax.numpy as jnp

from luna.cosmology import Cosmology

from ._grid import local_range


@dataclass(frozen=True)
class Params:
    """Zeldovich-Pancake's parameters, cfg.problem."""
    a_caustic: float = 1.5
    cs_frac: float = 1.0e-2


def exact(a, x, Lx, a_caustic, cosmo, H0, iterations=60):
    """Exact super-comoving (rho~, v~) at scale factor a and comoving positions x, until a_caustic (EdS only)."""
    if (cosmo.Omega_m, cosmo.Omega_L, cosmo.Omega_r) != (1.0, 0.0, 0.0):
        raise ValueError(f"Zel'dovich exact() takes the EdS growth D = a; got Omega_m={cosmo.Omega_m}, "
                         f"Omega_L={cosmo.Omega_L}, Omega_r={cosmo.Omega_r}")
    k = 2.0 * jnp.pi / Lx
    A = a / a_caustic
    # invert x = q - A sin(k q)/k (increasing for A < 1) by Newton safeguarded by bisection on the bracket
    # |q - x| <= A / k: plain Newton from q = x overshoots near the caustic, where 1 - A cos kq -> 0
    lo, hi, q = x - A / k, x + A / k, x
    for _ in range(iterations):
        f = q - A * jnp.sin(k * q) / k - x
        fp = 1.0 - A * jnp.cos(k * q)
        lo, hi = jnp.where(f < 0, q, lo), jnp.where(f > 0, q, hi)
        qn = q - f / fp
        q = jnp.where((qn > lo) & (qn < hi), qn, 0.5 * (lo + hi))
    rho = 1.0 / (1.0 - A * jnp.cos(k * q))                         # mean 1
    v = -(cosmo.dadtau(a, H0) / a_caustic) * jnp.sin(k * q) / k
    return rho, v


def zeldovich_pancake_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """U in super-comoving variables (rho~ = a^3 rho, v~ = a v_pec, P~ = a^5 P)."""
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx = cfg.Lx
    gamma = cfg.gamma
    H0 = cfg.cosmo_H0
    cosmo = Cosmology(Omega_m=cfg.cosmo_Omega_m,
                      Omega_L=cfg.cosmo_Omega_L)
    a_i = cfg.a_init
    a_c = cfg.problem.a_caustic
    cs_frac = cfg.problem.cs_frac

    k = 2.0 * jnp.pi / Lx
    dx = Lx / Nx

    xs, xe = local_range(Nx, x_slice)
    ys, ye = local_range(Ny, y_slice)
    zs, ze = local_range(Nz, z_slice)

    x = (jnp.arange(xs, xe) + 0.5) * dx
    rho_x, vx_x = exact(a_i, x, Lx, a_c, cosmo, H0)
    # cold gas: tiny polytropic pressure tied to the analytic peak velocity
    v_peak = cosmo.dadtau(a_i, H0) / (a_c * k)
    cs0 = cs_frac * v_peak
    P_x = (cs0 ** 2 / gamma) * rho_x ** gamma

    nx, ny, nz = xe - xs, ye - ys, ze - zs
    rho = jnp.broadcast_to(rho_x[:, None, None], (nx, ny, nz))
    vx = jnp.broadcast_to(vx_x[:, None, None], (nx, ny, nz))
    P = jnp.broadcast_to(P_x[:, None, None], (nx, ny, nz))
    z3 = jnp.zeros((nx, ny, nz))
    E = P / (gamma - 1.0) + 0.5 * rho * vx * vx
    return jnp.stack([rho, rho * vx, z3, z3, E])
