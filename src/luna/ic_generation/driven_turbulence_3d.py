"""Uniform medium at rest for driven turbulence; the OU forcing source drives it."""

from dataclasses import dataclass

import jax.numpy as jnp

from ._grid import local_range


@dataclass(frozen=True)
class Params:
    """Driven-Turbulence-3D's parameters, cfg.problem."""
    rho0: float = 1.0
    P0: float = 1.0


def driven_turbulence_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """U for a uniform medium at cfg.problem.rho0, P0."""
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    gamma = cfg.gamma

    x_start, x_end = local_range(Nx, x_slice)
    y_start, y_end = local_range(Ny, y_slice)
    z_start, z_end = local_range(Nz, z_slice)
    nx_local = x_end - x_start
    ny_local = y_end - y_start
    nz_local = z_end - z_start

    rho_bg = cfg.problem.rho0
    p_bg = cfg.problem.P0

    rho = jnp.full((nx_local, ny_local, nz_local), rho_bg)
    zeros = jnp.zeros_like(rho)
    E = jnp.full_like(rho, p_bg / (gamma - 1.0))
    return jnp.stack([rho, zeros, zeros, zeros, E], axis=0)
