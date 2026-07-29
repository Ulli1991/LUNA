"""Driven turbulence 3D initial condition.

Uniform background; the actual turbulence is driven by the OU module
(hydro/turbulence.py), which the runner applies as an operator-split
momentum source after each hydro step. This IC just hands the runner a
quiescent uniform medium to seed.
"""

import jax.numpy as jnp

from ._grid import local_range


def driven_turbulence_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    gamma = cfg.gamma

    x_start, x_end = local_range(Nx, x_slice)
    y_start, y_end = local_range(Ny, y_slice)
    z_start, z_end = local_range(Nz, z_slice)
    nx_local = x_end - x_start
    ny_local = y_end - y_start
    nz_local = z_end - z_start

    rho_bg = 1.0
    p_bg = 1.0

    rho = jnp.full((nx_local, ny_local, nz_local), rho_bg)
    zeros = jnp.zeros_like(rho)
    E = jnp.full_like(rho, p_bg / (gamma - 1.0))
    return jnp.stack([rho, zeros, zeros, zeros, E], axis=0)
