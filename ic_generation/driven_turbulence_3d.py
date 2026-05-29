"""Driven turbulence 3D initial condition.

Uniform background; the actual turbulence is driven by the OU module
(hydro/turbulence.py), which the runner applies as an operator-split
momentum source after each hydro step. This IC just hands the runner a
quiescent uniform medium to seed.
"""

import jax.numpy as jnp


def driven_turbulence_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    gamma = cfg.gamma

    x_start, x_end = x_slice if x_slice is not None else (0, Nx)
    y_start, y_end = y_slice if y_slice is not None else (0, Ny)
    z_start, z_end = z_slice if z_slice is not None else (0, Nz)
    nx_local = x_end - x_start
    ny_local = y_end - y_start
    nz_local = z_end - z_start

    rho_bg = 1.0
    p_bg = 1.0

    rho = jnp.full((nx_local, ny_local, nz_local), rho_bg)
    zeros = jnp.zeros_like(rho)
    E = jnp.full_like(rho, p_bg / (gamma - 1.0))
    return jnp.stack([rho, zeros, zeros, zeros, E], axis=0)
