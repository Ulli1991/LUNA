"""Zel'dovich pancake initial condition, in super-comoving variables.

A single sinusoidal mode along x, Zel'dovich-displaced, uniform in y and z -- a
thin 3D box so the run exercises the 3D multigrid Poisson solver. Built in
super-comoving variables (rho~ = a^3 rho, v~ = a v_pec, P~ = a^5 P) at the
initial scale factor a_init, so the comoving runner (setup/runner_cosmo.py)
integrates the plain Euler equations plus the (3/2) Omega_m a gravity source.

The Zel'dovich solution is exact for 1D pressureless collapse before shell
crossing; the gas is initialised cold so it tracks it until the caustic at
a = zel_a_caustic.

Config knobs:
    cosmo_Omega_m, cosmo_Omega_L, cosmo_H0 : background cosmology
    a_init        : initial scale factor (start redshift)
    zel_a_caustic : scale factor of first shell crossing (sets the amplitude)
    zel_cs_frac   : initial sound speed / peak peculiar velocity (keep gas cold)
"""
import jax.numpy as jnp

from cosmology import Cosmology


def zeldovich_pancake_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx = cfg.Lx
    gamma = cfg.gamma
    H0 = getattr(cfg, "cosmo_H0", 1.0)
    cosmo = Cosmology(Omega_m=getattr(cfg, "cosmo_Omega_m", 1.0),
                      Omega_L=getattr(cfg, "cosmo_Omega_L", 0.0))
    a_i = getattr(cfg, "a_init", 0.3)
    a_c = getattr(cfg, "zel_a_caustic", 1.5)
    cs_frac = getattr(cfg, "zel_cs_frac", 1.0e-2)

    k = 2.0 * jnp.pi / Lx
    dx = Lx / Nx
    A = a_i / a_c

    xs, xe = (0, Nx) if x_slice is None else x_slice
    ys, ye = (0, Ny) if y_slice is None else y_slice
    zs, ze = (0, Nz) if z_slice is None else z_slice

    x = (jnp.arange(xs, xe) + 0.5) * dx
    # Newton-invert  x = q - A sin(k q)/k  ->  q  (monotonic for A < 1)
    q = x
    for _ in range(60):
        f = q - A * jnp.sin(k * q) / k - x
        fp = 1.0 - A * jnp.cos(k * q)
        q = q - f / fp

    rho_x = 1.0 / (1.0 - A * jnp.cos(k * q))                       # rho~, mean 1
    vx_x = -(cosmo.dadtau(a_i, H0) / a_c) * jnp.sin(k * q) / k     # v~_x = a v_pec
    # cold gas: tiny polytropic pressure tied to the analytic peak velocity
    v_peak = cosmo.dadtau(a_i, H0) / (a_c * k)
    cs0 = cs_frac * v_peak
    P_x = (cs0 ** 2 / gamma) * rho_x ** gamma                     # P~

    nx, ny, nz = xe - xs, ye - ys, ze - zs
    rho = jnp.broadcast_to(rho_x[:, None, None], (nx, ny, nz))
    vx = jnp.broadcast_to(vx_x[:, None, None], (nx, ny, nz))
    P = jnp.broadcast_to(P_x[:, None, None], (nx, ny, nz))
    z3 = jnp.zeros((nx, ny, nz))
    E = P / (gamma - 1.0) + 0.5 * rho * vx * vx
    return jnp.stack([rho, rho * vx, z3, z3, E])
