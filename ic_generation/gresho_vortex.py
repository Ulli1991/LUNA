"""Gresho vortex (Gresho & Chan 1990, low-Mach variant à la Miczek+ 2015).

Steady-state rotating vortex on a periodic 2D box [0,1]^2, centered at
(0.5, 0.5). Velocity is piecewise-linear in radius; pressure is set so the
flow is centrifugally balanced. Density is uniform.

Velocity profile:
    v_phi(r) = 5 r                      for r < 0.2
    v_phi(r) = 2 - 5 r                  for 0.2 <= r < 0.4
    v_phi(r) = 0                        for r >= 0.4

Pressure (centrifugally balanced, dp/dr = rho v_phi^2 / r):
    p(r) = p0 + 12.5 r^2                                for r < 0.2
    p(r) = p0 + 9 + 4 ln(5 r) + 12.5 r^2 - 20 r + 4     for 0.2 <= r < 0.4
    p(r) = p0 + 3 + 4 ln 2                              for r >= 0.4

with the baseline p0 chosen so the maximum Mach number M_max = u_max/c_s is
the requested target (default M_max = 0.01 for deep low-Mach testing).

Density is uniform rho = 1.

Reads `cfg.gresho_Mmax` (defaults to 0.01) for the Mach scaling.
"""

import jax.numpy as jnp


def gresho_vortex(cfg, x_slice=None, y_slice=None, z_slice=None):
    Nx, Ny = cfg.Nx, cfg.Ny
    gamma = cfg.gamma
    Mmax = getattr(cfg, "gresho_Mmax", 0.01)
    rho0 = 1.0

    # p0 chosen so that c_s(p0, rho0) gives M_max = 1.0 / c_s = Mmax
    # i.e. c_s^2 = gamma p0 / rho0 = (1/Mmax)^2 -> p0 = rho0 / (gamma * Mmax^2)
    p0 = rho0 / (gamma * Mmax * Mmax)

    dx, dy = cfg.Lx / Nx, cfg.Ly / Ny
    x_start, x_end = x_slice if x_slice is not None else (0, Nx)
    y_start, y_end = y_slice if y_slice is not None else (0, Ny)
    x_loc = (jnp.arange(x_start, x_end) + 0.5) * dx
    y_loc = (jnp.arange(y_start, y_end) + 0.5) * dy
    X, Y = jnp.meshgrid(x_loc, y_loc, indexing="ij")
    cx, cy = 0.5 * cfg.Lx, 0.5 * cfg.Ly
    Xc = X - cx
    Yc = Y - cy
    R = jnp.sqrt(Xc * Xc + Yc * Yc)
    R_safe = jnp.where(R > 0, R, 1.0)

    # v_phi(r) piecewise
    v_phi = jnp.where(R < 0.2, 5.0 * R, jnp.where(R < 0.4, 2.0 - 5.0 * R, jnp.zeros_like(R)))

    # phi-hat = (-Yc/R, Xc/R)
    vx = -v_phi * Yc / R_safe
    vy = v_phi * Xc / R_safe

    # p(r) piecewise (centrifugally balanced, additive offset p0)
    p_inner = p0 + 12.5 * R * R
    p_mid = p0 + 9.0 + 4.0 * jnp.log(5.0 * jnp.maximum(R, 1e-12)) + 12.5 * R * R - 20.0 * R + 4.0
    p_outer = p0 + 3.0 + 4.0 * jnp.log(2.0)
    p = jnp.where(R < 0.2, p_inner, jnp.where(R < 0.4, p_mid, p_outer))

    rho = jnp.full_like(p, rho0)
    mom_x = rho * vx
    mom_y = rho * vy
    E = p / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy)
    return jnp.stack([rho, mom_x, mom_y, E], axis=0)
