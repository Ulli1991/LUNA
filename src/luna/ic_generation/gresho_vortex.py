"""Low-Mach Gresho vortex (Gresho & Chan 1990; Miczek+ 2015): a steady 2D rotating vortex."""

from dataclasses import dataclass

import jax.numpy as jnp

from ._grid import local_range


@dataclass(frozen=True)
class Params:
    """Gresho-Vortex-2D's parameters, cfg.problem."""
    Mmax: float = 0.01


def gresho_vortex(cfg, x_slice=None, y_slice=None, z_slice=None):
    """U with the background p0 set so the peak Mach number is cfg.problem.Mmax."""
    Nx, Ny = cfg.Nx, cfg.Ny
    gamma = cfg.gamma
    Mmax = cfg.problem.Mmax
    rho0 = 1.0

    # v_phi peaks at 1, so c_s = 1 / Mmax
    p0 = rho0 / (gamma * Mmax * Mmax)

    dx, dy = cfg.Lx / Nx, cfg.Ly / Ny
    x_start, x_end = local_range(Nx, x_slice)
    y_start, y_end = local_range(Ny, y_slice)
    x_loc = (jnp.arange(x_start, x_end) + 0.5) * dx
    y_loc = (jnp.arange(y_start, y_end) + 0.5) * dy
    X, Y = jnp.meshgrid(x_loc, y_loc, indexing="ij")
    cx, cy = 0.5 * cfg.Lx, 0.5 * cfg.Ly
    Xc = X - cx
    Yc = Y - cy
    R = jnp.sqrt(Xc * Xc + Yc * Yc)
    R_safe = jnp.where(R > 0, R, 1.0)

    v_phi = jnp.where(R < 0.2, 5.0 * R, jnp.where(R < 0.4, 2.0 - 5.0 * R, jnp.zeros_like(R)))

    vx = -v_phi * Yc / R_safe
    vy = v_phi * Xc / R_safe

    # centrifugally balanced and continuous: dp/dR = rho v_phi^2 / R
    p_inner = p0 + 12.5 * R * R
    p_mid = p0 + 4.0 + 4.0 * jnp.log(5.0 * jnp.maximum(R, 1e-12)) + 12.5 * R * R - 20.0 * R
    p_outer = p0 - 2.0 + 4.0 * jnp.log(2.0)
    p = jnp.where(R < 0.2, p_inner, jnp.where(R < 0.4, p_mid, p_outer))

    rho = jnp.full_like(p, rho0)
    mom_x = rho * vx
    mom_y = rho * vy
    E = p / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy)
    return jnp.stack([rho, mom_x, mom_y, E], axis=0)
