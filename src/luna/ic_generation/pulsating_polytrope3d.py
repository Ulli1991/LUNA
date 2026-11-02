"""Lane-Emden polytrope with a small radial velocity v_r ~ r approximating the fundamental mode.

It should oscillate without decay or run-away. Lane-Emden's parameters plus amp.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .lane_emden3d import _lane_emden_table

from ._grid import local_range


@dataclass(frozen=True)
class Params:
    """Pulsating-Polytrope-3D's parameters, cfg.problem."""
    n: float = 1.5
    rho_c: float = 1.0
    R: float = 1.0
    rho_ambient: float = 0.01
    amp: float = 0.01


def pulsating_polytrope_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """U for the perturbed polytrope."""
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    gamma = cfg.gamma

    n = float(cfg.problem.n)
    rho_c = float(cfg.problem.rho_c)
    R = float(cfg.problem.R)
    rho_amb_cfg = float(cfg.problem.rho_ambient)
    amp = float(cfg.problem.amp)
    G = float(cfg.GRAVITATIONAL_G)

    xi_grid, theta_grid = _lane_emden_table(n)
    xi_surface = float(xi_grid[-1])
    r_n = R / xi_surface
    K_polytrope = 4.0 * np.pi * G * r_n * r_n * rho_c ** (1.0 - 1.0 / n) / (n + 1.0)
    rho_amb = rho_amb_cfg if rho_amb_cfg > 0.0 else 1.0e-3 * rho_c

    dx = Lx / Nx; dy = Ly / Ny; dz = Lz / Nz
    cx, cy, cz = 0.5 * Lx, 0.5 * Ly, 0.5 * Lz
    xs, xe = local_range(Nx, x_slice)
    ys, ye = local_range(Ny, y_slice)
    zs, ze = local_range(Nz, z_slice)

    x_loc = (np.arange(xs, xe, dtype=np.float64) + 0.5) * dx - cx
    y_loc = (np.arange(ys, ye, dtype=np.float64) + 0.5) * dy - cy
    z_loc = (np.arange(zs, ze, dtype=np.float64) + 0.5) * dz - cz
    X, Y, Z = np.meshgrid(x_loc, y_loc, z_loc, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z)

    xi = r / r_n
    theta = np.interp(xi, xi_grid, theta_grid, left=1.0, right=0.0)
    rho_poly = rho_c * theta ** n
    # floor at the ambient so ρ meets it smoothly at the surface
    rho = np.maximum(rho_poly, rho_amb)
    P = K_polytrope * rho ** (1.0 + 1.0 / n)

    # v_r = amp sqrt(G/R) r/R inside R
    v_norm = amp * np.sqrt(G / R)
    v_r = v_norm * (r / R) * (r <= R)
    inv_r = 1.0 / np.maximum(r, 1e-30)
    vx = v_r * X * inv_r
    vy = v_r * Y * inv_r
    vz = v_r * Z * inv_r

    rho = jnp.asarray(rho); P = jnp.asarray(P)
    vx = jnp.asarray(vx); vy = jnp.asarray(vy); vz = jnp.asarray(vz)
    E = P / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz)

    return jnp.stack([rho, rho * vx, rho * vy, rho * vz, E])
