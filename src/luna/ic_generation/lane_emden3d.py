"""Lane-Emden polytrope (P = K ρ^(1+1/n)) in hydrostatic equilibrium; it should stay at rest.

K follows from R, ρ_c and n.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

from ._grid import local_range


@dataclass(frozen=True)
class Params:
    """Lane-Emden-3D's parameters, cfg.problem."""
    n: float = 1.5
    rho_c: float = 1.0
    R: float = 1.0
    rho_ambient: float = 0.01


def _lane_emden_table(n, xi_max=10.0, n_points=2000):
    """(xi, theta) of the Lane-Emden solution up to its first root (or xi_max)."""
    def rhs(xi, y):
        theta, dtheta = y
        # avoid 1/0 at the origin
        if xi < 1e-6:
            return [dtheta, -theta ** n]
        return [dtheta, -theta ** n - (2.0 / xi) * dtheta]

    sol = solve_ivp(
        rhs, (1e-6, xi_max), [1.0, 0.0],
        method="RK45", dense_output=True, max_step=xi_max / n_points,
        events=lambda xi, y: y[0],  # stop when theta = 0
        rtol=1e-10, atol=1e-12,
    )
    xi_grid = np.linspace(1e-6, sol.t[-1], n_points)
    theta_grid = sol.sol(xi_grid)[0]
    theta_grid = np.maximum(theta_grid, 0.0)
    return xi_grid, theta_grid


def lane_emden_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """U for the polytrope from cfg.problem."""
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    gamma = cfg.gamma

    n = float(cfg.problem.n)
    rho_c = float(cfg.problem.rho_c)
    R = float(cfg.problem.R)
    rho_amb_cfg = float(cfg.problem.rho_ambient)
    G = float(cfg.GRAVITATIONAL_G)

    xi_grid, theta_grid = _lane_emden_table(n)
    xi_surface = float(xi_grid[-1])

    r_n = R / xi_surface
    # K is fixed by consistency: r_n² = (n+1) K ρ_c^(1/n-1) / (4πG)
    K_polytrope = 4.0 * np.pi * G * r_n * r_n * rho_c ** (1.0 - 1.0 / n) / (n + 1.0)

    if rho_amb_cfg > 0.0:
        rho_amb = rho_amb_cfg
    else:
        rho_amb = 1.0e-3 * rho_c

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
    rho = np.where(r <= R, rho_poly, rho_amb)
    # the polytropic EOS for the ambient too
    P = K_polytrope * rho ** (1.0 + 1.0 / n)

    rho = jnp.asarray(rho); P = jnp.asarray(P)
    vx = jnp.zeros_like(rho); vy = jnp.zeros_like(rho); vz = jnp.zeros_like(rho)
    E = P / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz)

    return jnp.stack([rho, rho * vx, rho * vy, rho * vz, E])
