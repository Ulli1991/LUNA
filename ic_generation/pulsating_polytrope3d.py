"""Pulsating polytrope: Lane–Emden equilibrium plus a small radial velocity
perturbation v_r(r) = ε·v_max·(r/R) approximating the fundamental mode.

Check: the system oscillates (no decay, no run-away) with a period consistent
with the polytrope's free-fall time. Reuses the lane_* config knobs.
"""

import jax.numpy as jnp
import numpy as np

from .lane_emden3d import _lane_emden_table


def pulsating_polytrope_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    gamma = cfg.gamma

    n = float(getattr(cfg, "lane_n", 1.5))
    rho_c = float(getattr(cfg, "lane_rho_c", 1.0))
    R = float(getattr(cfg, "lane_R", 1.0))
    rho_amb_cfg = float(getattr(cfg, "lane_rho_ambient", 0.01))
    pulse_amp = float(getattr(cfg, "pulse_amp", 0.01))
    G = float(getattr(cfg, "GRAVITATIONAL_G", 1.0))

    xi_grid, theta_grid = _lane_emden_table(n)
    xi_surface = float(xi_grid[-1])
    r_n = R / xi_surface
    K_polytrope = 4.0 * np.pi * G * r_n * r_n * rho_c ** (1.0 - 1.0 / n) / (n + 1.0)
    rho_amb = rho_amb_cfg if rho_amb_cfg > 0.0 else 1.0e-3 * rho_c

    dx = Lx / Nx; dy = Ly / Ny; dz = Lz / Nz
    cx, cy, cz = 0.5 * Lx, 0.5 * Ly, 0.5 * Lz
    xs, xe = (0, Nx) if x_slice is None else x_slice
    ys, ye = (0, Ny) if y_slice is None else y_slice
    zs, ze = (0, Nz) if z_slice is None else z_slice

    x_loc = (np.arange(xs, xe, dtype=np.float64) + 0.5) * dx - cx
    y_loc = (np.arange(ys, ye, dtype=np.float64) + 0.5) * dy - cy
    z_loc = (np.arange(zs, ze, dtype=np.float64) + 0.5) * dz - cz
    X, Y, Z = np.meshgrid(x_loc, y_loc, z_loc, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z)

    xi = r / r_n
    theta = np.interp(xi, xi_grid, theta_grid, left=1.0, right=0.0)
    rho_poly = rho_c * theta ** n
    # Floor the polytrope density at the ambient level (instead of letting
    # ρ → 0 at the surface, which created the wrong-direction jump
    # diagnosed in the previous run). The polytrope's ρ smoothly meets
    # the ambient floor near the surface — no shock.
    rho = np.maximum(rho_poly, rho_amb)
    P = K_polytrope * rho ** (1.0 + 1.0 / n)

    # Small radial velocity perturbation: v_r = ε · √(GM/R) · (r/R).
    # Coefficient set so the perturbation kinetic energy is ε² · GM²/R
    # — a small fraction of the polytrope's binding energy.
    v_norm = pulse_amp * np.sqrt(G / R)
    v_r = v_norm * (r / R) * (r <= R)
    inv_r = 1.0 / np.maximum(r, 1e-30)
    vx = v_r * X * inv_r
    vy = v_r * Y * inv_r
    vz = v_r * Z * inv_r

    rho = jnp.asarray(rho); P = jnp.asarray(P)
    vx = jnp.asarray(vx); vy = jnp.asarray(vy); vz = jnp.asarray(vz)
    E = P / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz)

    return jnp.stack([rho, rho * vx, rho * vy, rho * vz, E])
