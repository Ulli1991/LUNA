"""Pulsating polytrope — Lane–Emden equilibrium + small radial velocity
perturbation at the expected fundamental mode shape.

The linear adiabatic radial mode for an n=1.5 polytrope has a known
dimensionless eigenfrequency. We seed an approximate fundamental-mode
velocity field

    v_r(r) = ε · v_max · (r/R)

(linear in r — the dominant fundamental-mode component is a uniform
expansion/contraction at small amplitudes), with ε = 0.01.

The simulation should oscillate at the fundamental frequency. The
analytic n=1.5 fundamental mode period (after Cox, *Theory of Stellar
Pulsation*) is

    T_1 = 2π · √( I / (4 R⁵ (3γ - 4) GM (some integral) ) )

The simple useful check: the system oscillates *at all* (no decay,
no run-away) and the period is consistent with the polytrope's
free-fall time scale.

Config knobs (reuses Lane-Emden):
    lane_n, lane_rho_c, lane_R, lane_rho_ambient (from lane_emden3d.py)
    pulse_amp  : velocity perturbation amplitude (default 0.01)
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
