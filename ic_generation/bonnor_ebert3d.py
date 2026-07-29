"""Bonnor–Ebert isothermal sphere: self-gravitating cloud truncated at ξ_t,
built from the isothermal Lane–Emden solution.

Stability: ξ_t < 6.451 is a stable equilibrium; ξ_t > 6.451 collapses.
Requires the runner's isothermal projection (isothermal_cs2 > 0) to enforce
P = c_s² ρ during evolution.
"""

import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

from ._grid import local_range


def _be_table(xi_max=20.0, n_points=4000):
    """Integrate the isothermal Lane–Emden equation."""
    def rhs(xi, y):
        psi, dpsi = y
        # Near origin use Taylor: ψ ≈ ξ²/6 + O(ξ⁴), so dψ/dξ ≈ ξ/3.
        if xi < 1e-6:
            return [dpsi, np.exp(-psi)]
        return [dpsi, np.exp(-psi) - (2.0 / xi) * dpsi]

    sol = solve_ivp(
        rhs, (1e-6, xi_max), [0.0, 0.0],
        method="RK45", dense_output=True, max_step=xi_max / n_points,
        rtol=1e-10, atol=1e-12,
    )
    xi_grid = np.linspace(1e-6, xi_max, n_points)
    psi_grid = sol.sol(xi_grid)[0]
    return xi_grid, psi_grid


def bonnor_ebert_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    gamma = cfg.gamma

    xi_t = float(getattr(cfg, "be_xi_t", 5.0))      # 5 stable, 10 super-critical
    rho_c = float(getattr(cfg, "be_rho_c", 1.0))
    cs2 = float(getattr(cfg, "be_cs2", 1.0))
    R = float(getattr(cfg, "be_R", 1.0))
    rho_amb_cfg = float(getattr(cfg, "be_rho_ambient", 0.0))
    G = float(getattr(cfg, "GRAVITATIONAL_G", 1.0))

    # Set rho_c so that r_BE * xi_t = R. r_BE = sqrt(cs² / (4πGρ_c)) →
    # physical R = ξ_t · sqrt(cs² / (4πGρ_c)). Solving for ρ_c:
    # ρ_c = cs² ξ_t² / (4πG R²)
    rho_c_for_R = cs2 * xi_t * xi_t / (4.0 * np.pi * G * R * R)
    rho_c = rho_c_for_R   # override the cfg value to match the geometry

    if rho_amb_cfg > 0.0:
        rho_amb = rho_amb_cfg
    else:
        rho_amb = 1.0e-3 * rho_c

    # BE table extends out to xi_t.
    xi_grid, psi_grid = _be_table(xi_max=max(xi_t + 1.0, 8.0))
    r_BE = np.sqrt(cs2 / (4.0 * np.pi * G * rho_c))

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
    xi = r / r_BE
    psi = np.interp(xi, xi_grid, psi_grid, left=0.0, right=psi_grid[-1])
    rho_be = rho_c * np.exp(-psi)

    # Match the ambient to the BE solution at ξ_t so there's no jump.
    rho_edge = rho_c * np.exp(-np.interp(xi_t, xi_grid, psi_grid))
    rho_amb = max(rho_amb, rho_edge * 0.5)   # ambient at most factor 2 below edge

    # Smoothstep transition over `w` cells centred at ξ = ξ_t. The cloud's
    # BE solution holds inside, the ambient holds well outside, and the
    # blend smoothly interpolates — no sharp pressure/density step that
    # would otherwise seed an outgoing shock at t = 0+.
    w_cells = 3.0
    w = w_cells * dx / r_BE   # in ξ units
    t_blend = np.clip((xi - (xi_t - w)) / (2.0 * w), 0.0, 1.0)
    smooth = 1.0 - t_blend * t_blend * (3.0 - 2.0 * t_blend)
    rho = rho_be * smooth + rho_amb * (1.0 - smooth)
    P = cs2 * rho   # isothermal EOS — pressure tracks ρ automatically

    rho = jnp.asarray(rho); P = jnp.asarray(P)
    vx = jnp.zeros_like(rho); vy = jnp.zeros_like(rho); vz = jnp.zeros_like(rho)
    E = P / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz)

    return jnp.stack([rho, rho * vx, rho * vy, rho * vz, E])
