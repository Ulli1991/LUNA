"""Lane–Emden polytropic equilibrium.

Self-gravitating polytrope in hydrostatic equilibrium:
    P = K ρ^(1 + 1/n)
    ∇²Φ = 4πGρ
    ∇P = -ρ ∇Φ        (HSE)

Substituting θⁿ for the normalised density (ρ = ρ_c · θⁿ) and
ξ = r/r_n where r_n = √((n+1) K ρ_c^(1/n - 1) / (4πG)), the
dimensionless Lane–Emden equation is

    d²θ/dξ² + (2/ξ) dθ/dξ + θⁿ = 0,
    θ(0) = 1,  θ'(0) = 0,

solved numerically. The surface ξ_1 is where θ first reaches zero;
for n = 1.5 (γ = 5/3), ξ_1 ≈ 3.6537.

Test: the polytrope should stay in HSE under self-gravity. Run for
several dynamical times and check max|v| remains small.

Config knobs (cfg fields):
    lane_n         : polytrope index (default 1.5 for γ=5/3)
    lane_rho_c     : central density (default 1.0)
    lane_R         : physical radius of polytrope (default 1.0)
    lane_K         : not used here — K is determined by R/ρ_c/n consistency
    lane_rho_ambient : low-density background outside the polytrope
"""

import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp


def _lane_emden_table(n, xi_max=10.0, n_points=2000):
    """Numerically integrate the LE equation. Return (xi, theta) arrays
    up to the first root (truncated at xi_max if no root found)."""
    def rhs(xi, y):
        theta, dtheta = y
        # Avoid 1/0 at the origin: use the Taylor expansion θ ≈ 1 - ξ²/6 + ...
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
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    gamma = cfg.gamma

    n = float(getattr(cfg, "lane_n", 1.5))
    rho_c = float(getattr(cfg, "lane_rho_c", 1.0))
    R = float(getattr(cfg, "lane_R", 1.0))
    rho_amb_cfg = float(getattr(cfg, "lane_rho_ambient", 0.0))
    G = float(getattr(cfg, "GRAVITATIONAL_G", 1.0))

    # Build 1-D Lane–Emden table and find the surface ξ_1.
    xi_grid, theta_grid = _lane_emden_table(n)
    xi_surface = float(xi_grid[-1])  # last point before θ crosses 0

    # Determine the scaling r_n so that the physical surface is at r = R.
    r_n = R / xi_surface
    # K is fixed by consistency: r_n² = (n+1) K ρ_c^(1/n-1) / (4πG)
    K_polytrope = 4.0 * np.pi * G * r_n * r_n * rho_c ** (1.0 - 1.0 / n) / (n + 1.0)

    if rho_amb_cfg > 0.0:
        rho_amb = rho_amb_cfg
    else:
        rho_amb = 1.0e-3 * rho_c

    dx = Lx / Nx; dy = Ly / Ny; dz = Lz / Nz
    cx, cy, cz = 0.5 * Lx, 0.5 * Ly, 0.5 * Lz

    if x_slice is None:
        xs, xe = 0, Nx
    else:
        xs, xe = x_slice
    if y_slice is None:
        ys, ye = 0, Ny
    else:
        ys, ye = y_slice
    if z_slice is None:
        zs, ze = 0, Nz
    else:
        zs, ze = z_slice

    x_loc = (np.arange(xs, xe, dtype=np.float64) + 0.5) * dx - cx
    y_loc = (np.arange(ys, ye, dtype=np.float64) + 0.5) * dy - cy
    z_loc = (np.arange(zs, ze, dtype=np.float64) + 0.5) * dz - cz
    X, Y, Z = np.meshgrid(x_loc, y_loc, z_loc, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z)

    # Map r to xi, interpolate theta(xi).
    xi = r / r_n
    theta = np.interp(xi, xi_grid, theta_grid, left=1.0, right=0.0)
    rho_poly = rho_c * theta ** n
    rho = np.where(r <= R, rho_poly, rho_amb)
    # Pressure from the polytropic EOS inside; same EOS for the ambient
    # gives P_amb = K ρ_amb^(1+1/n), keeping P/ρ^γ continuous in the
    # polytropic sense (not strictly the simulated γ unless n = 1/(γ-1)).
    P = K_polytrope * rho ** (1.0 + 1.0 / n)

    rho = jnp.asarray(rho); P = jnp.asarray(P)
    vx = jnp.zeros_like(rho); vy = jnp.zeros_like(rho); vz = jnp.zeros_like(rho)
    E = P / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz)

    return jnp.stack([rho, rho * vx, rho * vy, rho * vz, E])
