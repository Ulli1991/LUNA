"""Evrard collapse — 3D self-gravitating cold-cloud collapse test. Ref: Evrard 1988.

Cloud ρ(r) = M/(2πR²r) inside R, low-density ambient outside (no true vacuum
in Euler); u = 0.05, v = 0, M = R = G = 1 in code units.
Periodic BC needs L >> R (L = 4R usual minimum) so periodic images don't pollute.
"""

import jax.numpy as jnp


def evrard_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    gamma = cfg.gamma

    M = getattr(cfg, "evrard_M", 1.0)
    R = getattr(cfg, "evrard_R", 1.0)
    u_inside = getattr(cfg, "evrard_u", 0.05)
    rho_amb_cfg = getattr(cfg, "evrard_rho_ambient", 0.0)
    if rho_amb_cfg > 0.0:
        rho_amb = rho_amb_cfg
    else:
        rho_amb = 1.0e-3 * (3.0 * M / (4.0 * jnp.pi * R ** 3))

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

    x = (jnp.arange(xs, xe) + 0.5) * dx - cx
    y = (jnp.arange(ys, ye) + 0.5) * dy - cy
    z = (jnp.arange(zs, ze) + 0.5) * dz - cz
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    r = jnp.sqrt(X * X + Y * Y + Z * Z)

    # Inside cloud: ρ = M / (2π R²) · 1/r. Floor at small r to avoid
    # 1/0 at the central cell — that cell will have ρ ~ M/(2π R² dx_local).
    rho_inside = M / (2.0 * jnp.pi * R * R) / jnp.maximum(r, 0.5 * dx)
    rho = jnp.where(r <= R, rho_inside, rho_amb)

    # Pressure from cold internal energy: e_int = ρ u, P = (γ-1) ρ u.
    P_inside = (gamma - 1.0) * rho * u_inside
    # Ambient pressure: same temperature (u) for thermal balance at the
    # cloud edge — keeps the IC from having a sharp pressure discontinuity.
    # This is the standard choice in published Evrard implementations.
    P = (gamma - 1.0) * rho * u_inside

    vx = jnp.zeros_like(rho)
    vy = jnp.zeros_like(rho)
    vz = jnp.zeros_like(rho)
    E = P / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz)

    return jnp.stack([rho, rho * vx, rho * vy, rho * vz, E])
