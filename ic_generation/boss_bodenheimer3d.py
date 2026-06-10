"""Boss-Bodenheimer 1979 protostellar collapse test.

A rotating, isothermal, spherical cloud with an m=2 azimuthal density
perturbation collapses under self-gravity and fragments into a
protostellar binary. Classic 3D self-gravity + isothermal + rotation
+ nonlinear collapse test.

Standard parameters (Boss & Bodenheimer 1979; many SPH/grid reproductions):
    Mass             M = 1.0           (code units)
    Radius           R = 1.0
    Sound speed²     c_s² = 0.0025      (very cold, isothermal)
    Rotation         Ω: β = E_rot/|E_grav| = 0.16
                     Ω² = 0.16 · 8/3 · π G ρ̄  (since |E_grav| = 3GM²/(5R))
    m=2 perturb.     A = 0.1            (10 % amplitude)
                     ρ(φ) = ρ̄ · (1 + A cos(2φ))

Expected: cloud collapses → central density rises → m=2 perturbation
amplifies → fragments into 2 protostars. Tests:
  - Collapse driven by self-gravity
  - Isothermal cooling preserved by runner's isothermal_cs2 projection
  - Nonlinear instability of the m=2 mode (key paper result)

Config knobs:
    bb_M       : cloud mass (default 1.0)
    bb_R       : cloud radius (default 1.0)
    bb_cs2     : isothermal sound speed squared (default 0.0025)
    bb_beta    : E_rot / |E_grav| (default 0.16)
    bb_amp     : m=2 perturbation amplitude (default 0.1)
    bb_rho_ambient : low-density background (default 1e-3 * ρ̄)
"""

import jax.numpy as jnp
import numpy as np


def boss_bodenheimer_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    gamma = cfg.gamma

    M = float(getattr(cfg, "bb_M", 1.0))
    R = float(getattr(cfg, "bb_R", 1.0))
    cs2 = float(getattr(cfg, "bb_cs2", 0.0025))
    beta = float(getattr(cfg, "bb_beta", 0.16))
    amp = float(getattr(cfg, "bb_amp", 0.1))
    rho_amb_cfg = float(getattr(cfg, "bb_rho_ambient", 0.0))
    G = float(getattr(cfg, "GRAVITATIONAL_G", 1.0))

    rho_bar = 3.0 * M / (4.0 * np.pi * R ** 3)
    # |E_grav| = 3 G M² / (5 R). E_rot = β |E_grav|. For solid-body
    # rotation Ω about z, E_rot = (1/2) I Ω² with I = (2/5) M R²
    # → Ω² = 5 β G M / R³  (uniform sphere moment of inertia).
    omega2 = 5.0 * beta * G * M / R ** 3
    omega = np.sqrt(omega2)
    rho_amb = rho_amb_cfg if rho_amb_cfg > 0.0 else 1.0e-3 * rho_bar

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
    phi = np.arctan2(Y, X)

    # Smoothstep over r = R; suppresses the factor-1000 ρ jump that
    # produced negative density at step 1 in the previous run.
    w_cells = 3.0
    w = w_cells * dx
    t_blend = np.clip((r - (R - w)) / (2.0 * w), 0.0, 1.0)
    smooth = 1.0 - t_blend * t_blend * (3.0 - 2.0 * t_blend)

    rho_in = rho_bar * (1.0 + amp * np.cos(2.0 * phi))
    rho = rho_in * smooth + rho_amb * (1.0 - smooth)
    P = cs2 * rho  # isothermal EOS

    # Rigid-body rotation tapered with the same smoothstep so the ambient
    # is at rest (otherwise rotation × sharp boundary = supersonic shear).
    vx = -omega * Y * smooth
    vy = omega * X * smooth
    vz = np.zeros_like(rho)

    rho = jnp.asarray(rho); P = jnp.asarray(P)
    vx = jnp.asarray(vx); vy = jnp.asarray(vy); vz = jnp.asarray(vz)
    E = P / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz)

    return jnp.stack([rho, rho * vx, rho * vy, rho * vz, E])
