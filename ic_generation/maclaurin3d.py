"""Maclaurin spheroid — uniform-density self-gravitating ellipsoid in
rigid-body rotation.

A homogeneous-density oblate spheroid with semi-major axis a (equatorial)
and semi-minor axis c (polar), eccentricity e = √(1 - c²/a²), in
solid-body rotation Ω², satisfies the Maclaurin equilibrium

    Ω² / (πGρ) = (2 √(1 - e²) / e³) · (3 - 2e²) sin⁻¹ e − 6 (1 - e²) / e²

(Chandrasekhar 1969, *Ellipsoidal Figures of Equilibrium*, eq. 32a).

The interior pressure follows from hydrostatic equilibrium in the
rotating frame: P(x,y,z) = P_c · (1 − (x² + y²)/a² − z²/c²).

Test: the spheroid should preserve its shape under self-gravity +
rotation. A drift in axis ratio or non-zero radial velocity buildup
indicates an issue.

Config knobs:
    mac_e         : eccentricity (default 0.5 → moderately oblate)
    mac_a         : equatorial semi-major axis (default 1.0)
    mac_rho       : interior density (default 1.0)
    mac_rho_ambient : low-density background (default 1e-3 * mac_rho)
"""

import jax.numpy as jnp
import numpy as np


def _maclaurin_omega2(e, G, rho):
    """Chandrasekhar's relation between eccentricity and rotation Ω²."""
    if e < 1e-6:
        # Sphere limit: Ω² → 0.
        return 0.0
    e2 = e * e
    s = np.sqrt(1.0 - e2)
    term1 = (2.0 * s / e ** 3) * (3.0 - 2.0 * e2) * np.arcsin(e)
    term2 = 6.0 * (1.0 - e2) / e2
    return np.pi * G * rho * (term1 - term2)


def maclaurin_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    gamma = cfg.gamma

    e = float(getattr(cfg, "mac_e", 0.5))
    a = float(getattr(cfg, "mac_a", 1.0))
    c = a * np.sqrt(1.0 - e * e)
    rho_inside = float(getattr(cfg, "mac_rho", 1.0))
    rho_amb_cfg = float(getattr(cfg, "mac_rho_ambient", 0.0))
    G = float(getattr(cfg, "GRAVITATIONAL_G", 1.0))
    rho_amb = rho_amb_cfg if rho_amb_cfg > 0.0 else 1.0e-3 * rho_inside

    omega2 = _maclaurin_omega2(e, G, rho_inside)
    omega = np.sqrt(max(omega2, 0.0))

    # Central pressure from hydrostatic equilibrium at the spheroid centre.
    # Chandrasekhar gives the gravitational potential difference inside;
    # the simplest practical choice is P_c = π G ρ² a² · f(e) so that the
    # interior pressure stays positive everywhere. f(e) ~ 1; choose a
    # value that keeps c_s ~ a Ω so the system is well-resolved.
    P_c = np.pi * G * rho_inside * rho_inside * a * a

    dx = Lx / Nx; dy = Ly / Ny; dz = Lz / Nz
    cx, cy, cz = 0.5 * Lx, 0.5 * Ly, 0.5 * Lz
    xs, xe = (0, Nx) if x_slice is None else x_slice
    ys, ye = (0, Ny) if y_slice is None else y_slice
    zs, ze = (0, Nz) if z_slice is None else z_slice

    x_loc = (np.arange(xs, xe, dtype=np.float64) + 0.5) * dx - cx
    y_loc = (np.arange(ys, ye, dtype=np.float64) + 0.5) * dy - cy
    z_loc = (np.arange(zs, ze, dtype=np.float64) + 0.5) * dz - cz
    X, Y, Z = np.meshgrid(x_loc, y_loc, z_loc, indexing="ij")

    # Spheroid surface in dimensionless units: f = (x²+y²)/a² + z²/c²; surface at f=1.
    f = (X * X + Y * Y) / (a * a) + Z * Z / (c * c)
    inside = f <= 1.0

    # Smoothstep transition centred at the spheroid surface (f = 1) over
    # `w_cells` worth of dx (the local cell scale). Suppresses the
    # otherwise-factor-1000 ρ jump that drove spurious motion in the IC.
    w_cells = 3.0
    w = w_cells * dx / a
    sqrt_f = np.sqrt(f)
    t_blend = np.clip((sqrt_f - (1.0 - w)) / (2.0 * w), 0.0, 1.0)
    smooth = 1.0 - t_blend * t_blend * (3.0 - 2.0 * t_blend)

    P_inside = P_c * np.maximum(1.0 - f, 0.0)
    P_amb = P_c * 1e-3
    rho = rho_inside * smooth + rho_amb * (1.0 - smooth)
    P = P_inside * smooth + P_amb * (1.0 - smooth)

    # Rigid-body rotation tapered with the same smoothstep so the ambient
    # gas is at rest (no supersonic shear at the spheroid edge).
    vx = -omega * Y * smooth
    vy = omega * X * smooth
    vz = np.zeros_like(rho)

    rho = jnp.asarray(rho); P = jnp.asarray(P)
    vx = jnp.asarray(vx); vy = jnp.asarray(vy); vz = jnp.asarray(vz)
    E = P / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz)

    return jnp.stack([rho, rho * vx, rho * vy, rho * vz, E])
