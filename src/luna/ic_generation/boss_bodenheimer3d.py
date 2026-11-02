"""Boss & Bodenheimer 1979 rotating cloud with an m=2 perturbation (Hubber+ 2018 §6.12).

Needs cfg.units and the barotropic EOS; without it the collapse runs away.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ._grid import local_range


@dataclass(frozen=True)
class Params:
    """Boss-Bodenheimer-3D's parameters, cfg.problem."""
    rho0_cgs: float = 1.44e-17
    R_pc: float = 0.01
    omega_rad_s: float = 1.6e-12
    amp: float = 0.5
    m: int = 2
    rho_ambient_frac: float = 1.0e-3
    beta: float = 0.16


PC_CM = 3.0856775814913673e18
K_B = 1.380649e-16
M_H = 1.6726219e-24


def boss_bodenheimer_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """U for the Boss-Bodenheimer cloud from cfg.problem."""
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    gamma = cfg.gamma
    u = cfg.units
    if not u.is_enabled:
        raise ValueError("Boss-Bodenheimer (physical) IC requires cfg.units enabled")

    rho0_cgs = float(cfg.problem.rho0_cgs)
    R_pc = float(cfg.problem.R_pc)
    amp = float(cfg.problem.amp)
    m = int(cfg.problem.m)
    amb_frac = float(cfg.problem.rho_ambient_frac)
    beta = float(cfg.problem.beta)              # E_rot/|E_grav|
    omega_rad_s = float(cfg.problem.omega_rad_s)
    G_code = float(cfg.GRAVITATIONAL_G)

    rho_bar = rho0_cgs / u.UnitDensity_in_cgs
    R = R_pc * PC_CM / u.UnitLength_in_cm
    # omega_rad_s > 0 wins; else Ω² = 4πβGρ̄ (β=0.16 does not give SEREN's 1.6e-12 rad/s)
    if omega_rad_s > 0.0:
        omega = omega_rad_s * u.UnitTime_in_s
    else:
        omega = np.sqrt(4.0 * np.pi * beta * G_code * rho_bar)
    rho_amb = amb_frac * rho_bar

    # barotropic EOS parameters for the IC pressure
    T0 = float(cfg.barotropic_T0_K)
    rho_AD = float(cfg.barotropic_rho_AD_cgs)
    mu = float(cfg.barotropic_mu)
    gm1 = gamma - 1.0

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
    phi = np.arctan2(Y, X)

    # smoothstep over 3 cells around r = R
    w = 3.0 * dx
    t_blend = np.clip((r - (R - w)) / (2.0 * w), 0.0, 1.0)
    smooth = 1.0 - t_blend * t_blend * (3.0 - 2.0 * t_blend)

    # density: ρ₀[1 + A sin(m φ)] inside, blended to the ambient outside
    rho_in = rho_bar * (1.0 + amp * np.sin(m * phi))
    rho = rho_in * smooth + rho_amb * (1.0 - smooth)

    # rigid-body rotation about z, tapered so the ambient is at rest
    vx = -omega * Y * smooth
    vy = omega * X * smooth
    vz = np.zeros_like(rho)

    rho_cgs = rho * u.UnitDensity_in_cgs
    T = T0 * (1.0 + (rho_cgs / rho_AD) ** gm1)
    P = (K_B * T * rho_cgs / (mu * M_H)) / u.UnitPressure_in_cgs

    rho = jnp.asarray(rho); P = jnp.asarray(P)
    vx = jnp.asarray(vx); vy = jnp.asarray(vy); vz = jnp.asarray(vz)
    E = P / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz)

    return jnp.stack([rho, rho * vx, rho * vy, rho * vz, E])
