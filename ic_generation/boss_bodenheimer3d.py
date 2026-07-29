"""Boss–Bodenheimer 1979 protostellar collapse: rotating uniform cloud with an
m=2 density perturbation that fragments into a binary. Ref: Boss & Bodenheimer
1979; Hubber+ 2018 §6.12.

Requires cfg.units enabled (physical cgs parameters) and the barotropic EOS
(cfg.barotropic_eos = True; runner enforces it) — without the barotropic
stiffening the collapse runs away and the grid imprints a cross, not a binary.
"""

import jax.numpy as jnp
import numpy as np

from ._grid import local_range

PC_CM = 3.0856775814913673e18      # 1 parsec in cm
K_B = 1.380649e-16                 # erg / K
M_H = 1.6726219e-24                # g


def boss_bodenheimer_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    gamma = cfg.gamma
    u = cfg.units
    if not u.is_enabled:
        raise ValueError("Boss-Bodenheimer (physical) IC requires cfg.units enabled")

    # --- paper parameters → code units ---
    rho0_cgs = float(getattr(cfg, "bb_rho0_cgs", 1.44e-17))
    R_pc = float(getattr(cfg, "bb_R_pc", 0.01))
    amp = float(getattr(cfg, "bb_amp", 0.5))
    m = int(getattr(cfg, "bb_m", 2))
    amb_frac = float(getattr(cfg, "bb_rho_ambient_frac", 1.0e-3))
    beta = float(getattr(cfg, "bb_beta", 0.16))              # E_rot/|E_grav|
    omega_rad_s = float(getattr(cfg, "bb_omega_rad_s", 0.0))  # explicit physical Ω
    G_code = float(getattr(cfg, "GRAVITATIONAL_G", 1.0))

    rho_bar = rho0_cgs / u.UnitDensity_in_cgs                 # code density
    R = R_pc * PC_CM / u.UnitLength_in_cm                     # code length
    # Rigid-body Ω. Prefer an explicit physical rate (bb_omega_rad_s, rad/s) so
    # we can pin the reference value exactly (SEREN Ω=1.6e-12 rad/s); otherwise
    # derive from β (uniform sphere Ω²=3βGM/R³=4πβGρ̄). Note β=0.16 gives only
    # Ω=1.39e-12 here, so the two are NOT interchangeable.
    if omega_rad_s > 0.0:
        omega = omega_rad_s * u.UnitTime_in_s                 # rad/s -> rad/code-time
    else:
        omega = np.sqrt(4.0 * np.pi * beta * G_code * rho_bar)
    rho_amb = amb_frac * rho_bar

    # barotropic EOS parameters (for the IC pressure; runner enforces evolution)
    T0 = float(getattr(cfg, "barotropic_T0_K", 10.0))
    rho_AD = float(getattr(cfg, "barotropic_rho_AD_cgs", 1.0e-14))
    mu = float(getattr(cfg, "barotropic_mu", 2.35))
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

    # Smoothstep over r = R (3 cells) to avoid the sharp cloud/ambient jump.
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

    # barotropic IC pressure  P = k_B T(ρ) ρ / (μ m_H)
    rho_cgs = rho * u.UnitDensity_in_cgs
    T = T0 * (1.0 + (rho_cgs / rho_AD) ** gm1)
    P = (K_B * T * rho_cgs / (mu * M_H)) / u.UnitPressure_in_cgs

    rho = jnp.asarray(rho); P = jnp.asarray(P)
    vx = jnp.asarray(vx); vy = jnp.asarray(vy); vz = jnp.asarray(vz)
    E = P / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz)

    return jnp.stack([rho, rho * vx, rho * vy, rho * vz, E])
