"""Linear Jeans instability: uniform medium on a periodic box with a sinusoidal
density perturbation. Unstable mode should grow at the linear-theory rate
σ = √(4πGρ₀ − c_s² k²).
"""

import math

import jax.numpy as jnp


def jeans_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    gamma = cfg.gamma

    rho0 = getattr(cfg, "jeans_rho0", 1.0)
    P0 = getattr(cfg, "jeans_P0", 1.0)
    amp = getattr(cfg, "jeans_amp", 1.0e-4)
    kx = getattr(cfg, "jeans_kx", 0.0)
    ky = getattr(cfg, "jeans_ky", 0.0)
    kz = getattr(cfg, "jeans_kz", 0.0)
    # All-zero wavevector means "not set" (the template default). A Jeans IC
    # without a perturbation mode is never intended, so fall back to the
    # fundamental along x rather than silently producing a uniform box.
    if kx == 0.0 and ky == 0.0 and kz == 0.0:
        kx = 2.0 * jnp.pi / Lx

    dx = Lx / Nx; dy = Ly / Ny; dz = Lz / Nz

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

    x = (jnp.arange(xs, xe) + 0.5) * dx
    y = (jnp.arange(ys, ye) + 0.5) * dy
    z = (jnp.arange(zs, ze) + 0.5) * dz
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    theta = kx * X + ky * Y + kz * Z
    cphase = jnp.cos(theta)
    sphase = jnp.sin(theta)
    rho = rho0 * (1.0 + amp * cphase)

    # Pure growing Jeans eigenmode (from a static linear analysis): adiabatic
    # pressure δP = c_s²δρ  →  P = P0(1 + γ·amp·cos θ), plus the matching
    # longitudinal velocity  v = -(σ/k)·amp·sin θ · k̂  with
    # σ = √(4πGρ₀ − c_s²k²). This launches δρ(t) = A₀ e^{σt} (clean exponential),
    # so a trivial log-linear fit recovers σ — no cosh/offset artifact.
    eigenmode = getattr(cfg, "jeans_eigenmode", True)
    kmag2 = kx * kx + ky * ky + kz * kz
    kmag = kmag2 ** 0.5
    cs2 = gamma * P0 / rho0
    Gconst = getattr(cfg, "GRAVITATIONAL_G", 1.0)
    sig2 = 4.0 * math.pi * Gconst * rho0 - cs2 * kmag2

    if eigenmode and sig2 > 0.0 and kmag > 0.0:
        sigma = sig2 ** 0.5
        P = P0 * (1.0 + gamma * amp * cphase)
        vfac = -(sigma * amp / kmag)
        vx = (vfac * kx / kmag) * sphase
        vy = (vfac * ky / kmag) * sphase
        vz = (vfac * kz / kmag) * sphase
    else:
        # legacy static / stable fallback (equal-amplitude δP, v = 0)
        P = P0 * (1.0 + amp * cphase)
        vx = jnp.zeros_like(rho)
        vy = jnp.zeros_like(rho)
        vz = jnp.zeros_like(rho)

    E = P / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz)

    return jnp.stack([rho, rho * vx, rho * vy, rho * vz, E])
