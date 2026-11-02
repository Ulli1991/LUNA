"""Linear Jeans instability: a periodic sinusoidal perturbation growing at σ = √(4πGρ₀ − c_s²k²)."""

from dataclasses import dataclass
import math

import jax.numpy as jnp

from ._grid import local_range


@dataclass(frozen=True)
class Params:
    """Jeans-Instability-3D's parameters, cfg.problem."""
    rho0: float = 1.0
    P0: float = 1.0
    amp: float = 1.0e-4
    kx: float = 0.0
    ky: float = 0.0
    kz: float = 0.0
    eigenmode: bool = True


def wavevector(cfg):
    """(kx, ky, kz) of the mode; all zero in Params means the fundamental along x."""
    kx, ky, kz = cfg.problem.kx, cfg.problem.ky, cfg.problem.kz
    if kx == 0.0 and ky == 0.0 and kz == 0.0:
        kx = 2.0 * jnp.pi / cfg.Lx
    return kx, ky, kz


def sigma_squared(cfg):
    """4 pi G rho0 - cs^2 |k|^2: the growth rate squared, negative for a stable (oscillating) mode."""
    kx, ky, kz = wavevector(cfg)
    kmag2 = kx * kx + ky * ky + kz * kz
    cs2 = cfg.gamma * cfg.problem.P0 / cfg.problem.rho0
    return 4.0 * math.pi * cfg.GRAVITATIONAL_G * cfg.problem.rho0 - cs2 * kmag2


def jeans_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """U for the perturbation; with cfg.problem.eigenmode, the pure growing mode."""
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    gamma = cfg.gamma

    rho0 = cfg.problem.rho0
    P0 = cfg.problem.P0
    amp = cfg.problem.amp
    kx, ky, kz = wavevector(cfg)

    dx = Lx / Nx; dy = Ly / Ny; dz = Lz / Nz

    xs, xe = local_range(Nx, x_slice)
    ys, ye = local_range(Ny, y_slice)
    zs, ze = local_range(Nz, z_slice)

    x = (jnp.arange(xs, xe) + 0.5) * dx
    y = (jnp.arange(ys, ye) + 0.5) * dy
    z = (jnp.arange(zs, ze) + 0.5) * dz
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    theta = kx * X + ky * Y + kz * Z
    cphase = jnp.cos(theta)
    sphase = jnp.sin(theta)
    rho = rho0 * (1.0 + amp * cphase)

    # growing eigenmode: δP = c_s²δρ and v = -(σ/k) amp sin θ k̂, so δρ grows as e^{σt}
    eigenmode = cfg.problem.eigenmode
    kmag = (kx * kx + ky * ky + kz * kz) ** 0.5
    sig2 = sigma_squared(cfg)

    # adiabatic: δP = c_s²δρ = γ P0 amp cos θ, no entropy mode (P0 (1 + amp cos θ) carried one)
    P = P0 * (1.0 + gamma * amp * cphase)
    if eigenmode and sig2 > 0.0 and kmag > 0.0:
        sigma = sig2 ** 0.5
        vfac = -(sigma * amp / kmag)
        vx = (vfac * kx / kmag) * sphase
        vy = (vfac * ky / kmag) * sphase
        vz = (vfac * kz / kmag) * sphase
    else:
        # at rest: a stable mode's standing wave; unstable, growing + decaying modes
        vx = jnp.zeros_like(rho)
        vy = jnp.zeros_like(rho)
        vz = jnp.zeros_like(rho)

    E = P / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz)

    return jnp.stack([rho, rho * vx, rho * vy, rho * vz, E])
