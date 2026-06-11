"""Linear Jeans instability initial condition.

Uniform medium (ρ₀, P₀, v=0) on a periodic box, perturbed by a
sinusoidal density wave at wavenumber (jeans_kx, jeans_ky, jeans_kz)
with relative amplitude jeans_amp.

The Hancock predictor + multigrid self-gravity should grow the unstable
mode at the linear-theory rate σ = √(4πGρ₀ − c_s² k²) where k = |k|.

Config knobs expected:
    jeans_rho0 : background density
    jeans_P0   : background pressure (constant-T isothermal-like setup)
    jeans_amp  : initial relative density amplitude
    jeans_kx, jeans_ky, jeans_kz : perturbation wavenumber components
"""

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

    perturb = jnp.cos(kx * X + ky * Y + kz * Z)
    rho = rho0 * (1.0 + amp * perturb)
    P = P0 * (1.0 + amp * perturb)  # isothermal-like (c_s² ≈ γP/ρ stays ≈ const at linear order)
    vx = jnp.zeros_like(rho)
    vy = jnp.zeros_like(rho)
    vz = jnp.zeros_like(rho)
    E = P / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz)

    return jnp.stack([rho, rho * vx, rho * vy, rho * vz, E])
