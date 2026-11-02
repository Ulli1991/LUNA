"""3D linear sound wave along the box diagonal."""
import jax.numpy as jnp

from ._grid import local_range


def soundwave_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """U for a small diagonal density and pressure perturbation at rest; needs periodic boundaries."""
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    gamma = cfg.gamma

    x_start, x_end = local_range(Nx, x_slice)

    y_start, y_end = local_range(Ny, y_slice)

    z_start, z_end = local_range(Nz, z_slice)

    x_full = (jnp.arange(Nx) + 0.5) * (Lx / Nx)
    y_full = (jnp.arange(Ny) + 0.5) * (Ly / Ny)
    z_full = (jnp.arange(Nz) + 0.5) * (Lz / Nz)
    x_local = x_full[x_start:x_end]
    y_local = y_full[y_start:y_end]
    z_local = z_full[z_start:z_end]
    X, Y, Z = jnp.meshgrid(x_local, y_local, z_local, indexing="ij")

    rho0 = 1.0
    p0 = 1.0
    epsilon = 1e-6

    kx = 2.0 * jnp.pi / Lx
    ky = 2.0 * jnp.pi / Ly
    kz = 2.0 * jnp.pi / Lz

    phase = kx * X + ky * Y + kz * Z

    rho = rho0 * (1.0 + epsilon * jnp.sin(phase))
    p = p0 * (1.0 + epsilon * jnp.sin(phase))

    vx = jnp.zeros_like(X)
    vy = jnp.zeros_like(X)
    vz = jnp.zeros_like(X)

    E = p / (gamma - 1.0) + 0.5 * rho * (vx**2 + vy**2 + vz**2)

    U = jnp.stack([rho, rho * vx, rho * vy, rho * vz, E], axis=0)
    return U
