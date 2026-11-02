"""3D Kelvin-Helmholtz on a thin periodic z-slab."""
import jax.numpy as jnp

from ._grid import local_range


def transverse(cfg):
    """Nz/Lz for a thin z-slab of 16 cells with dz = dx."""
    dx = cfg.Lx / cfg.Nx
    return dict(Nz=16, Lz=16 * dx)


def kelvin_helmholtz_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """U for 2D KH plus a z-perturbation on the slab `transverse` chose."""
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    gamma = cfg.gamma
    dx = Lx / Nx

    if x_slice is None and y_slice is None and z_slice is None:
        print("3D Kelvin-Helmholtz Configuration:")
        print(f"  X-Y plane: {Nx} x {Ny}, Lx={Lx}, Ly={Ly}")
        print(f"  Z-direction: Nz={Nz}, Lz={Lz:.6f} (auto, 16 cells, dz={dx:.6f})")
        print(f"  Total cells: {Nx*Ny*Nz:,}")

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

    # same parameters as 2D, as fractions of Ly
    U0 = 1.0
    a = 0.025 * Ly
    A = 0.01
    sigma = 0.1 * Ly
    Y1, Y2 = 0.25 * Ly, 0.75 * Ly
    rho_low, rho_high = 1.0, 2.0
    delta = (rho_high - rho_low) / rho_low

    # shear layers at Y1 and Y2
    rho = 1.0 + 0.5 * delta * (jnp.tanh((Y - Y1) / a) - jnp.tanh((Y - Y2) / a))
    vx = -U0 * ((jnp.tanh((Y - Y1) / a) - jnp.tanh((Y - Y2) / a)) - 1.0)

    vy = (
        A
        * jnp.sin(2 * jnp.pi * X / Lx)
        * (jnp.exp(-((Y - Y1) ** 2) / sigma**2) + jnp.exp(-((Y - Y2) ** 2) / sigma**2))
    )

    vz = (
        0.5
        * A
        * jnp.sin(2 * jnp.pi * X / Lx)
        * jnp.sin(2 * jnp.pi * Z / Lz)
        * (jnp.exp(-((Y - Y1) ** 2) / sigma**2) + jnp.exp(-((Y - Y2) ** 2) / sigma**2))
    )

    Nx_local = x_local.shape[0]
    Ny_local = y_local.shape[0]
    Nz_local = z_local.shape[0]
    p = jnp.ones((Nx_local, Ny_local, Nz_local)) * 10.0

    E = p / (gamma - 1.0) + 0.5 * rho * (vx**2 + vy**2 + vz**2)

    U = jnp.stack([rho, rho * vx, rho * vy, rho * vz, E], axis=0)
    return U
