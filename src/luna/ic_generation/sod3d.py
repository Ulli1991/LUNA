"""3D Sod shock tubes: a thin tube, and the AMR variant on the configured grid."""
import jax.numpy as jnp

from ._grid import local_range


def transverse(cfg):
    """Ny/Nz/Ly/Lz for a thin tube of cubic cells, transverse size ~0.1."""
    dx = cfg.Lx / cfg.Nx
    n = max(1, round(0.1 / dx))
    return dict(Ny=n, Nz=n, Ly=n * dx, Lz=n * dx)


def sod_shock_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """U for a Sod shock along x on the thin tube `transverse` chose."""
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    gamma = cfg.gamma
    dx = Lx / Nx

    if x_slice is None and y_slice is None and z_slice is None:
        print("3D Sod Shock Tube Configuration:")
        print(f"  X-direction: Nx={Nx}, Lx={Lx:.6f}, dx={dx:.6f}")
        print(f"  Y-direction: Ny={Ny}, Ly={Ly:.6f}, dy={Ly/Ny:.6f}")
        print(f"  Z-direction: Nz={Nz}, Lz={Lz:.6f}, dz={Lz/Nz:.6f}")
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

    x0 = 0.5 * Lx

    rho_L, rho_R = 1.0, 0.125
    p_L, p_R = 1.0, 0.1
    vx_L, vx_R = 0.0, 0.0
    vy_L, vy_R = 0.0, 0.0
    vz_L, vz_R = 0.0, 0.0

    left = X < x0

    rho = jnp.where(left, rho_L, rho_R)
    vx = jnp.where(left, vx_L, vx_R)
    vy = jnp.where(left, vy_L, vy_R)
    vz = jnp.where(left, vz_L, vz_R)
    p = jnp.where(left, p_L, p_R)

    E = p / (gamma - 1.0) + 0.5 * rho * (vx**2 + vy**2 + vz**2)

    U = jnp.stack([rho, rho * vx, rho * vy, rho * vz, E], axis=0)
    return U


def sod_shock_3d_amr(cfg, x_slice=None, y_slice=None, z_slice=None):
    """U for a Sod shock along x on the grid as configured (Ny/Nz must fit the AMR blocks)."""
    import jax.numpy as jnp
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, gamma = cfg.Lx, cfg.gamma
    xs, xe = local_range(Nx, x_slice)
    ys, ye = local_range(Ny, y_slice)
    zs, ze = local_range(Nz, z_slice)
    x = (jnp.arange(xs, xe) + 0.5) * (Lx / Nx)
    left = (x < 0.5 * Lx)[:, None, None]
    ny, nz = ye - ys, ze - zs
    rho = jnp.where(left, 1.0, 0.125) * jnp.ones((xe - xs, ny, nz))
    p = jnp.where(left, 1.0, 0.1) * jnp.ones((xe - xs, ny, nz))
    U = jnp.zeros((5, xe - xs, ny, nz))
    U = U.at[0].set(rho)
    U = U.at[4].set(p / (gamma - 1.0))
    return U
