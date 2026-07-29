import jax.numpy as jnp

from ._grid import local_range


def kelvin_helmholtz_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """3D Kelvin-Helmholtz: 2D KH plus a z-perturbation; periodic all directions.
    Overwrites cfg.Nz/cfg.Lz with an auto thin z-slab (16 cells, dz ≈ dx).
    Returns U = [rho, rho*vx, rho*vy, rho*vz, E], shape (5, Nx_l, Ny_l, Nz_l).
    """
    Nx, Ny = cfg.Nx, cfg.Ny
    Lx, Ly = cfg.Lx, cfg.Ly
    gamma = cfg.gamma

    # Auto-set thin z-slab: 16 cells with dz ≈ dx
    dx = Lx / Nx
    Nz = 16
    Lz = Nz * dx
    cfg.Nz = Nz
    cfg.Lz = Lz

    if x_slice is None and y_slice is None and z_slice is None:
        print("3D Kelvin-Helmholtz Configuration:")
        print(f"  X-Y plane: {Nx} x {Ny}, Lx={Lx}, Ly={Ly}")
        print(f"  Z-direction: Nz={Nz}, Lz={Lz:.6f} (auto, 16 cells, dz={dx:.6f})")
        print(f"  Total cells: {Nx*Ny*Nz:,}")

    # Determine local ranges
    x_start, x_end = local_range(Nx, x_slice)

    y_start, y_end = local_range(Ny, y_slice)

    z_start, z_end = local_range(Nz, z_slice)

    # Create coordinate arrays for local slice only
    x_full = jnp.linspace(0, Lx, Nx, endpoint=False)
    y_full = jnp.linspace(0, Ly, Ny, endpoint=False)
    z_full = jnp.linspace(0, Lz, Nz, endpoint=False)
    x_local = x_full[x_start:x_end]
    y_local = y_full[y_start:y_end]
    z_local = z_full[z_start:z_end]
    X, Y, Z = jnp.meshgrid(x_local, y_local, z_local, indexing="ij")

    # KH parameters (same as 2D)
    U0 = 1.0
    a = 0.05
    A = 0.01
    sigma = 0.2
    Y1, Y2 = 0.5, 1.5
    rho_low, rho_high = 1.0, 2.0
    delta = (rho_high - rho_low) / rho_low

    # Density and x-velocity: shear layers at Y1 and Y2
    rho = 1.0 + 0.5 * delta * (jnp.tanh((Y - Y1) / a) - jnp.tanh((Y - Y2) / a))
    vx = -U0 * ((jnp.tanh((Y - Y1) / a) - jnp.tanh((Y - Y2) / a)) - 1.0)

    # y-perturbation: seed instability (same as 2D but with z-modulation)
    vy = (
        A
        * jnp.sin(2 * jnp.pi * X / Lx)
        * (jnp.exp(-((Y - Y1) ** 2) / sigma**2) + jnp.exp(-((Y - Y2) ** 2) / sigma**2))
    )

    # z-perturbation: seed 3D structure
    vz = (
        0.5
        * A
        * jnp.sin(2 * jnp.pi * X / Lx)
        * jnp.sin(2 * jnp.pi * Z / Lz)
        * (jnp.exp(-((Y - Y1) ** 2) / sigma**2) + jnp.exp(-((Y - Y2) ** 2) / sigma**2))
    )

    # Uniform pressure
    Nx_local = x_local.shape[0]
    Ny_local = y_local.shape[0]
    Nz_local = z_local.shape[0]
    p = jnp.ones((Nx_local, Ny_local, Nz_local)) * 10.0

    # Total energy
    E = p / (gamma - 1.0) + 0.5 * rho * (vx**2 + vy**2 + vz**2)

    U = jnp.stack([rho, rho * vx, rho * vy, rho * vz, E], axis=0)
    return U
