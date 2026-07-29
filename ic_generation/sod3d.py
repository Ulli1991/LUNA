import jax.numpy as jnp


def sod_shock_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """3D Sod shock tube: shock along x, uniform in y/z.
    Mutates cfg.Ny/Nz/Ly/Lz to keep cubic cells with transverse size ~0.1.
    Returns U = [rho, rho*vx, rho*vy, rho*vz, E], shape (5, Nx_l, Ny_l, Nz_l).
    """
    Nx = cfg.Nx
    Lx = cfg.Lx
    gamma = cfg.gamma

    # Compute grid spacing in x-direction
    dx = Lx / Nx

    # Target transverse physical size
    transverse_target = 0.1

    # Compute Ny, Nz to get cubic cells
    Ny = max(1, round(transverse_target / dx))
    Nz = max(1, round(transverse_target / dx))

    # Compute Ly, Lz to ensure exactly cubic cells
    Ly = Ny * dx
    Lz = Nz * dx

    # Update the config object so the rest of the code uses these values
    cfg.Ny = Ny
    cfg.Nz = Nz
    cfg.Ly = Ly
    cfg.Lz = Lz

    # Only print on main process (when generating full array, not slices)
    if x_slice is None and y_slice is None and z_slice is None:
        print("3D Sod Shock Tube Configuration:")
        print(f"  X-direction: Nx={Nx}, Lx={Lx:.6f}, dx={dx:.6f}")
        print(f"  Y-direction: Ny={Ny}, Ly={Ly:.6f}, dy={Ly/Ny:.6f}")
        print(f"  Z-direction: Nz={Nz}, Lz={Lz:.6f}, dz={Lz/Nz:.6f}")
        print(f"  Total cells: {Nx*Ny*Nz:,}")

    # Determine local ranges
    if x_slice is not None:
        x_start, x_end = x_slice
        Nx_local = x_end - x_start
    else:
        x_start, x_end = 0, Nx
        Nx_local = Nx

    if y_slice is not None:
        y_start, y_end = y_slice
        Ny_local = y_end - y_start
    else:
        y_start, y_end = 0, Ny
        Ny_local = Ny

    if z_slice is not None:
        z_start, z_end = z_slice
        Nz_local = z_end - z_start
    else:
        z_start, z_end = 0, Nz
        Nz_local = Nz

    # Create coordinate arrays for local slice only
    x_full = jnp.linspace(0, Lx, Nx, endpoint=False)
    y_full = jnp.linspace(0, Ly, Ny, endpoint=False)
    z_full = jnp.linspace(0, Lz, Nz, endpoint=False)
    x_local = x_full[x_start:x_end]
    y_local = y_full[y_start:y_end]
    z_local = z_full[z_start:z_end]

    # Create meshgrid
    X, Y, Z = jnp.meshgrid(x_local, y_local, z_local, indexing="ij")

    # Interface at center
    x0 = 0.5 * Lx

    # Left and right states (same as 1D Sod)
    rho_L, rho_R = 1.0, 0.125
    p_L, p_R = 1.0, 0.1
    vx_L, vx_R = 0.0, 0.0
    vy_L, vy_R = 0.0, 0.0
    vz_L, vz_R = 0.0, 0.0

    # Condition based on x-coordinate
    left = X < x0

    rho = jnp.where(left, rho_L, rho_R)
    vx = jnp.where(left, vx_L, vx_R)
    vy = jnp.where(left, vy_L, vy_R)
    vz = jnp.where(left, vz_L, vz_R)
    p = jnp.where(left, p_L, p_R)

    # Total energy
    E = p / (gamma - 1.0) + 0.5 * rho * (vx**2 + vy**2 + vz**2)

    # Conservative vars [rho, rho*vx, rho*vy, rho*vz, E]
    U = jnp.stack([rho, rho * vx, rho * vy, rho * vz, E], axis=0)
    return U


def sod_shock_3d_amr(cfg, x_slice=None, y_slice=None, z_slice=None):
    """Planar Sod shock tube on the grid AS CONFIGURED (no Ny/Nz mutation) --
    the AMR benchmark variant: cfg.Ny/Nz must be block-shape multiples, so the
    auto-cubic resizing of :func:`sod_shock_3d` is skipped. Discontinuity at
    x = 0.5*Lx; classic states L=(1,0,1), R=(0.125,0,0.1)."""
    import jax.numpy as jnp
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, gamma = cfg.Lx, cfg.gamma
    xs, xe = x_slice if x_slice is not None else (0, Nx)
    ys, ye = y_slice if y_slice is not None else (0, Ny)
    zs, ze = z_slice if z_slice is not None else (0, Nz)
    x = (jnp.arange(xs, xe) + 0.5) * (Lx / Nx)
    left = (x < 0.5 * Lx)[:, None, None]
    ny, nz = ye - ys, ze - zs
    rho = jnp.where(left, 1.0, 0.125) * jnp.ones((xe - xs, ny, nz))
    p = jnp.where(left, 1.0, 0.1) * jnp.ones((xe - xs, ny, nz))
    U = jnp.zeros((5, xe - xs, ny, nz))
    U = U.at[0].set(rho)
    U = U.at[4].set(p / (gamma - 1.0))
    return U
