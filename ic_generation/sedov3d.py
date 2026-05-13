import jax.numpy as jnp


def sedov_blast_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """
    Sedov-Taylor blast wave in 3D.

    Uniform background with energy deposited in a central sphere.
    The blast wave expands self-similarly with known analytic solution.

    Parameters
    ----------
    cfg : Config
        Configuration object containing Nx, Ny, Nz, Lx, Ly, Lz, gamma.
    x_slice : tuple (x_start, x_end), optional
        Generate only this x-range for distributed initialization
    y_slice : tuple (y_start, y_end), optional
        Generate only this y-range for distributed initialization
    z_slice : tuple (z_start, z_end), optional
        Generate only this z-range for distributed initialization

    Returns
    -------
    U : jax.numpy.ndarray
        Conservative variables [rho, rho*vx, rho*vy, rho*vz, E]
        with shape (5, Nx_local, Ny_local, Nz_local)
    """
    Nx = cfg.Nx
    Ny = cfg.Ny
    Nz = cfg.Nz
    Lx = cfg.Lx
    Ly = cfg.Ly
    Lz = cfg.Lz
    gamma = cfg.gamma

    # Grid spacing
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz

    # Only print on main process
    if x_slice is None and y_slice is None and z_slice is None:
        print(f"3D Sedov Blast Configuration:")
        print(f"  Domain: [{Lx} x {Ly} x {Lz}]")
        print(f"  Grid: [{Nx} x {Ny} x {Nz}]")
        print(f"  dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}")
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

    # Create coordinate arrays for local slice
    x_full = jnp.linspace(0, Lx, Nx, endpoint=False) + 0.5 * dx
    y_full = jnp.linspace(0, Ly, Ny, endpoint=False) + 0.5 * dy
    z_full = jnp.linspace(0, Lz, Nz, endpoint=False) + 0.5 * dz
    x_local = x_full[x_start:x_end]
    y_local = y_full[y_start:y_end]
    z_local = z_full[z_start:z_end]

    # Create meshgrid
    X, Y, Z = jnp.meshgrid(x_local, y_local, z_local, indexing='ij')

    # Domain center
    x0 = 0.5 * Lx
    y0 = 0.5 * Ly
    z0 = 0.5 * Lz

    # Background state
    rho_bg = 1.0
    p_bg = 1e-2

    # Blast energy: deposited in a sphere of radius r_blast
    r_blast = 6.0 * max(dx, dy, dz)
    E_blast = 0.5  # Total energy deposited

    # Distance from center
    r = jnp.sqrt((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2)

    # Top-hat energy deposition inside blast sphere
    V_blast = (4.0 / 3.0) * jnp.pi * r_blast**3
    e_blast = E_blast / V_blast
    p_blast = (gamma - 1.0) * e_blast

    # Pressure: high inside blast sphere, background outside
    p = jnp.where(r < r_blast, p_blast + p_bg, p_bg)

    # Uniform density, zero velocity
    rho = jnp.full_like(X, rho_bg)
    vx = jnp.zeros_like(X)
    vy = jnp.zeros_like(X)
    vz = jnp.zeros_like(X)

    # Total energy
    E = p / (gamma - 1.0) + 0.5 * rho * (vx**2 + vy**2 + vz**2)

    # Conservative vars [rho, rho*vx, rho*vy, rho*vz, E]
    U = jnp.stack([rho, rho*vx, rho*vy, rho*vz, E], axis=0)
    return U
