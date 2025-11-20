import jax.numpy as jnp

def sod_shock_3d(cfg):
    """
    Sod shock tube test problem in 3D.

    Shock propagates in x-direction, uniform in y and z directions.
    Automatically computes Ly, Lz, Ny, Nz to maintain cubic cells (dx = dy = dz)
    with transverse dimensions close to 0.1.

    Parameters
    ----------
    cfg : Config
        Configuration object containing Nx, Lx, gamma.
        Note: Ny, Nz, Ly, Lz are modified by this function.

    Returns
    -------
    U : jax.numpy.ndarray
        Conservative variables [rho, rho*vx, rho*vy, rho*vz, E] with shape (5, Nx, Ny, Nz)
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

    print(f"3D Sod Shock Tube Configuration:")
    print(f"  X-direction: Nx={Nx}, Lx={Lx:.6f}, dx={dx:.6f}")
    print(f"  Y-direction: Ny={Ny}, Ly={Ly:.6f}, dy={Ly/Ny:.6f}")
    print(f"  Z-direction: Nz={Nz}, Lz={Lz:.6f}, dz={Lz/Nz:.6f}")
    print(f"  Total cells: {Nx*Ny*Nz:,}")

    # 3D coordinates
    x = jnp.linspace(0, Lx, Nx, endpoint=False)
    y = jnp.linspace(0, Ly, Ny, endpoint=False)
    z = jnp.linspace(0, Lz, Nz, endpoint=False)

    # Create meshgrid
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')

    # Interface at center
    x0 = 0.5 * Lx

    # Left and right states (same as 1D Sod)
    rho_L, rho_R = 1.0, 0.125
    p_L,   p_R   = 1.0, 0.1
    vx_L,  vx_R  = 0.0, 0.0
    vy_L,  vy_R  = 0.0, 0.0
    vz_L,  vz_R  = 0.0, 0.0

    # Condition based on x-coordinate
    left = X < x0

    rho = jnp.where(left, rho_L, rho_R)
    vx  = jnp.where(left, vx_L,  vx_R)
    vy  = jnp.where(left, vy_L,  vy_R)
    vz  = jnp.where(left, vz_L,  vz_R)
    p   = jnp.where(left, p_L,   p_R)

    # Total energy
    E = p / (gamma - 1.0) + 0.5 * rho * (vx**2 + vy**2 + vz**2)

    # Conservative vars [rho, rho*vx, rho*vy, rho*vz, E]
    U = jnp.stack([rho, rho*vx, rho*vy, rho*vz, E], axis=0)
    return U
