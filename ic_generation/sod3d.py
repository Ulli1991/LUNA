import jax.numpy as jnp

def sod_shock_3d(cfg):
    """
    3D Sod shock tube test - shock propagates in x-direction,
    uniform in y and z directions.
    """
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    gamma = cfg.gamma

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
