import jax.numpy as jnp

def rotation_slotted_disk(cfg):
    Nx, Ny, Lx, Ly = cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly
    gamma = cfg.gamma

    # 2D coordinates
    x = jnp.linspace(0, Lx, Nx, endpoint=False)
    y = jnp.linspace(0, Ly, Ny, endpoint=False)
    X, Y = jnp.meshgrid(x, y)

    # Polar coordinates
    R = jnp.sqrt(X**2 + Y**2)
    theta = jnp.arctan2(Y, X)

    # Disk with slotted profile
    disk = (R < 0.4 * Lx)

    # Angular velocity (constant)
    omega = 2.0 * jnp.pi  # 1 complete rotation per unit time

    # Initial conditions
    rho = disk.astype(jnp.float32) + 1.0  # uniform density inside disk
    u = -omega * Y  # velocity components due to rotation
    v = omega * X
    p = jnp.ones_like(rho)  # constant pressure

    # Total energy
    E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2)

    # Conservative variables [rho, rho*u, rho*v, E]
    U = jnp.stack([rho, rho*u, rho*v, E], axis=0)
    return U

