import jax.numpy as jnp

def rotation_slotted_disk(cfg):
    Nx, Ny, Lx, Ly = cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly
    gamma = cfg.gamma

    # Grid coordinates
    x = jnp.linspace(0, Lx, Nx, endpoint=False)
    y = jnp.linspace(0, Ly, Ny, endpoint=False)
    X, Y = jnp.meshgrid(x, y, indexing='xy')

    # Centered coordinates
    xc = X - 0.5 * Lx
    yc = Y - 0.5 * Ly

    # Disk region
    R2 = xc**2 + yc**2
    inner = 0.15**2
    outer = 0.25**2
    disk = (R2 > inner) & (R2 < outer)

    # Slot condition
    slot_vertical = jnp.abs(xc) <= 0.025
    slot_upper = Y >= 0.85
    slot = slot_vertical & slot_upper

    # Slotted disk = disk with slot cut out
    slotted = disk & slot

    # Density
    rho = 1.0 + slotted.astype(jnp.float32)

    # Velocities (rigid rotation)
    u = -yc
    v =  xc

    # Pressure
    p = jnp.ones_like(rho)

    # Total energy
    E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2)

    # Conservative variables
    U = jnp.stack([rho, rho*u, rho*v, E], axis=0)
    return U

