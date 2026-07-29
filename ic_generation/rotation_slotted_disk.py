import jax.numpy as jnp

from ._grid import local_range


def rotation_slotted_disk(cfg, x_slice=None, y_slice=None):
    """2D Zalesak slotted disk in rigid rotation (advection of sharp features).
    Uses a normalized [0,1]^2 domain regardless of Lx/Ly.
    Returns U = [rho, rho*ux, rho*uy, E], shape (4, Nx_local, Ny_local).
    """
    Nx, Ny, Lx, Ly = cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly
    gamma = cfg.gamma

    # Determine local ranges
    x_start, x_end = local_range(Nx, x_slice)

    y_start, y_end = local_range(Ny, y_slice)

    # Create coordinate arrays for local slice only (normalized to [0,1] domain)
    x_full = jnp.linspace(0, 1, Nx, endpoint=False)
    y_full = jnp.linspace(0, 1, Ny, endpoint=False)
    x_local = x_full[x_start:x_end]
    y_local = y_full[y_start:y_end]
    X, Y = jnp.meshgrid(x_local, y_local, indexing="ij")

    # Zalesak disk: solid circle centered at (0.5, 0.75) with radius 0.15
    disk_center_x = 0.5
    disk_center_y = 0.75
    disk_radius = 0.15

    # Distance from disk center
    R2 = (X - disk_center_x) ** 2 + (Y - disk_center_y) ** 2
    disk = R2 <= disk_radius**2

    # Slot: rectangular notch with |x - 0.5| < 0.025 AND y < 0.85
    slot_width_half = 0.025
    slot_top = 0.85
    slot = (jnp.abs(X - disk_center_x) < slot_width_half) & (Y < slot_top)

    # Slotted disk = disk with slot cut out
    slotted = disk & ~slot

    # Density: rho=2 inside slotted disk, rho=1 outside
    rho = jnp.where(slotted, 2.0, 1.0)

    # Velocities: rigid rotation around (0.5, 0.5) with angular velocity ω
    # For one full rotation in time T, we need: T = 2π/ω, so ω = 2π/T
    # Standard Zalesak disk: one rotation in T = 2π
    rotation_center_x = 0.5
    rotation_center_y = 0.5
    omega = -1.0  # negative for clockwise rotation (viewed from above)
    u = omega * (rotation_center_y - Y)
    v = omega * (X - rotation_center_x)

    # Pressure: must balance centrifugal force for equilibrium
    # For solid body rotation: dp/dr = ρ*ω²*r → p(r) = p_center + 0.5*ρ*ω²*r²
    # CRITICAL: Must use LOCAL density ρ, not a constant!
    # The pressure discontinuity at the disk boundary is physically required
    r_squared = (X - rotation_center_x) ** 2 + (Y - rotation_center_y) ** 2
    p_center = 10.0  # pressure at rotation center
    p = p_center + 0.5 * rho * (omega**2) * r_squared

    # Total energy
    E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2)

    # Conservative variables
    U = jnp.stack([rho, rho * u, rho * v, E], axis=0)
    return U
