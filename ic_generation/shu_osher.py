import jax.numpy as jnp

from ._grid import local_range


def shu_osher(cfg, x_slice=None):
    """1D Shu-Osher test: Mach 3 shock (x < -4) hitting a sinusoidal density field.
    Domain is centered at 0 ([-Lx/2, Lx/2]).
    Returns U = [rho, rho*u, E], shape (3, Nx_local).
    """
    Nx, Lx = cfg.Nx, cfg.Lx
    gamma = cfg.gamma

    # Determine local range
    x_start, x_end = local_range(Nx, x_slice)

    # Create coordinate array for local slice only
    x_full = jnp.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
    x = x_full[x_start:x_end]

    # Left state (for xc < -4)
    rho_L = 3.857143
    u_L = 2.629369
    p_L = 10.33333

    # Right state (for xc > -4)
    rho_R = 1.0 + 0.2 * jnp.sin(5.0 * x)
    u_R = 0.0
    p_R = 1.0

    # Define the shock location at x = -4. We use xc = x - 5 for the domain centered at 0.
    left = x < -4.0  # Shock is at x = -4

    # Create density, velocity, pressure using conditions on left or right of shock
    rho = jnp.where(left, rho_L, rho_R)
    u = jnp.where(left, u_L, u_R)
    p = jnp.where(left, p_L, p_R)

    # Calculate the total energy using the formula: E = p / (gamma - 1) + 0.5 * rho * u^2
    E = p / (gamma - 1) + 0.5 * rho * u**2

    # Return the conservative variables: [rho, rho*u, E]
    return jnp.stack([rho, rho * u, E], axis=0)
