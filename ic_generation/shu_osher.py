import jax.numpy as jnp

def shu_osher(cfg, x_slice=None):
    """
    Shu-Osher shock-density wave interaction test problem (1D).

    A Mach 3 shock (left state) interacts with a sinusoidal density field (right state).
    Tests the scheme's ability to resolve small-scale features in the presence of shocks.

    Left state (x < -4): Mach 3 shock
    Right state (x >= -4): Sinusoidal density perturbations

    Parameters
    ----------
    cfg : Config
        Configuration object containing Nx, Lx, gamma
    x_slice : tuple (x_start, x_end), optional
        Generate only this x-range for distributed initialization

    Returns
    -------
    U : jax.numpy.ndarray
        Conservative variables [rho, rho*u, E] with shape (3, Nx_local)
    """
    Nx, Lx = cfg.Nx, cfg.Lx
    gamma = cfg.gamma

    # Determine local range
    if x_slice is not None:
        x_start, x_end = x_slice
    else:
        x_start, x_end = 0, Nx

    # Create coordinate array for local slice only
    x_full = jnp.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
    x = x_full[x_start:x_end]

    # Left state (for xc < -4)
    rho_L = 3.857143
    u_L   = 2.629369
    p_L   = 10.33333

    # Right state (for xc > -4)
    rho_R = 1.0 + 0.2 * jnp.sin(5.0 * x)
    u_R   = 0.0
    p_R   = 1.0

    # Define the shock location at x = -4. We use xc = x - 5 for the domain centered at 0.
    left = x < -4.0  # Shock is at x = -4

    # Create density, velocity, pressure using conditions on left or right of shock
    rho = jnp.where(left, rho_L, rho_R)
    u   = jnp.where(left, u_L, u_R)
    p   = jnp.where(left, p_L, p_R)

    # Calculate the total energy using the formula: E = p / (gamma - 1) + 0.5 * rho * u^2
    E = p / (gamma - 1) + 0.5 * rho * u**2

    # Return the conservative variables: [rho, rho*u, E]
    return jnp.stack([rho, rho*u, E], axis=0)

