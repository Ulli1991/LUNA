import jax.numpy as jnp

def shu_osher(cfg):
    Nx, Lx = cfg.Nx, cfg.Lx
    gamma = cfg.gamma

    # Define x values to cover the domain from -5 to 5
    x = jnp.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)  # Adjusted to go from -5 to 5

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

