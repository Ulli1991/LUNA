import jax.numpy as jnp

def compute_dt(U, dx, dy, gamma, CFL):
    """
    Compute timestep based on CFL condition.
    U: (4, Nx, Ny) array of conservative variables
    dx, dy: grid spacing in x and y directions
    gamma: adiabatic index
    CFL: CFL condition constant
    """
    rho, rhou, rhov, E = U

    ux = rhou / rho
    uy = rhov / rho
    p  = (gamma - 1.0) * (E - 0.5 * rho * (ux**2 + uy**2))
    c  = jnp.sqrt(gamma * p / rho)

    # max signal speed in x and y
    Sx = jnp.max(jnp.abs(ux) + c)
    Sy = jnp.max(jnp.abs(uy) + c)

    dt_x = CFL * dx / Sx
    dt_y = CFL * dy / Sy

    return float(jnp.minimum(dt_x, dt_y))

