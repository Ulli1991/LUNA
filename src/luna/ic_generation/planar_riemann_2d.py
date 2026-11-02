"""2D planar Riemann problem along x: Sod's left state (1, 0, 1) against (0.125, 2, 0.1), split at Lx/2.

Two rarefactions about a contact, uniform in y; exact solution: luna.hydro.exact_riemann. (Formerly
registered as Double-Mach-Reflection, which it is not.)
"""
import jax.numpy as jnp

from ._grid import local_range


def planar_riemann_2d(cfg, x_slice=None, y_slice=None):
    """U for the planar Riemann state split at Lx/2."""
    Nx, Ny, Lx, Ly = cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly
    gamma = cfg.gamma

    x_start, x_end = local_range(Nx, x_slice)

    y_start, y_end = local_range(Ny, y_slice)

    x_full = (jnp.arange(Nx) + 0.5) * (Lx / Nx)
    y_full = (jnp.arange(Ny) + 0.5) * (Ly / Ny)
    x_local = x_full[x_start:x_end]
    y_local = y_full[y_start:y_end]
    X, Y = jnp.meshgrid(x_local, y_local, indexing="ij")

    shock_location = 0.5 * Lx

    rho_left = 1.0
    rho_right = 0.125
    p_left = 1.0
    p_right = 0.1
    u_left = 0.0
    u_right = 2.0

    rho = jnp.where(X < shock_location, rho_left, rho_right)
    u = jnp.where(X < shock_location, u_left, u_right)
    p = jnp.where(X < shock_location, p_left, p_right)
    v = jnp.zeros_like(rho)

    E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2)

    U = jnp.stack([rho, rho * u, rho * v, E], axis=0)
    return U


def double_mach_reflection(cfg, x_slice=None, y_slice=None):
    """Refused: the real double Mach reflection (Woodward & Colella 1984) is not implemented yet."""
    raise NotImplementedError(
        "Double-Mach-Reflection (Woodward & Colella 1984) needs per-face and time-dependent boundaries "
        "(a reflecting wall for x > 1/6, inflow, the exact shock on the top face), which LUNA does not "
        "have yet (docs/TEST_PLAN.md); the planar 2D Riemann problem formerly registered under this "
        "name is Planar-Riemann-2D")
