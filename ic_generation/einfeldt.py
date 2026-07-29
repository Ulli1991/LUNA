"""Einfeldt rarefaction test problems (1D/2D), Athena test suite.

The three registered variants share one implementation and differ only in
the (m_L/m_R, n_L/n_R, e_L/e_R) left/right state constants.
"""
import jax.numpy as jnp

from ._grid import local_range


def _einfeldt(cfg, x_slice, m_L, m_R, n_L, n_R, e_L, e_R):
    """Shared Einfeldt rarefaction IC: left/right states split at Lx/2.

    Returns conservative variables [rho, rho*u, rho*v, E], shape (4, Nx_local).
    """
    Nx, Lx = cfg.Nx, cfg.Lx

    x_start, x_end = local_range(Nx, x_slice)

    # Create coordinate array for local slice only
    x_full = jnp.linspace(0, Lx, Nx, endpoint=False)
    x = x_full[x_start:x_end]
    x0 = 0.5 * Lx
    left = x < x0

    rho_L, rho_R = 1.0, 1.0
    u_L, u_R = m_L / rho_L, m_R / rho_R
    v_L, v_R = n_L / rho_L, n_R / rho_R
    E_L, E_R = e_L, e_R

    rho = jnp.where(left, rho_L, rho_R)
    u = jnp.where(left, u_L, u_R)
    v = jnp.where(left, v_L, v_R)
    E = jnp.where(left, E_L, E_R)

    # 2D conservative variables: [rho, rho*u, rho*v, E]
    U = jnp.stack([rho, rho * u, rho * v, E], axis=0)
    return U


def einfeldt_strong_rarefaction(cfg, x_slice=None):
    """Strong opposing velocities (Athena 1-2-0-3):
    L (ρ=1, m=-2, n=0, e=3) | R (ρ=1, m=+2, n=0, e=3)."""
    return _einfeldt(cfg, x_slice, -2.0, 2.0, 0.0, 0.0, 3.0, 3.0)


def einfeldt_shear_rarefaction(cfg, x_slice=None):
    """Moderate opposing velocities (Athena 1-1-0-5):
    L (ρ=1, m=-1, n=0, e=5) | R (ρ=1, m=+1, n=0, e=5)."""
    return _einfeldt(cfg, x_slice, -1.0, 1.0, 0.0, 0.0, 5.0, 5.0)


def einfeldt_transverse_rarefaction(cfg, x_slice=None):
    """Rarefaction with strong transverse shear (Athena 1-1-2-5):
    L (ρ=1, m=-1, n=+2, e=5) | R (ρ=1, m=+1, n=-2, e=5)."""
    return _einfeldt(cfg, x_slice, -1.0, 1.0, 2.0, -2.0, 5.0, 5.0)
