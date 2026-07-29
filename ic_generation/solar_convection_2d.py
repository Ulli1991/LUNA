"""2D solar convection IC: polytropic atmosphere in hydrostatic equilibrium,
optionally with an entropy seed. Same recipe as the 3D version.

Vertical axis is y; gravity still lives in cfg.gravity_z (legacy naming) and
must be < 0 (downward). State has 4 fields (rho, vx, vy, E).
"""

import jax.numpy as jnp
import numpy as np


def _polytropic_profile(cfg):
    rho0 = getattr(cfg, "solar_rho0", 1.0)
    p0 = getattr(cfg, "solar_p0", 1.0)
    Gamma = getattr(cfg, "solar_Gamma", 0.0)
    if Gamma == 0.0:
        Gamma = cfg.gamma
    g = -cfg.gravity_z
    if g <= 0:
        raise ValueError("solar_convection_2d expects cfg.gravity_z < 0 (downward).")
    H = Gamma * p0 / ((Gamma - 1.0) * rho0 * g)
    return rho0, p0, Gamma, H


def _equilibrium_arrays(cfg, y_slice):
    rho0, p0, Gamma, H = _polytropic_profile(cfg)
    Ny, Ly = cfg.Ny, cfg.Ly
    dy = Ly / Ny
    y_start, y_end = y_slice if y_slice is not None else (0, Ny)
    y_local = (jnp.arange(y_start, y_end) + 0.5) * dy
    base = jnp.maximum(1.0 - y_local / H, 1e-8)
    rho_eq = rho0 * base ** (1.0 / (Gamma - 1.0))
    p_eq = p0 * base ** (Gamma / (Gamma - 1.0))
    return rho_eq, p_eq


def solar_convection_2d_equilibrium(cfg, x_slice=None, y_slice=None, z_slice=None):
    """Hydrostatic equilibrium U_eq for the 2D polytropic atmosphere."""
    Nx, Ny = cfg.Nx, cfg.Ny
    x_start, x_end = x_slice if x_slice is not None else (0, Nx)
    nx_l = x_end - x_start
    rho_y, p_y = _equilibrium_arrays(cfg, y_slice)
    ny_l = rho_y.shape[0]
    rho = jnp.broadcast_to(rho_y[None, :], (nx_l, ny_l)).copy()
    p = jnp.broadcast_to(p_y[None, :], (nx_l, ny_l)).copy()
    zeros = jnp.zeros_like(rho)
    E = p / (cfg.gamma - 1.0)
    return jnp.stack([rho, zeros, zeros, E], axis=0)


def solar_convection_2d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """2D solar convection IC = equilibrium + optional entropy seed."""
    U_eq = solar_convection_2d_equilibrium(cfg, x_slice, y_slice, z_slice)
    amp = getattr(cfg, "solar_pert_amp", 0.0)
    if amp == 0.0:
        return U_eq
    seed = getattr(cfg, "solar_pert_seed", 11)
    shape = U_eq[0].shape
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(shape).astype(np.float32)
    dE = amp * jnp.asarray(noise) * U_eq[3]
    return U_eq.at[3].add(dE)
