"""3D solar convection IC: polytropic atmosphere in hydrostatic equilibrium,
optionally with an entropy seed (z is up; cfg.gravity_z must be < 0).

Gamma = gamma gives an isentropic Schwarzschild-stable profile (for HSE
validation); to drive convection use Gamma > gamma or the entropy seed.
"""

import jax.numpy as jnp
import numpy as np


def _polytropic_profile(cfg):
    rho0 = getattr(cfg, "solar_rho0", 1.0)
    p0 = getattr(cfg, "solar_p0", 1.0)
    Gamma = getattr(cfg, "solar_Gamma", 0.0)
    if Gamma == 0.0:
        Gamma = cfg.gamma
    g = -cfg.gravity_z  # downward magnitude (>0)
    if g <= 0:
        raise ValueError("solar_convection_3d expects cfg.gravity_z < 0 (downward).")
    H = Gamma * p0 / ((Gamma - 1.0) * rho0 * g)
    return rho0, p0, Gamma, H


def _equilibrium_arrays(cfg, z_slice):
    """Build (ρ_eq, p_eq) per local cell along z, broadcast-ready."""
    rho0, p0, Gamma, H = _polytropic_profile(cfg)
    Nz = cfg.Nz
    Lz = cfg.Lz
    dz = Lz / Nz
    z_start, z_end = z_slice if z_slice is not None else (0, Nz)
    z_local = (jnp.arange(z_start, z_end) + 0.5) * dz  # cell-centered

    base = 1.0 - z_local / H  # (Nz_local,)
    # Guard against negative base if box too tall — clip to a small positive.
    base = jnp.maximum(base, 1e-8)
    rho_eq = rho0 * base ** (1.0 / (Gamma - 1.0))
    p_eq = p0 * base ** (Gamma / (Gamma - 1.0))
    return rho_eq, p_eq  # (Nz_local,)


def solar_convection_3d_equilibrium(cfg, x_slice=None, y_slice=None, z_slice=None):
    """Hydrostatic equilibrium state for the polytropic atmosphere.

    Returns U_eq with v=0 sharded the same as U. Used by the well-balanced
    integrator path.
    """
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    x_start, x_end = x_slice if x_slice is not None else (0, Nx)
    y_start, y_end = y_slice if y_slice is not None else (0, Ny)
    nx_l = x_end - x_start
    ny_l = y_end - y_start

    rho_z, p_z = _equilibrium_arrays(cfg, z_slice)  # (Nz_local,)
    nz_l = rho_z.shape[0]

    rho = jnp.broadcast_to(rho_z[None, None, :], (nx_l, ny_l, nz_l)).copy()
    p = jnp.broadcast_to(p_z[None, None, :], (nx_l, ny_l, nz_l)).copy()
    zeros = jnp.zeros_like(rho)
    E = p / (cfg.gamma - 1.0)  # KE=0
    return jnp.stack([rho, zeros, zeros, zeros, E], axis=0)


def solar_convection_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """Solar convection IC = hydrostatic equilibrium + optional entropy seed."""
    U_eq = solar_convection_3d_equilibrium(cfg, x_slice, y_slice, z_slice)
    amp = getattr(cfg, "solar_pert_amp", 0.0)
    if amp == 0.0:
        return U_eq

    # Add a small random entropy seed to E (which then perturbs T at fixed rho).
    seed = getattr(cfg, "solar_pert_seed", 11)
    # Match the slice shape
    shape = U_eq[0].shape
    # Reproducible RNG independent of process layout: numpy global seed +
    # slice-aware offset would be ideal; using simple seed for now.
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(shape).astype(np.float32)
    # Scale by amplitude and equilibrium internal energy at each cell.
    e_int_eq = U_eq[4] / U_eq[0] - 0.5 * 0.0  # = p_eq / ((gamma-1)*rho_eq)
    dE = amp * jnp.asarray(noise) * (U_eq[4])  # ~amp * E_eq
    return U_eq.at[4].add(dE)
