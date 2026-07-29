"""CC85-style starburst wind (M82 parameters). Ref: Schneider, Robertson &
Thompson 2018.

This IC only sets up the hot-CGM ambient medium; the wind is injected every
step by the runner's cc85_source term, not as initial mass. Requires cfg.units
enabled.
"""

import jax.numpy as jnp

# Physical-units knobs (read from cfg, with sensible M82 defaults below).
_PROTON_MASS = 1.67262e-24  # g
_BOLTZMANN = 1.38065e-16  # erg/K
_SEC_PER_YR = 3.15576e7
_MSUN = 1.98892e33  # g
_KPC = 3.08568e21  # cm
_PC = 3.08568e18  # cm


def _ambient_state_cgs(n_amb, T_amb, mu=0.6):
    """Return rho and p of ambient gas in cgs given (n_amb, T_amb)."""
    rho = n_amb * mu * _PROTON_MASS  # g/cm^3
    p = n_amb * _BOLTZMANN * T_amb  # erg/cm^3
    return rho, p


def cc85_wind_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """Uniform hot CGM ambient. The wind appears via a runner-side source."""
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    gamma = cfg.gamma

    n_amb = getattr(cfg, "cc85_n_ambient_cgs", 1.0e-4)
    T_amb = getattr(cfg, "cc85_T_ambient_K", 1.0e6)
    mu = getattr(cfg, "cc85_mu", 0.6)

    rho_cgs, p_cgs = _ambient_state_cgs(n_amb, T_amb, mu)

    # Convert to code units via cfg.units (must be enabled).
    if not cfg.units.is_enabled:
        raise ValueError("cc85_wind_3d requires cfg.units enabled (Arepo unit system).")
    rho_code = rho_cgs / cfg.units.UnitDensity_in_cgs
    p_code = p_cgs / cfg.units.UnitPressure_in_cgs

    x_start, x_end = x_slice if x_slice is not None else (0, Nx)
    y_start, y_end = y_slice if y_slice is not None else (0, Ny)
    z_start, z_end = z_slice if z_slice is not None else (0, Nz)
    nx_l = x_end - x_start
    ny_l = y_end - y_start
    nz_l = z_end - z_start

    rho = jnp.full((nx_l, ny_l, nz_l), rho_code)
    zeros = jnp.zeros_like(rho)
    E = jnp.full_like(rho, p_code / (gamma - 1.0))
    return jnp.stack([rho, zeros, zeros, zeros, E], axis=0)
