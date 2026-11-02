"""Hot-CGM ambient for the CC85 starburst wind (Schneider, Robertson & Thompson 2018).

The wind itself is injected each step by the cc85 source; needs cfg.units.
"""

from dataclasses import dataclass

import jax.numpy as jnp

from ._grid import local_range


@dataclass(frozen=True)
class Params:
    """CC85-Wind-3D's parameters, cfg.problem."""
    n_ambient_cgs: float = 1.0e-4
    T_ambient_K: float = 1.0e6


_PROTON_MASS = 1.67262e-24
_BOLTZMANN = 1.38065e-16
_SEC_PER_YR = 3.15576e7
_MSUN = 1.98892e33
_KPC = 3.08568e21
_PC = 3.08568e18


def _ambient_state_cgs(n_amb, T_amb, mu=0.6):
    """Ambient (rho, p) in cgs."""
    rho = n_amb * mu * _PROTON_MASS
    p = n_amb * _BOLTZMANN * T_amb
    return rho, p


def cc85_wind_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """U for the uniform hot CGM."""
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    gamma = cfg.gamma

    n_amb = cfg.problem.n_ambient_cgs
    T_amb = cfg.problem.T_ambient_K
    mu = cfg.cooling_mu

    rho_cgs, p_cgs = _ambient_state_cgs(n_amb, T_amb, mu)

    if not cfg.units.is_enabled:
        raise ValueError("cc85_wind_3d requires cfg.units enabled (Arepo unit system).")
    rho_code = rho_cgs / cfg.units.UnitDensity_in_cgs
    p_code = p_cgs / cfg.units.UnitPressure_in_cgs

    x_start, x_end = local_range(Nx, x_slice)
    y_start, y_end = local_range(Ny, y_slice)
    z_start, z_end = local_range(Nz, z_slice)
    nx_l = x_end - x_start
    ny_l = y_end - y_start
    nz_l = z_end - z_start

    rho = jnp.full((nx_l, ny_l, nz_l), rho_code)
    zeros = jnp.zeros_like(rho)
    E = jnp.full_like(rho, p_code / (gamma - 1.0))
    return jnp.stack([rho, zeros, zeros, zeros, E], axis=0)
