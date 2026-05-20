"""CC85-style starburst wind (M82-nuclear-region parameters).

Setup follows Schneider, Robertson, Thompson 2018 ("Production of Cool Gas in
Thermally Driven Outflows") with M82 loading factors:

    SFR        = 12 Msun/yr   (M82 nuclear starburst)
    eta_M      = 1            (mass loading)
    eta_E      = 0.6          (energy loading)
    => Mdot   = eta_M * SFR                       = 12 Msun/yr
    => Edot   = eta_E * (SFR / 100 Msun) * 1e51 erg / yr ~ 2.28e42 erg/s
    => v_inf  = sqrt(2*Edot/Mdot)                 ~ 776 km/s

Injection region: uniform mass/energy deposition inside a sphere of radius
R_* (here 300 pc).

This IC just sets up the AMBIENT medium and the injection-region book-keeping
(the central source term is applied at every step in the runner — see the
cc85_source machinery elsewhere). The injected gas itself appears only via
the operator-split source, not as initial mass inside R_*.

Ambient: low-density hot CGM ~ (n = 1e-4 cm^-3, T = 1e6 K).
"""
import jax.numpy as jnp


# Physical-units knobs (read from cfg, with sensible M82 defaults below).
_PROTON_MASS = 1.67262e-24      # g
_BOLTZMANN   = 1.38065e-16      # erg/K
_SEC_PER_YR  = 3.15576e7
_MSUN        = 1.98892e33       # g
_KPC         = 3.08568e21       # cm
_PC          = 3.08568e18       # cm


def _ambient_state_cgs(n_amb, T_amb, mu=0.6):
    """Return rho and p of ambient gas in cgs given (n_amb, T_amb)."""
    rho = n_amb * mu * _PROTON_MASS                # g/cm^3
    p   = n_amb * _BOLTZMANN * T_amb               # erg/cm^3
    return rho, p


def cc85_wind_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """Uniform hot CGM ambient. The wind appears via a runner-side source."""
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    gamma = cfg.gamma

    n_amb = getattr(cfg, "cc85_n_ambient_cgs", 1.0e-4)
    T_amb = getattr(cfg, "cc85_T_ambient_K",   1.0e6)
    mu    = getattr(cfg, "cc85_mu",            0.6)

    rho_cgs, p_cgs = _ambient_state_cgs(n_amb, T_amb, mu)

    # Convert to code units via cfg.units (must be enabled).
    if not cfg.units.is_enabled:
        raise ValueError("cc85_wind_3d requires cfg.units enabled (Arepo unit system).")
    rho_code = rho_cgs / cfg.units.UnitDensity_in_cgs
    p_code   = p_cgs   / cfg.units.UnitPressure_in_cgs

    x_start, x_end = (x_slice if x_slice is not None else (0, Nx))
    y_start, y_end = (y_slice if y_slice is not None else (0, Ny))
    z_start, z_end = (z_slice if z_slice is not None else (0, Nz))
    nx_l = x_end - x_start; ny_l = y_end - y_start; nz_l = z_end - z_start

    rho = jnp.full((nx_l, ny_l, nz_l), rho_code)
    zeros = jnp.zeros_like(rho)
    E = jnp.full_like(rho, p_code / (gamma - 1.0))
    return jnp.stack([rho, zeros, zeros, zeros, E], axis=0)
