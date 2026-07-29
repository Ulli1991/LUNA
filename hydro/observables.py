"""Synthetic observables computed from the conserved state."""

import jax
import jax.numpy as jnp

from hydro.cooling import BOLTZMANN_CGS, PROTON_MASS_CGS


def xray_luminosity(U, cfg, mu=0.6):
    """Soft-band X-ray luminosity L_X = sum n^2 sqrt(T) band(T) dV, physical
    T [K] from the cfg unit system. Absolute scale is a bremsstrahlung proxy;
    the logarithmic sensitivity (elasticity) is scale-free."""
    gamma = cfg.gamma
    u = cfg.units
    rho = U[0]
    v2 = (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / (rho ** 2)
    P = (gamma - 1.0) * (U[4] - 0.5 * rho * v2)
    T = (mu * PROTON_MASS_CGS * jnp.clip(P, 1e-30) * u.UnitPressure_in_cgs
         / (BOLTZMANN_CGS * rho * u.UnitDensity_in_cgs))
    band = jax.nn.sigmoid((T - 1.0e6) / 3.0e5)
    emiss = rho ** 2 * jnp.sqrt(jnp.clip(T, 1.0)) * band
    dV = (cfg.Lx / cfg.Nx) * (cfg.Ly / cfg.Ny) * (cfg.Lz / cfg.Nz)
    return jnp.sum(emiss) * dV
