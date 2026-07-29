"""Operator-split EOS projections: reset total energy so the pressure
follows an isothermal (p = cs^2 rho) or barotropic (Boss-Bodenheimer style
T(rho)) equation of state. Applied after the hydro step by the runners.
"""

import jax
import jax.numpy as jnp


@jax.jit
def _isothermal_project_jit(U, cs2_iso, gm1):
    """Reset total energy so p = cs2_iso * rho exactly (any dimensionality).

    E_new = p / (gamma - 1) + 0.5 |mom|^2 / rho with p = cs2_iso * rho.
    Momentum is U[1:-1], energy is U[-1] — so this is correct in 1D/2D/3D
    (the previous hard-coded U[1],U[2],U[3]/U[4] was a silent no-op in 2D, where
    U[3] is energy and .at[4] is an out-of-bounds update JAX drops). rho and
    momentum are left untouched.
    """
    rho = U[0]
    mom = U[1:-1]
    p_iso = cs2_iso * rho
    KE = 0.5 * jnp.sum(mom * mom, axis=0) / rho
    E_new = p_iso / gm1 + KE
    return U.at[-1].set(E_new)


def isothermal_project(U, cs2_iso, gamma):
    return _isothermal_project_jit(U, jnp.float32(cs2_iso), jnp.float32(gamma - 1.0))


@jax.jit
def _barotropic_project_jit(U, unit_density, unit_pressure, T0, rho_AD, mu, gm1):
    """Reset total energy so p follows the barotropic EOS (cgs):
        T(rho) = T0 [1 + (rho/rho_AD)^gm1],  p = k_B T rho / (mu m_H).
    Leaves rho and momentum untouched. dtype follows U (run in fp64).
    Momentum is U[1:-1], energy U[-1] — dimension-generic, same fix as
    _isothermal_project_jit (hard-coded U[1..3]/.at[4] was a silent no-op
    in 2D)."""
    K_B = 1.380649e-16
    M_H = 1.6726219e-24
    rho = U[0]
    mom = U[1:-1]
    rho_cgs = rho * unit_density
    T = T0 * (1.0 + (rho_cgs / rho_AD) ** gm1)
    p = (K_B * T * rho_cgs / (mu * M_H)) / unit_pressure
    KE = 0.5 * jnp.sum(mom * mom, axis=0) / rho
    E_new = p / gm1 + KE
    return U.at[-1].set(E_new)


def barotropic_project(U, cfg):
    u = cfg.units
    return _barotropic_project_jit(
        U, u.UnitDensity_in_cgs, u.UnitPressure_in_cgs,
        float(cfg.barotropic_T0_K), float(cfg.barotropic_rho_AD_cgs),
        float(cfg.barotropic_mu), cfg.gamma - 1.0)
