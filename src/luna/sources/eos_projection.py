"""Operator-split EOS projections: reset total energy to an isothermal or barotropic pressure."""

import jax
import jax.numpy as jnp


@jax.jit
def _isothermal_project_jit(U, cs2_iso, gm1):
    """Reset total energy so p = cs2_iso * rho; rho and momentum untouched (any dimension)."""
    rho = U[0]
    mom = U[1:-1]
    p_iso = cs2_iso * rho
    KE = 0.5 * jnp.sum(mom * mom, axis=0) / rho
    E_new = p_iso / gm1 + KE
    return U.at[-1].set(E_new)


def isothermal_project(U, cs2_iso, gamma):
    return _isothermal_project_jit(U, jnp.asarray(cs2_iso, U.dtype), jnp.asarray(gamma - 1.0, U.dtype))


@jax.jit
def _barotropic_project_jit(U, unit_density, unit_pressure, T0, rho_AD, mu, gm1):
    """Reset total energy to the barotropic EOS T = T0 [1 + (rho/rho_AD)^gm1] (cgs); rho, momentum untouched."""
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


@jax.jit
def _pressure_floor_project_jit(U, jeans_coeff, p_floor, rho_floor, gm1):
    """Floor the stored state to rho >= rho_floor and p >= max(p_floor, jeans_coeff rho^2).

    jeans_coeff = G (N_J dx)^2 / pi keeps lambda_J >= N_J dx (Truelove+1997); momentum untouched.
    """
    rho = jnp.maximum(U[0], rho_floor)
    mom = U[1:-1]
    KE = 0.5 * jnp.sum(mom * mom, axis=0) / rho
    p = gm1 * (U[-1] - KE)
    p_new = jnp.maximum(p, jnp.maximum(p_floor, jeans_coeff * rho * rho))
    return U.at[0].set(rho).at[-1].set(p_new / gm1 + KE)


def pressure_floor_project(U, jeans_coeff, p_floor, rho_floor, gamma):
    return _pressure_floor_project_jit(
        U, jnp.float64(jeans_coeff), jnp.float64(p_floor),
        jnp.float64(rho_floor), jnp.float64(gamma - 1.0))
