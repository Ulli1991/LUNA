"""Barotropic equation of state for star-formation collapse (Boss-Bodenheimer).

Following GANDALF (Hubber, Rosotti & Booth 2018, MNRAS 473, 1603, sec. 6.12)
and the Boss & Bodenheimer (1979) test, the temperature is a fixed function of
density:

    T(ρ) = T0 [ 1 + (ρ / ρ_AD)^(γ-1) ],   P(ρ) = k_B T(ρ) ρ / (μ m_H)

so the gas is isothermal at T0 for ρ ≪ ρ_AD (the optically-thin regime) and
adiabatically stiffens above ρ_AD (the optically-thick adiabatic core). The
stiffening is what halts the runaway isothermal collapse and lets the m=2
perturbation fragment into a binary instead of producing a grid-aligned
central spike.

Requires cfg.units (cgs scales). The projection is operator-split after the
hydro step exactly like the isothermal thermostat: it resets the total energy
so the pressure follows the barotropic relation, leaving ρ and momentum
untouched.
"""
import jax.numpy as jnp

K_B = 1.380649e-16     # erg / K
M_H = 1.6726219e-24    # g (hydrogen mass)


def barotropic_pressure_code(rho_code, cfg):
    """Barotropic pressure in CODE units for a code-unit density field."""
    u = cfg.units
    gm1 = cfg.gamma - 1.0
    rho_cgs = rho_code * u.UnitDensity_in_cgs
    T = cfg.barotropic_T0_K * (1.0 + (rho_cgs / cfg.barotropic_rho_AD_cgs) ** gm1)
    P_cgs = K_B * T * rho_cgs / (cfg.barotropic_mu * M_H)
    return P_cgs / u.UnitPressure_in_cgs


def isothermal_cs2_code(cfg):
    """Code-unit c_s² of the low-density (T0) isothermal floor — handy for CFL
    sanity and for an isothermal cross-check run."""
    cs2_cgs = K_B * cfg.barotropic_T0_K / (cfg.barotropic_mu * M_H)
    return cs2_cgs / cfg.units.UnitVelocity_in_cm_per_s ** 2
