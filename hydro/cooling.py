"""
ISM-style radiative cooling + heating with Townsend (2009) exact integration.

Provides two cooling functions out of the box, optionally combined:

  * Koyama & Inutsuka (2002, ApJ 564 L97): analytic two-phase fit valid for
    T ~ 30 K - 10^4 K with a constant photoelectric heating rate
        Gamma_KI = 2e-26 erg / s
        Lambda_KI(T) / Gamma_KI = 10^7 * exp(-118400 / (T + 1000))
                                + 14   * sqrt(T) * exp(-92 / T)
    so Lambda has units erg cm^3 / s.

  * Sutherland & Dopita (1993): collisional ionization equilibrium cooling
    above ~10^4 K. Embedded as a compact log T - log Lambda table at solar
    metallicity. For production runs swap in a higher-resolution table.

The Townsend (2009) scheme advances temperature exactly under piecewise
power-law Lambda(T) by integrating the cooling ODE in closed form.

For one zone at fixed density (rho constant during the cooling step):
    de_int/dt = -n_H^2 Lambda(T) + n_H Gamma                 (cgs)
    T         = (gamma - 1) mu m_p e_int / (k_B rho)
"""
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Physical constants (cgs)
# ---------------------------------------------------------------------------
PROTON_MASS_CGS = 1.6726219e-24
BOLTZMANN_CGS = 1.380649e-16


# ---------------------------------------------------------------------------
# Cooling table builder
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CoolingTable:
    """Tabulated Lambda(T) on a log-T grid.

    On bin [T_k, T_{k+1}] we treat Lambda as power-law
        Lambda(T) = Lambda_k (T / T_k)^alpha_k.
    """
    log_T: np.ndarray         # (N,)
    log_Lambda: np.ndarray    # (N,)
    heating: float = 0.0      # erg / s per H atom

    def __post_init__(self):
        if self.log_T.shape != self.log_Lambda.shape:
            raise ValueError("log_T and log_Lambda must have the same shape")
        if np.any(np.diff(self.log_T) <= 0):
            raise ValueError("log_T must be strictly increasing")

    @property
    def T(self):
        return 10.0 ** self.log_T

    @property
    def Lambda(self):
        return 10.0 ** self.log_Lambda

    @property
    def alpha(self) -> np.ndarray:
        return np.diff(self.log_Lambda) / np.diff(self.log_T)


def koyama_inutsuka_2002(T_min: float = 30.0,
                         T_max: float = 1.0e8,
                         n_pts: int = 256) -> CoolingTable:
    """Build a tabulated CoolingTable from the KI 2002 analytic formula.

    Default T_min = 30 K: below this Lambda is effectively zero (heating
    dominates) and the Townsend integrator's Y integral becomes ill-conditioned.
    Set T_min lower at your own risk.
    """
    log_T = np.linspace(np.log10(T_min), np.log10(T_max), n_pts)
    T = 10.0 ** log_T
    gamma_ki = 2.0e-26
    Lambda = gamma_ki * (
        1.0e7 * np.exp(-1.184e5 / (T + 1000.0))
        + 14.0 * np.sqrt(T) * np.exp(-92.0 / T)
    )
    # Floor at 0.01 of the local KI cooling at T_min so log10 is well-defined
    # and the per-bin integral never blows up.
    Lambda_floor = 0.01 * gamma_ki * 14.0 * np.sqrt(T_min) * np.exp(-92.0 / T_min)
    log_Lambda = np.log10(np.maximum(Lambda, Lambda_floor))
    return CoolingTable(log_T=log_T, log_Lambda=log_Lambda, heating=gamma_ki)


# Sutherland & Dopita 1993 CIE at Z=Z_sun, every 0.25 dex from log T = 4.0 to 8.5.
_SD93_SOLAR_LOG_T = np.array([
    4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75,
    6.00, 6.25, 6.50, 6.75, 7.00, 7.25, 7.50, 7.75,
    8.00, 8.25, 8.50,
])
_SD93_SOLAR_LOG_L = np.array([
    -25.20, -21.85, -21.30, -21.40, -21.50, -21.65, -21.80, -21.95,
    -22.10, -22.25, -22.45, -22.65, -22.85, -23.00, -23.10, -23.15,
    -23.10, -23.05, -23.00,
])


def sutherland_dopita_cie(metallicity: str = "solar") -> CoolingTable:
    if metallicity != "solar":
        raise NotImplementedError(
            "Only solar metallicity is embedded; load your own table for other Z"
        )
    return CoolingTable(
        log_T=_SD93_SOLAR_LOG_T.copy(),
        log_Lambda=_SD93_SOLAR_LOG_L.copy(),
        heating=0.0,
    )


def combine(ki: CoolingTable, sd: CoolingTable,
            T_split: float = 1.5e4, n_pts: int = 512) -> CoolingTable:
    """Merge KI (T < T_split) with SD-CIE (T > T_split) on a common log-T grid."""
    T_min = min(ki.log_T[0], sd.log_T[0])
    T_max = max(ki.log_T[-1], sd.log_T[-1])
    log_T = np.linspace(T_min, T_max, n_pts)
    log_L_ki = np.interp(log_T, ki.log_T, ki.log_Lambda)
    log_L_sd = np.interp(log_T, sd.log_T, sd.log_Lambda)
    log_T_split = np.log10(T_split)
    log_Lambda = np.where(log_T < log_T_split, log_L_ki, log_L_sd)
    return CoolingTable(log_T=log_T, log_Lambda=log_Lambda, heating=ki.heating)


# ---------------------------------------------------------------------------
# Townsend (2009) helpers
#
# Convention: Y is dimensionless and *strictly decreasing in T*.
#     Y(T) = (Lambda_ref / T_ref) * integral_T^{T_ref} dT' / Lambda(T')
# So Y(T_ref) = 0, Y(T_min) = Y_max > 0.
#
# Under a piecewise power-law Lambda(T) = L_k (T/T_k)^a_k on [T_k, T_{k+1}]:
#     integral_T^{T_{k+1}} dT' / Lambda(T') = (T_k / L_k) * g(a_k, T/T_k, T_{k+1}/T_k)
# where
#     g(a, x_lo, x_hi) = (x_hi^{1-a} - x_lo^{1-a}) / (1 - a)    if a != 1
#                     = log(x_hi / x_lo)                          if a == 1.
#
# Then Y(T) = Y(T_{k+1}) + (Lambda_ref/T_ref) * (T_k/L_k) * g(a_k, T/T_k, T_{k+1}/T_k).
# ---------------------------------------------------------------------------
def _townsend_Y_grid(table: CoolingTable, T_ref: Optional[float] = None) -> np.ndarray:
    T = table.T
    L = table.Lambda
    alpha = table.alpha
    N = len(T)
    if T_ref is None:
        T_ref = T[-1]
    Lambda_ref = L[-1]

    Y = np.zeros(N, dtype=np.float64)
    # Walk from top down; Y at the highest-T node is zero.
    for k in range(N - 2, -1, -1):
        a = alpha[k]
        x_hi = T[k + 1] / T[k]
        x_lo = 1.0
        if abs(a - 1.0) < 1.0e-12:
            seg = (T[k] / L[k]) * np.log(x_hi)
        else:
            seg = (T[k] / L[k]) * ((x_hi ** (1.0 - a) - x_lo ** (1.0 - a)) / (1.0 - a))
        Y[k] = Y[k + 1] + (Lambda_ref / T_ref) * seg
    return Y


def _g_segment(a, x_lo, x_hi):
    """int_{x_lo}^{x_hi} y^{-a} dy. Branch on a==1.

    Note: x_lo and x_hi are positive scalars (or jnp arrays). For a!=1 we use
    the closed form (x_hi^{1-a} - x_lo^{1-a}) / (1-a). For a==1 we use the log.
    """
    one_minus_a = 1.0 - a
    # Use safe value of a to avoid 0/0 in the unused branch
    safe = jnp.where(jnp.abs(one_minus_a) < 1.0e-10, 1.0e-3, one_minus_a)
    pwr = (x_hi ** safe - x_lo ** safe) / safe
    lin = jnp.log(x_hi / x_lo)
    return jnp.where(jnp.abs(one_minus_a) < 1.0e-10, lin, pwr)


def _Y_of_T(T, T_grid, L_grid, alpha_grid, Y_grid, T_ref, Lambda_ref):
    """Y(T). Clamps T to the table range."""
    T_clip = jnp.clip(T, T_grid[0], T_grid[-1])
    k = jnp.clip(jnp.searchsorted(T_grid, T_clip) - 1, 0, T_grid.shape[0] - 2)
    T_k = T_grid[k]
    L_k = L_grid[k]
    a = alpha_grid[k]
    g = _g_segment(a, T_clip / T_k, T_grid[k + 1] / T_k)
    seg = (T_k / L_k) * g
    return Y_grid[k + 1] + (Lambda_ref / T_ref) * seg


def _T_of_Y(y, T_grid, L_grid, alpha_grid, Y_grid, T_ref, Lambda_ref):
    """Invert Y -> T. Clamps to table.

    Y_grid is descending in index; we locate the bin via -Y_grid (ascending).
    """
    y_clip = jnp.clip(y, Y_grid[-1], Y_grid[0])
    neg_Y = -Y_grid
    k = jnp.clip(jnp.searchsorted(neg_Y, -y_clip) - 1, 0, T_grid.shape[0] - 2)
    T_k = T_grid[k]
    L_k = L_grid[k]
    a = alpha_grid[k]

    # delta = (y - Y_grid[k+1]) * (T_ref / Lambda_ref) = (T_k/L_k) * g(a, T/T_k, T_{k+1}/T_k)
    # Solve g(a, x, x_hi) = delta * L_k / T_k for x = T/T_k.
    delta = (y_clip - Y_grid[k + 1]) * (T_ref / Lambda_ref)
    rhs = delta * (L_k / T_k)
    x_hi = T_grid[k + 1] / T_k

    one_minus_a = 1.0 - a
    safe = jnp.where(jnp.abs(one_minus_a) < 1.0e-10, 1.0e-3, one_minus_a)
    # Power-law branch: x^{1-a} = x_hi^{1-a} - (1-a) * rhs
    x_pwr_arg = jnp.maximum(x_hi ** safe - safe * rhs, 1.0e-30)
    x_pwr = x_pwr_arg ** (1.0 / safe)
    # Log branch (a == 1): log(x_hi / x) = rhs -> x = x_hi / exp(rhs)
    x_lin = x_hi / jnp.exp(rhs)
    x = jnp.where(jnp.abs(one_minus_a) < 1.0e-10, x_lin, x_pwr)
    T_new = T_k * x
    return jnp.clip(T_new, T_grid[0], T_grid[-1])


# ---------------------------------------------------------------------------
# Townsend cooling solver
# ---------------------------------------------------------------------------
@dataclass
class TownsendCoolingSolver:
    """Townsend (2009) cooling + explicit forward-Euler heating, per cell."""

    table: CoolingTable
    mu: float = 1.27

    def jax_pack(self):
        T = self.table.T.astype(np.float64)
        L = self.table.Lambda.astype(np.float64)
        a = self.table.alpha.astype(np.float64)
        Y = _townsend_Y_grid(self.table).astype(np.float64)
        return {
            "T_grid": jnp.asarray(T),
            "L_grid": jnp.asarray(L),
            "alpha_grid": jnp.asarray(a),
            "Y_grid": jnp.asarray(Y),
            "T_ref": jnp.float64(T[-1]),
            "Lambda_ref": jnp.float64(L[-1]),
            "heating": jnp.float64(self.table.heating),
            "mu": jnp.float64(self.mu),
        }

    @staticmethod
    def step(T_cgs, dt_eff, params):
        """Advance T_cgs over an effective dt under pure Townsend cooling.

        dt_eff must already absorb the (gamma-1) mu m_p n_H Lambda_ref / (k_B T_ref)
        prefactor so that Y_new = Y_old + dt_eff is the correct exact update.
        """
        y = _Y_of_T(T_cgs,
                    params["T_grid"], params["L_grid"], params["alpha_grid"],
                    params["Y_grid"], params["T_ref"], params["Lambda_ref"])
        return _T_of_Y(y + dt_eff,
                       params["T_grid"], params["L_grid"], params["alpha_grid"],
                       params["Y_grid"], params["T_ref"], params["Lambda_ref"])


# ---------------------------------------------------------------------------
# High-level interface
# ---------------------------------------------------------------------------
def apply_cooling(U, dt_code, cfg, solver: TownsendCoolingSolver):
    """Apply Townsend cooling + explicit heating to U for one hydro step.

    rho, momentum, kinetic energy unchanged. Only total energy E (cell 4) is
    updated, via a Strang split: half-step heating, full-step cooling,
    half-step heating.
    """
    if not cfg.units.is_enabled:
        raise ValueError("apply_cooling requires cfg.units enabled (cgs scales)")

    rho = U[0]
    mx, my, mz = U[1], U[2], U[3]
    E = U[4]
    gamma = cfg.gamma
    mu = solver.mu

    kin = 0.5 * (mx * mx + my * my + mz * mz) / rho
    u_int = E - kin  # internal energy per unit volume (code units)

    rho_cgs = rho * cfg.units.UnitDensity_in_cgs
    u_int_cgs = u_int * cfg.units.UnitPressure_in_cgs
    T_cgs = (gamma - 1.0) * mu * PROTON_MASS_CGS * u_int_cgs / (BOLTZMANN_CGS * rho_cgs)
    n_H_cgs = rho_cgs / (mu * PROTON_MASS_CGS)
    dt_cgs = dt_code * cfg.units.UnitTime_in_s

    params = solver.jax_pack()
    T_ref = params["T_ref"]
    Lambda_ref = params["Lambda_ref"]
    Gamma = params["heating"]

    # Effective Townsend dt. The mu*m_p factor that converts u_int<->T
    # cancels with the n_H = rho/(mu*m_p) factor in the cooling rate, leaving
    #   dT/dt = -(gamma-1) n_H Lambda(T) / k_B,
    # so dY/dt = (gamma-1) n_H Lambda_ref / (k_B T_ref).
    dt_eff = (dt_cgs * (gamma - 1.0) * n_H_cgs * Lambda_ref
              / (BOLTZMANN_CGS * T_ref))

    # Strang-split heating: half before, full cooling, half after.
    # dT/dt|_heat = (gamma-1) Gamma / (k_B mu)    (independent of n_H since
    # rho = n_H mu m_p and Gamma is per-H heating).
    dT_heat_half = 0.5 * (gamma - 1.0) * Gamma * dt_cgs / (BOLTZMANN_CGS * mu)
    T_after_h1 = T_cgs + dT_heat_half
    T_after_c = TownsendCoolingSolver.step(T_after_h1, dt_eff, params)
    T_new = T_after_c + dT_heat_half

    u_int_cgs_new = BOLTZMANN_CGS * rho_cgs * T_new / ((gamma - 1.0) * mu * PROTON_MASS_CGS)
    u_int_new = u_int_cgs_new / cfg.units.UnitPressure_in_cgs
    E_new = u_int_new + kin
    return U.at[4].set(E_new)
