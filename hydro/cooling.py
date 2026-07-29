"""ISM-style radiative cooling + heating (Schneider+2018 Cholla/CGOLS style):
Koyama-Inutsuka (T ~ 30 K - 1e4 K, constant photoelectric heating) and
Sutherland-Dopita CIE (> ~1e4 K) tables, optionally combined. Operator-split
after the hydro update, n_sub forward-Euler sub-steps with a temperature
floor; the smooth log-log Lambda(T) interpolant keeps the step differentiable.
"""

from dataclasses import dataclass

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

    log_T: np.ndarray  # (N,)
    log_Lambda: np.ndarray  # (N,)
    heating: float = 0.0  # erg / s per H atom

    def __post_init__(self):
        if self.log_T.shape != self.log_Lambda.shape:
            raise ValueError("log_T and log_Lambda must have the same shape")
        if np.any(np.diff(self.log_T) <= 0):
            raise ValueError("log_T must be strictly increasing")

    @property
    def T(self):
        return 10.0**self.log_T

    @property
    def Lambda(self):
        return 10.0**self.log_Lambda

    @property
    def alpha(self) -> np.ndarray:
        return np.diff(self.log_Lambda) / np.diff(self.log_T)


def koyama_inutsuka_2002(
    T_min: float = 30.0, T_max: float = 1.0e8, n_pts: int = 256
) -> CoolingTable:
    """Build a tabulated CoolingTable from the KI 2002 analytic formula.

    Default T_min = 30 K: below this Lambda is effectively zero (heating
    dominates) and log10(Lambda) is ill-defined. Set T_min lower at your
    own risk.
    """
    log_T = np.linspace(np.log10(T_min), np.log10(T_max), n_pts)
    T = 10.0**log_T
    gamma_ki = 2.0e-26
    Lambda = gamma_ki * (
        1.0e7 * np.exp(-1.184e5 / (T + 1000.0)) + 14.0 * np.sqrt(T) * np.exp(-92.0 / T)
    )
    # Floor at 0.01 of the local KI cooling at T_min so log10 is well-defined
    # and the per-bin integral never blows up.
    Lambda_floor = 0.01 * gamma_ki * 14.0 * np.sqrt(T_min) * np.exp(-92.0 / T_min)
    log_Lambda = np.log10(np.maximum(Lambda, Lambda_floor))
    return CoolingTable(log_T=log_T, log_Lambda=log_Lambda, heating=gamma_ki)


# Sutherland & Dopita 1993 CIE at Z=Z_sun, every 0.25 dex from log T = 4.0 to 8.5.
_SD93_SOLAR_LOG_T = np.array(
    [
        4.00,
        4.25,
        4.50,
        4.75,
        5.00,
        5.25,
        5.50,
        5.75,
        6.00,
        6.25,
        6.50,
        6.75,
        7.00,
        7.25,
        7.50,
        7.75,
        8.00,
        8.25,
        8.50,
    ]
)
_SD93_SOLAR_LOG_L = np.array(
    [
        -25.20,
        -21.85,
        -21.30,
        -21.40,
        -21.50,
        -21.65,
        -21.80,
        -21.95,
        -22.10,
        -22.25,
        -22.45,
        -22.65,
        -22.85,
        -23.00,
        -23.10,
        -23.15,
        -23.10,
        -23.05,
        -23.00,
    ]
)


def sutherland_dopita_cie(metallicity: str = "solar") -> CoolingTable:
    """Sutherland & Dopita (1993) CIE cooling table (solar metallicity only)."""
    if metallicity != "solar":
        raise NotImplementedError(
            "Only solar metallicity is embedded; load your own table for other Z"
        )
    return CoolingTable(
        log_T=_SD93_SOLAR_LOG_T.copy(),
        log_Lambda=_SD93_SOLAR_LOG_L.copy(),
        heating=0.0,
    )


def combine(
    ki: CoolingTable, sd: CoolingTable, T_split: float = 1.5e4, n_pts: int = 512
) -> CoolingTable:
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
# Explicit sub-cycled cooling solver
# ---------------------------------------------------------------------------
@dataclass
class ExplicitCoolingSolver:
    """Forward-Euler sub-cycled radiative cooling (Schneider+2018 style) with
    a temperature floor; the smooth log-log Lambda(T) interpolant keeps it
    fully differentiable."""

    table: CoolingTable
    mu: float = 0.6
    n_sub: int = 16
    T_floor: float = 1.0e4

    def jax_pack(self):
        return {
            "logT": jnp.asarray(self.table.log_T, dtype=jnp.float64),
            "logL": jnp.asarray(self.table.log_Lambda, dtype=jnp.float64),
            "heating": jnp.float64(self.table.heating),
            "mu": jnp.float64(self.mu),
            "n_sub": int(self.n_sub),
            "T_floor": jnp.float64(self.T_floor),
        }


# ---------------------------------------------------------------------------
# High-level interface
# ---------------------------------------------------------------------------
def _uniform_interp(x, xs, ys):
    """Linear interp on a UNIFORM grid ``xs`` by index arithmetic — NO
    searchsorted (ROCm's searchsorted kernel faults inside the cooling scan).
    Clamped ends reproduce jnp.interp's flat extrapolation."""
    n = xs.shape[0]
    dx = (xs[-1] - xs[0]) / (n - 1)
    f = jnp.clip((x - xs[0]) / dx, 0.0, n - 1.0)
    i0 = jnp.clip(jnp.floor(f).astype(jnp.int32), 0, n - 2)
    w = f - i0
    return ys[i0] * (1.0 - w) + ys[i0 + 1] * w


@jax.custom_vjp
def _cooling_lambda(T, logT, logL):
    """Lambda(T) = 10**interp(log10 T; logT, logL) on the uniform log-T table.
    Custom VJP: the backward uses a central difference of the FORWARD interp
    (gradient-equivalent; the table is constant -> zero cotangent)."""
    return 10.0 ** _uniform_interp(jnp.log10(T), logT, logL)


def _cooling_lambda_fwd(T, logT, logL):
    return _cooling_lambda(T, logT, logL), (T, logT, logL)


def _cooling_lambda_bwd(res, g):
    T, logT, logL = res
    h = jnp.maximum(1.0e-4 * T, 1.0e-30)
    lp = 10.0 ** _uniform_interp(jnp.log10(T + h), logT, logL)
    lm = 10.0 ** _uniform_interp(jnp.log10(T - h), logT, logL)
    dT = g * (lp - lm) / (2.0 * h)
    return dT, jnp.zeros_like(logT), jnp.zeros_like(logL)


_cooling_lambda.defvjp(_cooling_lambda_fwd, _cooling_lambda_bwd)


def apply_cooling(U, dt_code, cfg, solver: ExplicitCoolingSolver):
    """Apply explicit sub-cycled cooling (+ heating) for one hydro step: only
    total energy E changes, via du/dt = -n_e n_H Lambda(T) + n_H Gamma with
    n_sub forward-Euler sub-steps and a temperature floor."""
    if not cfg.units.is_enabled:
        raise ValueError("apply_cooling requires cfg.units enabled (cgs scales)")

    rho = U[0]
    kin = 0.5 * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / rho
    u_int = U[4] - kin
    gamma = cfg.gamma

    p = solver.jax_pack()
    logT, logL = p["logT"], p["logL"]
    Gamma, mu, n_sub, T_floor = p["heating"], p["mu"], p["n_sub"], p["T_floor"]

    # L = n_e n_H Lambda(T): two-body rate matching the Sutherland-Dopita
    # Lambda_N (per n_e*n_H) normalization. n_tot=rho/(mu*m_p) sets u<->T.
    X_H, NE_PER_NH = 0.7154, 1.189
    rho_cgs = rho * cfg.units.UnitDensity_in_cgs
    u_int_cgs = u_int * cfg.units.UnitPressure_in_cgs
    n_tot = rho_cgs / (mu * PROTON_MASS_CGS)
    n_H = X_H * rho_cgs / PROTON_MASS_CGS
    n_e = NE_PER_NH * n_H
    T0 = (gamma - 1.0) * mu * PROTON_MASS_CGS * jnp.clip(u_int_cgs, 1e-100) \
        / (BOLTZMANN_CGS * rho_cgs)
    dt_sub = (dt_code * cfg.units.UnitTime_in_s) / n_sub

    def Lam(T):
        return _cooling_lambda(T, logT, logL)

    def sub(T, _):
        # dT/dt = (gamma-1)/(n_tot k_B) * (n_H Gamma - n_e n_H Lambda(T))
        dTdt = (gamma - 1.0) / (n_tot * BOLTZMANN_CGS) * (n_H * Gamma - n_e * n_H * Lam(T))
        return jnp.maximum(T + dTdt * dt_sub, T_floor), None

    T_new, _ = jax.lax.scan(sub, T0, None, length=n_sub)
    u_int_cgs_new = BOLTZMANN_CGS * rho_cgs * T_new / ((gamma - 1.0) * mu * PROTON_MASS_CGS)
    u_int_new = u_int_cgs_new / cfg.units.UnitPressure_in_cgs
    return U.at[4].set(u_int_new + kin)


# ----------------------------------------------------------------------
# Runner-side setup helper
# ----------------------------------------------------------------------
