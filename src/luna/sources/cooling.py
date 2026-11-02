"""ISM-style radiative cooling + heating (Schneider+2018 Cholla/CGOLS style):
Koyama-Inutsuka (T ~ 30 K - 1e4 K, constant photoelectric heating) and
Sutherland-Dopita CIE (> ~1e4 K) tables, optionally combined. Operator-split
after the hydro update, n_sub forward-Euler sub-steps with a temperature
floor; the smooth log-log Lambda(T) interpolant keeps the step differentiable.
`setup_cooling(cfg)` builds the solver a run uses (luna/setup/assemble.py).
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (cgs)
# ---------------------------------------------------------------------------
from luna.units import BOLTZMANN_CGS, PROTON_MASS_CGS


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

    The low-T coefficient is 1.4e-2 (the printed 14 is a typo): with it the published two-phase
    states are thermal equilibria (tests/test_cooling_tables.py).

    Default T_min = 30 K: below this Lambda is effectively zero (heating
    dominates) and log10(Lambda) is ill-defined. Set T_min lower at your
    own risk.
    """
    log_T = np.linspace(np.log10(T_min), np.log10(T_max), n_pts)
    T = 10.0**log_T
    gamma_ki = 2.0e-26
    Lambda = gamma_ki * (
        1.0e7 * np.exp(-1.184e5 / (T + 1000.0)) + 1.4e-2 * np.sqrt(T) * np.exp(-92.0 / T)
    )
    # Floor at 0.01 of the local KI cooling at T_min so log10 is well-defined
    # and the per-bin integral never blows up.
    Lambda_floor = 0.01 * gamma_ki * 1.4e-2 * np.sqrt(T_min) * np.exp(-92.0 / T_min)
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
    n_sub: int = 16              # CAP on the per-cell adaptive sub-steps per hydro step
    dt_frac: float = 0.1         # sub-step <= dt_frac * t_cool(T), t_cool = T / |dT/dt|
    T_floor: float = 1.0e4
    X_H: float = 0.7154          # hydrogen mass fraction -> n_H
    ne_per_nH: float = 1.189     # free electrons per H nucleus -> n_e (full ionisation)

    def jax_pack(self):
        return {
            "logT": jnp.asarray(self.table.log_T, dtype=jnp.float64),
            "logL": jnp.asarray(self.table.log_Lambda, dtype=jnp.float64),
            "heating": jnp.float64(self.table.heating),
            "mu": jnp.float64(self.mu),
            "n_sub": int(self.n_sub),
            "dt_frac": jnp.float64(self.dt_frac),
            "T_floor": jnp.float64(self.T_floor),
            "X_H": jnp.float64(self.X_H),
            "ne_per_nH": jnp.float64(self.ne_per_nH),
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


def _thermo(U, dt_code, cfg, solver):
    """Per-cell temperature T0 [K], the rate function dT/dt(T) [K/s], the
    step length [s] and the packed parameters shared by apply_cooling and
    cooling_substeps_needed."""
    if not cfg.units.is_enabled:
        raise ValueError("apply_cooling requires cfg.units enabled (cgs scales)")
    rho = U[0]
    kin = 0.5 * sum(U[d] ** 2 for d in range(1, U.shape[0] - 1)) / rho   # rows rho, m_1..m_ndim, E
    u_int = U[-1] - kin
    gamma = cfg.gamma
    p = solver.jax_pack()
    logT, logL = p["logT"], p["logL"]
    Gamma, mu = p["heating"], p["mu"]
    # L = n_e n_H Lambda(T): two-body rate matching the Sutherland-Dopita
    # Lambda_N (per n_e*n_H) normalization. n_tot=rho/(mu*m_p) sets u<->T.
    X_H, NE_PER_NH = p["X_H"], p["ne_per_nH"]     # Config: cooling_X_H, cooling_ne_per_nH
    rho_cgs = rho * cfg.units.UnitDensity_in_cgs
    u_int_cgs = u_int * cfg.units.UnitPressure_in_cgs
    n_tot = rho_cgs / (mu * PROTON_MASS_CGS)
    n_H = X_H * rho_cgs / PROTON_MASS_CGS
    n_e = NE_PER_NH * n_H
    T0 = (gamma - 1.0) * mu * PROTON_MASS_CGS * jnp.clip(u_int_cgs, 1e-100) \
        / (BOLTZMANN_CGS * rho_cgs)
    coef = (gamma - 1.0) / (n_tot * BOLTZMANN_CGS)

    def dTdt(T):
        # dT/dt = (gamma-1)/(n_tot k_B) * (n_H Gamma - n_e n_H Lambda(T))
        return coef * (n_H * Gamma - n_e * n_H * _cooling_lambda(T, logT, logL))

    dt_tot = dt_code * cfg.units.UnitTime_in_s
    return T0, dTdt, dt_tot, p, rho_cgs, kin


def apply_cooling(U, dt_code, cfg, solver: ExplicitCoolingSolver):
    """Apply explicit cooling (+ heating) for one hydro step: only the total
    energy E changes, via du/dt = -n_e n_H Lambda(T) + n_H Gamma.

    Cooling-time control: every cell advances by forward-Euler sub-steps of
    length min(remaining, dt_frac * t_cool(T)), t_cool = T / |dT/dt|, inside
    one static scan of n_sub iterations (a cell that finishes early takes
    zero-length steps). A cell that needs more than n_sub sub-steps gets the
    remainder as a single floored forward-Euler step -- the old behaviour --
    and cooling_substeps_needed() lets the runner report it. The previous
    scheme took n_sub EQUAL sub-steps whatever t_cool was (audit sec.3)."""
    T0, dTdt, dt_tot, p, rho_cgs, kin = _thermo(U, dt_code, cfg, solver)
    n_sub, frac, T_floor = p["n_sub"], p["dt_frac"], p["T_floor"]
    gamma, mu = cfg.gamma, p["mu"]

    def sub(carry, _):
        T, rem = carry
        r = dTdt(T)
        t_cool = T / jnp.maximum(jnp.abs(r), 1e-300)
        h = jnp.minimum(rem, frac * t_cool)
        T = jnp.maximum(T + r * h, T_floor)
        return (T, rem - h), None

    # Differentiated, the sub-steps are unrolled (n_sub is a small static
    # cap): as a device loop nested in the rollout's, the gradient segfaulted
    # on ROCm (128^3 x 20 steps, third call) unless launches were serialized,
    # which hangs across nodes. Unrolled it runs async, 30% faster than the
    # serialized scan (256^3, 1 GPU); the forward alone is 17% faster as a
    # scan, so plain runs keep the scan (docs/AUTODIFF.md, "Across nodes").
    unroll = bool(getattr(cfg, "autodiff_mode", False)) and n_sub <= 32
    (T_new, rem), _ = jax.lax.scan(sub, (T0, jnp.full_like(T0, dt_tot)), None,
                                   length=n_sub, unroll=unroll)
    T_new = jnp.maximum(T_new + dTdt(T_new) * rem, T_floor)   # cap hit: remainder, floored
    u_int_cgs_new = BOLTZMANN_CGS * rho_cgs * T_new / ((gamma - 1.0) * mu * PROTON_MASS_CGS)
    u_int_new = u_int_cgs_new / cfg.units.UnitPressure_in_cgs
    return U.at[-1].set(u_int_new + kin)


def cooling_substeps_needed(U, dt_code, cfg, solver: ExplicitCoolingSolver, mesh=None):
    """Max over cells of ceil(dt / (dt_frac * t_cool(T))) at the current
    state: above solver.n_sub, apply_cooling's cap is hit somewhere. With
    ``mesh`` (a sharded U) the per-cell part runs as a shard_map program and
    the max is a mesh reduction; returns a scalar."""
    def _n(U_):
        T0, dTdt, dt_tot, p, _, _ = _thermo(U_, dt_code, cfg, solver)
        r = dTdt(T0)
        t_cool = T0 / jnp.maximum(jnp.abs(r), 1e-300)
        return jnp.ceil(dt_tot / (p["dt_frac"] * t_cool))

    if mesh is None:
        return jnp.max(jax.jit(_n)(U))
    from jax.sharding import PartitionSpec as P
    from luna.utils.mesh_reduce import grid_reduce
    from luna.utils.shmap import shard_map
    axes = ("x", "y", "z")[:cfg.SPATIAL_DIMENSION]
    n_cell = jax.jit(shard_map(_n, mesh=mesh, in_specs=(P(None, *axes),),
                               out_specs=P(*axes)))(U)
    return grid_reduce(mesh, "max", n_cell)


# ---------------------------------------------------------------------------
# The solver a run uses, from cfg
# ---------------------------------------------------------------------------
def setup_cooling(cfg, is_main_process: bool = True):
    """Return the ExplicitCoolingSolver or None when disabled. Tables:
    "sd93_solar" (SD CIE, T>1e4 K), "koyama_inutsuka" (two-phase ISM),
    "combined" (KI below 1.5e4 K, SD above)."""
    if not cfg.cooling_enabled:
        return None
    if not cfg.units.is_enabled:
        raise ValueError("cooling_enabled requires cfg.units (Arepo system).")
    ct = cfg.cooling_table
    if ct == "sd93_solar":
        table = sutherland_dopita_cie(metallicity="solar")
    elif ct == "koyama_inutsuka":
        table = koyama_inutsuka_2002()
    elif ct == "combined":
        table = combine(koyama_inutsuka_2002(),
                        sutherland_dopita_cie(metallicity="solar"))
    else:
        raise ValueError(f"Unknown cooling_table: {ct}")
    mu = cfg.cooling_mu
    n_sub = cfg.cooling_n_sub
    solver = ExplicitCoolingSolver(
        table=table, mu=mu, n_sub=n_sub,
        dt_frac=float(cfg.cooling_dt_frac),
        T_floor=float(cfg.cooling_T_floor),
        X_H=float(cfg.cooling_X_H),
        ne_per_nH=float(cfg.cooling_ne_per_nH))
    if is_main_process:
        print(
            f"✓ Cooling enabled: table={cfg.cooling_table}, "
            f"T range [{table.T.min():.2e}, {table.T.max():.2e}] K, mu={mu}"
        )
    return solver
