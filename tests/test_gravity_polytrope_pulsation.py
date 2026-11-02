"""Pulsating n = 1 polytrope under isolated self-gravity against the linear adiabatic radial pulsation eigenfrequency.

Star: rho_c sinc(r / R), P = K rho^2 (R = 0.35 in a unit box, G = rho_c = 1), responding with Gamma_1 = gamma = 5/3,
plus P0 = 1e-4 K rho_c^2 inside (hydrostatics unchanged, P continuous at R), in an isothermal atmosphere at the central
temperature (1e-4 rho_c at R), hydrostatic in -G M / r. A uniform hot ambient as in test_hydro_hard_well_balanced.py
puts a pressure jump at R whose inward wave focuses at the centre near t = 1.5 (+9 % in rho_c at 64^3); a cold one
drains through the diode faces. Perturbation: the fundamental-mode eigenfunction of the LAWE (tests/_polytrope_lawe.py)
as a velocity, v_r = AMP / 3 omega zeta(r) r, zeta(0) = 1, so rho_c(t) = rho_c (1 - AMP sin omega t) in linear theory.
The registered Pulsating-Polytrope-3D IC is not used: its v ~ r is not the eigenmode and its ambient is cold.
Signal: central density (8 central cells) of the perturbed run minus that of the unperturbed run on the same grid,
which removes the scheme's own relaxation (the well-balance drift); fitted with A e^(-gamma t) sin(omega t + phi) + a.
Headline: the worst over three fit windows (1, 1.25, 1.5 periods, fixed beforehand) of |omega / omega_LAWE - 1| over N;
also the signed damping rate gamma / omega (measured negative: apparent growth), the unperturbed central density
drift and the energy budget E_kin + E_th + W (W = 1/2 sum rho phi dV, phi LUNA's isolated solve; gravity_energy auto
resolves to flux here). Run through the production loop (tests/_helpers.evolve_staged), hllc + minmod, FP64. The
reference (LAWE) is independent; the bounds are ~1.3x LUNA's own measured errors (regression of an honest metric).
"""
import functools

import numpy as np
import pytest
from scipy.optimize import least_squares

from tests._helpers import _where, accuracy_cfg, check, evolve_staged, gpu_only
from tests._polytrope_lawe import fundamental_zeta, modes, variational_bounds

R = 0.35
RHO_C, G, GAMMA = 1.0, 1.0, 5.0 / 3.0
AMB = 1e-4                                  # ambient density / rho_c and pressure / (K rho_c^2)
K = 2.0 * G * R * R / np.pi                 # n = 1: R = pi r_n, r_n^2 = K / (2 pi G)
MASS = 4.0 * RHO_C * R ** 3 / np.pi         # 4 pi rho_c r_n^3 xi_1^2 |theta'(xi_1)| = 4 pi^2 rho_c r_n^3
AMP = 0.01                                  # linear central-density amplitude delta rho_c / rho_c


def omega_lawe(gamma1=GAMMA):
    """Fundamental angular frequency of the n = 1 polytrope for Gamma_1, in code units."""
    return float(np.sqrt(modes(1, gamma1)[0] * G * MASS / R ** 3))


def radius(n):
    """Cell-centre distance from the box centre on an n^3 unit box."""
    x = (np.arange(n) + 0.5) / n - 0.5
    return np.sqrt(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2)


def star_U(n, amp=AMP, gamma=GAMMA):
    """Conserved U: the n = 1 polytrope in the hot ambient, with the LAWE fundamental mode's velocity of amplitude amp."""
    r = radius(n)
    inside = r < R
    # P0 inside keeps P continuous at R (a pressure jump there sends a wave that focuses at the centre);
    # outside an isothermal atmosphere at the central temperature, hydrostatic in -G M / r
    P0 = AMB * K * RHO_C ** 2
    atm = AMB * RHO_C * np.exp(G * MASS * RHO_C * AMB / P0 * (1.0 / np.maximum(r, R) - 1.0 / R))
    rho = np.where(inside, np.maximum(RHO_C * np.sinc(r / R), AMB * RHO_C), atm)
    P = np.where(inside, K * rho ** 2 + P0, atm * P0 / (AMB * RHO_C))
    xz, z = fundamental_zeta(1, GAMMA)
    vr = np.where(inside, amp / 3.0 * omega_lawe() * np.interp(r / R, xz, z) * r, 0.0)
    x = (np.arange(n) + 0.5) / n - 0.5
    rs = np.maximum(r, 1e-30)
    v = [vr * x[:, None, None] / rs, vr * x[None, :, None] / rs, vr * x[None, None, :] / rs]
    v = [np.broadcast_to(c, rho.shape) for c in v]
    E = P / (gamma - 1.0) + 0.5 * rho * vr * vr
    return np.stack([rho, rho * v[0], rho * v[1], rho * v[2], E])


def cfg_for(n, gamma=GAMMA, t_end=1.0, **kw):
    """Isolated self-gravity on the diode box, hllc-minmod by default; the registered IC is a placeholder for U0."""
    from luna.ic_generation import pulsating_polytrope3d
    return accuracy_cfg(test_problem="Pulsating-Polytrope-3D", SPATIAL_DIMENSION=3, Nx=n, Ny=n, Nz=n,
                        Lx=1.0, Ly=1.0, Lz=1.0, BOUNDARY=4, POISSON_BOUNDARY=3, SELF_GRAVITY=True,
                        GRAVITATIONAL_G=G, gamma=gamma, write_snapshots=False, t_end=t_end,
                        problem=pulsating_polytrope3d.Params(n=1.0, rho_c=RHO_C, R=R, rho_ambient=AMB, amp=0.0),
                        **kw)


@functools.lru_cache(maxsize=None)
def run(n, amp, periods, per_period=24, gamma=GAMMA, **kw):
    """(t, rho_c / rho_c0, E_kin + E_th + W) at per_period samples per LAWE period up to `periods` periods."""
    import jax
    from luna.gravity.particle_mesh import ParticleMeshGravity
    period = 2 * np.pi / omega_lawe()
    times = period * np.arange(1, int(round(periods * per_period)) + 1) / per_period
    cfg = cfg_for(n, gamma=gamma, t_end=float(times[-1]), **kw)
    U0 = star_U(n, amp, gamma)
    c = slice(n // 2 - 1, n // 2 + 1)
    phi = jax.jit(ParticleMeshGravity(cfg).gas_potential)
    dV = 1.0 / n ** 3

    def obs(U, t=0.0):
        rho = U[0]
        ek = 0.5 * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / rho
        W = 0.5 * float((rho * np.asarray(phi(rho))).sum()) * dV
        return float(rho[c, c, c].mean()), float(ek.sum() * dV), float((U[4] - ek).sum() * dV), W

    o0 = obs(U0)
    out = [o0] + evolve_staged(cfg, U0, times, conserve=False, observe=obs)
    t = np.concatenate([[0.0], times])
    a = np.array(out)
    print(f"RUN [{_where()}] pulsation N={n} amp={amp} gamma={gamma:.4f} {kw} t={t[-1]:.6f} "
          f"({t[-1] / period:.3f} P) reached=True")
    return t, a[:, 0] / o0[0], a[:, 1] + a[:, 2] + a[:, 3], o0[3]


def fit_mode(t, s, omega0):
    """(omega, gamma, A) of s = A e^(-gamma t) sin(omega t + phi) + a, least squares from omega0."""
    def res(p):
        A, om, ga, ph, a = p
        return A * np.exp(-ga * t) * np.sin(om * t + ph) + a - s
    p0 = [-AMP, omega0, 0.0, 0.0, 0.0]
    p = least_squares(res, p0).x
    if p[0] < 0:
        p[0] = -p[0]
    return float(p[1]), float(p[2]), float(p[0])


# Fit windows, fixed before any run: one full period (the least that resolves omega) to PERIODS (where the drift
# starts, below), in quarter periods. Every window is reported and the worst one is bounded, so no window is picked.
WINDOWS = (1.0, 1.25, 1.5)


@functools.lru_cache(maxsize=None)
def measure(n, periods, gamma=GAMMA, **kw):
    """Fitted mode in every window against LAWE, plus the unperturbed hold: period error, damping, drift, energy."""
    om0 = omega_lawe()
    t, rp, Ep, W0 = run(n, AMP, periods, gamma=gamma, **kw)
    _, r0, E0, _ = run(n, 0.0, periods, gamma=gamma, **kw)
    period = 2 * np.pi / om0
    fits = [fit_mode(t[t <= w * period * (1 + 1e-9)], (rp - r0)[t <= w * period * (1 + 1e-9)], om0)
            for w in WINDOWS if w <= periods]
    om = [f[0] for f in fits]
    damp = [f[1] / om0 for f in fits]
    d = dict(period_err=max(abs(o / om0 - 1.0) for o in om), damping_min=min(damp), damping_max=max(damp),
             drho_c=float(np.abs(r0 - 1.0).max()), dE_eq=float(np.abs(E0 - E0[0]).max() / abs(W0)),
             dE_pert=float(np.abs(Ep - Ep[0]).max() / abs(W0)), omega=om)
    print(f"MEASURED pulsation N={n} gamma={gamma:.4f} {kw} windows={WINDOWS} P: omega_LAWE={om0:.6f} "
          f"omega/omega_LAWE - 1 = {[f'{o / om0 - 1:+.4e}' for o in om]}, damping/omega = "
          f"{[f'{g:+.4e}' for g in damp]}, amp = {[f'{f[2] / AMP:.4f}' for f in fits]}; {d}", flush=True)
    return d


# -- the reference --------------------------------------------------------------------------------------------------

def test_lawe_reference():
    """LAWE solver: Pekeris n = 0 (exact), Cox 1980 Table 8.1 periods (n = 1.5, 3), n = 1 variational bounds."""
    for g in (5.0 / 3.0, 1.4):
        om = modes(0, g)
        # Pekeris 1938: Omega_k^2 = Gamma (k + 1)(2k + 3) - 4
        for k in range(3):
            check(f"lawe n=0 Gamma={g:.3f} k={k} |Omega^2 / Pekeris - 1|", abs(om[k] / (g * (k + 1) * (2 * k + 3) - 4) - 1),
                  1e-7)
    # Cox 1980 Table 8.1 via Goodman's Princeton A403 notes: Pi_0 = C (R^3 / G M)^(1/2), C = 3.815 (n = 1.5) and
    # 2.065 (n = 3); Pi_1 / Pi_0 = 0.465 and 0.738; 3-4 digits published
    for n, c, ratio in ((1.5, 3.815, 0.465), (3.0, 2.065, 0.738)):
        om = modes(n, 5.0 / 3.0)
        check(f"lawe n={n} |C / Cox - 1|", abs(2 * np.pi / np.sqrt(om[0]) / c - 1), 1.5e-3)
        check(f"lawe n={n} |Pi1/Pi0 - Cox|", abs(np.sqrt(om[0] / om[1]) - ratio), 1.5e-3)
    # n = 1, Gamma_1 = 5/3: no tabulated value found (Hurley, Roberts & Wright 1966 not read); Cox's bounds instead
    lo, hi = variational_bounds(1, GAMMA)
    om = modes(1, GAMMA)
    print(f"MEASURED lawe n=1 Gamma=5/3 Omega^2 = {om[0]:.8f}, {om[1]:.6f}, {om[2]:.6f}; bounds {lo:.6f} .. {hi:.6f}")
    assert lo < om[0] < hi
    check("lawe n=1 |Omega^2(4000) - Omega^2(2000)| / Omega^2", abs(modes(1, GAMMA, n_el=2000)[0] / om[0] - 1), 1e-8)


# -- a coarse, short run ----------------------------------------------------------------------------------------

# 32^3 to P/4 (8 samples): the deviation from linear theory is the coarse grid's own (14 %), a regression bound
# ~1.3x the A100 FP64 value 0.142 (job 31655473; CPU-8 gives the same); the unperturbed drift likewise, 0.0792
COARSE_BOUND, COARSE_DRIFT = 0.19, 0.10


@gpu_only
def test_pulsation_coarse():
    """Regression only: 32^3 over P/4, perturbed minus unperturbed rho_c against -AMP sin(omega t) (14 % off), drift."""
    t, rp, _, _ = run(32, AMP, 0.25, per_period=8)
    _, r0, _, _ = run(32, 0.0, 0.25, per_period=8)
    s = rp - r0
    lin = -AMP * np.sin(omega_lawe() * t)
    err = float(np.abs(s - lin).sum() / np.abs(lin).sum())
    print(f"MEASURED pulsation N=32 s(t)/AMP = {np.round(s / AMP, 4)}; linear {np.round(lin / AMP, 4)}")
    check("pulsation N=32 L1 |s - linear| / |linear| over P/4", err, COARSE_BOUND)
    check("pulsation N=32 max |rho_c / rho_c0 - 1| unperturbed", float(np.abs(r0 - 1).max()), COARSE_DRIFT)


# -- GPU: the ladder ------------------------------------------------------------------------------------------------

# 1.5 LAWE periods: after ~2 periods (t ~ 8, ~8 central sound crossings) the central density of both runs climbs
# (+15 % at 64^3, +30 % at 128^3 by t = 12; also with gravity_energy='kick' and with R = 0.25); the run stops before
PERIODS = 1.5
# Regression bounds ~1.3x measured, A100 FP64, job 31655473. period_err is the worst of WINDOWS (64: +4.87e-3,
# -2.35e-3, -2.54e-4; 128: +3.79e-3, +2.55e-4, +1.61e-3; 256: +1.21e-3, +4.88e-4, +6.65e-4): the old 1.5-period-only
# bound (3.3e-4 at 64) sat on the luckiest window. Damping gamma / omega is signed and two-sided, [1.3 min, max +
# 0.3 |max|]: measured 64: -3.27e-2 .. -6.52e-4; 128: -1.43e-2 .. +1.66e-3; 256: -7.02e-3 .. -3.49e-4. It is NEGATIVE,
# an apparent growth the linear eigenmode should not have (the old abs() hid the sign); it falls with N (about first
# order over the 1-period window) and is not resolved here: <= 1.5 periods condition the e-folding fit poorly.
# The worst-window period error falls with N (orders 0.36, 1.64), not cleanly enough to bound an order.
BOUNDS = {64: dict(period_err=6.3e-3, damping_min=-4.26e-2, damping_max=-4.6e-4, drho_c=0.043, dE_eq=7.5e-5,
                   dE_pert=8.0e-5),
          128: dict(period_err=4.9e-3, damping_min=-1.86e-2, damping_max=2.15e-3, drho_c=0.0156, dE_eq=4.9e-5,
                    dE_pert=5.1e-5),
          256: dict(period_err=1.6e-3, damping_min=-9.1e-3, damping_max=-2.4e-4, drho_c=5.4e-3, dE_eq=2.6e-5,
                    dE_pert=2.6e-5)}
DRIFT_ORDER = 1.35                   # fitted order of the unperturbed drift, 64-256: measured 1.50


def _check(n, d, bounds):
    """Every bounded quantity of a ladder point; damping_min is a lower bound."""
    for k, b in bounds.items():
        check(f"pulsation N={n} {k}", d[k], b, lower=(k == "damping_min"))


@gpu_only
@pytest.mark.parametrize("n", (64, 128))
def test_pulsation_ladder_gpu(n):
    """64^3, 128^3 to 1.5 periods: worst-window period error, signed damping, drift, energy (regression bounds)."""
    _check(n, measure(n, PERIODS), BOUNDS[n])


@gpu_only
def test_pulsation_256_gpu():
    """The same at 256^3 (each run ~3 min on an A100), and the unperturbed drift's fitted order over 64-256."""
    _check(256, measure(256, PERIODS), BOUNDS[256])
    ns = (64, 128, 256)
    errs = [measure(n, PERIODS)["period_err"] for n in ns]
    print(f"MEASURED pulsation worst-window period_err 64/128/256 {errs}, orders {np.round(-np.diff(np.log2(errs)), 3)}")
    drift = [measure(n, PERIODS)["drho_c"] for n in ns]
    check("pulsation unperturbed drift order 64-256", float(-np.polyfit(np.log2(ns), np.log2(drift), 1)[0]),
          DRIFT_ORDER, lower=True)


TEETH_GAMMA = 1.55
TEETH_OWN = 2.5e-2                   # worst-window |omega / omega_LAWE(1.55) - 1|, measured 1.91e-2 (job 31655473)


@gpu_only
def test_wrong_gamma_is_caught_gpu():
    """Teeth: gamma = 1.55 fails the 64^3 worst-window period bound and follows its own Gamma_1 = 1.55 LAWE."""
    d = measure(64, PERIODS, gamma=TEETH_GAMMA)
    own = max(abs(o / omega_lawe(TEETH_GAMMA) - 1) for o in d["omega"])
    check(f"pulsation N=64 gamma={TEETH_GAMMA} worst |omega / omega_LAWE({TEETH_GAMMA}) - 1|", own, TEETH_OWN)
    check(f"pulsation N=64 gamma={TEETH_GAMMA} period_err", d["period_err"], BOUNDS[64]["period_err"], lower=True)
