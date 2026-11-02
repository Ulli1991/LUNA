"""Guderley (1942) converging shock before collapse: the self-similar exact solution (Lazarus 1981, SIAM J. Numer.
Anal. 18, 316, eqs. 2.5-2.9; the ODE numerators and state transform as in ExactPack `solvers/guderley/ramsey.py`,
github.com/lanl/ExactPack, BSD-3).

Lazarus time: the shock is at R_s(t) = (-t)^alpha, alpha = 1 / lambda, t in [-1, 0); x = t / r^lambda, x = -1 at
the shock. Ahead of it (r < R_s) cold gas at rest, rho0; behind it rho = rho0 R(x), u = -V(x) r / (lambda t),
c = -C(x) r / (lambda t), p = rho c^2 / gamma. lambda is the eigenvalue for which the trajectory from the strong-
shock point passes the sonic singularity C = 1 + V (Ramsey, Kamm & Bolstad 2012, LA-UR-09-05853, eq. 42); found
here by shooting, alpha checked against their Table 1. The ODE is integrated in a regular parameter s (dV/ds =
n0, dC/ds = C n1, d ln(-x)/ds = lambda D), the saddle crossed along its eigen-line; R from the adiabatic integral
(Lazarus eq. 2.7), checked against its own ODE. A reference for tests; the hydro step never calls it."""
import functools

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

# Ramsey, Kamm & Bolstad 2012 (LA-UR-09-05853) Table 1, "new value": alpha for (j, gamma), j = 2 cyl., 3 sph.
ALPHA_RKB = {(2, 1.4): 0.835323192, (2, 5.0 / 3.0): 0.815624901, (2, 2.0): 0.800112351, (2, 3.0): 0.775666619,
             (3, 1.4): 0.717174501, (3, 5.0 / 3.0): 0.688376823, (3, 2.0): 0.667046070, (3, 3.0): 0.636410594}


def numerators(V, C, lam, gamma, j):
    """(n0, n1, n2, D) of Lazarus eqs. 2.8-2.9 and the R equation: dV/dx = n0 / (D x lam), dC/dx = C n1 / (D x lam)."""
    nu, fac, Vp1, C2 = j - 1, (lam - 1.0) / gamma, 1.0 + V, C * C
    n0 = ((nu + 1) * V + 2.0 * fac) * C2 - V * Vp1 * (V + lam)
    n1 = ((1.0 + fac / Vp1) * C2 - 0.5 * nu * (gamma - 1.0) * V * Vp1 - Vp1 ** 2
          - 0.5 * (lam - 1.0) * ((3.0 - gamma) * V + 2.0))
    n2 = -2.0 * fac * C2 / Vp1 + V * (V + lam) - (nu + 1) * V * Vp1
    return n0, n1, n2, C2 - Vp1 ** 2


def shock_state(gamma):
    """(V, C, R) behind the strong shock at x = -1 (Lazarus eq. 2.6)."""
    return -2.0 / (gamma + 1.0), np.sqrt(2.0 * gamma * (gamma - 1.0)) / (gamma + 1.0), (gamma + 1.0) / (gamma - 1.0)


def _rhs(lam, gamma, j, sgn):
    """d(V, C, ln(-x), ln R)/ds, sign sgn on s."""
    def f(s, y):
        n0, n1, n2, D = numerators(y[0], y[1], lam, gamma, j)
        return [sgn * n0, sgn * y[1] * n1, sgn * lam * D, sgn * n2]
    return f


def _classify(lam, gamma, j, rtol=1e-13):
    """+1 if the trajectory from the shock reaches the sonic line D = 0 first, -1 if n0 = 0 (V turns) first."""
    def sonic(s, y):
        return numerators(y[0], y[1], lam, gamma, j)[3]

    def turn(s, y):
        return numerators(y[0], y[1], lam, gamma, j)[0]
    sonic.terminal = turn.terminal = True
    V0, C0, _ = shock_state(gamma)
    sol = solve_ivp(_rhs(lam, gamma, j, -1.0), (0.0, 1e4), [V0, C0, 0.0, 0.0], method="DOP853", rtol=rtol,
                    atol=rtol * 1e-3, events=[sonic, turn])
    ts, tt = [e[0] if len(e) else np.inf for e in sol.t_events]
    if not np.isfinite(min(ts, tt)):
        raise RuntimeError(f"Guderley shooting: no event for lambda={lam}")
    return 1 if ts < tt else -1


@functools.lru_cache(maxsize=None)
def exponent(j, gamma):
    """lambda = 1 / alpha by bisection between alpha_RKB -+ 1e-6 on the phase-plane classification."""
    a = ALPHA_RKB.get((j, gamma))
    if a is None:
        raise ValueError(f"no Ramsey-Kamm-Bolstad bracket for j={j}, gamma={gamma}")
    lo, hi = 1.0 / (a + 1e-6), 1.0 / (a - 1e-6)
    clo, chi = _classify(lo, gamma, j), _classify(hi, gamma, j)
    if clo == chi:
        raise RuntimeError(f"Guderley shooting: no sign change in [{lo}, {hi}]")
    while hi - lo > 2e-16 * hi:
        mid = 0.5 * (lo + hi)
        if mid in (lo, hi):
            break
        if _classify(mid, gamma, j) == clo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _saddles(lam, gamma, j):
    """The two (V, C = 1 + V) roots of n0 = D = 0: nu V^2 + (nu + 1 + 2 fac - lam) V + 2 fac = 0."""
    nu, fac = j - 1, (lam - 1.0) / gamma
    return [(v, 1.0 + v) for v in np.roots([nu, nu + 1 + 2 * fac - lam, 2 * fac]).real]


class Similarity:
    """V, C, R on x in [-1, x_min] for (j, gamma): splines in ln(-x)."""

    def __init__(self, j, gamma, x_min=-1e-6, eps=1e-5):
        self.j, self.gamma = j, gamma
        lam = self.lam = exponent(j, gamma)
        V0, C0, R0 = shock_state(gamma)
        sad = _saddles(lam, gamma, j)

        def near(k):
            def ev(s, y):
                return np.hypot(y[0] - sad[k][0], y[1] - sad[k][1]) - eps
            ev.terminal = True
            return ev
        tol = dict(method="DOP853", rtol=1e-13, atol=1e-15, dense_output=True)
        a = solve_ivp(_rhs(lam, gamma, j, -1.0), (0.0, 1e4), [V0, C0, 0.0, np.log(R0)], events=[near(0), near(1)],
                      **tol)
        hit = [k for k in (0, 1) if len(a.t_events[k])]
        if not hit:
            raise RuntimeError("Guderley: trajectory missed the sonic saddle")
        S = np.array(sad[hit[0]])
        y1 = a.y[:, -1]
        # across the saddle: point reflection in (V, C); ln(-x), ln R by their slopes dl/dV there (O(eps^2))
        n0, n1, n2, D = numerators(y1[0], y1[1], lam, gamma, j)
        dV = 2.0 * (S[0] - y1[0])
        y2 = [2 * S[0] - y1[0], 2 * S[1] - y1[1], y1[2] + lam * D / n0 * dV, y1[3] + n2 / n0 * dV]
        sgn = -np.sign(numerators(y2[0], y2[1], lam, gamma, j)[3])     # keep ln(-x) decreasing

        def done(s, y):
            return y[2] - np.log(-x_min)
        done.terminal = True
        b = solve_ivp(_rhs(lam, gamma, j, sgn), (0.0, 1e4), y2, events=[done], **tol)
        if not len(b.t_events[0]):
            raise RuntimeError("Guderley: integration past the saddle did not reach x_min")
        self.x_sonic = -np.exp(0.5 * (y1[2] + y2[2]))
        self.saddle = (float(S[0]), float(S[1]))
        ys = [a.sol(np.linspace(0.0, a.t[-1], 6000)), b.sol(np.linspace(0.0, b.t[-1], 6000))]
        Y = np.concatenate([ys[0], ys[1][:, 1:]], axis=1)
        ell = -Y[2]                                            # -ln(-x), increasing from 0
        keep = np.concatenate([[True], np.diff(ell) > 1e-14])
        ell, Y = ell[keep], Y[:, keep]
        self._V, self._C, self._lnR_ode = (CubicSpline(ell, Y[0]), CubicSpline(ell, Y[1]), CubicSpline(ell, Y[3]))
        self.ell_max = float(ell[-1])
        q = 2.0 * (lam - 1.0) / j
        self._q = q
        self._K = C0 ** 2 * (1.0 + V0) ** q * R0 ** (q - gamma + 1.0)

    def vcr(self, x):
        """(V, C, R) at x in [-1, x_min]; R from the adiabatic integral (C / x)^2 (1 + V)^q R^(q - gamma + 1) = K."""
        ell = -np.log(-np.asarray(x, dtype=float))
        if np.any(ell < -1e-12) or np.any(ell > self.ell_max):
            raise ValueError("x outside [-1, x_min]")
        V, C = self._V(ell), self._C(ell)
        R = (self._K * np.exp(-2.0 * ell) / (C * C * (1.0 + V) ** self._q)) ** (1.0 / (self._q - self.gamma + 1.0))
        return V, C, R

    def r_ode(self, x):
        """R from integrating its own ODE (the check on the adiabatic integral)."""
        return np.exp(self._lnR_ode(-np.log(-np.asarray(x, dtype=float))))


@functools.lru_cache(maxsize=None)
def similarity(j, gamma):
    """The cached Similarity for (j, gamma)."""
    return Similarity(j, gamma)


def shock_radius(t, j, gamma):
    """R_s(t) = (-t)^(1 / lambda), t < 0."""
    return (-np.asarray(t, dtype=float)) ** (1.0 / exponent(j, gamma))


def solution(r, t, j, gamma, rho0=1.0, p0=0.0):
    """(rho, u, p) at radii r and time t < 0; ahead of the shock rho0, 0, p0."""
    r = np.asarray(r, dtype=float)
    sim = similarity(j, gamma)
    lam = sim.lam
    behind = r > shock_radius(t, j, gamma)
    rb = np.where(behind, r, 1.0)
    x = np.where(behind, t / rb ** lam, -1.0)
    V, C, R = sim.vcr(x)
    u = -V * r / (lam * t)
    c = -C * r / (lam * t)
    rho = np.where(behind, rho0 * R, rho0)
    return rho, np.where(behind, u, 0.0), np.where(behind, rho * c * c / gamma, p0)
