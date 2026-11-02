"""Radial adiabatic pulsation modes of a polytrope: the LAWE as a Sturm-Liouville problem, linear finite elements.

LAWE (Cox 1980, Theory of Stellar Pulsation, ch. 8; Unno et al. 1989, Nonradial Oscillations of Stars, sec. 13),
for zeta = delta r / r with Gamma_1 constant:

    d/dr (Gamma_1 P r^4 zeta') + r^3 [d((3 Gamma_1 - 4) P)/dr + omega^2 rho r] zeta = 0,

i.e. -(p zeta')' + q zeta = omega^2 w zeta with p = Gamma_1 P r^4, q = (3 Gamma_1 - 4) r^3 rho g, w = rho r^4.
p vanishes at both ends (r^4 at the centre, P at the surface), so the natural boundary conditions of the
Rayleigh quotient (Cox's variational principle) are the regularity conditions; zeta'(0) = 0 and delta P / P
finite at the surface. Units G = rho_c = r_n = 1 (P = 4 pi / (n + 1) theta^(n+1)); frequencies are returned as
Omega^2 = omega^2 R^3 / (G M).
"""
import functools

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


@functools.lru_cache(maxsize=None)
def lane_emden(n):
    """(xi_1, theta(xi), theta'(xi)) of the Lane-Emden solution, closed form for n = 0, 1, else integrated."""
    if n == 0:
        return np.sqrt(6.0), lambda x: 1.0 - x * x / 6.0, lambda x: -x / 3.0
    if n == 1:
        return np.pi, lambda x: np.sinc(x / np.pi), lambda x: np.where(
            x > 1e-8, (x * np.cos(x) - np.sin(x)) / np.maximum(x, 1e-300) ** 2, -x / 3.0)
    x0 = 1e-4                       # series start: theta = 1 - x^2/6 + n x^4/120
    y0 = [1 - x0 ** 2 / 6 + n * x0 ** 4 / 120, -x0 / 3 + n * x0 ** 3 / 30]
    ev = lambda x, y: y[0]          # noqa: E731
    ev.terminal = True
    sol = solve_ivp(lambda x, y: [y[1], -np.maximum(y[0], 0.0) ** n - 2 * y[1] / x], (x0, 20.0), y0,
                    method="DOP853", rtol=1e-13, atol=1e-15, dense_output=True, events=ev)
    x1 = float(sol.t_events[0][0])

    def ev_sol(x, i):
        x = np.asarray(x, float)
        return sol.sol(np.minimum(np.maximum(x, x0), x1).ravel())[i].reshape(x.shape)

    def th(x):
        x = np.asarray(x, float)
        return np.where(x < x0, 1 - x * x / 6 + n * x ** 4 / 120, ev_sol(x, 0))

    def dth(x):
        x = np.asarray(x, float)
        return np.where(x < x0, -x / 3 + n * x ** 3 / 30, ev_sol(x, 1))
    return x1, th, dth


def _solve(n, gamma1, n_el, k):
    """(Omega^2 of the k lowest modes, nodes xi / xi_1, zeta on the nodes with zeta(0) = 1) on n_el elements."""
    x1, th, dth = lane_emden(n)
    xn = np.linspace(0.0, x1, n_el + 1)
    h = np.diff(xn)
    gq, gw = np.polynomial.legendre.leggauss(4)
    xq = 0.5 * (xn[:-1, None] + xn[1:, None]) + 0.5 * h[:, None] * gq[None, :]      # (n_el, 4)
    wq = 0.5 * h[:, None] * gw[None, :]
    t, dt = np.maximum(th(xq), 0.0), dth(xq)
    rho, P = t ** n, 4 * np.pi / (n + 1) * t ** (n + 1)
    g = 4 * np.pi * (-dt)                           # G m(xi) / xi^2 = 4 pi xi^2 (-theta') / xi^2
    p, q, w = gamma1 * P * xq ** 4, (3 * gamma1 - 4) * xq ** 3 * rho * g, rho * xq ** 4
    a = (xq - xn[:-1, None]) / h[:, None]           # the element's shape functions: 1 - a, a
    kp = (p * wq).sum(1) / h ** 2
    q00, q01, q11 = ((q * (1 - a) ** 2) * wq).sum(1), ((q * a * (1 - a)) * wq).sum(1), ((q * a * a) * wq).sum(1)
    w00, w01, w11 = ((w * (1 - a) ** 2) * wq).sum(1), ((w * a * (1 - a)) * wq).sum(1), ((w * a * a) * wq).sum(1)
    dK = np.zeros(n_el + 1); dM = np.zeros(n_el + 1)
    dK[:-1] += kp + q00; dK[1:] += kp + q11
    dM[:-1] += w00; dM[1:] += w11
    K = diags([-kp + q01, dK, -kp + q01], [-1, 0, 1], format="csc")
    M = diags([w01, dM, w01], [-1, 0, 1], format="csc")
    vals, vecs = eigsh(K, k=k, M=M, sigma=0.0, which="LM")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    unit = 4 * np.pi * (-float(dth(x1))) / x1       # G M / R^3 in G rho_c units
    return vals / unit, xn / x1, vecs[:, 0] / vecs[0, 0]


@functools.lru_cache(maxsize=None)
def modes(n, gamma1, k=3, n_el=4000):
    """Omega^2 = omega^2 R^3 / (G M) of the k lowest radial modes, Richardson-extrapolated from n_el and 2 n_el."""
    a, _, _ = _solve(n, gamma1, n_el, k)
    b, _, _ = _solve(n, gamma1, 2 * n_el, k)
    return tuple((4 * b - a) / 3)


@functools.lru_cache(maxsize=None)
def fundamental_zeta(n, gamma1, n_el=8000):
    """(x = r / R nodes, zeta(x)) of the fundamental mode, zeta(0) = 1."""
    _, x, z = _solve(n, gamma1, n_el, 1)
    return x, z


def variational_bounds(n, gamma1):
    """Cox 1980 ch. 8 bounds (3 Gamma_1 - 4) G M / R^3 <= omega^2 <= (3 Gamma_1 - 4) |W| / I, in G M / R^3."""
    x1, th, dth = lane_emden(n)
    x = np.linspace(0.0, x1, 200001)
    dm = 4 * np.pi * x * x * np.maximum(th(x), 0.0) ** n
    m = 4 * np.pi * x * x * (-dth(x))
    W = np.trapezoid(m * dm / np.maximum(x, 1e-300), x)          # |W| = int G m dm / r
    inertia = np.trapezoid(x * x * dm, x)
    M = m[-1]
    return 3 * gamma1 - 4, (3 * gamma1 - 4) * W / inertia / (M / x1 ** 3)
