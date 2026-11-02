"""Sedov-Taylor blast wave, closed form (Kamm 2000, LA-UR-00-6055 eqs. 29-47; Kamm & Timmes 2007,
LA-UR-07-2849; uniform ambient, omega = 0, planar j = 1, cylindrical j = 2, spherical j = 3):
R_s(t) = (E0 / (alpha rho0))^(1/(j+2)) t^(2/(j+2)) and the interior profiles.

A reference solution for tests and validation; the hydro step never calls it. Checked against Kamm & Timmes
2007 Tables 1-5 in tests/test_reference_solvers.py."""
import numpy as np


def similarity(V, gamma, j=3):
    """(lam, f, g, h) = r/R_s, v/v2, rho/rho2, p/p2 at similarity velocity V (Kamm 2000, omega = 0)."""
    return _similarity_dv(np.asarray(V, dtype=np.float64) - v_range(gamma, j)[0], gamma, j)


def _similarity_dv(dV, gamma, j):
    """(lam, f, g, h) at V = V0 + dV; x2 = b c dV exactly (c V0 = 1), no cancellation towards the origin."""
    dV = np.asarray(dV, dtype=np.float64)
    V = v_range(gamma, j)[0] + dV
    a = (j + 2) * (gamma + 1) / 4.0
    b = (gamma + 1) / (gamma - 1)
    c = (j + 2) * gamma / 2.0
    d = ((j + 2) * (gamma + 1)
         / ((j + 2) * (gamma + 1) - 2.0 * (2.0 + j * (gamma - 1))))
    e = (2.0 + j * (gamma - 1)) / 2.0

    a0 = 2.0 / (j + 2)
    a2 = -(gamma - 1) / (2.0 * (gamma - 1) + j)
    a1 = ((j + 2) * gamma / (2.0 + j * (gamma - 1))
          * (2.0 * j * (2.0 - gamma) / (gamma * (j + 2) ** 2) - a2))
    a3 = j / (2.0 * (gamma - 1) + j)
    a4 = a1 * (j + 2) / (2.0 - gamma)
    a5 = -2.0 / (2.0 - gamma)

    x1 = a * V
    x2 = b * c * dV
    x3 = d * (1.0 - e * V)
    x4 = b * (1.0 - (c / gamma) * V)

    lam = x1 ** (-a0) * x2 ** (-a2) * x3 ** (-a1)
    f = x1 * lam
    g = x2 ** a3 * x3 ** a4 * x4 ** a5
    h = x1 ** (a0 * j) * x3 ** (a4 - 2.0 * a1) * x4 ** (1.0 + a5)
    return lam, f, g, h


def v_range(gamma, j=3):
    """(V0, V2): V at the origin (lambda -> 0) and at the shock (lambda = 1)."""
    return 2.0 / ((j + 2) * gamma), 4.0 / ((j + 2) * (gamma + 1))


def _sedov_profiles(gamma, j=3, n=200_000):
    """Closed-form interior profiles on a dense V grid, geometric towards the origin (x2 -> 0 there).
    Returns (lam, f, g, h): r/R_s, v/v2, rho/rho2, p/p2, increasing in lam."""
    V0, V2 = v_range(gamma, j)
    return _similarity_dv((V2 - V0) * np.geomspace(1e-12, 1.0, n), gamma, j)


# A_j (4 / (j + 2)^2): the measure A_j r^(j-1) dr (A = 2, 2 pi, 4 pi) times (U t / R_s)^2
_ALPHA_PREFACTOR = {1: 8.0 / 9.0, 2: np.pi / 2.0, 3: 16.0 * np.pi / 25.0}


def energy_integrals(gamma, j=3):
    """(kinetic, thermal) parts of alpha by quadrature in sigma, V = V0 + (V2 - V0) sigma^m, m = 1 / |a2|:
    lam ~ sigma at the origin, so the integrand is regular on [0, 1] (a lam grid misses lam < s_min^|a2|)."""
    from scipy.integrate import quad
    if j not in _ALPHA_PREFACTOR:
        raise NotImplementedError(f"j={j}: the energy integral is implemented for j = 1, 2, 3")
    V0, V2 = v_range(gamma, j)
    a2 = -(gamma - 1) / (2.0 * (gamma - 1) + j)
    a1 = ((j + 2) * gamma / (2.0 + j * (gamma - 1))
          * (2.0 * j * (2.0 - gamma) / (gamma * (j + 2) ** 2) - a2))
    e, m = (2.0 + j * (gamma - 1)) / 2.0, -1.0 / a2

    def dens(sig, part):
        dV = (V2 - V0) * sig ** m
        lam, f, g, h = _similarity_dv(dV, gamma, j)
        V = V0 + dV
        # d lam / d sigma = lam (dln lam / dV) dV/dsigma; the x2 term -a2 / dV times dV/dsigma is -a2 m / sigma
        dlam = lam * ((-2.0 / (j + 2) / V + a1 * e / (1.0 - e * V)) * (V2 - V0) * m * sig ** (m - 1) - a2 * m / sig)
        w = (0.5 * (gamma + 1) / (gamma - 1) * g * (2.0 / (gamma + 1)) ** 2 * f ** 2 if part == 0
             else (2.0 / (gamma + 1)) / (gamma - 1) * h)
        return w * lam ** (j - 1) * dlam
    return tuple(_ALPHA_PREFACTOR[j] * quad(dens, 0.0, 1.0, args=(k,), epsabs=0.0, epsrel=1e-13, limit=400)[0]
                 for k in (0, 1))


def sedov_alpha(gamma, j=3):
    """Dimensionless energy constant alpha with E0 = alpha rho0 R_s^(j+2)/t^2 (planar: E0 of the two-sided slab),
    E0 = int (rho v^2/2 + p/(gamma-1)) dVol over the blast."""
    return sum(energy_integrals(gamma, j))


def xi0(gamma, j=3):
    """The Sedov constant xi_0 = alpha^(-1/(j+2)), R_s = xi_0 (E0 t^2 / rho0)^(1/(j+2))."""
    return sedov_alpha(gamma, j) ** (-1.0 / (j + 2))


def solution(r, t, E0, rho0, gamma, j=3):
    """(rho, u, p, R_s) at radii r and time t; ambient rho0 at rest and pressure 0 outside R_s."""
    r = np.asarray(r, dtype=np.float64)
    alpha = sedov_alpha(gamma, j)
    Rs = (E0 / (alpha * rho0)) ** (1.0 / (j + 2)) * t ** (2.0 / (j + 2))
    U = (2.0 / (j + 2)) * Rs / t
    v2 = 2.0 * U / (gamma + 1.0)
    rho2 = rho0 * (gamma + 1.0) / (gamma - 1.0)
    p2 = 2.0 * rho0 * U ** 2 / (gamma + 1.0)

    lam, f, g, h = _sedov_profiles(gamma, j)
    x = r / Rs
    inside = x <= 1.0
    fv = np.interp(x, lam, f, right=1.0)
    fv = np.where(x < lam[0], f[0] * x / max(lam[0], 1e-300), fv)  # v ~ r at centre
    gv = np.interp(x, lam, g, right=1.0)
    small = (x < lam[0]) & (x > 0)
    if lam[1] > lam[0] and g[0] > 0 and g[1] > 0:
        slope = np.log(g[1] / g[0]) / np.log(lam[1] / lam[0])      # rho power law
        gv = np.where(small, g[0] * (x / lam[0]) ** slope, gv)
    hv = np.interp(x, lam, h, right=1.0)
    hv = np.where(x < lam[0], h[0], hv)                            # p flat at centre

    rho = np.where(inside, rho2 * gv, rho0)
    v_r = np.where(inside, v2 * fv, 0.0)
    P = np.where(inside, p2 * hv, 0.0)
    return rho, v_r, P, Rs
