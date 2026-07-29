"""Sedov-Taylor blast wave: closed-form analytic solution.

Transcribed from Kamm (2000), LA-UR-00-6055 eqs. 29-47 (uniform ambient,
omega=0, spherical j=3): R_s(t) = (E0/(alpha rho0))^(1/5) t^(2/5) plus the
interior rho/v/P profiles. alpha self-check: 0.8511 (gamma=1.4),
xi_0 ~ 1.15 (gamma=5/3)."""
import numpy as np


def _sedov_profiles(gamma, j=3, n=200_000):
    """Closed-form interior profiles on a dense V grid (Kamm 2000, omega=0).
    Returns (lam, f, g, h): r/R_s, v/v2, rho/rho2, p/p2, increasing in lam."""
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

    V0 = 2.0 / ((j + 2) * gamma)              # origin (lambda -> 0)
    V2 = 4.0 / ((j + 2) * (gamma + 1))        # shock  (lambda = 1)
    # open interval at the origin end (x2 -> 0 there); geometric clustering
    s = np.geomspace(1e-12, 1.0, n)
    V = V0 + (V2 - V0) * s

    x1 = a * V
    x2 = b * (c * V - 1.0)
    x3 = d * (1.0 - e * V)
    x4 = b * (1.0 - (c / gamma) * V)

    lam = x1 ** (-a0) * x2 ** (-a2) * x3 ** (-a1)
    f = x1 * lam
    g = x2 ** a3 * x3 ** a4 * x4 ** a5
    h = x1 ** (a0 * j) * x3 ** (a4 - 2.0 * a1) * x4 ** (1.0 + a5)
    return lam, f, g, h


def _trapz(y, x):
    return np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)


def sedov_alpha(gamma, j=3):
    """Dimensionless energy constant alpha with E0 = alpha rho0 R_s^(j+2)/t^2,
    by quadrature of E0 = int (rho v^2/2 + p/(gamma-1)) dVol over the blast."""
    if j != 3:
        raise NotImplementedError("energy integral implemented for j=3")
    lam, f, g, h = _sedov_profiles(gamma, j)
    # E0 = 4 pi r2^3 U^2 rho0 * I with U = (2/(j+2)) r2/t -> alpha = (16 pi/25) I
    integrand = (0.5 * (gamma + 1) / (gamma - 1) * g
                 * (2.0 / (gamma + 1)) ** 2 * f ** 2
                 + (2.0 / (gamma + 1)) / (gamma - 1) * h) * lam ** 2
    return (16.0 * np.pi / 25.0) * _trapz(integrand, lam)


def sedov_taylor_3d(r, t, E0=1.0, rho0=1.0, gamma=5.0 / 3.0):
    """Sedov-Taylor self-similar solution at time t on radius array r.

    Returns dict {"rho", "v_r", "P", "R_shock"} (ambient rho0, v=0, P~0
    outside the shock; pre-shock pressure assumed negligible)."""
    r = np.asarray(r, dtype=np.float64)
    alpha = sedov_alpha(gamma)
    Rs = (E0 / (alpha * rho0)) ** 0.2 * t ** 0.4
    U = 0.4 * Rs / t
    v2 = 2.0 * U / (gamma + 1.0)
    rho2 = rho0 * (gamma + 1.0) / (gamma - 1.0)
    p2 = 2.0 * rho0 * U ** 2 / (gamma + 1.0)

    lam, f, g, h = _sedov_profiles(gamma)
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
    return {"rho": rho, "v_r": v_r, "P": P, "R_shock": Rs}


if __name__ == "__main__":
    # self-check against the classic tabulated energy constants
    for gam, alpha_ref in ((1.4, 0.8511), (5.0 / 3.0, 0.4936)):
        al = sedov_alpha(gam)
        xi0 = (1.0 / al) ** 0.2
        print(f"gamma={gam:.4f}: alpha={al:.5f} (ref {alpha_ref})  "
              f"xi0={xi0:.4f}  rel_err={abs(al - alpha_ref) / alpha_ref:.2e}")
