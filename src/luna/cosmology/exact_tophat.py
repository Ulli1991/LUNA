"""Pressureless top-hat in Einstein-de Sitter (H0 = 1, t = 2/3 a^(3/2)): a uniform sphere of comoving overdensity
delta_i at a_i, started on the exact closed-FRW solution R = A (1 - cos th), t = B (th - sin th), checked against
a direct integration of R'' = -G M / R^2. Gives 1 + delta(a) and the interior peculiar-velocity gradient.
The reference of tests/test_cosmo_tophat_collapse.py; `python tools/tophat.py` prints the cross-check."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def _one_plus_delta(th):
    """Mean density inside the shell over the background, 9/2 (th - sin th)^2 / (1 - cos th)^3."""
    return 4.5 * (th - np.sin(th)) ** 2 / (1.0 - np.cos(th)) ** 3


class TopHat:
    """The exact interior solution for a top-hat of initial overdensity delta_i at a_i."""

    def __init__(self, delta_i, a_i):
        self.a_i = float(a_i)
        self.th_i = brentq(lambda th: _one_plus_delta(th) - 1.0 - delta_i, 1e-6, np.pi)
        self.t_i = 2.0 / 3.0 * self.a_i ** 1.5
        self.B = self.t_i / (self.th_i - np.sin(self.th_i))
        self.R_i = 1.0                                       # physical radius scale (drops out of every ratio)
        self.A = self.R_i / (1.0 - np.cos(self.th_i))
        self.GM = self.A ** 3 / self.B ** 2                  # A^3 = G M B^2

    def theta(self, a):
        """th at scale factor a (EdS: t = 2/3 a^(3/2)), on the expanding and collapsing branch (th < 2 pi)."""
        t = 2.0 / 3.0 * np.asarray(a, float) ** 1.5
        return np.array([brentq(lambda th: self.B * (th - np.sin(th)) - ti, 1e-9, 2 * np.pi - 1e-9)
                         for ti in np.atleast_1d(t)])

    def one_plus_delta(self, a):
        """Comoving density of the interior, rho~ = 1 + delta(a)."""
        return _one_plus_delta(self.theta(a))

    def vpec_over_r(self, a):
        """Interior peculiar velocity / physical radius, R_dot / R - H (uniform: v_pec = (R_dot/R - H) r)."""
        th = self.theta(a)
        rdot_over_r = np.sin(th) / (self.B * (1.0 - np.cos(th)) ** 2)
        return rdot_over_r - np.asarray(a, float) ** -1.5     # EdS: H = a^(-3/2)

    def a_turnaround(self):
        """a at th = pi."""
        return (1.5 * self.B * np.pi) ** (2.0 / 3.0)

    def integrate(self, a):
        """1 + delta(a) by integrating R'' = -G M / R^2 from the same initial state (the cross-check)."""
        th = self.th_i
        R0 = self.A * (1 - np.cos(th))
        V0 = self.A / self.B * np.sin(th) / (1 - np.cos(th))
        t = 2.0 / 3.0 * np.asarray(a, float) ** 1.5
        sol = solve_ivp(lambda _, y: [y[1], -self.GM / y[0] ** 2], (self.t_i, float(t.max())), [R0, V0],
                        t_eval=np.sort(t), rtol=1e-13, atol=1e-16, method="DOP853")
        R = sol.y[0]
        rho_bg = 1.0 / (6.0 * np.pi * t[np.argsort(np.argsort(t))] ** 2)   # EdS: 4 pi G rho = 2/(3 t^2), G = 1
        rho = 3.0 * self.GM / (4.0 * np.pi * R[np.argsort(np.argsort(t))] ** 3)
        return rho / rho_bg
