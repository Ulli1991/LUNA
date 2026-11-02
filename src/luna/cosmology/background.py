"""Flat-FLRW background as pure functions of the scale factor ``a`` (a0 = 1); H0 comes from the caller."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

@dataclass(frozen=True)
class Cosmology:
    Omega_m: float = 0.3      # total matter (CDM + baryons), z = 0
    Omega_L: float = 0.7      # cosmological constant / dark energy
    Omega_b: float = 0.049    # baryons (a subset of Omega_m)
    Omega_r: float = 0.0      # radiation
    h: float = 0.6711         # H0 = 100 h km/s/Mpc
    n_s: float = 0.9624       # primordial tilt (initial conditions)
    sigma8: float = 0.8       # power-spectrum normalisation (initial conditions)

    @property
    def Omega_k(self) -> float:
        return 1.0 - self.Omega_m - self.Omega_L - self.Omega_r

    def E(self, a):
        return np.sqrt(self.Omega_m * a ** -3 + self.Omega_r * a ** -4
                       + self.Omega_k * a ** -2 + self.Omega_L)

    def H(self, a, H0):
        return H0 * self.E(a)

    def adot(self, a, H0):
        return a * self.H(a, H0)

    def Omega_m_a(self, a):
        return self.Omega_m * a ** -3 / self.E(a) ** 2

    def growth_factor(self, a):
        def _Dtilde(av):
            ap = np.linspace(1e-8, av, 4096)
            return self.E(av) * _trapz(1.0 / (ap * self.E(ap)) ** 3, ap)

        a_arr = np.asarray(a, dtype=float)
        D = np.vectorize(_Dtilde)(a_arr) / _Dtilde(1.0)
        return D if a_arr.ndim else float(D)

    def growth_rate(self, a):
        a_arr = np.asarray(a, dtype=float)
        la, d = np.log(a_arr), 1e-4
        f = (np.log(self.growth_factor(np.exp(la + d)))
             - np.log(self.growth_factor(np.exp(la - d)))) / (2 * d)
        return f if a_arr.ndim else float(f)

    # super-comoving integration as in RAMSES (Martel & Shapiro 1998)
    def dadtau(self, a, H0):
        return a ** 3 * self.H(a, H0)

    def poisson_coeff(self, a):
        return 1.5 * self.Omega_m * a

    def build_a_of_tau(self, a_start, a_end, H0, n=4096):
        a_tab = np.linspace(a_start, a_end, n)
        integ = 1.0 / (a_tab ** 3 * self.H(a_tab, H0))
        tau = np.concatenate(([0.0], np.cumsum(
            0.5 * (integ[1:] + integ[:-1]) * np.diff(a_tab))))
        return a_tab, tau

    def a_after_dloga(self, a, dloga):
        return a * np.exp(dloga)


def EdS(**kw) -> Cosmology:
    return Cosmology(Omega_m=1.0, Omega_L=0.0, Omega_r=0.0, **kw)


def redshift_to_a(z):
    return 1.0 / (1.0 + np.asarray(z, dtype=float))


def a_to_redshift(a):
    return 1.0 / np.asarray(a, dtype=float) - 1.0
