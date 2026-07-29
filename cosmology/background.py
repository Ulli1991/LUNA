"""Background cosmology: pure functions of the scale factor ``a`` for a flat
FLRW model (``a0 = 1`` today) -- no JAX, no solver state. ``E(a) = H/H0`` is
dimensionless; ``H0`` (in the run's code units) is supplied by the caller.
The arithmetic-only methods accept floats or arrays and may run inside the JIT
time loop; ``growth_factor``/``growth_rate`` use numpy quadrature and are
IC-time helpers only. Defaults: CAMELS / IllustrisTNG fiducial cosmology.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:                       # numpy>=2 renamed trapz -> trapezoid (trapz removed)
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


@dataclass(frozen=True)
class Cosmology:
    """Flat-FLRW background parameters (CAMELS fiducial by default)."""

    Omega_m: float = 0.3      # total matter (CDM + baryons), z = 0
    Omega_L: float = 0.7      # cosmological constant / dark energy
    Omega_b: float = 0.049    # baryons (a subset of Omega_m)
    Omega_r: float = 0.0      # radiation (negligible for the pancake / low-z box)
    h: float = 0.6711         # H0 = 100 h km/s/Mpc
    n_s: float = 0.9624       # primordial tilt (initial conditions)
    sigma8: float = 0.8       # power-spectrum normalisation (initial conditions)

    @property
    def Omega_k(self) -> float:
        """Curvature term that closes the budget (0 for a flat model)."""
        return 1.0 - self.Omega_m - self.Omega_L - self.Omega_r

    # -- dimensionless expansion rate -------------------------------------
    def E(self, a):
        """``E(a) = H(a)/H0 = sqrt(Om a^-3 + Or a^-4 + Ok a^-2 + OL)``."""
        return np.sqrt(self.Omega_m * a ** -3 + self.Omega_r * a ** -4
                       + self.Omega_k * a ** -2 + self.Omega_L)

    def H(self, a, H0):
        """Hubble parameter in the caller's units (``H0`` in those units)."""
        return H0 * self.E(a)

    def adot(self, a, H0):
        """``da/dt = a H(a)`` (proper-time expansion rate)."""
        return a * self.H(a, H0)

    def Omega_m_a(self, a):
        """Matter density parameter at ``a``: ``Om a^-3 / E(a)^2``."""
        return self.Omega_m * a ** -3 / self.E(a) ** 2

    # -- linear growth (IC-time; numpy quadrature) ------------------------
    def growth_factor(self, a):
        """Linear growth factor ``D(a)``, normalised so ``D(1) = 1``.

        ``D(a) propto E(a) * int_0^a da' / (a' E(a'))^3`` (Heath 1977). For
        Einstein--de Sitter this reduces exactly to ``D(a) = a``.
        """
        def _Dtilde(av):
            ap = np.linspace(1e-8, av, 4096)
            return self.E(av) * _trapz(1.0 / (ap * self.E(ap)) ** 3, ap)

        a_arr = np.asarray(a, dtype=float)
        D = np.vectorize(_Dtilde)(a_arr) / _Dtilde(1.0)
        return D if a_arr.ndim else float(D)

    def growth_rate(self, a):
        """Linear growth rate ``f = dlnD/dlna`` (``~ Omega_m(a)^0.55``)."""
        a_arr = np.asarray(a, dtype=float)
        la, d = np.log(a_arr), 1e-4
        f = (np.log(self.growth_factor(np.exp(la + d)))
             - np.log(self.growth_factor(np.exp(la - d)))) / (2 * d)
        return f if a_arr.ndim else float(f)

    # -- RAMSES super-comoving integration (Martel & Shapiro 1998) --------
    # LUNA is a grid + multigrid-Poisson code, so we follow RAMSES rather than
    # Gadget. In super-comoving variables (rho~=a^3 rho, v~=a v_pec, P~=a^5 P)
    # with super-comoving time dtau = dt/a^2, the comoving Euler equations
    # reduce to the *plain* Euler equations -- the flux kernel is unchanged --
    # and the only explicit a-dependence is the Poisson source coefficient.
    def dadtau(self, a, H0):
        """``da/dtau = a^3 H(a)`` for super-comoving time ``dtau = dt/a^2``.
        Integrating this once gives the a(tau) table (the RAMSES 'friedman')."""
        return a ** 3 * self.H(a, H0)

    def poisson_coeff(self, a):
        """Coefficient of ``delta`` in the super-comoving Poisson equation
        ``lap(phi~) = (3/2) Omega_m a delta`` -- replaces ``4 pi G`` in the
        existing FMG solve."""
        return 1.5 * self.Omega_m * a

    def build_a_of_tau(self, a_start, a_end, H0, n=4096):
        """Tabulate super-comoving time ``tau(a) = int_{a_start}^{a} da'/(a'^3
        H)`` by integrating the Friedmann ODE; returns ``(a_table, tau_table)``
        with ``tau = 0`` at ``a_start``, for interpolating ``a(tau)`` in the
        time loop (``a = numpy.interp(tau, tau_table, a_table)``)."""
        a_tab = np.linspace(a_start, a_end, n)
        integ = 1.0 / (a_tab ** 3 * self.H(a_tab, H0))
        tau = np.concatenate(([0.0], np.cumsum(
            0.5 * (integ[1:] + integ[:-1]) * np.diff(a_tab))))
        return a_tab, tau

    def a_after_dloga(self, a, dloga):
        """Scale factor after a logarithmic step: ``a1 = a exp(dloga)`` (used
        for the ``Delta ln a <= dloga_max`` expansion cap on the timestep)."""
        return a * np.exp(dloga)


def EdS(**kw) -> Cosmology:
    """Einstein--de Sitter (``Omega_m = 1``): ``D(a) = a`` -- the clean universe
    for the Zel'dovich pancake's analytic check."""
    return Cosmology(Omega_m=1.0, Omega_L=0.0, Omega_r=0.0, **kw)


def redshift_to_a(z):
    """Scale factor from redshift: ``a = 1 / (1 + z)``."""
    return 1.0 / (1.0 + np.asarray(z, dtype=float))


def a_to_redshift(a):
    """Redshift from scale factor: ``z = 1/a - 1``."""
    return 1.0 / np.asarray(a, dtype=float) - 1.0
