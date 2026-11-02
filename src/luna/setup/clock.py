"""The run's time variable: proper time, or super-comoving tau with a(tau).

`ProperClock` and `CosmoClock` answer the same five questions, so the loop
never asks which one it holds: `scale(t)` is the scale factor a (1 for proper
time), `cap(dt, t)` applies the clock's own limits (the d ln a cap and the
table end for comoving; none for proper), `kick_scale(a)` is what multiplies
the unit-coefficient potential in a gravity kick ((3/2) Omega_m H0^2 a for
comoving, 1 for proper), `target_of(a)` turns a snapshot scale factor into a
landing time, and `finished(t)`.
"""
import numpy as np

# Pad the a(tau) table past a_end so interpolation never runs off its end.
A_TAB_PAD = 0.05


class ProperClock:
    """Newtonian time: no scale factor, no caps, unit kick scale."""
    comoving = False

    def __init__(self, t_start, t_end):
        self.t_start = float(t_start)
        self.t_end = float(t_end)

    def scale(self, t):
        return 1.0

    def cap(self, dt, t):
        return dt

    def kick_scale(self, a):
        return 1.0

    def target_of(self, a):
        raise ValueError("a proper-time run has no scale-factor snapshot targets; "
                         "use snapshot times")

    def finished(self, t):
        return t >= self.t_end

    def label(self, t):
        return f"t={t:.5e}"


class CosmoClock:
    """Super-comoving time tau (Martel & Shapiro 1998): the a(tau) table is
    the RAMSES 'friedman' integration, dtau = dt / a^2."""
    comoving = True

    def __init__(self, cosmo, a_init, a_end, H0, dloga_max):
        self.cosmo = cosmo
        self.H0 = float(H0)
        self.a_init = float(a_init)
        self.a_end = float(a_end)
        self.dloga_max = float(dloga_max)
        self.a_tab, self.tau_tab = cosmo.build_a_of_tau(
            self.a_init, self.a_end + A_TAB_PAD, self.H0)
        self.coeff_fac = 1.5 * float(cosmo.Omega_m) * self.H0 * self.H0
        self.t_start = 0.0
        self.t_end = self.tau_of_a(self.a_end)

    def scale(self, t):
        return float(np.interp(t, self.tau_tab, self.a_tab))

    def tau_of_a(self, a):
        return float(np.interp(a, self.a_tab, self.tau_tab))

    def cap(self, dt, t):
        """Never expand by more than dloga_max in one step, never leave the table."""
        a = self.scale(t)
        dt = min(dt, self.dloga_max / (a * a * self.cosmo.H(a, self.H0)))
        return min(dt, float(self.tau_tab[-1]) - t)

    def kick_scale(self, a):
        """lap(psi) = delta is solved with a unit coefficient; the kick multiplies
        its gradient by (3/2) Omega_m H0^2 a."""
        return self.coeff_fac * a

    def target_of(self, a):
        return self.tau_of_a(a)

    def finished(self, t):
        return t >= self.t_end

    def label(self, t):
        a = self.scale(t)
        return f"a={a:.4f} z={1.0 / a - 1.0:.4f}"


def make_clock(cfg):
    """The clock a config asks for."""
    if not cfg.CosmologicalIntegration:
        return ProperClock(cfg.t_start, cfg.t_end)
    from luna.cosmology import Cosmology
    cosmo = Cosmology(Omega_m=cfg.cosmo_Omega_m, Omega_L=cfg.cosmo_Omega_L)
    return CosmoClock(cosmo, cfg.a_init, cfg.a_end, cfg.cosmo_H0, cfg.dloga_max)
