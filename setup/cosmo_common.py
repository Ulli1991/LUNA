"""Shared super-comoving cosmology machinery for the cosmo runners.

Single source for the pieces that were copy-pasted five times (and had
drifted) across ``runner_cosmo`` / ``runner_cosmo_dm`` / ``runner_cosmo_hydro``:
the a(tau) lookup table (the RAMSES 'friedman' integration), the per-step
timestep caps, the comoving Poisson coefficient, and the DM softening default.

NOTE on defaults that intentionally differ between runners (pre-existing
behavior, preserved at the call sites): ``dloga_max`` defaults to 0.01 in the
gas runners but 0.02 in the DM-only runner; only the coupled runner reads
``POISSON_FMG_N_COARSEST``.
"""
import numpy as np

# Pad the a(tau) table past a_end so interpolation never runs off the end of
# the integration range (every runner used this same magic +0.05).
A_TAB_PAD = 0.05


class CosmoClock:
    """a(tau) bookkeeping for super-comoving integration.

    Builds the a(tau) table once and provides the per-step operations every
    cosmo runner shares: scale-factor lookup, the d(ln a) expansion cap, and
    the never-step-past-the-table clamp.
    """

    def __init__(self, cosmo, a_init, a_end, H0):
        self.cosmo = cosmo
        self.H0 = float(H0)
        self.a_tab, self.tau_tab = cosmo.build_a_of_tau(
            a_init, a_end + A_TAB_PAD, H0)

    def a_of_tau(self, tau):
        """Scale factor at super-comoving time tau (host-side interp)."""
        return float(np.interp(tau, self.tau_tab, self.a_tab))

    def dtau_expansion(self, a, dloga_max):
        """The dtau that advances d(ln a) by dloga_max at scale factor a."""
        return dloga_max / (a * a * self.cosmo.H(a, self.H0))

    def cap_expansion(self, dtau, a, dloga_max):
        """Cap dtau so a single step never exceeds d(ln a) = dloga_max."""
        return min(dtau, self.dtau_expansion(a, dloga_max))

    def cap_to_table(self, dtau, tau):
        """Clamp dtau so tau + dtau stays inside the a(tau) table."""
        return min(dtau, float(self.tau_tab[-1]) - tau)

    def tau_of_a(self, a):
        """Super-comoving time at scale factor a (inverse of a_of_tau)."""
        return float(np.interp(a, self.a_tab, self.tau_tab))

    def cap_to_target(self, dtau, tau, a_target):
        """Clamp dtau so the step lands exactly on a_target, never past it
        (the comoving analog of the dense runner's t_end/snapshot clamp)."""
        return min(dtau, self.tau_of_a(a_target) - tau)


def dtau_accel(eta, eps_soft, gmax):
    """Gravitational acceleration timestep: dtau ~ eta * sqrt(eps / |g|_max).

    eps is the force softening (PM cell scale) -- this is the N-body accuracy
    criterion, not a hydro Courant condition.
    """
    return eta * np.sqrt(eps_soft / (gmax + 1e-30))


def poisson_coeff_fac(Om, H0):
    """Comoving Poisson coefficient prefactor (3/2) Omega_m H0^2.

    The per-step coefficient is ``poisson_coeff_fac(Om, H0) * a``; for
    dimensionless pancake units (H0 = 1) it reduces to (3/2) Omega_m a.
    """
    return 1.5 * float(Om) * float(H0) * float(H0)


def softening_from_config(cfg, dx):
    """DM force softening: cfg.dm_softening, defaulting to the PM cell dx."""
    import builtins
    return builtins.float(getattr(cfg, "dm_softening", 0.0)) or dx
