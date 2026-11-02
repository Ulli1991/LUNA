"""The super-comoving internal-energy term for gamma != 5/3 (Martel & Shapiro 1998, eq. 41):
d eps~/d tau = -H (3 gamma - 5) eps~ with H = a^-1 da/dtau, integrated exactly over a step,
rho~ eps~ -> rho~ eps~ (a_1 / a_0)^(5 - 3 gamma); kinetic energy and momenta are untouched."""

import jax
import jax.numpy as jnp


def setup_comoving_energy(cfg, clock, is_main=True):
    """The stage for a comoving run with gamma != 5/3, None otherwise (the term vanishes at 5/3)."""
    gamma = float(cfg.gamma)
    if clock is None or not clock.comoving or abs(gamma - 5.0 / 3.0) < 1e-12:
        return None
    from luna.sources.stages import ComovingEnergy
    ndim = int(cfg.SPATIAL_DIMENSION)
    exponent = 5.0 - 3.0 * gamma

    @jax.jit
    def scale(U, factor):
        """Scale the internal energy of conserved U by `factor`."""
        mom = U[1:1 + ndim]
        ke = 0.5 * jnp.sum(mom * mom, axis=0) / U[0]
        return U.at[-1].set(ke + factor * (U[-1] - ke))

    if is_main:
        print(f"  comoving energy term: gamma = {gamma:.6g}, eps~ ~ a^{exponent:.6g} (Martel & Shapiro 1998)",
              flush=True)
    return ComovingEnergy(clock, exponent, scale)
