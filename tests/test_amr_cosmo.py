"""Cosmo-AMR hooks (amr/cosmo.py), gas-only Phase 1.

Covers the three things the cosmology kit is responsible for and that nothing
else tested: the conformal-time / Poisson-coefficient scalings, the dual-energy
guard (asserted off in Phase 1), and the per-level floor stack (buffered density
floor + momentum zeroing, velocity ceiling). Runs on the ROCm GPU backend on
viper; CPU on GitHub CI.
"""
import types
import numpy as np
import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from amr.config import AMRConfig
from amr import driver, cosmo

GAMMA = 5.0 / 3.0


def _cfg(**kw):
    """A minimal cfg stand-in: amr/cosmo.py reads gamma directly and everything
    else via getattr-with-default, so a SimpleNamespace is enough."""
    base = dict(gamma=GAMMA, cosmo_Omega_m=1.0, cosmo_Omega_L=0.0, cosmo_H0=1.0,
                a_init=0.1, a_end=1.0, dloga_max=0.01)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _acfg():
    return AMRConfig(ndim=3, box=(1.,) * 3, base_grid=(16,) * 3, block_shape=(16,) * 3,
                     max_levels=0, boundary=0, gamma=GAMMA)


def test_dual_energy_guarded():
    """Phase 1 does not port dual-energy -> constructing the kit must raise."""
    with pytest.raises(NotImplementedError):
        cosmo.make_cosmo_amr(_cfg(DualEnergy=True), _acfg(), 1.0, is_main_process=False)


def test_dtau_and_coeff_scalings():
    """dtau_cap = dloga_max / (a^2 H0 E(a)); coeff = 1.5 Om H0^2 a / rho_bar."""
    Om, OL, H0, rho_bar = 0.3, 0.7, 0.7, 2.0
    kit = cosmo.make_cosmo_amr(
        _cfg(cosmo_Omega_m=Om, cosmo_Omega_L=OL, cosmo_H0=H0), _acfg(), rho_bar,
        is_main_process=False)
    a = 0.5
    Ok = 1.0 - Om - OL
    E = np.sqrt(Om * a ** -3 + Ok * a ** -2 + OL)
    assert abs(float(kit["dtau_cap"](a)) - 0.01 / (a * a * H0 * E)) < 1e-12
    assert abs(float(kit["coeff"](a)) - 1.5 * Om * H0 * H0 * a / rho_bar) < 1e-12
    # a(tau) table brackets the requested span and is monotone.
    assert float(kit["a_tab"][0]) <= 0.1 and float(kit["a_tab"][-1]) >= 1.0
    assert bool(jnp.all(jnp.diff(kit["a_tab"]) > 0))


def test_floor_stack_clamps():
    """Buffered density floor clamps rho to rho_floor and zeros its momentum;
    the velocity ceiling caps |v| = |p|/rho on an otherwise-good cell."""
    rho_bar = 1.0
    rho_floor_frac, vmax = 1e-3, 5.0
    kit = cosmo.make_cosmo_amr(
        _cfg(cosmo_rho_floor=rho_floor_frac, cosmo_p_floor=1e-8, cosmo_vmax=vmax,
             cosmo_floor_buffer=1.0), _acfg(), rho_bar, is_main_process=False)
    rho_floor = rho_floor_frac * rho_bar

    # 1-level uniform hierarchy; inject two pathological cells into the one block.
    U = np.zeros((5, 16, 16, 16)); U[0] = 1.0; U[4] = 1.0
    h = driver.build_hierarchy(jnp.asarray(U), _acfg(), thresholds=[0.0],
                               criterion="static", periodic=True, pad_to=1)
    d = np.array(jax.device_get(h[0].data))     # writable copy
    d[0, 0, 0, 0, 0] = 1e-5                       # under-floor density (cell A)
    d[0, 0, 1, 1, 1] = 1.0                        # good density   (cell B)
    d[0, 1, 1, 1, 1] = 100.0                      # but huge x-momentum -> v=100
    h = (h[0]._replace(data=jnp.asarray(d)),) + h[1:]

    hf = kit["floors"](h, 0.5)
    df = np.asarray(jax.device_get(hf[0].data))

    # cell A: density clamped up to the floor, momentum zeroed
    assert abs(df[0, 0, 0, 0, 0] - rho_floor) < 1e-14
    assert df[0, 1, 0, 0, 0] == 0.0
    # cell B: |v| = |mom|/rho ceiled to vmax
    vB = df[0, 1, 1, 1, 1] / df[0, 0, 1, 1, 1]
    assert vB <= vmax * (1 + 1e-9)
    assert abs(vB - vmax) < 1e-9                  # was 100, must be pulled to 5
    # NaN sentinel slot passes through untouched
    assert bool(np.all(np.isnan(df[-1])))


if __name__ == "__main__":
    test_dual_energy_guarded()
    test_dtau_and_coeff_scalings()
    test_floor_stack_clamps()
    print("PASS test_amr_cosmo")
