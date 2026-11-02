"""Differentiable OU driving (forcing_theta): the unit-state form reproduces the driver, and its derivative is right.

a = sigma_k P_k(zeta) b_k must equal init_state/evolve for the same keys at theta = (weight, zeta), and
d/dtheta of a scalar of a must match a central difference.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from luna.hydro.turbulence import OUDriver, OUDriverConfig

pytestmark = pytest.mark.fast


def _driver(zeta=0.3):
    cfg = OUDriverConfig(k_min=1.0, k_max=3.0, solenoidal_weight=zeta, rms_acceleration=2.0, seed=0,
                         autocorrelation_time=0.125)
    return OUDriver(cfg, dtype=float)


def test_unit_state_reproduces_the_driver_amplitudes():
    drv, zeta = _driver(), 0.3
    key = jax.random.PRNGKey(3)
    a, b = drv.init_state(key), drv.unit_state(key)
    np.testing.assert_allclose(drv.amplitudes(b, drv.weight, zeta), a, rtol=0, atol=1e-14)
    for k in jax.random.split(jax.random.PRNGKey(4), 5):         # the same draws step by step
        a, b = drv.evolve(a, 1e-3, k), drv.unit_evolve(b, 1e-3, k)
        np.testing.assert_allclose(drv.amplitudes(b, drv.weight, zeta), a, rtol=0, atol=1e-14)


def test_amplitude_gradient_matches_central_differences():
    drv = _driver()
    b = drv.unit_evolve(drv.unit_state(jax.random.PRNGKey(5)), 0.05, jax.random.PRNGKey(6))
    w = jnp.arange(1.0, 4.0)

    def J(theta):
        a = drv.amplitudes(b, theta[:-1], theta[-1])
        return jnp.sum(jnp.abs(a[:3]) ** 2 * w[:, None])         # weights a few modes unevenly
    theta = jnp.concatenate([drv.weight, jnp.asarray([0.3])])
    g = jax.grad(J)(theta)
    for i in (0, 1, 2, theta.size - 1):                          # three mode amplitudes and zeta
        e = jnp.zeros_like(theta).at[i].set(1e-6)
        fd = (J(theta + e) - J(theta - e)) / 2e-6
        assert abs(float(g[i]) - float(fd)) <= 1e-7 * max(1.0, abs(float(fd))), (i, float(g[i]), float(fd))
    assert abs(float(g[-1])) > 1e-6                              # zeta enters the amplitudes


def test_correlation_time_enters_the_unit_evolution():
    drv = _driver()
    b = drv.unit_state(jax.random.PRNGKey(7))
    k = jax.random.PRNGKey(8)
    np.testing.assert_array_equal(drv.unit_evolve(b, 1e-3, k), drv.unit_evolve(b, 1e-3, k, 0.125))
    J = lambda T: jnp.sum(jnp.abs(drv.unit_evolve(b, 1e-3, k, T)) ** 2)  # noqa: E731
    fd = (J(0.125 + 1e-7) - J(0.125 - 1e-7)) / 2e-7
    assert abs(float(jax.grad(J)(0.125)) - float(fd)) <= 1e-6 * max(1.0, abs(float(fd)))
