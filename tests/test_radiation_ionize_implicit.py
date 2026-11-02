"""The implicit ionization update: it solves its equation, conserves photons,
reaches the ionization equilibrium, and does not depend on how the step is
cut -- across the extremes a cloud and STARBENCH reach (c sigma n_H dt up to
1e10, recombination b up to 1e6, photons per hydrogen from 1e-8 to 1e6)."""
import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp  # noqa: E402

from luna.radiation.ionization import ionize_implicit  # noqa: E402

pytestmark = pytest.mark.fast


def _cells(seed=0, n=4000):
    """Per-cell (a, b, photons per H, x0), log-uniform over the extremes; the
    physical constants are folded into a and b by chat = sigma = m_H = 1."""
    r = np.random.default_rng(seed)
    a = 10.0 ** r.uniform(-4, 10, n)
    b = 10.0 ** r.uniform(-6, 6, n)
    P = 10.0 ** r.uniform(-8, 6, n)
    x0 = np.concatenate([r.uniform(0, 1, n // 2), 1.0 - 10.0 ** r.uniform(-12, 0, n - n // 2)])
    return a, b, P, x0


def _update(a, b, P, x0, n_H=1.0):
    """One call with sigma = chat = m_H = 1, so a = n_H dt and b = alpha_B n_H dt."""
    dt = a / n_H
    alpha_B = b / (n_H * dt)
    rho = jnp.full(a.shape, n_H)
    R = jnp.zeros((1, 4) + a.shape).at[0, 0].set(P * n_H).at[0, 1].set(0.5 * P * n_H)
    R1, x1 = ionize_implicit(R, jnp.asarray(x0), rho, jnp.asarray(dt), chat=1.0,
                             sigma=1.0, alpha_B=jnp.asarray(alpha_B), m_H=1.0)
    return np.asarray(R1), np.asarray(x1)


def test_it_solves_its_equation_and_conserves_photons():
    """The balance, read off the outputs themselves: photons absorbed (from
    N1, exact) are the ionizations plus the recombinations (from x1). Not
    through z = 1 - x1, which cannot carry a neutral fraction of 1e-15."""
    a, b, P, x0 = _cells()
    R1, x1 = _update(a, b, P, x0)
    absorbed = P - R1[0, 0]                         # per hydrogen (n_H = 1)
    spent = x1 - x0 + b * x1 ** 2
    scale = np.maximum.reduce([np.abs(absorbed), np.abs(spent), np.abs(x0),
                               np.full_like(x0, 1e-300)])
    assert np.max(np.abs(absorbed - spent) / scale) < 1e-9
    assert np.all((x1 >= 0.0) & (x1 <= 1.0))
    assert np.all(R1[0, 0] >= 0.0) and np.all(R1[0, 0] <= P * (1 + 1e-14))
    # the flux keeps the photon density's direction and its |F|/N
    np.testing.assert_allclose(R1[0, 1], 0.5 * R1[0, 0], rtol=1e-12, atol=1e-300)


def test_an_opaque_photon_rich_cell_does_not_leak():
    """a ~ 1e9 and 1e5 photons per hydrogen: the root sits at z ~ 1e-15, and
    the cell must take exactly the last neutrals and the recombinations -- a
    few tenths of a photon per hydrogen of the 7e5 on offer. A solve that
    stopped at z ~ 1e-12 (linear bisection, 40 steps) absorbed a thousand
    times too much here, and the balance was off by 100%."""
    a, b, P, x0 = (np.array([1e9]), np.array([0.03]), np.array([7e5]), np.array([0.41]))
    R1, x1 = _update(a, b, P, x0)
    absorbed = P - R1[0, 0]
    np.testing.assert_allclose(absorbed, x1 - x0 + b * x1 ** 2, rtol=1e-9)
    assert absorbed[0] < 1.0                        # of 7e5 photons per H


def test_a_steady_supply_reaches_the_ionization_equilibrium():
    """Photons topped up to P each step: the fixed point is where the
    absorptions balance recombinations, P a z / (1 + a z) = b (1 - z)^2."""
    a = np.array([1e2, 1e5, 1e8]); b = np.array([1e-2, 1.0, 1e2]); P = np.array([1.0, 1.0, 1e3])
    x = np.zeros(3)
    for _ in range(400):
        _, x = _update(a, b, P, x)
    R1, x1 = _update(a, b, P, x)
    np.testing.assert_allclose(x1, x, rtol=1e-12)            # a fixed point
    np.testing.assert_allclose(P - R1[0, 0], b * x1 ** 2, rtol=1e-8)   # absorbed = recombined


def test_the_step_can_be_cut_in_pieces():
    """Half the photons twice over half the time lands where the whole does,
    to first order in a step that is long against the recombination time."""
    a, b, P, x0 = (np.array([1e6]), np.array([10.0]), np.array([2.0]), np.array([0.3]))
    R1, x1 = _update(a, b, P, x0)
    Rh, xh = _update(a / 2, b / 2, P / 2, x0)
    Rh, xh = _update(a / 2, b / 2, P / 2 + Rh[0, 0], xh)
    assert abs(xh[0] - x1[0]) < 0.05 * x1[0]


def test_a_dark_neutral_cell_stays_exactly_neutral():
    """No photons and x0 = 0: z = 1 is the root from the start, and the solve
    keeps it. It used to throw z to the bracket's middle and come back only
    to 1 - 1.1e-16, an x of 1.1e-16 in every dark cell."""
    n = 64
    a = np.full(n, 3.0)
    b = 10.0 ** np.linspace(-6, 6, n)
    R1, x1 = _update(a, b, np.zeros(n), np.zeros(n))
    np.testing.assert_array_equal(x1, 0.0)
    np.testing.assert_array_equal(R1, 0.0)


def test_nan_photons_make_a_nan_fraction_not_an_ionised_cell():
    """A NaN photon density used to bisect z down to the bracket floor: x = 1,
    a fully ionised cell from garbage, and the NaN gone from what the hydro
    reads. The NaN must reach x."""
    a, b, P, x0 = (np.full(4, 3.0), np.full(4, 1e-2), np.array([1.0, np.nan, np.inf, 0.0]),
                   np.zeros(4))
    R1, x1 = _update(a, b, P, x0)
    assert np.isnan(x1[1]) and np.isnan(x1[2])
    assert np.isfinite(x1[0]) and x1[3] == 0.0
