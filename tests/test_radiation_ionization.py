"""Hydrogen ionization against what it is supposed to reproduce.

The end of this file is the one that counts: an ionizing source in a uniform
medium must settle at the Strömgren radius, the radius at which the
recombinations inside balance the photons emitted. That number
(0.314 pc for the STARBENCH parameters) follows from the photon bookkeeping
alone, so it tests the transport, the absorption and the recombination
together, and it is the starting point of the D-type expansion the benchmark
is about.
"""

import numpy as np
import pytest


import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402

from luna.radiation.ionization import (inject, ionize,        # noqa: E402
                                       ionized_radius, source_weights,
                                       starbench_units, transport_and_ionize,
                                       two_temperature)
from tests._helpers import check                              # noqa: E402

pytestmark = pytest.mark.fast


def test_recombination_is_the_exact_solution():
    """With no photons, dx/dt = -alpha n_H x^2 has a closed form, and the
    update must be it -- not a forward Euler step, because inside an H II
    region the recombination time is far below the hydrodynamic step."""
    n_H, alpha = 1.0e3, 2.5e-13
    rho, m_H = 1.0, 1.0 / n_H
    R = jnp.zeros((1, 4, 3))
    x0 = 0.8
    for dt in (1.0e6, 1.0e9, 1.0e12):
        _, x = ionize(R, jnp.full((3,), x0), jnp.full((3,), rho), dt,
                      chat=1.0, sigma=0.0, alpha_B=alpha, m_H=m_H)
        exact = x0 / (1.0 + alpha * n_H * x0 * dt)
        assert abs(float(x[0]) - exact) < 1e-14 * max(exact, 1e-30)


def test_every_absorbed_photon_is_one_ionization():
    """Photon conservation, cell by cell: the drop in photon number density
    equals the rise in ionized hydrogen. This is what keeps the front in the
    right place when a cell is optically thick to the whole step -- at
    benchmark resolution tau per cell is of order 1e3."""
    rng = np.random.default_rng(0)
    n_H, m_H = 1.0e3, 1.0e-3
    rho = jnp.full((8,), n_H * m_H)
    x0 = jnp.asarray(rng.uniform(0.0, 0.9, size=8))
    R = jnp.zeros((1, 4, 8)).at[0, 0].set(jnp.asarray(rng.uniform(1.0, 50.0, size=8)))
    for dt, sigma, rtol in ((1e-6, 1e-4, 1e-7), (1.0, 1e-2, 1e-9),
                           (1e3, 1.0, 1e-13)):
        R2, x = ionize(R, x0, rho, dt, chat=1.0, sigma=sigma, alpha_B=0.0,
                       m_H=m_H)
        lost = np.asarray(R[0, 0] - R2[0, 0])
        made = np.asarray((x - x0) * n_H)
        # Both sides are differences of nearly equal numbers -- photon
        # densities on one side, ionization fractions times n_H on the other
        # -- so the accuracy of the comparison is set by eps times the large
        # quantities, not by eps times the small answer. That bound is
        # written out rather than replaced by a relative tolerance chosen to
        # pass.
        eps = np.finfo(np.float64).eps
        bound = 8.0 * eps * (np.asarray(R[0, 0]) + n_H * np.asarray(x0))
        assert np.all(np.abs(lost - made) <= bound), (dt, lost - made, bound)
        assert np.allclose(lost, made, rtol=max(rtol, 1e-6))
        assert float(jnp.max(x)) <= 1.0 and float(jnp.min(R2[0, 0])) >= 0.0
    # and the flux is attenuated by exactly the factor the density was
    R3, _ = ionize(R.at[0, 1].set(R[0, 0] * 0.5), x0, rho, 1.0, chat=1.0,
                   sigma=1e-2, alpha_B=0.0, m_H=m_H)
    assert np.allclose(np.asarray(R3[0, 1] / R3[0, 0]), 0.5, rtol=1e-12)


def test_a_cell_relaxes_to_photoionization_equilibrium():
    """Held in a fixed photon bath, a cell must reach Gamma (1-x) =
    alpha n_H x^2 -- the balance the Strömgren radius is built from -- and
    reach it to first order in the step.

    The update absorbs N (1 - exp(-Gamma dt)) rather than N Gamma dt, which
    is right (the cell's photons deplete within the step) and means the
    discrete fixed point sits below the continuous one by O(Gamma dt). So
    the gate is the convergence, not one number: halving dt must halve the
    error. Measured at dt = 0.2 / 0.1 / 0.05 (2026-09-21).
    """
    n_H, m_H, alpha, sigma, chat = 1.0e3, 1.0e-3, 2.7e-4, 1.0e-3, 1.0
    N = 5.0
    rho = jnp.full((1,), n_H * m_H)
    gamma_rate = chat * sigma * N
    # the root of Gamma (1-x) = alpha n_H x^2
    a, b, c = alpha * n_H, gamma_rate, -gamma_rate
    x_exact = (-b + (b * b - 4 * a * c) ** 0.5) / (2 * a)

    @jax.jit
    def equilibrium(dt, steps):
        def body(_, x):
            R = jnp.zeros((1, 4, 1)).at[0, 0].set(N)       # the bath refills
            return ionize(R, x, rho, dt, chat=chat, sigma=sigma,
                          alpha_B=alpha, m_H=m_H)[1]
        return jax.lax.fori_loop(0, steps, body, jnp.zeros(1))[0]

    errs = []
    for dt in (0.2, 0.1, 0.05):
        x = float(equilibrium(dt, int(4000 / dt)))         # 4000 time units
        errs.append(abs(x - x_exact))
    assert errs[0] > errs[1] > errs[2], errs
    assert errs[2] / errs[1] < 0.6, f"not first order: {errs}"
    assert errs[2] < 0.02 * x_exact, f"x = {x}, exact {x_exact}"


def test_the_source_injects_exactly_the_photon_rate():
    n = 16
    dx = (1.0 / n,) * 3
    w = source_weights((n, n, n), dx, (0.5, 0.5, 0.5), 2.5 / n)
    cell = dx[0] * dx[1] * dx[2]
    assert abs(float(jnp.sum(w) * cell) - 1.0) < 1e-12
    R = jnp.zeros((1, 4, n, n, n))
    R = inject(R, w, 7.0, 0.5)
    assert abs(float(jnp.sum(R[0, 0]) * cell) - 3.5) < 1e-12


def test_the_prescribed_temperature_moves_only_the_internal_energy():
    rng = np.random.default_rng(1)
    n = 6
    rho = rng.uniform(0.5, 1.5, size=(n, n, n))
    mom = rng.normal(0.0, 0.2, size=(3, n, n, n))
    U = jnp.asarray(np.concatenate([rho[None], mom,
                                    (np.ones((1, n, n, n)))], axis=0))
    x = jnp.asarray(rng.uniform(0.0, 1.0, size=(n, n, n)))
    cs2_n, cs2_i, gamma = 0.83, 165.0, 5.0 / 3.0
    U2 = two_temperature(U, x, cs2_neutral=cs2_n, cs2_ionized=cs2_i, gamma=gamma)
    assert np.array_equal(np.asarray(U2[:4]), np.asarray(U[:4]))
    kin = 0.5 * jnp.sum(U2[1:4] ** 2, axis=0) / U2[0]
    cs2 = (gamma - 1.0) * (U2[4] - kin) / U2[0]
    assert np.allclose(np.asarray(cs2), cs2_n + (cs2_i - cs2_n) * np.asarray(x),
                       rtol=1e-12)


def test_the_benchmark_numbers_are_self_consistent():
    """The Strömgren radius that comes out of the converted units has to be
    the 0.314 pc the paper quotes; if a conversion is wrong, this is where it
    shows."""
    u = starbench_units("early")
    assert abs(u["R_St"] - 0.314) < 0.001, u["R_St"]
    assert abs(u["cs2_ionized"] ** 0.5 * 3.0856775814913673e18 / 3.1556952e13
               / 1e5 - 12.85) < 0.01          # c_i back in km/s
    assert abs(u["t_end"] - 0.141) < 1e-12
    # the two classical solutions must start at R_St and the HI one must lead
    from luna.radiation.ionization import (hosokawa_inutsuka_radius,
                                           spitzer_radius)
    c_i = u["cs2_ionized"] ** 0.5
    assert abs(spitzer_radius(0.0, u["R_St"], c_i) - u["R_St"]) < 1e-12
    for t in (0.01, 0.05, 0.141):
        assert (hosokawa_inutsuka_radius(t, u["R_St"], c_i)
                > spitzer_radius(t, u["R_St"], c_i))


STROMGREN_HELD = 0.15     # measured 0.170 (CPU, A100): 83 % recombined (or left) by the end


def _stromgren_run(n=48, box=1.5, chat_over_ci=100.0, steps=400):
    """The radiation stage alone on a fixed uniform medium: no hydrodynamics,
    so what is measured is only the photon bookkeeping."""
    u = starbench_units("early")
    dx = box / n
    dxv = (dx, dx, dx)
    c_i = u["cs2_ionized"] ** 0.5
    chat = chat_over_ci * c_i
    rho = jnp.full((n, n, n), u["rho0"])
    x = jnp.zeros((n, n, n))
    R = jnp.zeros((1, 4, n, n, n))
    w = source_weights((n, n, n), dxv, (0.5 * box,) * 3, 2.0 * dx)
    h = 0.4 * dx / (3.0 * chat)                    # one transport substep

    @jax.jit
    def step(carry):
        R, x = carry
        return transport_and_ionize(
            R, x, rho, h, dx, n_sub=1, chat=chat, sigma=u["sigma"],
            alpha_B=u["alpha_B"], m_H=u["m_H"], weight=w,
            rate=u["photon_rate"], boundary=1, order=2)

    for _ in range(steps):
        R, x = step((R, x))
    # photons stored plus ionisations held, over those injected: <= 1, the rest recombined or escaped
    held = float((jnp.sum(R[0, 0]) + jnp.sum(rho / u["m_H"] * x)) * dx ** 3)
    return u, float(ionized_radius(x, dxv)), h * steps, held / (u["photon_rate"] * h * steps)


def test_the_ionized_sphere_settles_at_the_stromgren_radius():
    """The one that matters: photons in, recombinations out, and the front
    stops where they balance. 0.314 pc for these parameters, measured as the
    volume-equivalent radius of the cells with x > 0.5 on a grid whose cells
    are 0.031 pc, so a cell is a tenth of the answer."""
    u, R_num, t_run, held = _stromgren_run()
    # not conserved by design: recombinations (not exposed by transport_and_ionize) take photons out
    check("stromgren (stored + ionised) / injected", held, 1.0 + 1e-12)
    check("stromgren (stored + ionised) / injected", held, STROMGREN_HELD, lower=True)
    rel = (R_num - u["R_St"]) / u["R_St"]
    # Measured 2026-09-21, and it converges: +5.6% at 48^3 (10 cells per
    # R_St), +2.4% at 96^3, +1.6% at 144^3. The gate is the 48^3 number with
    # a little room, not a tolerance chosen to pass.
    assert abs(rel) < 0.08, (f"R = {R_num:.4f} pc after {t_run:.2e} Myr, "
                             f"Stromgren {u['R_St']:.4f} pc ({rel * 100:+.1f}%)")
