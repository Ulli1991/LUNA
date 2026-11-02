"""The M1 radiation solver against its two analytic limits, plus the
properties that make it multigroup and couplable.

The module had no test at all: its only caller was a script in one person's
checkout, and the audit had it down as dead code. The two physics checks
below are the ones that script ran (free-streaming front position, diffusion
limit growth); the rest are what wiring it into the step requires.
"""

import numpy as np
import pytest


import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402

from luna.radiation.m1 import (m1_advance, m1_damp, m1_exchange,  # noqa: E402
                               m1_step, m1_substeps)
from tests._helpers import check                              # noqa: E402

pytestmark = pytest.mark.fast

CHAT = 1.0
CFL = 0.4


def _beam_1d(n, g=1):
    """A sharp front at x = 0 moving right at c: E = 1 behind it, F = c E."""
    E = np.zeros((g, n))
    E[:, : n // 8] = 1.0 / g          # split evenly over groups
    R = np.zeros((g, 2, n))
    R[:, 0] = E
    R[:, 1] = CHAT * E
    return jnp.asarray(R)


def _gaussian_1d(n, sigma, g=1):
    x = (np.arange(n) + 0.5) / n - 0.5
    E = np.exp(-0.5 * (x / sigma) ** 2)[None, :] / g
    R = np.zeros((g, 2, n))
    R[:, 0] = E
    return jnp.asarray(R), x


def test_free_streaming_front_moves_at_the_reduced_light_speed():
    """kappa = 0: the front must sit at x = c_hat t, within two cells."""
    n, steps = 256, 80
    dx = 1.0 / n
    dt = CFL * dx / CHAT
    R = _beam_1d(n)
    step = jax.jit(lambda r: m1_step(r, dt, dx, chat=CHAT, boundary=1, order=2))
    for _ in range(steps):
        R = step(R)
    E = np.asarray(R[0, 0])
    front = np.searchsorted(-E, -0.5 * E.max())          # half-max crossing
    expect = n // 8 + CHAT * dt * steps / dx
    assert abs(front - expect) <= 2.0, f"front at {front}, expected {expect:.1f}"
    # exact inflow budget: the zero-gradient ghost repeats (1, c), so the left face carries c, the right is dark.
    # Measured 7.99e-13 on CPU and A100, the same without m1_floor: the round-off of 160 stage updates
    budget = (n // 8) * dx + CHAT * dt * steps
    check("m1 free streaming |sum E dx - inflow| / inflow", abs(E.sum() * dx - budget) / budget, 1.5e-12)


def test_the_diffusion_limit_spreads_at_the_analytic_rate():
    """Optically thick, kappa large: the M1 system relaxes to diffusion with
    D = c / (3 kappa), so sigma^2 grows as sigma0^2 + 2 D t."""
    n, sigma0 = 256, 0.05
    dx = 1.0 / n
    kappa = 300.0
    D = CHAT / (3.0 * kappa)
    R, x = _gaussian_1d(n, sigma0)
    dt = CFL * dx / CHAT
    steps = 400

    @jax.jit
    def step(r):
        r = m1_step(r, dt, dx, chat=CHAT, boundary=1, order=2)
        return m1_damp(r, dt, chat=CHAT, kappa=kappa)

    for _ in range(steps):
        R = step(R)
    E = np.asarray(R[0, 0])
    E0 = np.asarray(_gaussian_1d(n, sigma0)[0][0, 0])
    # damping touches F only and the tails at the edges are ~1e-14: sum E is fixed (measured 7.8e-15)
    check("m1 diffusion |d sum E| / sum E", abs(E.sum() / E0.sum() - 1.0), 1e-13)
    var = np.sum(E * x ** 2) / np.sum(E)
    expect = sigma0 ** 2 + 2.0 * D * dt * steps
    assert abs(var - expect) / expect < 0.05, (
        f"sigma^2 = {var:.5e}, diffusion predicts {expect:.5e}")


def test_groups_are_independent_and_reduce_to_the_grey_case():
    """Four groups each carrying a quarter of the energy, with the same
    opacity, must evolve exactly like one group carrying all of it. This is
    what makes the solver multigroup-ready rather than grey with an extra
    axis: if the group loop leaked, the sum would drift."""
    n, steps = 128, 40
    dx = 1.0 / n
    dt = CFL * dx / CHAT
    one, four = _beam_1d(n, g=1), _beam_1d(n, g=4)
    step = jax.jit(lambda r: m1_step(r, dt, dx, chat=CHAT, boundary=1, order=2))
    for _ in range(steps):
        one, four = step(one), step(four)
    total = np.asarray(jnp.sum(four[:, 0], axis=0))
    assert np.allclose(total, np.asarray(one[0, 0]), rtol=1e-12, atol=1e-14)


def test_groups_with_different_opacities_separate():
    """And they must NOT be interchangeable when the opacities differ: a
    transparent group keeps its flux, an opaque one loses it."""
    n = 64
    dx = 1.0 / n
    dt = CFL * dx / CHAT
    R = _beam_1d(n, g=2)
    R = m1_damp(R, dt * 50, chat=CHAT, kappa=jnp.array([0.0, 1000.0]))
    assert float(jnp.max(jnp.abs(R[0, 1]))) > 0.1        # transparent: flux kept
    assert float(jnp.max(jnp.abs(R[1, 1]))) < 1e-6       # opaque: flux gone


def test_the_matter_exchange_conserves_energy_exactly():
    """Absorption and emission move energy between radiation and gas and
    nowhere else: the sum is unchanged to round-off, cell by cell, at any
    optical depth and any step size. This is the property the coupled stage
    rests on."""
    rng = np.random.default_rng(0)
    R = jnp.asarray(rng.uniform(0.5, 1.5, size=(3, 2, 32)))
    e_gas = jnp.asarray(rng.uniform(1.0, 2.0, size=(32,)))
    kappa = jnp.asarray(rng.uniform(0.0, 500.0, size=(3, 32)))
    source = jnp.asarray(rng.uniform(0.0, 1.0, size=(3, 32)))
    total0 = jnp.sum(R[:, 0]) + jnp.sum(e_gas)
    for dt in (1e-6, 1e-3, 1.0, 1e3):        # optically thin to very thick
        R2, e2 = m1_exchange(R, e_gas, dt, chat=CHAT, kappa_a=kappa, source=source)
        total = jnp.sum(R2[:, 0]) + jnp.sum(e2)
        rel = float(abs(total - total0) / total0)
        assert rel < 1e-14, f"dt={dt}: energy changed by {rel:.2e}"


def test_the_exchange_relaxes_towards_the_emission_source():
    """With a fixed source the exact update must approach it monotonically,
    and reach it in the optically thick limit."""
    R = jnp.zeros((1, 2, 4)).at[:, 0].set(0.0)
    src = jnp.full((1, 4), 2.0)
    short, _ = m1_exchange(R, jnp.zeros(4), 1e-3, chat=CHAT, kappa_a=1.0, source=src)
    long_, _ = m1_exchange(R, jnp.zeros(4), 1e3, chat=CHAT, kappa_a=1.0, source=src)
    assert 0.0 < float(short[0, 0, 0]) < 2.0
    assert abs(float(long_[0, 0, 0]) - 2.0) < 1e-12


def test_the_substep_count_is_the_multi_dimensional_bound():
    """The transport is unsplit, so the stable step divides by ndim. A count
    taken from the 1D bound is unstable in 3D; this is a Python int so the
    scan length stays static."""
    n1 = m1_substeps(0.1, 0.01, chat=1.0, ndim=1, cfl=0.4)
    n3 = m1_substeps(0.1, 0.01, chat=1.0, ndim=3, cfl=0.4)
    assert isinstance(n1, int) and n1 == 25
    # exactly three times, not 76: the ratio is 75.00000000000001 in floating
    # point, and a ceiling with no slack charges a whole extra substep for it
    assert n3 == 75
    # and the slack must not swallow a real fraction of a step
    assert m1_substeps(0.1001, 0.01, chat=1.0, ndim=3, cfl=0.4) == 76
    assert m1_substeps(1e-12, 0.01, chat=1.0, ndim=3) == 1     # never zero


def test_a_grey_state_needs_no_group_axis():
    """The shape the solver was written with, and the shape its existing
    validation script passes: (1+ndim, N...), no group axis. It must behave
    exactly like a single group and come back in the same shape."""
    n = 64
    dx = 1.0 / n
    dt = CFL * dx / CHAT
    grey = np.asarray(_beam_1d(n, g=1))[0]               # drop the group axis
    grouped = jnp.asarray(grey[None])
    a = m1_step(jnp.asarray(grey), dt, dx, chat=CHAT, boundary=1, order=2)
    b = m1_step(grouped, dt, dx, chat=CHAT, boundary=1, order=2)
    assert a.shape == grey.shape and b.shape == grouped.shape
    assert np.array_equal(np.asarray(a), np.asarray(b[0]))
    c = m1_damp(jnp.asarray(grey), dt, chat=CHAT, kappa=3.0)
    assert c.shape == grey.shape
    d = m1_advance(jnp.asarray(grey), 4 * dt, dx, chat=CHAT, kappa=1.0,
                   boundary=1, n_sub=4)
    assert d.shape == grey.shape
    e, gas = m1_exchange(jnp.asarray(grey), jnp.zeros(n), dt, chat=CHAT,
                         kappa_a=1.0, source=0.0)
    assert e.shape == grey.shape and gas.shape == (n,)


def test_advance_is_differentiable_end_to_end():
    """The radiation advance has to survive reverse mode, because it is about
    to sit inside the differentiated step next to the hydro."""
    n = 32
    dx = 1.0 / n
    R = _beam_1d(n)

    def loss(R):
        out = m1_advance(R, 0.5 * dx / CHAT, dx, chat=CHAT, kappa=1.0,
                         boundary=1, n_sub=4)
        return jnp.sum(out[:, 0] ** 2)

    g = jax.jit(jax.grad(loss))(R)
    bad = int(jnp.sum(~jnp.isfinite(g)))
    # this is what caught the |F| square root: the forward pass was finite
    # everywhere and the gradient was NaN in every cell the beam had not
    # reached, because d sqrt(F.F)/dF is infinite at F = 0
    assert bad == 0, f"{bad}/{g.size} gradient entries are not finite"
    assert float(jnp.max(jnp.abs(g))) > 0.0


def test_the_division_free_closure_is_levermore_to_round_off():
    """chi = (5 - 2s)/3 and one reciprocal for f and 1/|F| (the guards kept) against the textbook form."""
    from luna.radiation.m1 import E_FLOOR, _closure
    rng = np.random.default_rng(11)
    chat = 7.0
    E = np.concatenate([np.exp(rng.normal(scale=20.0, size=4000)), [0.0, 1e-40, 1.0, 1.0]])
    F = chat * E[None] * rng.uniform(-0.6, 0.6, size=(3, E.size))
    F[:, -2] = 0.0                                          # |F| = 0: the isotropic fallback
    F[:, -1] = [chat, 0.0, 0.0]                             # f = 1: free streaming
    s, inv = (np.asarray(v) for v in _closure(jnp.asarray(E), [jnp.asarray(c) for c in F], chat))
    Fmag = np.sqrt((F ** 2).sum(axis=0) + E_FLOOR ** 2)
    f = np.clip(Fmag / (chat * np.maximum(E, E_FLOOR)), 0.0, 1.0 - 1e-10)
    chi = (3.0 + 4.0 * f * f) / (5.0 + 2.0 * np.sqrt(4.0 - 3.0 * f * f))
    np.testing.assert_allclose((5.0 - 2.0 * s) / 3.0, chi, rtol=1e-15, atol=0)
    np.testing.assert_allclose(inv, 1.0 / np.maximum(Fmag, E_FLOOR), rtol=1e-15, atol=0)
    assert abs((5.0 - 2.0 * s[-2]) / 3.0 - 1.0 / 3.0) < 1e-16 and abs((5.0 - 2.0 * s[-1]) / 3.0 - 1.0) < 1e-9
