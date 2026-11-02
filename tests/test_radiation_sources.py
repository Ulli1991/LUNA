"""Many ionizing sources on the grid.

A benchmark has one star at the box centre and `source_weights` is enough for
it. A collapsing cloud has hundreds, each with a rate stellar evolution hands
over, appearing and disappearing as the run goes. `deposit_sources` is that
case, and the things worth checking are the ones that would be silently wrong
rather than loud: does it put the right NUMBER of photons in, does it put them
in the right PLACE, do empty slots contribute nothing, and does the shape stay
fixed when a star forms.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402

from luna.radiation.ionization import (deposit_sources, inject,  # noqa: E402
                                       source_weights)

pytestmark = pytest.mark.fast

SHAPE = (16, 16, 16)
DX = (1.0 / 16, 1.0 / 16, 1.0 / 16)
DV = DX[0] * DX[1] * DX[2]


def test_it_puts_in_exactly_the_photons_it_was_given():
    """sum(field) * dV == sum of the active rates. If this drifts, the run
    quietly gains or loses ionizing photons and the front is wrong."""
    rng = np.random.default_rng(0)
    pos = jnp.asarray(rng.uniform(0.2, 0.8, size=(12, 3)))
    rates = jnp.asarray(rng.uniform(0.5, 4.0, size=12))
    for boundary in (0, 1):
        f = deposit_sources(SHAPE, DX, pos, rates, boundary=boundary)
        assert abs(float(jnp.sum(f)) * DV - float(jnp.sum(rates))) < 1e-12


def test_inactive_slots_contribute_nothing():
    """The capacity is fixed and mostly empty; empty slots must be free."""
    rng = np.random.default_rng(1)
    pos = jnp.asarray(rng.uniform(0.2, 0.8, size=(8, 3)))
    rates = jnp.asarray(rng.uniform(1.0, 3.0, size=8))
    active = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    got = deposit_sources(SHAPE, DX, pos, rates, active=active)
    want = deposit_sources(SHAPE, DX, pos[:3], rates[:3])
    assert float(jnp.max(jnp.abs(got - want))) < 1e-14


def test_the_photons_land_at_the_star():
    """A single source, and the deposited mass sits in the cells around it --
    not somewhere else by half a cell, which a wrong index offset would do."""
    pos = jnp.asarray([[0.5 + 0.5 * DX[0], 0.5 + 0.5 * DX[1],
                        0.5 + 0.5 * DX[2]]])          # a cell centre
    f = np.asarray(deposit_sources(SHAPE, DX, pos, jnp.array([1.0])))
    i, j, k = np.unravel_index(np.argmax(f), f.shape)
    centre = [(c + 0.5) * d for c, d in zip((i, j, k), DX)]
    for c, p in zip(centre, np.asarray(pos)[0]):
        assert abs(c - p) <= DX[0] * 1.001


def test_one_source_agrees_with_the_benchmark_weights_on_the_total():
    """The single-source limit. `source_weights` spreads over a top-hat and
    this over eight cells, so the FIELDS differ by construction -- but both
    are normalised, and a benchmark run must inject the same number of
    photons whichever it uses."""
    centre = (0.5, 0.5, 0.5)
    w = source_weights(SHAPE, DX, centre, 2.0 * DX[0])
    f = deposit_sources(SHAPE, DX, jnp.asarray([centre]), jnp.array([3.0]))
    assert abs(float(jnp.sum(w)) * DV - 1.0) < 1e-12
    assert abs(float(jnp.sum(f)) * DV - 3.0) < 1e-12


def test_it_drops_into_inject():
    """`inject` is unchanged: a rate-density field goes in with rate 1."""
    R = jnp.zeros((1, 4) + SHAPE)
    pos = jnp.asarray([[0.3, 0.5, 0.5], [0.7, 0.5, 0.5]])
    f = deposit_sources(SHAPE, DX, pos, jnp.array([2.0, 5.0]))
    out = inject(R, f, 1.0, 0.1)
    assert abs(float(jnp.sum(out[0, 0])) * DV - 0.7) < 1e-12   # (2+5)*0.1


def test_a_star_forming_does_not_change_the_shape():
    """Fixed capacity plus a mask: turning a slot on must not retrace. If it
    does, every star formation event recompiles the step."""
    f = jax.jit(lambda p, r, a: deposit_sources(SHAPE, DX, p, r, active=a))
    rng = np.random.default_rng(2)
    pos = jnp.asarray(rng.uniform(0.2, 0.8, size=(16, 3)))
    rates = jnp.asarray(rng.uniform(1.0, 2.0, size=16))

    before = f._cache_size()
    for n_on in (1, 4, 9, 16):
        a = jnp.asarray([1.0] * n_on + [0.0] * (16 - n_on))
        out = f(pos, rates, a)
        assert abs(float(jnp.sum(out)) * DV
                   - float(jnp.sum(rates[:n_on]))) < 1e-12
    assert f._cache_size() == before + 1, "recompiled when a star turned on"
