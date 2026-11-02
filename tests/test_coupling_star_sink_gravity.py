"""Stars and LUNA's sinks attract each other in the coupled bridge (worker's
`_pair_accel`): the gas field is solved from the gas alone, the star field kicks
only the gas, sink-sink gravity is off in coupled runs and PeTar does not hold
the sinks, so the pairs are added in the two bridge kicks. Checked against the
softened two-body force, for momentum conservation over a kick pair, and for
the switch."""
import os
from dataclasses import replace

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tests.test_coupling_sink_to_star import _inject_sink, _worker  # noqa: E402
from luna.species.sinks import Sinks  # noqa: E402

G = 1.0


def _uniform_gas(w):
    """Uniform density at rest: its periodic field is zero, only the pairs act."""
    U = np.zeros(w.state.U.shape)
    U[0] = 1.0
    U[4] = 1.0
    w.state.U = jnp.asarray(U)


def _setup(on=True):
    w = _worker()
    w.cfg = replace(w.cfg, star_sink_gravity=on)
    _uniform_gas(w)
    _inject_sink(w, [10.0, 10.0, 10.0], mass=50.0)
    tags = [w.new_particle(x, 10.0, 10.0) for x in (13.0, 6.0)]
    w.set_particle_mass(tags, [2.0, 3.0])
    return w, tags


def _eps(w):
    return float(getattr(w.cfg, "sink_softening", 0.0) or 0.0) or 2.0 * float(w.dx)


def test_a_star_feels_the_sink_as_a_softened_point_mass():
    w, tags = _setup()
    v0 = np.array(w.stars.arrays["velocity"])
    dt = 0.01
    w.apply_gas_gravity_on_particles(dt)
    dv = np.array(w.stars.arrays["velocity"]) - v0
    eps2 = _eps(w) ** 2
    for x, slot in zip((13.0, 6.0), w.stars.slots_for(tags)):
        d = 10.0 - x
        want = dt * G * 50.0 * d / (d * d + eps2) ** 1.5
        np.testing.assert_allclose(dv[slot], [want, 0.0, 0.0], rtol=1e-10, atol=1e-14)


def test_a_kick_pair_conserves_star_plus_sink_momentum():
    w, tags = _setup()
    part = w.state.species[Sinks.name]
    p0_sink = float(jax.device_get(part.mass[0])) * np.array(jax.device_get(part.p[0]))
    m = np.array(w.stars.arrays["mass"])
    p0_star = np.sum(m[:, None] * np.array(w.stars.arrays["velocity"]), axis=0)
    dt = 0.01
    w.apply_gas_gravity_on_particles(dt)
    w.apply_particle_gravity_on_gas(dt)
    part = w.state.species[Sinks.name]
    p1_sink = float(jax.device_get(part.mass[0])) * np.array(jax.device_get(part.p[0]))
    p1_star = np.sum(m[:, None] * np.array(w.stars.arrays["velocity"]), axis=0)
    d_star, d_sink = p1_star - p0_star, p1_sink - p0_sink
    assert np.abs(d_star).max() > 1e-4                 # something happened
    np.testing.assert_allclose(d_star + d_sink, 0.0, atol=1e-12 * np.abs(d_star).max())


def test_switched_off_the_pairs_are_gone():
    w, tags = _setup(on=False)
    v0 = np.array(w.stars.arrays["velocity"])
    w.apply_gas_gravity_on_particles(0.01)
    w.apply_particle_gravity_on_gas(0.01)
    assert np.abs(np.array(w.stars.arrays["velocity"]) - v0).max() < 1e-12
    assert float(np.abs(np.array(jax.device_get(w.state.species[Sinks.name].p))).max()) == 0.0
