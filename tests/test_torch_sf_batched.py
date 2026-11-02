"""Torch's star formation, with LUNA's sinks fetched in one call.

torch_sf.queue_stars / make_stars_from_sinks ask the hydro code for each
sink's mass, position, velocity and mean_cs one sink and one attribute at a
time: 58% of a coupled 256^3 bridge step with 26 sinks. Our Torch branch
fetches them all at once when the hydro code offers `get_sink_state`.

The stars must come out the same, bit for bit: same IMF queues, same spawns,
same positions and velocities, same sink masses left behind. A stand-in hydro
code with the call surface torch_sf uses runs both paths from the same seed.

Needs amuse and a Torch whose torch_sf has the batched read (`_sink_state`, Torch branch
luna-hydro-code) at $TORCH_SRC; without it every test here fails (never skips): a Torch without the batched read
runs the per-sink path twice and the bitwise comparison would compare a path with itself.
Stock Torch (main 744bcae) has no batched read.
"""
import copy
import os
import sys

import numpy as np
import pytest

from amuse.units import units
from amuse.units.quantities import as_vector_quantity

TORCH_SRC = os.environ.get("TORCH_SRC", "")
NEED = ("a Torch clone with the batched sink read (torch_sf._sink_state, branch luna-hydro-code) at $TORCH_SRC")


@pytest.fixture(scope="module")
def torch_sf():
    """Torch's torch_sf with the batched read, else the test fails naming what is missing."""
    if not os.path.isfile(os.path.join(TORCH_SRC, "torch_sf.py")):
        pytest.fail(f"no Torch clone at $TORCH_SRC={TORCH_SRC!r}; needs {NEED}")
    sys.path.insert(0, TORCH_SRC)
    import torch_sf
    if not hasattr(torch_sf, "_sink_state"):
        pytest.fail(f"Torch at {TORCH_SRC} has no batched sink read; needs {NEED}")
    return torch_sf


class Hydro:
    """The calls torch_sf makes, over a dict of sinks; counts every call."""

    def __init__(self, sinks):
        self.sinks = copy.deepcopy(sinks)          # tag -> [mass, pos, vel, cs]
        self.stars = {}                            # tag -> dict
        self.ptr, self.calls, self.reads, self.batched, self._next = "mass", 0, 0, 0, 1000

    def _c(self):
        self.calls += 1

    def _r(self):
        self.calls += 1
        self.reads += 1

    def set_particle_pointers(self, p): self._c(); self.ptr = p
    def get_number_of_particles(self):
        self._c(); return len(self.sinks) if self.ptr == "sink" else len(self.stars)
    def get_particle_tags(self, idx):
        self._c(); return np.array(sorted(self.sinks if self.ptr == "sink" else self.stars))

    def get_particle_mass(self, tag):
        self._r(); assert self.ptr == "sink"; return self.sinks[tag][0] | units.MSun
    def get_particle_position(self, tag):
        self._r(); return self.sinks[tag][1] | units.parsec
    def get_particle_velocity(self, tag):
        self._r(); return self.sinks[tag][2] | units.kms
    def get_sink_mean_cs(self, tag):
        self._r(); return self.sinks[tag][3] | units.kms

    def set_particle_mass(self, tag, m):
        self._c()
        if self.ptr == "sink":
            self.sinks[tag][0] = m.value_in(units.MSun)
        else:
            for t, mm in zip(np.atleast_1d(tag), np.atleast_1d(m.value_in(units.MSun))):
                self.stars[t]["mass"] = mm

    def add_particles(self, x, y, z):
        self._c(); assert self.ptr == "mass"
        tags = []
        for xi, yi, zi in zip(x, y, z):
            self._next += 1
            self.stars[self._next] = {"pos": [q.value_in(units.parsec) for q in (xi, yi, zi)]}
            tags.append(self._next)
        return np.array(tags)

    def set_particle_velocity(self, tag, vx, vy, vz):
        self._c()
        for t, a, b, c in zip(tag, vx, vy, vz):
            self.stars[t]["vel"] = [q.value_in(units.kms) for q in (a, b, c)]

    def set_particle_oldmass(self, tag, m): self._c()


class BatchedHydro(Hydro):
    def get_sink_state(self, tags):
        self._r()
        self.batched += 1
        rows = [self.sinks[t] for t in tags]
        col = lambda k, j, u: as_vector_quantity([r[k][j] | u for r in rows])   # noqa: E731
        return (as_vector_quantity([r[0] | units.MSun for r in rows]),
                col(1, 0, units.parsec), col(1, 1, units.parsec), col(1, 2, units.parsec),
                col(2, 0, units.kms), col(2, 1, units.kms), col(2, 2, units.kms),
                as_vector_quantity([r[3] | units.kms for r in rows]))


class State:
    def __init__(self):
        self.all_masses = {}
        self.user = {"with_lyc": False, "with_pe_heat": False}


SINKS = {3: [120.0, [1.0, 2.0, 3.0], [0.1, 0.2, -0.3], 0.4],
         7: [0.3, [4.0, 1.0, 2.0], [0.0, -0.5, 0.2], 0.3],
         9: [260.0, [2.5, 2.5, 0.5], [-0.2, 0.0, 0.1], 0.0]}
STEPS = 3                                   # bridge steps


def _run(torch_sf, cls):
    np.random.seed(5)
    h, s = cls(SINKS), State()
    for _ in range(STEPS):                  # three bridge steps, the sinks growing
        torch_sf.queue_stars(s, h, min_imf_mass=0.08 | units.MSun,
                             max_imf_mass=120.0 | units.MSun)
        torch_sf.make_stars_from_sinks(s, h, sink_rad=0.05 | units.parsec)
        h.ptr = "sink"
        for t in h.sinks:
            h.sinks[t][0] += 40.0
        h.ptr = "mass"
    return h, s


def test_the_batched_sinks_make_the_same_stars(torch_sf):
    """Same stars, sink masses and IMF queues bit for bit, with the batched read used in every call."""
    a, sa = _run(torch_sf, Hydro)
    b, sb = _run(torch_sf, BatchedHydro)
    assert a.batched == 0
    assert b.batched == 2 * STEPS and b.reads == b.batched, (b.batched, b.reads)   # else a path against itself
    assert len(a.stars) > 0
    assert sorted(a.stars) == sorted(b.stars)
    for t in a.stars:
        for k in ("mass", "pos", "vel"):
            np.testing.assert_array_equal(np.asarray(a.stars[t][k]), np.asarray(b.stars[t][k]))
    for t in SINKS:
        assert a.sinks[t][0] == b.sinks[t][0]
        np.testing.assert_array_equal(sa.all_masses[t], sb.all_masses[t])


def test_the_batched_sinks_read_each_sink_once_per_call_not_five_times(torch_sf):
    """One read per queue_stars / make_stars_from_sinks call, against 1 (queue) + 4 (make) per sink per call."""
    a, _ = _run(torch_sf, Hydro)
    b, _ = _run(torch_sf, BatchedHydro)
    assert b.reads == 2 * STEPS
    assert a.reads >= STEPS * len(SINKS) * (1 + 4), a.reads
