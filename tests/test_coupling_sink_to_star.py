"""Sink → star handoff (Phase 5 / #2): LUNA sink mass becomes star particles.

Verifies the coupling logic without the MPI codes (se=grav=None): a sink with
enough mass spawns a star (mass moves sink→star, star lands in LUNA's store),
and a sink below threshold does not.
"""
import os

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp  # noqa: E402
from amuse.units import units as u  # noqa: E402

from tests.test_regressions import _build_cfg_for_test  # noqa: E402
from luna.setup.state import Particles  # noqa: E402
from luna.species.sinks import Sinks  # noqa: E402
from amuse_luna.worker import LunaWorker  # noqa: E402
from amuse_luna.sink_to_star import spawn_stars_from_sinks  # noqa: E402


def _worker(N=16):
    cfg = _build_cfg_for_test(
        test_problem="Sod-Shock-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=N, Ny=N, Nz=N, Lx=20.0, Ly=20.0, Lz=20.0, gamma=5.0 / 3.0,
        ORDER=1, LIMITER=1, GRAVITATIONAL_G=1.0, POISSON_BOUNDARY=0,
    )
    w = LunaWorker(cfg)
    w.initialize_code()
    w.commit_parameters()
    return w


def _inject_sink(w, pos, mass, cs=1.0):
    """Put one sink in front of the worker.

    Sinks are a SPECIES now -- `state.species[Sinks.name]`, a `Particles`, with the
    accreted-gas moments in `extra` -- not the source carry they were when
    these tests were written (main deleted `sources.sinks.setup_sinks` in
    4848abe). Only the injection moved; what the tests assert did not.
    """
    Nmax = int(getattr(w.cfg, "sink_max_count", 4))
    p = np.zeros((Nmax, 3)); p[0] = pos
    m = np.zeros(Nmax); m[0] = mass
    act = np.zeros(Nmax); act[0] = 1.0
    macc = np.zeros(Nmax); macc[0] = mass
    scs = np.zeros(Nmax); scs[0] = mass * cs          # -> mean_cs = cs
    mom = {"macc": jnp.asarray(macc), "sv": jnp.zeros((Nmax, 3)),
           "sv2": jnp.zeros((Nmax, 3)), "scs": jnp.asarray(scs), "sL": jnp.zeros((Nmax, 3))}
    w.state.species[Sinks.name] = Particles(
        pos=jnp.asarray(p), p=jnp.zeros((Nmax, 3)),
        mass=jnp.asarray(m), active=jnp.asarray(act), extra=mom)


def test_sink_above_threshold_spawns_star():
    w = _worker()
    _inject_sink(w, [10.0, 10.0, 10.0], mass=50.0)

    new = spawn_stars_from_sinks(w, m_star=10.0, se=None, grav=None, store=True)
    assert len(new) == 1
    np.testing.assert_allclose(new.mass.value_in(u.MSun)[0], 10.0)
    np.testing.assert_allclose(new.position.value_in(u.parsec)[0], [10.0, 10.0, 10.0])

    # sink mass decremented by the star mass
    sel, spos, svel, smass = w.get_active_sinks()
    np.testing.assert_allclose(smass[0], 40.0, rtol=1e-9)
    # star landed in LUNA's 'mass' particle store
    w.set_particle_pointers("mass")
    assert w.get_number_of_particles() == 1


def test_sink_below_threshold_spawns_nothing():
    w = _worker()
    _inject_sink(w, [10.0, 10.0, 10.0], mass=5.0)
    new = spawn_stars_from_sinks(w, m_star=10.0, se=None, grav=None, store=True)
    assert len(new) == 0
    sel, spos, svel, smass = w.get_active_sinks()
    np.testing.assert_allclose(smass[0], 5.0)          # untouched
    w.set_particle_pointers("mass")
    assert w.get_number_of_particles() == 0


def test_repeated_spawn_drains_sink():
    w = _worker()
    _inject_sink(w, [10.0, 10.0, 10.0], mass=25.0)
    total = 0
    for _ in range(5):
        total += len(spawn_stars_from_sinks(w, m_star=10.0, se=None, grav=None, store=True))
    assert total == 2                                   # 25 -> 15 -> 5 (stops; 5 < 10)
    sel, spos, svel, smass = w.get_active_sinks()
    np.testing.assert_allclose(smass[0], 5.0, rtol=1e-9)


def test_a_sink_mass_torch_changes_reaches_the_carry_and_nothing_else_is_written():
    """Torch's make_stars_from_sinks lowers a sink's mass through the particle
    interface with the pointer on 'sink'; switching it back pushes that into
    LUNA's carry. A switch with nothing changed writes nothing (the push back
    is skipped: it was a device scatter on every pointer switch)."""
    w = _worker()
    _inject_sink(w, [10.0, 10.0, 10.0], mass=50.0)
    w.set_particle_pointers("sink")
    w.set_particle_pointers("mass")
    before = w.state.species[Sinks.name]
    w.set_particle_pointers("sink")
    w.set_particle_pointers("mass")
    assert w.state.species[Sinks.name] is before           # untouched: nothing to push
    w.set_particle_pointers("sink")
    tags = w.sinks.active_tags()
    w.set_particle_mass(tags, [30.0])
    w.set_particle_pointers("mass")
    _, _, _, smass = w.get_active_sinks()
    np.testing.assert_allclose(smass[0], 30.0, rtol=1e-12)
