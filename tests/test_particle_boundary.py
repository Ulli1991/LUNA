"""Sinks and coupled stars at the box faces (`particle_boundary`, audit C7/M3).

Periodic: a sink that drifts out comes back in on the other side, and every
distance -- its field on the gas, the gas field it samples, sink-sink, the
creation separation -- goes to the nearest image. Outflow: it leaves the run,
its mass counted in `m_out`. The worker tells Torch the box is unbounded when
periodic (Torch deletes every star outside the bounds it reads) and wraps the
stars for every grid-facing use.
"""
from types import SimpleNamespace

import numpy as np
import pytest

import jax

import jax.numpy as jnp  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from luna.setup.state import Particles  # noqa: E402
from luna.species.sinks import Sinks, particles_periodic  # noqa: E402

N, L = 16, 1.0
DX = L / N
C = (8 + 0.5) * DX                        # a cell centre on y and z


def _sinks(periodic, n_max=4, sink_sink=False, rho_create=1.0):
    return Sinks(N3=(N, N, N), box3=(L, L, L), G=1.0, boundary=0, n_max=n_max,
                 softening=0.0, accretion_radius=2.0 * DX, creation_density=rho_create,
                 creation_separation=0.0, create_frac=0.9, sink_sink=sink_sink,
                 periodic_particles=periodic, is_main=False)


def _part(sk, positions, masses):
    n = sk.n_max
    pos = np.zeros((n, 3)); m = np.zeros(n); act = np.zeros(n)
    for i, (p, mi) in enumerate(zip(positions, masses)):
        pos[i], m[i], act[i] = p, mi, 1.0
    return Particles(jnp.asarray(pos), jnp.zeros((n, 3)), jnp.asarray(m), jnp.asarray(act),
                     jnp.zeros((n, 3)), extra=sk.zero_moments())


def _thin_gas(rho=0.5):
    """Below rho_create: nothing accretes, nothing forms."""
    U = np.zeros((5, N, N, N))
    U[0] = rho
    U[4] = 1.0
    return jnp.asarray(U)


def test_the_switch_follows_the_gas_unless_set():
    cfg = lambda b, pb: SimpleNamespace(BOUNDARY=b, particle_boundary=pb)   # noqa: E731
    assert particles_periodic(cfg(0, "")) is True
    assert particles_periodic(cfg(1, "")) is False
    assert particles_periodic(cfg(0, "outflow")) is False
    assert particles_periodic(cfg(0, "periodic")) is True
    with pytest.raises(ValueError):
        particles_periodic(cfg(0, "reflect"))
    with pytest.raises(ValueError):        # periodic particles in a walled box
        Sinks(N3=(N, N, N), box3=(L, L, L), G=1.0, boundary=1, n_max=2, softening=0.0,
              accretion_radius=0.0, creation_density=1.0, creation_separation=0.0,
              create_frac=0.9, periodic_particles=True, is_main=False)


def test_outflow_a_sink_that_left_is_off_and_its_mass_counted():
    sk = _sinks(periodic=False)
    part = _part(sk, [(L + 0.02, C, C), (0.5, C, C)], [3.0, 2.0])
    U = _thin_gas()
    U2, p2 = sk.finish(U, part, jnp.ones((N, N, N)), sk.observe(U))
    np.testing.assert_array_equal(np.asarray(p2.active)[:2], [0.0, 1.0])
    np.testing.assert_array_equal(np.asarray(p2.mass)[:2], [0.0, 2.0])
    np.testing.assert_array_equal(np.asarray(p2.extra["m_out"])[:2], [3.0, 0.0])
    np.testing.assert_array_equal(np.asarray(U2), np.asarray(U))


def test_periodic_a_sink_that_left_comes_back_on_the_other_side():
    sk = _sinks(periodic=True)
    part = _part(sk, [(L + 0.02, C, C), (0.5, -0.1, C)], [3.0, 2.0])
    U = _thin_gas()
    _, p2 = sk.finish(U, part, jnp.ones((N, N, N)), sk.observe(U))
    np.testing.assert_allclose(np.asarray(p2.pos)[:2], [[0.02, C, C], [0.5, L - 0.1, C]],
                               rtol=0, atol=1e-14)
    np.testing.assert_array_equal(np.asarray(p2.active)[:2], [1.0, 1.0])
    np.testing.assert_array_equal(np.asarray(p2.extra["m_out"]), 0.0)


def test_periodic_the_field_reaches_across_the_face():
    """A sink just inside x = 0 pulls the cells just inside x = L towards +x,
    across the face; without the image they are pulled back through the box."""
    pos = jnp.asarray([[0.02, C, C]])
    mass, act = jnp.ones(1), jnp.ones(1)
    for periodic, sign in ((True, 1.0), (False, -1.0)):
        sk = _sinks(periodic, n_max=1)
        gx = np.asarray(sk._sink_field_on_grid(pos, mass, act)[0])
        assert sign * gx[N - 1, 8, 8] > 0.0, periodic
    sk = _sinks(True, n_max=1)
    shifted = pos + jnp.asarray([[L, -L, 0.0]])      # the same sink, one box over
    for a, b in zip(sk._sink_field_on_grid(pos, mass, act),
                    sk._sink_field_on_grid(shifted, mass, act)):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-12, atol=1e-12)


def test_periodic_a_sink_by_the_face_samples_the_cells_on_both_sides():
    """x = 0.01 lies between the centres of cell N-1 (at -dx/2 by the wrap) and
    cell 0: the weights are (dx/2 - x)/dx and (dx/2 + x)/dx."""
    f = np.zeros((N, N, N))
    f[N - 1] = 5.0
    f[0] = 1.0
    x = 0.01
    t = (x + 0.5 * DX) / DX
    for periodic, want in ((True, (1 - t) * 5.0 + t * 1.0), (False, 1.0)):
        sk = _sinks(periodic)
        got = float(sk._interp(jnp.asarray(f), jnp.asarray([x, C, C])))
        assert abs(got - want) < 1e-12, (periodic, got, want)


def test_periodic_two_sinks_across_the_face_attract_through_it():
    sk = _sinks(True, n_max=2, sink_sink=True)
    pos = jnp.asarray([[0.02, C, C], [L - 0.02, C, C]])
    zero = tuple(jnp.zeros((N, N, N)) for _ in range(3))
    a = np.asarray(sk._sink_accel(pos, jnp.ones(2), jnp.ones(2), zero))
    assert a[0, 0] < 0.0 < a[1, 0]
    np.testing.assert_allclose(abs(a[0, 0]), 1.0 / (0.04 ** 2 + sk.eps ** 2) ** 1.5 * 0.04,
                               rtol=1e-12)


def _blob(c):
    """A dense, bound, converging blob centred on cell c: a sink forms there
    unless an existing one is too close."""
    x = (np.arange(N) + 0.5) * DX
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    d = [np.mod(Q - q + 0.5 * L, L) - 0.5 * L for Q, q in zip((X, Y, Z), c)]
    r2 = d[0] ** 2 + d[1] ** 2 + d[2] ** 2
    U = np.zeros((5, N, N, N))
    U[0] = 1.0 + 30.0 * np.exp(-r2 / 0.01)
    U[4] = 1.0
    phi = -50.0 * np.exp(-r2 / 0.02)
    return jnp.asarray(U), jnp.asarray(phi), -jnp.ones((N, N, N))


@pytest.mark.parametrize("periodic,formed", [(True, 1.0), (False, 2.0)])
def test_the_creation_separation_is_measured_to_the_nearest_image(periodic, formed):
    """The blob peaks in cell 0; an existing sink sits 0.2 away through the x
    face -- inside r_sep = 4 dx = 0.25, outside its own r_acc = 2 dx, so it
    cannot drain the peak -- and 0.8 away through the box."""
    sk = _sinks(periodic)
    c = (0.5 * DX, C, C)
    U, phi, divv = _blob(c)
    part = _part(sk, [(L - 0.2 + 0.5 * DX, C, C)], [1.0])
    _, p2 = sk.finish(U, part, phi, divv)
    assert float(np.asarray(p2.active).sum()) == formed


def test_on_the_mesh_the_faces_match_one_device():
    """Sinks outside, on and by the faces: wrap, accretion, the force."""
    from tests.test_sinks_sharded import KW, _gas, _mesh, _shard
    from luna.species.sinks_sharded import ShardedSinks
    mesh = _mesh()
    pts = [(L + 0.01, 0.5, 0.5), (0.03, -0.02, 0.97), (0.5, 0.5, 0.999), (0.2, 0.2, 0.8)]
    rng = np.random.default_rng(5)
    g = tuple(jnp.asarray(rng.normal(size=(N, N, N))) for _ in range(3))
    for periodic in (True, False):
        sk = Sinks(**{**KW, "periodic_particles": periodic})
        sh = ShardedSinks(mesh, **{**KW, "periodic_particles": periodic})
        part = _part(sk, pts, [1.0, 2.0, 3.0, 4.0])
        U = jnp.asarray(_gas(seed=4))
        phi = jnp.ones((N, N, N))
        divv = sk.observe(U)
        Ua, pa = sk.finish(U, part, phi, divv)
        Ub, pb = sh.finish(_shard(mesh, U, P(None, "x", "y", "z")), sh._replicate(part),
                           _shard(mesh, phi, P("x", "y", "z")),
                           _shard(mesh, divv, P("x", "y", "z")))
        np.testing.assert_allclose(np.asarray(Ub), np.asarray(Ua), rtol=1e-12, atol=1e-14)
        for k in ("pos", "mass", "active"):
            np.testing.assert_allclose(np.asarray(getattr(pb, k)), np.asarray(getattr(pa, k)),
                                       rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(np.asarray(pb.extra["m_out"]), np.asarray(pa.extra["m_out"]))
        assert (float(np.asarray(pa.extra["m_out"]).sum()) > 0.0) is (not periodic)
        fa = sk.force(pa, g)
        fb = sh.force(sh._replicate(pa), tuple(_shard(mesh, q, P("x", "y", "z")) for q in g))
        np.testing.assert_array_equal(np.asarray(fa), np.asarray(fb))


# -- the coupled worker's stars ------------------------------------------------
def _star_worker(pb):
    from dataclasses import replace
    from tests.test_coupling_sink_to_star import _worker
    w = _worker()
    w.cfg = replace(w.cfg, particle_boundary=pb)
    rng = np.random.default_rng(7)
    U = np.zeros(w.state.U.shape)
    U[0] = 1.0 + rng.random(U.shape[1:])
    U[4] = 1.0
    w.state.U = jnp.asarray(U)
    return w


def test_torch_is_told_the_box_is_unbounded_only_when_periodic():
    w = _star_worker("periodic")
    assert w.get_runtime_parameter("xmin") == -np.inf
    assert w.get_runtime_parameter("zmax") == np.inf
    w = _star_worker("outflow")
    assert w.get_runtime_parameter("xmin") == 0.0
    assert w.get_runtime_parameter("zmax") == float(w.cfg.Lz)


def test_a_periodic_star_outside_the_box_acts_as_its_image():
    """PeTar keeps the star at x - Lx; the gas it feels, the gas that feels it
    and the sink by the other face it pulls on are those of the star at x."""
    from tests.test_coupling_sink_to_star import _inject_sink
    from luna.species.sinks import Sinks as _S
    got = {}
    for where in ("in", "out"):
        w = _star_worker("periodic")
        Lx = float(w.cfg.Lx)
        _inject_sink(w, [1.0, 10.0, 10.0], mass=50.0)        # 2 from the star through x = 0
        x = 19.0 if where == "in" else 19.0 - Lx
        tag = w.new_particle(x, 10.0, 7.0 if where == "in" else 7.0 + Lx)
        w.set_particle_mass([tag], [5.0])
        w.apply_gas_gravity_on_particles(0.01)
        slot = w.stars.slots_for([tag])[0]
        w.apply_particle_gravity_on_gas(0.01)
        sink_p = np.asarray(jax.device_get(w.state.species[_S.name].p))[0]
        got[where] = (np.array(w.stars.arrays["velocity"][slot]), np.asarray(w.state.U), sink_p)
    assert got["in"][2][0] < 0.0          # the sink at x = 1 is pulled through x = 0
    for a, b in zip(got["out"], got["in"]):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-15)


def test_a_supernova_by_a_periodic_face_fills_its_whole_sphere():
    """Standalone too: the sphere continues across the face, the same number
    of cells as in the middle of the box, and the ejecta land on both sides."""
    from luna.sources.feedback import inject_energy
    cfg = SimpleNamespace(Lx=1.0, Ly=1.0, Lz=1.0, Nx=N, Ny=N, Nz=N, BOUNDARY=0)
    U = jnp.zeros((5, N, N, N)).at[0].set(1.0).at[4].set(1.0)

    def cells(x):
        dU = np.asarray(inject_energy(U, cfg, x, C, C, 1.0, mass=1.0, radius=2.5 * DX))[0] - 1.0
        return dU > 0.0
    mid, face = cells(0.5 + 0.5 * DX), cells(0.5 * DX)
    assert face.sum() == mid.sum()
    assert face[N - 1].any() and face[0].any()


def test_outflow_the_box_is_half_open_for_sinks_and_stars_alike():
    """One rule for both: a particle is in the box on [0, L). A sink and a
    star exactly at x = L are out, one exactly at x = 0 is in (stars used to
    leave only at x > L)."""
    sk = _sinks(periodic=False)
    part = _part(sk, [(L, C, C), (0.0, C, C), (0.5, C, L)], [3.0, 2.0, 1.0])
    U = _thin_gas()
    _, p2 = sk.finish(U, part, jnp.ones((N, N, N)), sk.observe(U))
    np.testing.assert_array_equal(np.asarray(p2.active)[:3], [0.0, 1.0, 0.0])

    w = _star_worker("outflow")
    Lx, Lz = float(w.cfg.Lx), float(w.cfg.Lz)
    tags = w.new_particle([Lx, 0.0, 5.0], [10.0, 10.0, 10.0], [10.0, 10.0, Lz])
    w.set_particle_mass(tags, [1.0, 1.0, 1.0])
    w.particles_sort()
    assert sorted(int(t) for t in w.stars.active_tags()) == [int(tags[1])]
