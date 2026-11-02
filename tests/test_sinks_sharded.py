"""Sinks on a 2x2x2 device mesh against the single-device kernels.

Same state, same sinks -- placed on block faces and corners, where a
window straddles up to eight blocks -- and every hook has to agree: the
field on the gas and div v bitwise, the force on the sinks bitwise (the
other ranks add exact zeros), accretion and creation to round-off (the
partial sums cross the ranks in a different order).
"""
import numpy as np
import pytest

import jax

import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from luna.setup.state import Particles  # noqa: E402
from luna.species.sinks import Sinks  # noqa: E402
from luna.species.sinks_sharded import ShardedSinks  # noqa: E402

N, L = 16, 1.0
DX = L / N
KW = dict(N3=(N, N, N), box3=(L, L, L), G=1.0, boundary=0, n_max=4, softening=0.0,
          accretion_radius=2.0 / N, creation_density=1.0, creation_separation=0.0,
          create_frac=0.9, sink_sink=False, is_main=False)


def _mesh():
    if jax.device_count() < 8:
        pytest.skip("needs 8 devices (the suite's CPU backend has them)")
    return Mesh(np.array(jax.devices()[:8]).reshape(2, 2, 2), ("x", "y", "z"))


def _pair():
    mesh = _mesh()
    return Sinks(**KW), ShardedSinks(mesh, **KW), mesh


def _shard(mesh, a, spec):
    return jax.device_put(a, NamedSharding(mesh, spec))


def _gas(seed=0, converging=None):
    rng = np.random.default_rng(seed)
    x = (np.arange(N) + 0.5) * DX
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    rho = 1.0 + 4.0 * rng.random((N, N, N))
    v = rng.normal(0.0, 0.3, (3, N, N, N))
    if converging is not None:                     # v = -(r - c): div v < 0 everywhere
        cx, cy, cz = converging
        v = -np.stack([X - cx, Y - cy, Z - cz])
    e = 1.0 + rng.random((N, N, N))
    U = np.stack([rho, rho * v[0], rho * v[1], rho * v[2],
                  e + 0.5 * rho * (v ** 2).sum(axis=0)])
    return U


# sinks on a corner of all eight blocks, on a face, in an interior, at a box edge
SINKS = [(0.5, 0.5, 0.5), (0.5, 0.27, 0.73), (0.2, 0.2, 0.8), (0.03, 0.97, 0.5)]


def _state(sk, positions, masses=None):
    n = sk.n_max
    pos = np.zeros((n, 3)); act = np.zeros(n); m = np.zeros(n); vel = np.zeros((n, 3))
    for i, p in enumerate(positions):
        pos[i], act[i] = p, 1.0
        m[i] = 1.0 + i if masses is None else masses[i]
        vel[i] = (0.1 * i, -0.2, 0.3)
    return jnp.asarray(pos), jnp.asarray(vel), jnp.asarray(m), jnp.asarray(act)


def _block_vs_global(periodic):
    mesh = _mesh()
    kw = {**KW, "periodic_particles": periodic}
    sk, sh = Sinks(**kw), ShardedSinks(mesh, **kw)
    pos, _, mass, act = _state(sk, SINKS)
    full = sk._sink_field_on_grid(pos, mass, act)
    n = N // 2
    for ox in (0, n):
        for oy in (0, n):
            for oz in (0, n):
                loc = sh._sink_field_on_grid(pos, mass, act, origin=(ox, oy, oz))
                for a, b in zip(loc, full):
                    yield np.asarray(a), np.asarray(b)[ox:ox + n, oy:oy + n, oz:oz + n]


def test_the_field_on_a_block_is_the_global_field_there():
    """Outflow sinks (the plain distance): bitwise."""
    for a, b in _block_vs_global(periodic=False):
        np.testing.assert_array_equal(a, b)


def test_the_periodic_field_on_a_block_is_the_global_field_to_round_off():
    """Periodic sinks take the nearest image, d - L floor(d/L + 1/2): the
    same expression on the same coordinates, but XLA compiles it differently
    for an 8^3 block and the 16^3 grid -- measured, 59 of 4096 values one ulp
    apart, all on the block's first z plane. Round-off on the field's scale."""
    worst = 0.0
    for a, b in _block_vs_global(periodic=True):
        scale = float(np.abs(b).max())
        assert scale > 0.0
        worst = max(worst, float(np.abs(a - b).max()) / scale)
    assert worst <= 1e-15


def test_div_v_is_bitwise_the_single_device_one():
    sk, sh, mesh = _pair()
    U = jnp.asarray(_gas())
    a = sk.observe(U)
    b = sh.observe(_shard(mesh, U, P(None, "x", "y", "z")))
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_the_force_on_the_sinks_is_bitwise_the_single_device_one():
    sk, sh, mesh = _pair()
    rng = np.random.default_rng(3)
    g = tuple(jnp.asarray(rng.normal(size=(N, N, N))) for _ in range(3))
    pos, vel, mass, act = _state(sk, SINKS)
    part = Particles(pos, vel, mass, act, jnp.zeros((4, 3)), extra=None)
    a = sk.force(part, g)
    b = sh.force(sh._replicate(part), tuple(_shard(mesh, q, P("x", "y", "z")) for q in g))
    assert float(jnp.max(jnp.abs(a))) > 0.0
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _finish_both(positions, U_np, phi_np, masses=None):
    sk, sh, mesh = _pair()
    pos, vel, mass, act = _state(sk, positions, masses)
    U, phi = jnp.asarray(U_np), jnp.asarray(phi_np)
    divv = sk.observe(U)
    mom0 = sk.zero_moments()
    part = Particles(pos, vel, mass, act, jnp.zeros((sk.n_max, 3)), extra=mom0)
    Ua, pa = sk.finish(U, part, phi, divv)
    Ub, pb = sh.finish(_shard(mesh, U, P(None, "x", "y", "z")), sh._replicate(part),
                       _shard(mesh, phi, P("x", "y", "z")),
                       _shard(mesh, divv, P("x", "y", "z")))
    return (Ua, pa), (Ub, pb), U


def _assert_same(a, b, U):
    (Ua, pa), (Ub, pb) = a, b
    dU = np.abs(np.asarray(Ua) - np.asarray(U)).max()
    assert dU > 0.0                                   # something was accreted
    np.testing.assert_allclose(np.asarray(Ub), np.asarray(Ua), rtol=1e-12, atol=1e-14)
    for k in ("pos", "p", "mass", "active"):
        np.testing.assert_allclose(np.asarray(getattr(pb, k)), np.asarray(getattr(pa, k)),
                                   rtol=1e-12, atol=1e-14)
    for k in pa.extra:
        np.testing.assert_allclose(np.asarray(pb.extra[k]), np.asarray(pa.extra[k]),
                                   rtol=1e-12, atol=1e-14)


def test_accretion_across_block_faces_and_corners_matches():
    U = _gas(seed=1)
    phi = np.zeros((N, N, N)) + 1.0                    # no minimum worth a sink
    a, b, U0 = _finish_both(SINKS, U, phi)
    _assert_same(a, b, U0)
    assert float(np.asarray(a[1].active).sum()) == 4.0   # nothing created


def test_creation_at_a_block_corner_matches():
    """A bound, converging, dense blob whose potential minimum sits on the
    corner cell shared by all eight blocks."""
    c = (0.5, 0.5, 0.5)
    U = _gas(seed=2, converging=c)
    x = (np.arange(N) + 0.5) * DX
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2
    U[0] = 1.0 + 30.0 * np.exp(-r2 / 0.01)                # above rho_create = 1 near c
    U[4] = 1.0 + 0.5 * U[0] * ((U[1:4] / U[0]) ** 2).sum(axis=0)
    phi = -50.0 * np.exp(-r2 / 0.02)
    a, b, U0 = _finish_both([(0.15, 0.15, 0.15)], U, phi, masses=[0.5])
    _assert_same(a, b, U0)
    assert float(np.asarray(a[1].active).sum()) == 2.0    # one sink was created
    assert float(np.asarray(a[1].mass)[1]) > 0.0


@pytest.mark.parametrize("eint,formed", [(0.05, 2.0), (50.0, 1.0)])
def test_a_hot_core_at_a_block_corner_makes_no_sink(eint, formed):
    """The bound check on the mesh is the single-device one: a dense core at
    rest in a well, on the corner of all eight blocks, forms a sink when cold
    and none when 1000x hotter."""
    x = (np.arange(N) + 0.5) * DX
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2
    U = np.zeros((5, N, N, N))
    U[0] = 1.0 + 30.0 * np.exp(-r2 / 0.01)
    U[1] = -U[0] * (X - 0.5)                               # converging: div v < 0
    U[2] = -U[0] * (Y - 0.5)
    U[3] = -U[0] * (Z - 0.5)
    U[4] = eint * U[0] + 0.5 * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / U[0]
    phi = -50.0 * np.exp(-r2 / 0.02)
    a, b, U0 = _finish_both([(0.15, 0.15, 0.15)], U, phi, masses=[0.5])
    _assert_same(a, b, U0)
    assert float(np.asarray(a[1].active).sum()) == formed


def test_the_field_inside_a_shard_map_is_the_global_field():
    """As the sharded gas kick evaluates it: in a shard_map body, the block
    origin from the axis indices (the chunked loop's carry has to vary per
    device there, or the program does not trace). To round-off, not bitwise:
    compiled in a shard_map the sums fuse differently (up to 2e-14 absolute,
    measured on the unrolled kernel as well)."""
    from luna.gravity.particle_mesh_blocks import _block_origin
    from luna.utils.shmap import shard_map
    sk, sh, mesh = _pair()
    pos, _, mass, act = _state(sk, SINKS)
    full = sk._sink_field_on_grid(pos, mass, act)
    nloc = (N // 2,) * 3

    def body(p, m, a):
        return sh._sink_field_on_grid(p, m, a, origin=_block_origin((2, 2, 2), nloc))
    g = P("x", "y", "z")
    got = jax.jit(shard_map(body, mesh=mesh, in_specs=(P(), P(), P()),
                            out_specs=(g, g, g)))(pos, mass, act)
    for a, b in zip(got, full):
        b = np.asarray(b)
        np.testing.assert_allclose(np.asarray(a), b, rtol=1e-12, atol=1e-13 * np.abs(b).max())


# two sinks 1.5 dx apart (their r_acc = 2 dx spheres overlap), one pair on a
# block face, one across the periodic x edge
CLOSE = [(0.5, 0.52, 0.5), (0.5, 0.52 + 1.5 * DX, 0.5),
         (0.01, 0.3, 0.3), (0.99, 0.3, 0.3)]


def test_overlapping_sinks_share_the_gas_they_both_reach():
    """Audit C4: each sink used to take every cell's full excess, so a cell two
    sinks reach lost it twice (dropping below rho_create) and the sinks gained
    more mass and momentum than the gas lost."""
    sk = Sinks(**KW)
    U = jnp.asarray(_gas(seed=3))
    pos, vel, mass, act = _state(sk, CLOSE)
    part = Particles(pos, vel, mass, act, jnp.zeros((sk.n_max, 3)), extra=sk.zero_moments())
    phi = jnp.ones((N, N, N))
    U2, p2 = sk.finish(U, part, phi, sk.observe(U))
    assert float(np.asarray(p2.active).sum()) == 4.0          # accretion only
    dV = DX ** 3
    gas_lost = float(jnp.sum(U[0] - U2[0])) * dV
    gained = float(jnp.sum(p2.mass - mass))
    assert gas_lost > 0.0
    np.testing.assert_allclose(gained, gas_lost, rtol=1e-12)
    mom_lost = np.asarray(jnp.sum(U[1:4] - U2[1:4], axis=(1, 2, 3))) * dV
    mom_gained = np.asarray(jnp.sum(p2.mass[:, None] * p2.p - mass[:, None] * vel, axis=0))
    np.testing.assert_allclose(mom_gained, mom_lost, rtol=1e-10, atol=1e-14)
    drained = np.asarray(U2[0] < U[0])
    assert np.asarray(U2[0])[drained].min() >= 1.0 - 1e-12   # never below rho_create


def test_overlapping_accretion_matches_on_the_mesh():
    U = _gas(seed=3)
    phi = np.zeros((N, N, N)) + 1.0
    a, b, U0 = _finish_both(CLOSE, U, phi)
    _assert_same(a, b, U0)
