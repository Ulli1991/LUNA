"""The fused gas kick (gravity/kick_pallas.py) against the XLA kick with the
sinks' field it replaces (backend.kick_gas_fields_body + Sinks._field_at):
live and empty sink slots mixed, none live, and -- on two GPUs -- the block
form inside a shard_map with the axis-index origin. Same terms in the same
order, a different compiler: round-off, normalised by the kick's size."""
import numpy as np
import pytest

import jax
import jax.numpy as jnp


from luna.gravity import kick_pallas                             # noqa: E402
from luna.gravity.backend import kick_gas_fields_body            # noqa: E402
from luna.species.sinks import Sinks                             # noqa: E402

GPU = jax.default_backend() in ("gpu", "rocm", "cuda")
N = 16 if not GPU else 64


def _sinks(n, n_max, periodic=True, L=1.0):
    return Sinks(N3=(n, n, n), box3=(L, L, L), G=1.0, boundary=0, n_max=n_max,
                 softening=0.05, accretion_radius=0.0, creation_density=0.0,
                 creation_separation=0.0, create_frac=0.0,
                 periodic_particles=periodic, is_main=False)


def _state(n, seed=0):
    rng = np.random.default_rng(seed)
    rho = 1.0 + rng.random((n, n, n))
    m = 0.3 * rng.standard_normal((3, n, n, n)) * rho
    E = 2.0 + 0.5 * (m ** 2).sum(0) / rho
    U = jnp.asarray(np.concatenate([rho[None], m, E[None]]))
    g = tuple(jnp.asarray(0.1 * rng.standard_normal((n, n, n))) for _ in range(3))
    return U, g


def _slots(n_max, live, seed=1, L=1.0):
    rng = np.random.default_rng(seed)
    pos = jnp.asarray(L * rng.random((n_max, 3)))
    mass = jnp.asarray(rng.random(n_max) + 0.5)
    active = np.zeros(n_max)
    active[list(live)] = 1.0
    return pos, mass, jnp.asarray(active)


def _both(live, n_max=10, periodic=True, L=1.0):
    sk = _sinks(N, n_max, periodic, L)
    U, g = _state(N)
    pos, mass, active = _slots(n_max, live, L=L)
    s, dt = 0.7, 0.01
    xla = jax.jit(lambda U: kick_gas_fields_body(
        U, g, s, dt, (sk._sink_field_on_grid,), ((pos, mass, active),)))(U)
    pal = jax.jit(lambda U: kick_pallas.kick(U, g, s, dt, pos, mass, active, origin=(0, 0, 0),
                                             dxyz=sk.dxyz, G=sk.G, eps=sk.eps,
                                             box=sk.image_box))(U)
    return np.asarray(U), np.asarray(xla), np.asarray(pal)


@pytest.mark.parametrize("L", [1.0, 3.0])
@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("live", [(), (5,), (0, 3, 4, 9)])
def test_the_fused_kick_is_the_xla_kick(live, periodic, L):
    """Periodic particles: both take the nearest image (the slots sit
    anywhere in the box, most within half a box of some face); L=3: an inexact 1/L."""
    U, a, b = _both(live, periodic=periodic, L=L)
    dU = np.abs(a - U).max()
    assert dU > 0.0
    assert float(np.abs(a - b).max()) <= 1e-12 * dU
    np.testing.assert_array_equal(b[0], a[0])            # density untouched


@pytest.mark.skipif(not GPU or jax.device_count() < 2, reason="the block form: two GPUs")
def test_the_block_form_on_two_gpus_is_the_one_device_kick():
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from luna.gravity.particle_mesh_blocks import _block_origin
    from luna.utils.shmap import shard_map
    n_max = 10
    sk = _sinks(N, n_max)
    U, g = _state(N)
    pos, mass, active = _slots(n_max, (1, 2, 7))
    s, dt = 0.7, 0.01
    one = np.asarray(jax.jit(lambda U: kick_pallas.kick(
        U, g, s, dt, pos, mass, active, origin=(0, 0, 0), dxyz=sk.dxyz, G=sk.G, eps=sk.eps))(U))
    mesh = Mesh(np.array(jax.devices()[:2]).reshape(2, 1, 1), ("x", "y", "z"))
    PU, Pg = P(None, "x", "y", "z"), P("x", "y", "z")

    def body(U, gx, gy, gz):
        org = _block_origin((2, 1, 1), (N // 2, N, N))
        return kick_pallas.kick(U, (gx, gy, gz), s, dt, pos, mass, active, origin=org,
                                dxyz=sk.dxyz, G=sk.G, eps=sk.eps)
    f = jax.jit(shard_map(body, mesh=mesh, in_specs=(PU, Pg, Pg, Pg), out_specs=PU))
    two = np.asarray(f(jax.device_put(U, NamedSharding(mesh, PU)),
                       *(jax.device_put(q, NamedSharding(mesh, Pg)) for q in g)))
    np.testing.assert_array_equal(two, one)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_the_plain_pallas_kick_is_the_xla_kick_bitwise(dtype):
    from luna.gravity.backend import kick_gas_body
    U, g = _state(N)
    U, g = U.astype(dtype), tuple(q.astype(dtype) for q in g)
    assert kick_pallas.applies_plain(U, g)
    s, dt = 0.7, 0.01
    xla = np.asarray(jax.jit(kick_gas_body)(U, g, s, dt))
    pal = np.asarray(jax.jit(kick_pallas.kick_plain)(U, g, s, dt))
    assert pal.dtype == xla.dtype == np.dtype(dtype)
    assert np.abs(xla - np.asarray(U)).max() > 0.0
    np.testing.assert_array_equal(pal, xla)


def test_the_plain_kick_refuses_what_it_cannot_do():
    from luna.gravity.backend import _plain_kick_fn, kick_gas_body
    U, g = _state(N)
    assert _plain_kick_fn("xla", False) is kick_gas_body
    assert _plain_kick_fn("pallas", True) is kick_gas_body          # autodiff: XLA
    assert not kick_pallas.applies_plain(U, tuple(q.astype(jnp.float32) for q in g))
    with pytest.raises(ValueError, match="gravity_kernel='pallas'"):
        _plain_kick_fn("pallas", False)(U, tuple(q.astype(jnp.float32) for q in g), 0.7, 0.01)


@pytest.mark.skipif(not GPU or jax.device_count() < 2, reason="the block form: two GPUs (interpret mode refuses mixed vma)")
def test_the_plain_kick_on_a_mesh_is_the_xla_kick():
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from luna.gravity.backend import _plain_kick_fn, kick_gas_body
    from luna.utils.shmap import shard_map
    U, g = _state(N)
    mesh = Mesh(np.array(jax.devices()[:2]).reshape(2, 1, 1), ("x", "y", "z"))
    PU, Pg = P(None, "x", "y", "z"), P("x", "y", "z")
    Us = jax.device_put(U, NamedSharding(mesh, PU))
    gs = tuple(jax.device_put(q, NamedSharding(mesh, Pg)) for q in g)

    def run(fn):
        f = jax.jit(shard_map(fn, mesh=mesh, in_specs=(PU, (Pg, Pg, Pg), P(), P()), out_specs=PU))
        return np.asarray(f(Us, gs, 0.7, 0.01))
    np.testing.assert_array_equal(run(_plain_kick_fn("pallas", False)), run(kick_gas_body))
