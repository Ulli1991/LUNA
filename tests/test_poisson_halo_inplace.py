"""The in-place Poisson halos (one padded buffer, ghosts written x -> y -> z) give the bits of the concatenating halos
they replaced, corners included: dense periodic, Neumann, odd and extrapolated faces, the 4th-order gradient's
2-cell pad, and on device meshes the scalar shard_map halo, the isolated pads and the block-PM halo."""
from functools import partial

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from luna.gravity import isolated
from luna.gravity.particle_mesh_blocks import _halo as blocks_halo
from luna.gravity.poisson_bc import make_halo_3d, make_odd_dirichlet_halo_3d
from luna.gravity.poisson_bc_shardmap import make_scalar_halo_3d_shardmap
from luna.utils.halo_exchange_shardmap import _axis_ghosts, _pad_axis
from luna.utils.shmap import shard_map

AXES = ("x", "y", "z")
G = P("x", "y", "z")


def _field(shape=(12, 10, 8), seed=0):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape))


def _concat_pad(phi, ghosts):
    """The replaced pattern: concatenate (lo, phi, hi) one axis at a time."""
    for ax in range(3):
        lo, hi = ghosts(phi, ax)
        phi = jnp.concatenate([lo, phi, hi], axis=ax)
    return phi


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    assert np.array_equal(a, b)


@pytest.mark.parametrize("mode,boundary", [("wrap", 0), ("edge", 1), ("edge", 2)])
def test_dense_halo_equals_jnp_pad(mode, boundary):
    phi = _field()
    _same(make_halo_3d(boundary)(phi), jnp.pad(phi, 1, mode=mode))


@pytest.mark.parametrize("boundary", [0, 1, 2])
def test_dense_halo_inplace_flag_gives_the_same_bits(boundary):
    phi = _field()
    _same(make_halo_3d(boundary, inplace=False)(phi), make_halo_3d(boundary)(phi))
    _same(make_odd_dirichlet_halo_3d(inplace=False)(phi), make_odd_dirichlet_halo_3d()(phi))


def test_odd_halo_equals_concatenation():
    phi = _field()
    ref = _concat_pad(phi, lambda a, ax: (-isolated._take(a, ax, 0), -isolated._take(a, ax, -1)))
    _same(make_odd_dirichlet_halo_3d()(phi), ref)


@pytest.mark.parametrize("width", [1, 2])
def test_extrap_pad_equals_concatenation(width):
    phi = _field()
    _same(isolated.extrap_pad(phi, width),
          _concat_pad(phi, lambda a, ax: isolated._extrap_ghosts(a, ax, width)))


@pytest.mark.parametrize("boundary,mode", [(0, "wrap"), (1, "edge")])
def test_gradient_pad2_equals_jnp_pad(boundary, mode):
    from luna.gravity.particle_mesh import ParticleMeshGravity
    from tests._helpers import _build_cfg_for_test
    cfg = _build_cfg_for_test(test_problem="Soundwave-3D", SPATIAL_DIMENSION=3, Nx=8, Ny=8, Nz=8,
                              Lx=1.0, Ly=1.0, Lz=1.0, BOUNDARY=0 if boundary == 0 else 4,
                              POISSON_BOUNDARY=boundary, SELF_GRAVITY=True, enable_x64=True)
    phi = _field((8, 8, 8))
    _same(ParticleMeshGravity(cfg)._pad2(phi), jnp.pad(phi, 2, mode=mode))


def _mesh(shape):
    n = int(np.prod(shape))
    if jax.device_count() < n:
        pytest.skip(f"needs {n} devices")
    return Mesh(np.asarray(jax.devices()[:n]).reshape(shape), AXES)


def _sharded(body, mesh, phi):
    """body(local block) under shard_map, gathered to one host array."""
    f = jax.jit(shard_map(body, mesh=mesh, in_specs=(G,), out_specs=G))
    return np.asarray(f(jax.device_put(phi, NamedSharding(mesh, G))))


MESHES = [(2, 2, 2), (4, 1, 1), (1, 2, 1), (1, 1, 1)]


@pytest.mark.parametrize("shape", MESHES)
@pytest.mark.parametrize("boundary", [0, 1])
@pytest.mark.parametrize("width", [1, 2, 4])
def test_sharded_scalar_halo_equals_pad_axis(shape, boundary, width):
    mesh = _mesh(shape)
    phi = _field((16, 16, 16))
    halo = make_scalar_halo_3d_shardmap(mesh)(boundary)

    def ref(p):
        for a, name in enumerate(AXES):
            p = _pad_axis(p, axis=a, axis_name=name, n_proc=int(mesh.shape[name]),
                          periodic=boundary == 0, width=width)
        return p
    _same(_sharded(lambda p: halo(p, width), mesh, phi), _sharded(ref, mesh, phi))


@pytest.mark.parametrize("shape", MESHES)
@pytest.mark.parametrize("boundary,width", [(0, 1), (1, 2), (3, 1), (3, 2)])
def test_pad_sharded_equals_concatenation(shape, boundary, width):
    mesh = _mesh(shape)
    phi = _field((16, 16, 16))

    def ref(p):
        for ax, name in enumerate(AXES):
            n = int(mesh.shape[name])
            lo, hi = _axis_ghosts(p, axis=ax, axis_name=name, n_proc=n,
                                  periodic=boundary in (0, 3), width=width)
            if boundary == 3:
                elo, ehi = isolated._extrap_ghosts(p, ax, width)
                idx = jax.lax.axis_index(name)
                lo = jnp.where(idx == 0, elo, lo)
                hi = jnp.where(idx == n - 1, ehi, hi)
            p = jnp.concatenate([lo, p, hi], axis=ax)
        return p
    _same(_sharded(partial(isolated.pad_sharded, mesh=mesh, width=width, boundary=boundary), mesh, phi),
          _sharded(ref, mesh, phi))


@pytest.mark.parametrize("shape", MESHES)
def test_odd_pad_sharded_equals_concatenation(shape):
    mesh = _mesh(shape)
    phi = _field((16, 16, 16))

    def ref(p):
        for ax, name in enumerate(AXES):
            n = int(mesh.shape[name])
            lo, hi = _axis_ghosts(p, axis=ax, axis_name=name, n_proc=n, periodic=True, width=1)
            idx = jax.lax.axis_index(name)
            lo = jnp.where(idx == 0, -isolated._take(p, ax, 0), lo)
            hi = jnp.where(idx == n - 1, -isolated._take(p, ax, -1), hi)
            p = jnp.concatenate([lo, p, hi], axis=ax)
        return p
    _same(_sharded(lambda p: isolated._odd_pad(p, mesh), mesh, phi), _sharded(ref, mesh, phi))


@pytest.mark.parametrize("shape", MESHES)
def test_blocks_halo_equals_pad_axis(shape):
    mesh = _mesh(shape)
    phi = _field((16, 16, 16))

    def ref(p):
        for a, name in enumerate(AXES):
            p = _pad_axis(p, axis=a, axis_name=name, n_proc=int(mesh.shape[name]), periodic=True, width=2)
        return p
    _same(_sharded(lambda p: blocks_halo(p, mesh, 2), mesh, phi), _sharded(ref, mesh, phi))
