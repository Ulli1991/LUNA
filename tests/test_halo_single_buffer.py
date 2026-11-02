"""The halo built in one buffer against the successive concatenates it replaces.

`make_halo_shardmap`'s halo used to concatenate ghosts onto the state one axis
at a time -- a full copy of the state per axis, and at 512^3 FP64 a large
share of the hydro step's temporaries. It now pads once and writes each axis's
ghost slabs in place. The values must be the same, bit for bit, corners
included, on a real device mesh (ghosts from the neighbours) and on one device.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, PartitionSpec as P  # noqa: E402

from luna.utils.halo_exchange_shardmap import (_AXIS_NAMES, _fix_face,  # noqa: E402
                                               _pad_axis, make_halo_shardmap)
from luna.utils.shmap import shard_map  # noqa: E402


def _reference_halo(mesh, sdim):
    """The halo as it was: one concatenate per axis, then the face fixes."""
    names = _AXIS_NAMES[sdim]
    n_proc = {a: int(mesh.shape[a]) for a in names}

    def halo(U, boundary, width):
        for axis, name in zip(range(1, sdim + 1), names):
            U = _pad_axis(U, axis=axis, axis_name=name, n_proc=n_proc[name],
                          periodic=boundary == 0, width=width)
        if boundary in (2, 4):
            for axis, name in zip(range(1, sdim + 1), names):
                U = _fix_face(U, axis=axis, axis_name=name, n_proc=n_proc[name],
                              comp=axis, op="reflect" if boundary == 2 else "diode")
        return U
    return halo


def _mesh(shape):
    n = int(np.prod(shape))
    if jax.device_count() < n:
        pytest.skip(f"needs {n} devices")
    return Mesh(np.array(jax.devices()[:n]).reshape(shape), _AXIS_NAMES[len(shape)])


CASES = [(sdim, shape, bnd, w)
         for sdim, shapes in ((1, [(1,), (4,)]), (2, [(1, 1), (2, 2)]),
                              (3, [(1, 1, 1), (2, 2, 2)]))
         for shape in shapes
         for bnd in (0, 1, 2, 4)
         for w in (1, 2)
         if not (w == 2 and bnd in (2, 4))]


@pytest.mark.parametrize("sdim,shape,boundary,width", CASES)
def test_single_buffer_halo_matches_successive_concatenates(sdim, shape, boundary, width):
    mesh = _mesh(shape)
    n = (8, 12, 16)[:sdim]
    rng = np.random.default_rng(sdim * 100 + boundary * 10 + width)
    U = jnp.asarray(rng.normal(size=(sdim + 2,) + n))
    spec = P(None, *_AXIS_NAMES[sdim])
    new = make_halo_shardmap(mesh, sdim)
    ref = _reference_halo(mesh, sdim)

    def run(fn):
        return jax.jit(shard_map(fn, mesh=mesh, in_specs=(spec,), out_specs=spec))(U)

    a = run(lambda u: new(u, boundary, sdim, width=width))
    b = run(lambda u: ref(u, boundary, width))
    assert a.shape == b.shape
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
