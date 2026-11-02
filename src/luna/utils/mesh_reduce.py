"""Explicit shard_map reductions (psum/pmax/pmin) on sharded arrays, cached per mesh and op.

Use these instead of eager jnp.sum/max on sharded arrays; see docs/RUNNING.md (Multi-process rules).
"""
from functools import partial

import jax
import jax.numpy as jnp
from luna.utils.shmap import shard_map
from jax.sharding import PartitionSpec as P

_CACHE = {}

_LOCAL = {"sum": jnp.sum, "max": jnp.max, "min": jnp.min}
_JOIN = {"sum": jax.lax.psum, "max": jax.lax.pmax, "min": jax.lax.pmin}


def _mesh_key(mesh):
    return (tuple(sorted((k, int(v)) for k, v in mesh.shape.items())),
            tuple(str(d) for d in mesh.devices.flat))


def _build(mesh, spec, op):
    local, join = _LOCAL[op], _JOIN[op]
    names = tuple(mesh.axis_names)

    @partial(jax.jit, static_argnames=())
    @partial(shard_map, mesh=mesh, in_specs=(spec,), out_specs=P())
    def _red(x_l):
        return join(local(x_l), names)

    return _red


def grid_reduce(mesh, op, field):
    """Reduce a P('x','y','z')-sharded field with op in ('sum','max','min'); returns a 0-d replicated array."""
    key = ("grid", _mesh_key(mesh), op)
    fn = _CACHE.get(key)
    if fn is None:
        _CACHE[key] = fn = _build(mesh, P(*mesh.axis_names), op)
    return fn(field)


def flat_reduce(mesh, op, arr):
    """Reduce a flat particle array sharded P(('x','y','z')) or P(('x','y','z'), None) with op."""
    key = ("flat", _mesh_key(mesh), op, arr.ndim)
    fn = _CACHE.get(key)
    if fn is None:
        names = tuple(mesh.axis_names)
        spec = P(names) if arr.ndim == 1 else P(names, None)
        _CACHE[key] = fn = _build(mesh, spec, op)
    return fn(arr)


def grid_mean(mesh, field):
    """Return the mean of a P('x','y','z')-sharded field."""
    return grid_reduce(mesh, "sum", field) / float(field.size)
