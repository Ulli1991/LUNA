"""Explicit shard_map reductions for the multi-GPU runner loops.

Eagerly calling ``jnp.max``/``jnp.sum``/``jnp.mean`` on a sharded array asks
GSPMD to auto-build a collective, and splicing those between the explicit
shard_map executables of a step loop wedges the ROCm runtime (silent deadlock;
pinned twice on 2026-08-27 — the coupled blocks loop at step 2 and the 64-GPU
bench's step_warm section). These helpers are the replacement: every reduction
is an explicit psum/pmax/pmin inside a shard_map program, built once per
(mesh, spec, op, dtype) and cached — the ``field_stats`` rule from the DM
blocks runner ("NO auto-sharding, reductions are psum/pmax") as a library.

``grid_reduce``  — fields sharded one mesh axis per array axis (a component
                   of U, rho, phi; dim-generic via mesh.axis_names).
``flat_reduce``  — flat particle arrays sharded P((*axis_names,)) or
                   P((*axis_names,), None).
Both return a 0-d replicated array; wrap in float()/int() at the call site
(that sync is the caller's explicit choice, not a hidden collective).
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
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
    """op in ('sum','max','min') over a P('x','y','z')-sharded field."""
    key = ("grid", _mesh_key(mesh), op)
    fn = _CACHE.get(key)
    if fn is None:
        _CACHE[key] = fn = _build(mesh, P(*mesh.axis_names), op)
    return fn(field)


def flat_reduce(mesh, op, arr):
    """op over a flat particle array: 1-D P(('x','y','z')) or 2-D
    P(('x','y','z'), None)."""
    key = ("flat", _mesh_key(mesh), op, arr.ndim)
    fn = _CACHE.get(key)
    if fn is None:
        names = tuple(mesh.axis_names)
        spec = P(names) if arr.ndim == 1 else P(names, None)
        _CACHE[key] = fn = _build(mesh, spec, op)
    return fn(arr)


def grid_mean(mesh, field):
    """Mean of a P('x','y','z')-sharded field (psum of sums / global size)."""
    return grid_reduce(mesh, "sum", field) / float(field.size)
