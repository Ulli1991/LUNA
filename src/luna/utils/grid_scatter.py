"""Scatter-add / gather a few cells of a possibly sharded grid.

Each device handles its own block in a shard_map; GSPMD-partitioned scatters can go wrong."""
import numpy as np

import jax
import jax.numpy as jnp


def _bucket(m):
    """Pad size: a power of two >= 64, so one program compiles per size class."""
    return max(64, 1 << int(np.ceil(np.log2(max(int(m), 1)))))


def make_grid_scatter(sharding, n_comp=None, after=None):
    """`scatter(field, idx, vals)` adding vals (n_comp, M) or (M,) at global cells idx (M, 3).

    `after(old, new)`, if given, maps the touched cells' old/new values to their final values."""
    progs = {}
    vec = n_comp is not None

    def add_local(field, loc, vals, own):
        v = vals * own
        i, j, k = loc[:, 0], loc[:, 1], loc[:, 2]
        if not vec:
            return field.at[i, j, k].add(v)
        new = field.at[:, i, j, k].add(v)
        if after is None:
            return new
        # touched cells only; a repeated cell gathers the same sums, so its
        # duplicate writes agree
        old_c = field[:, i, j, k]
        new_c = new[:, i, j, k]
        return new.at[:, i, j, k].set(after(old_c, new_c))

    mesh = getattr(sharding, "mesh", None)
    if mesh is None or mesh.size == 1:
        def one_device(field, idx, vals):
            return add_local(field, idx, vals, jnp.ones(idx.shape[0], vals.dtype))

        def make(m):
            return jax.jit(one_device) if sharding is None else jax.jit(
                one_device, out_shardings=sharding)
    else:
        from jax.sharding import PartitionSpec as P
        from luna.utils.shmap import shard_map
        # a spec may leave trailing axes out (P("x") on a (2, 1, 1) mesh): pad to 3
        spec = tuple(sharding.spec)[1:] if vec else tuple(sharding.spec)
        axes = (spec + (None,) * 3)[:3]
        names = list(axes)

        def body(field, idx, vals):
            nloc = field.shape[1:] if vec else field.shape
            org = jnp.stack([jax.lax.axis_index(names[a]) * nloc[a] if names[a] is not None
                             else 0 for a in range(3)])
            loc = idx - org
            own = jnp.all((loc >= 0) & (loc < jnp.asarray(nloc)), axis=1)
            loc = jnp.clip(loc, 0, jnp.asarray(nloc) - 1)
            return add_local(field, loc, vals, own.astype(vals.dtype))

        fspec = P(None, *axes) if vec else P(*axes)

        def make(m):
            return jax.jit(shard_map(body, mesh=mesh, in_specs=(fspec, P(), P()),
                                     out_specs=fspec))

    def scatter(field, idx, vals):
        idx = np.asarray(idx, dtype=np.int32).reshape(-1, 3)
        vals = np.asarray(vals, dtype=np.float64)
        m = idx.shape[0]
        b = _bucket(m)
        pad = b - m
        if pad:
            idx = np.concatenate([idx, np.zeros((pad, 3), np.int32)])
            z = np.zeros((vals.shape[0], pad) if vec else (pad,))
            vals = np.concatenate([vals, z], axis=-1)
        if b not in progs:
            progs[b] = make(b)
        return progs[b](field, jnp.asarray(idx), jnp.asarray(vals))

    return scatter


def make_grid_gather(sharding, comp=0):
    """`gather(U, idx)`: host values of U[comp] at global cells idx (M, 3)."""
    progs = {}
    mesh = getattr(sharding, "mesh", None)
    if mesh is None or mesh.size == 1:
        def make(m):
            return jax.jit(lambda U, idx: U[comp, idx[:, 0], idx[:, 1], idx[:, 2]])
    else:
        from jax.sharding import PartitionSpec as P
        from luna.utils.shmap import shard_map
        spec = tuple(sharding.spec)[1:]
        axes = (spec + (None,) * 3)[:3]
        names = [a for a in axes if a is not None]

        def body(U, idx):
            nloc = U.shape[1:]
            org = jnp.stack([jax.lax.axis_index(axes[a]) * nloc[a] if axes[a] is not None
                             else 0 for a in range(3)])
            loc = idx - org
            own = jnp.all((loc >= 0) & (loc < jnp.asarray(nloc)), axis=1)
            loc = jnp.clip(loc, 0, jnp.asarray(nloc) - 1)
            v = jnp.where(own, U[comp, loc[:, 0], loc[:, 1], loc[:, 2]], 0.0)
            return jax.lax.psum(v, tuple(names))

        def make(m):
            return jax.jit(shard_map(body, mesh=mesh, in_specs=(P(None, *axes), P()),
                                     out_specs=P()))

    def gather(U, idx):
        idx = np.asarray(idx, dtype=np.int32).reshape(-1, 3)
        m = idx.shape[0]
        b = _bucket(m)
        if b > m:
            idx = np.concatenate([idx, np.zeros((b - m, 3), np.int32)])
        if b not in progs:
            progs[b] = make(b)
        return np.asarray(progs[b](U, jnp.asarray(idx)))[:m]

    return gather
