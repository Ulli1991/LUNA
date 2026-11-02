"""Velocity structure functions on the fly: lags along the grid axes, increments from neighbour blocks."""
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from luna.analysis.fields import AXES, primitives, roll_ahead
from luna.utils.shmap import shard_map


def default_lags(n_global):
    """Powers of two from one cell to half the box."""
    lags, lag = [], 1
    while lag <= n_global // 2:
        lags.append(lag)
        lag *= 2
    return tuple(lags)


def build_structure_functions(mesh, gamma, n_global, orders=8, lags=None):
    """Return sf(U) -> dict: longitudinal and transverse S_p(l) of v and rho^(1/3) v, p = 1..orders."""
    mesh_shape = tuple(int(mesh.shape[a]) for a in AXES)
    lags = tuple(lags or default_lags(n_global))
    p = jnp.arange(1, orders + 1, dtype=jnp.float32)
    n = float(n_global) ** 3

    def _moments(du, d):
        """sum |du_L|^p, sum |du_T|^p over this block, and sum du_L^3 (signed)."""
        L = jnp.abs(du[d]).ravel()
        T = jnp.abs(jnp.stack([du[e] for e in range(3) if e != d])).ravel()
        pp = p.astype(du.dtype)[:, None]
        return (jnp.sum(L[None, :] ** pp, axis=1), jnp.sum(T[None, :] ** pp, axis=1),
                jnp.sum(du[d] ** 3))

    def _sf(U):
        rho, v, _ = primitives(U, gamma)
        w = rho ** (1.0 / 3.0) * v
        acc = {k: [] for k in ("SL_v", "ST_v", "S3L_v", "SL_w", "ST_w", "S3L_w")}
        for lag in lags:
            sums = {k: 0.0 for k in acc}
            for d in range(3):
                for name, f in (("v", v), ("w", w)):
                    du = roll_ahead(f, d, lag, mesh_shape, lead=1) - f
                    sl, st, s3 = _moments(du, d)
                    sums["SL_" + name] = sums["SL_" + name] + sl
                    sums["ST_" + name] = sums["ST_" + name] + st
                    sums["S3L_" + name] = sums["S3L_" + name] + s3
            for k in acc:
                acc[k].append(sums[k])
        out = {"lags": jnp.asarray(lags), "orders": p}
        for k, vals in acc.items():
            total = jax.lax.psum(jnp.stack(vals), AXES)
            out[k] = total / (3.0 * n) if k.startswith(("SL", "S3L")) else total / (6.0 * n)
        return out

    spec = P(None, *AXES)
    return jax.jit(shard_map(_sf, mesh=mesh, in_specs=(spec,), out_specs=P()))
