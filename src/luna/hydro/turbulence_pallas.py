"""The OU kick as one Pallas pass over U: the force built per tile from 1-D tables, never stored."""

import functools

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from luna.hydro.muscl_hancock_pallas import tile_for


def tables(driver, a, x, y, z):
    """Per-pair factors of F_c = 2 sum_p [Gr_cp(x) Er_p(y,z) - Gi_cp(x) Ei_p(y,z)], with Er, Ei by angle addition."""
    hi = jax.lax.Precision.HIGHEST
    dt = driver.dtype
    ar, ai = jnp.real(a).astype(dt), jnp.imag(a).astype(dt)
    px = driver.kx[:, None] * x[None, :]
    cx, sx = jnp.cos(px), jnp.sin(px)
    S = driver.pair_of_mode
    Gr = (jnp.einsum("mp,mc,mx->cpx", S, ar, cx, precision=hi)
          - jnp.einsum("mp,mc,mx->cpx", S, ai, sx, precision=hi))
    Gi = (jnp.einsum("mp,mc,mx->cpx", S, ar, sx, precision=hi)
          + jnp.einsum("mp,mc,mx->cpx", S, ai, cx, precision=hi))
    py = driver.k_pair[:, 0, None] * y[None, :]
    pz = driver.k_pair[:, 1, None] * z[None, :]
    return Gr, Gi, jnp.cos(py), jnp.sin(py), jnp.cos(pz), jnp.sin(pz)


@functools.lru_cache(maxsize=None)
def _call(shape, n_pairs, dtype, interpret, mat=None):
    """pallas_call(h, Gr, Gi, cy, sy, cz, sz, U) -> U with rho F h on the momenta and the matching E."""
    from jax.experimental import pallas as pl
    if not interpret:
        jax.config.update("jax_pallas_use_mosaic_gpu", False)   # ROCm: Triton only
    tx, ty, tz = tile_for(shape)

    def kernel(h_ref, gr_ref, gi_ref, cy_ref, sy_ref, cz_ref, sz_ref, u_ref, o_ref):
        h = h_ref[0]
        bx, by, bz = pl.program_id(0) * tx, pl.program_id(1) * ty, pl.program_id(2) * tz
        sx, sy_, sz_ = pl.ds(bx, tx), pl.ds(by, ty), pl.ds(bz, tz)
        F = [None, None, None]
        for p in range(n_pairs):
            cy, sy = cy_ref[p, sy_][:, None], sy_ref[p, sy_][:, None]
            cz, sz = cz_ref[p, sz_][None, :], sz_ref[p, sz_][None, :]
            er = (cy * cz - sy * sz)[None]                     # cos(ky y + kz z)
            ei = (sy * cz + cy * sz)[None]                     # sin(ky y + kz z)
            for c in range(3):
                t = gr_ref[c, p, sx][:, None, None] * er - gi_ref[c, p, sx][:, None, None] * ei
                F[c] = t if F[c] is None else F[c] + t
        Fx, Fy, Fz = (2.0 * f for f in F)
        Fx, Fy, Fz = Fx.astype(dtype), Fy.astype(dtype), Fz.astype(dtype)
        idx = (sx, sy_, sz_)
        rho, mx, my, mz, E = (u_ref[(c,) + idx] for c in range(5))
        dE = (mx * Fx + my * Fy + mz * Fz) * h + 0.5 * rho * (Fx * Fx + Fy * Fy + Fz * Fz) * h * h
        o_ref[(0,) + idx] = rho
        o_ref[(1,) + idx] = mx + rho * Fx * h
        o_ref[(2,) + idx] = my + rho * Fy * h
        o_ref[(3,) + idx] = mz + rho * Fz * h
        o_ref[(4,) + idx] = E + dE

    out = jax.ShapeDtypeStruct((5,) + tuple(shape), jnp.dtype(dtype), manual_axis_type=mat)
    return pl.pallas_call(kernel, out_shape=out, grid=(shape[0] // tx, shape[1] // ty, shape[2] // tz),
                          input_output_aliases={7: 0}, interpret=interpret)


def apply_local(driver, U, a, h, x, y, z):
    """The kick on one device's block U (5, nx, ny, nz) with its own cell-centre coordinates."""
    if U.ndim != 4 or U.shape[0] != 5:
        raise NotImplementedError(f"Pallas OU kick: a 3-D gas block (5, nx, ny, nz), got {U.shape}")
    Gr, Gi, cy, sy, cz, sz = tables(driver, a, x, y, z)
    interpret = jax.default_backend() == "cpu"
    mat = getattr(jax.typeof(U), "mat", None)
    call = _call(tuple(int(n) for n in U.shape[1:]), int(Gr.shape[1]), jnp.dtype(U.dtype).name, interpret, mat)
    return call(jnp.reshape(jnp.asarray(h, U.dtype), (1,)), Gr, Gi, cy, sy, cz, sz, U)


def make_kick(driver, mesh, spec, n, d):
    """kick(U, a, h) under shard_map on `mesh` with U laid out as `spec`; n, d: global cells and spacing per axis."""
    from luna.utils.shmap import shard_map
    axes = tuple(spec[1:])

    def local(U, a, h):
        coords = []
        for ax, dd, m in zip(axes, d, U.shape[1:]):
            off = jax.lax.axis_index(ax) * m if ax is not None else 0
            coords.append((jnp.arange(m, dtype=driver.dtype) + off + 0.5) * dd)
        return apply_local(driver, U, a, h, *coords)

    return jax.jit(shard_map(local, mesh=mesh, in_specs=(spec, P(), P()), out_specs=spec))
