"""Isolated gravity boundaries on a device mesh: the multipole-Dirichlet FMG and extrapolated face ghosts.

See docs/WALL_M3.md.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from luna.gravity.poisson_bc import pad_inplace
from luna.utils.halo_exchange_shardmap import _axis_ghosts

AXES = ("x", "y", "z")


def _take(a, axis, i):
    sl = [slice(None)] * a.ndim
    sl[axis] = slice(i, i + 1) if i >= 0 else slice(i, i + 1 if i < -1 else None)
    return a[tuple(sl)]


def _extrap_ghosts(phi, axis, width):
    """(low, high) ghost slabs along `axis`: width 1 quadratic, width 2 cubic extrapolation."""
    k = 3 if width == 1 else 4
    f = [_take(phi, axis, i) for i in range(k)]
    b = [_take(phi, axis, -1 - i) for i in range(k)]
    if width == 1:
        return 3 * f[0] - 3 * f[1] + f[2], 3 * b[0] - 3 * b[1] + b[2]
    lo1 = 4 * f[0] - 6 * f[1] + 4 * f[2] - f[3]
    lo2 = 4 * lo1 - 6 * f[0] + 4 * f[1] - f[2]
    hi1 = 4 * b[0] - 6 * b[1] + 4 * b[2] - b[3]
    hi2 = 4 * hi1 - 6 * b[0] + 4 * b[1] - b[2]
    return (jnp.concatenate([lo2, lo1], axis=axis),
            jnp.concatenate([hi1, hi2], axis=axis))


def extrap_pad(phi, width):
    """`width`-cell ghosts on all faces by polynomial extrapolation of the interior."""
    return pad_inplace(phi, width, lambda core, ax: _extrap_ghosts(core, ax, width))


def pad_sharded(phi_l, mesh, width, boundary):
    """`width`-cell scalar halo in a shard_map body; box faces periodic (0), edge (1, 2) or extrapolated (3)."""
    def ghosts(core, ax):
        name = AXES[ax]
        n = int(mesh.shape[name])
        lo, hi = _axis_ghosts(core, axis=ax, axis_name=name, n_proc=n,
                              periodic=boundary in (0, 3), width=width)
        if boundary == 3:
            elo, ehi = _extrap_ghosts(core, ax, width)
            idx = jax.lax.axis_index(name)
            lo = jnp.where(idx == 0, elo, lo)
            hi = jnp.where(idx == n - 1, ehi, hi)
        return lo, hi
    return pad_inplace(phi_l, width, ghosts)


def _odd_pad(phi_l, mesh):
    """One-cell halo with -edge ghosts at the box faces (Φ = 0 on the face)."""
    def ghosts(core, ax):
        name = AXES[ax]
        n = int(mesh.shape[name])
        lo, hi = _axis_ghosts(core, axis=ax, axis_name=name, n_proc=n,
                              periodic=True, width=1)
        idx = jax.lax.axis_index(name)
        lo = jnp.where(idx == 0, -_take(core, ax, 0), lo)
        hi = jnp.where(idx == n - 1, -_take(core, ax, -1), hi)
        return lo, hi
    return pad_inplace(phi_l, 1, ghosts)


def make_odd_halo_sharded(mesh):
    return lambda phi_l: _odd_pad(phi_l, mesh)


def _face_phi(rho_l, dx, fourpiG, mesh, n_glob):
    """Monopole + quadrupole Φ of the whole grid on this shard's part of the faces: {(axis, side): 2-D}."""
    G = fourpiG / (4.0 * jnp.pi)
    nloc = rho_l.shape
    coords = []
    for ax, name in enumerate(AXES):
        org = jax.lax.axis_index(name) * nloc[ax]
        coords.append((org + jnp.arange(nloc[ax], dtype=rho_l.dtype) + 0.5) * dx)
    X = coords[0][:, None, None]
    Y = coords[1][None, :, None]
    Z = coords[2][None, None, :]
    w = rho_l * dx ** 3
    ps = partial(jax.lax.psum, axis_name=AXES)
    M = ps(jnp.sum(w))
    inv = 1.0 / (ps(jnp.sum(rho_l)) + 1e-30)
    cx = ps(jnp.sum(rho_l * X)) * inv
    cy = ps(jnp.sum(rho_l * Y)) * inv
    cz = ps(jnp.sum(rho_l * Z)) * inv
    dX, dY, dZ = X - cx, Y - cy, Z - cz
    r2 = dX * dX + dY * dY + dZ * dZ
    Q = dict(xx=ps(jnp.sum(w * (3 * dX * dX - r2))), yy=ps(jnp.sum(w * (3 * dY * dY - r2))),
             zz=ps(jnp.sum(w * (3 * dZ * dZ - r2))), xy=ps(jnp.sum(w * 3 * dX * dY)),
             xz=ps(jnp.sum(w * 3 * dX * dZ)), yz=ps(jnp.sum(w * 3 * dY * dZ)))

    def phi(rx, ry, rz):
        r = jnp.maximum(jnp.sqrt(rx * rx + ry * ry + rz * rz), 0.5 * dx)
        quad = (Q["xx"] * rx * rx + Q["yy"] * ry * ry + Q["zz"] * rz * rz
                + 2 * Q["xy"] * rx * ry + 2 * Q["xz"] * rx * rz + 2 * Q["yz"] * ry * rz)
        return -G * M / r - G * 0.5 * quad / r ** 5

    x, y, z = coords
    L = [n * dx for n in n_glob]
    out = {}
    for side, fx in ((0, 0.0), (1, L[0])):
        out[(0, side)] = phi(fx - cx, y[:, None] - cy, z[None, :] - cz)
    for side, fy in ((0, 0.0), (1, L[1])):
        out[(1, side)] = phi(x[:, None] - cx, fy - cy, z[None, :] - cz)
    for side, fz in ((0, 0.0), (1, L[2])):
        out[(2, side)] = phi(x[:, None] - cx, y[None, :] - cy, fz - cz)
    return out


def make_multipole_halo_sharded(rho_l, dx, fourpiG, mesh, n_glob):
    """Face-centred Dirichlet halo on a mesh: ghost = 2 Φ_face − φ_edge on boundary shards (Guillet & Teyssier 2011)."""
    faces = _face_phi(rho_l, dx, fourpiG, mesh, n_glob)
    idx = [jax.lax.axis_index(n) for n in AXES]
    last = [int(mesh.shape[n]) - 1 for n in AXES]

    def halo(phi_l):
        pp = _odd_pad(phi_l, mesh)
        nx, ny, nz = phi_l.shape
        I = slice(1, None)
        sets = (((0, 0), (0, I, I), phi_l[0]), ((0, 1), (nx + 1, I, I), phi_l[-1]),
                ((1, 0), (I, 0, I), phi_l[:, 0]), ((1, 1), (I, ny + 1, I), phi_l[:, -1]),
                ((2, 0), (I, I, 0), phi_l[:, :, 0]), ((2, 1), (I, I, nz + 1), phi_l[:, :, -1]))
        for (ax, side), where, edge in sets:
            sl = tuple(slice(1, -1) if s is I else s for s in where)
            on_face = idx[ax] == (0 if side == 0 else last[ax])
            pp = pp.at[sl].set(jnp.where(on_face, 2.0 * faces[(ax, side)] - edge, pp[sl]))
        return pp

    return halo


def isolated_fmg_local(rho_l, dx, fourpiG, mesh, n_glob, n_levels, n_pre, n_post,
                       n_coarsest, n_vcycles_per_level=1, n_extra_vcycles=0,
                       phi_init=None, warm_cycles=1, agg_n=32, warm=None):
    """Isolated lap(phi) = fourpiG rho in a shard_map body, root grid gathered; warm: `cycles_to_tol` args, its tuple."""
    from luna.gravity.poisson_bc import make_odd_dirichlet_halo_3d
    from luna.gravity.poisson_multigrid import (_fmg_ladder, _solve_poisson_fmg_isolated, cycles_to_tol,
                                                extrapolated_seed, gather_grid, gathered_coarse, own_block,
                                                restrict, solve_poisson, split_levels, vcycle)
    # the extrapolated face ghosts need 3 cells: an isolated shard level keeps 4 or more
    n_sh, n_levels = split_levels(n_glob, mesh, n_levels, agg_n, min_local=4)
    if warm is not None:
        mixed, prev, q, tol, max_cycles = warm
        seed = extrapolated_seed(phi_init, prev, q)
        n_max = int(max_cycles) if tol > 0 else max(int(warm_cycles), 1)
    if n_sh == 1:
        rho_g = gather_grid(rho_l, mesh)
        if warm is not None:
            from luna.gravity.poisson_bc import (make_linear_extrap_dirichlet_halo_3d,
                                                 multipole_dirichlet_face_values)
            top = make_linear_extrap_dirichlet_halo_3d(multipole_dirichlet_face_values(rho_g, dx, fourpiG))
            phi, k, rel = cycles_to_tol(gather_grid(seed, mesh), fourpiG * rho_g, dx, top, n_levels - 1,
                                        n_pre, n_post, n_coarsest, tol, n_max, mixed=mixed,
                                        halo_inner=make_odd_dirichlet_halo_3d())
            # every device solved the same gathered grid: mark k and rel replicated
            return own_block(phi, mesh), jax.lax.pmax(k, AXES), jax.lax.pmax(rel, AXES)
        if phi_init is None:
            phi = _solve_poisson_fmg_isolated(rho_g, dx, fourpiG, n_levels, n_vcycles_per_level,
                                              n_extra_vcycles, n_pre, n_post, n_coarsest, False)
        else:
            phi = solve_poisson(rho_g, dx, fourpiG, 3, n_levels=n_levels, n_cycles=max(int(warm_cycles), 1),
                                n_pre=n_pre, n_post=n_post, n_coarsest=n_coarsest,
                                phi_init=gather_grid(phi_init, mesh))
        return own_block(phi, mesh)
    n_below = n_levels - n_sh
    inner = make_odd_halo_sharded(mesh)
    # the physical (multipole) BC on the finest level only; corrections are zero on the faces
    coarse, _ = gathered_coarse(mesh, make_odd_dirichlet_halo_3d(inplace=False), n_below, n_pre, n_post, n_coarsest)
    rhs = fourpiG * rho_l
    if warm is not None:
        def norm(x):
            return jnp.sqrt(jax.lax.psum(jnp.sum(x * x), AXES) / float(np.prod(n_glob)))
        top = make_multipole_halo_sharded(rho_l, dx, fourpiG, mesh, n_glob)
        return cycles_to_tol(seed, rhs, dx, top, n_sh - 1, n_pre, n_post, n_coarsest, tol, n_max,
                             mixed=mixed, halo_inner=inner, coarse_fn=coarse, norm=norm)
    if phi_init is not None:
        top = make_multipole_halo_sharded(rho_l, dx, fourpiG, mesh, n_glob)
        body = lambda _, ph: vcycle(ph, rhs, dx, top, n_sh - 1, n_pre, n_post,   # noqa: E731
                                    n_coarsest, halo_inner=inner, coarse_fn=coarse)
        return jax.lax.fori_loop(0, max(int(warm_cycles), 1), body, phi_init)
    rho_levels = [rho_l]
    for _ in range(n_sh - 1):
        rho_levels.append(restrict(rho_levels[-1]))
    dx_levels = [dx * (2 ** L) for L in range(n_sh)]
    rhs_levels = [fourpiG * r for r in rho_levels]
    tops = [make_multipole_halo_sharded(rho_levels[L], dx_levels[L], fourpiG, mesh,
                                        tuple(n >> L for n in n_glob))
            for L in range(n_sh)]
    coarse_top = None
    if n_below > 0:
        def coarse_top(phi_c, rhs_c, dxc):
            return own_block(_solve_poisson_fmg_isolated(
                gather_grid(rho_levels[-1], mesh), dxc, fourpiG, n_below + 1, n_vcycles_per_level, 0,
                n_pre, n_post, n_coarsest, False), mesh)
    phi = _fmg_ladder(rhs_levels, dx_levels, tops, inner, n_pre, n_post, n_coarsest,
                      n_vcycles_per_level, coarse_fn=coarse, coarse_top_fn=coarse_top,
                      halo_prolong=lambda f: pad_sharded(f, mesh, 1, 3))
    for _ in range(n_extra_vcycles):
        phi = vcycle(phi, rhs_levels[0], dx, tops[0], n_sh - 1, n_pre, n_post,
                     n_coarsest, halo_inner=inner, coarse_fn=coarse)
    return phi


_PROGRAMS = {}


def solve_isolated_shardmap(rho, dx, fourpiG, mesh, n_levels=None, n_vcycles_per_level=1,
                            n_extra_vcycles=0, n_pre=2, n_post=2, n_coarsest=50,
                            phi_init=None, warm_cycles=1, agg_n=32, mixed=False, phi_prev=None, q=0.0,
                            tol=0.0, max_cycles=8):
    """`isolated_fmg_local` as one jitted shard_map program (memoized); mixed or tol > 0 run `cycles_to_tol`."""
    from jax.sharding import PartitionSpec as P
    from luna.gravity.poisson_multigrid import _max_levels
    from luna.utils.shmap import shard_map
    n_glob = tuple(int(n) for n in rho.shape)
    if n_levels is None:
        n_levels = _max_levels(n_glob)
    key = (tuple(sorted((k, int(v)) for k, v in mesh.shape.items())),
           tuple(str(d) for d in mesh.devices.flat), n_glob, float(dx), float(fourpiG),
           int(n_levels), int(n_vcycles_per_level), int(n_extra_vcycles), int(n_pre),
           int(n_post), int(n_coarsest), phi_init is not None, int(warm_cycles),
           str(rho.dtype), int(agg_n or 0), bool(mixed), phi_prev is not None, float(tol), int(max_cycles))
    adaptive = phi_init is not None and (mixed or tol > 0 or phi_prev is not None)
    prog = _PROGRAMS.get(key)
    if prog is None:
        sp = P("x", "y", "z")
        if adaptive:
            def body(r, ph, *extrap):
                prev, qq = extrap if extrap else (None, 0.0)
                out = isolated_fmg_local(r, dx, fourpiG, mesh, n_glob, n_levels, n_pre, n_post,
                                         n_coarsest, phi_init=ph, warm_cycles=warm_cycles, agg_n=agg_n,
                                         warm=(mixed, prev, qq, tol, max_cycles))
                return out if tol > 0 else out[0]
            ins = (sp, sp, sp, P()) if phi_prev is not None else (sp, sp)
            prog = jax.jit(shard_map(body, mesh=mesh, in_specs=ins,
                                     out_specs=(sp, P(), P()) if tol > 0 else sp))
        elif phi_init is None:
            def body(r):
                return isolated_fmg_local(r, dx, fourpiG, mesh, n_glob, n_levels, n_pre,
                                          n_post, n_coarsest, n_vcycles_per_level,
                                          n_extra_vcycles, agg_n=agg_n)
            prog = jax.jit(shard_map(body, mesh=mesh, in_specs=(sp,), out_specs=sp))
        else:
            def body(r, ph):
                return isolated_fmg_local(r, dx, fourpiG, mesh, n_glob, n_levels, n_pre,
                                          n_post, n_coarsest, phi_init=ph,
                                          warm_cycles=warm_cycles, agg_n=agg_n)
            prog = jax.jit(shard_map(body, mesh=mesh, in_specs=(sp, sp), out_specs=sp))
        _PROGRAMS[key] = prog
    if phi_init is None:
        return prog(rho)
    if adaptive and phi_prev is not None:
        return prog(rho, phi_init, phi_prev, jnp.asarray(q, rho.dtype))
    return prog(rho, phi_init)
