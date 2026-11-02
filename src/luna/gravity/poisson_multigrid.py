"""Multigrid (V-cycle + FMG) Poisson solver for ∇²Φ = 4πGρ on a uniform grid, one device or shard_map-sharded.

`boundary`: 0 periodic, 1 Neumann, 2 reflective (= Neumann for Φ), 3 isolated (multipole Dirichlet faces).
"""

import jax
import jax.numpy as jnp

from luna.gravity.poisson_bc import make_halo_3d


# Memo of the sharded solver's jitted programs, keyed on every option: a rebuild per call would retrace every step.
_SHARDMAP_PROGRAM_CACHE = {}
_SHARDMAP_PROGRAM_CACHE_MAX = 64


def _cache_put(key, prog):
    """Memoize `prog`, dropping the oldest entry beyond the bound."""
    while len(_SHARDMAP_PROGRAM_CACHE) >= _SHARDMAP_PROGRAM_CACHE_MAX:
        _SHARDMAP_PROGRAM_CACHE.pop(next(iter(_SHARDMAP_PROGRAM_CACHE)))
    _SHARDMAP_PROGRAM_CACHE[key] = prog


def laplacian_7pt(phi, dx, halo_fn):
    """7-point Laplacian: ∇²Φ_ijk ≈ (Σ_neighbours - 6Φ_ijk) / dx²."""
    phi_pad = halo_fn(phi)
    nbr_sum = (
        phi_pad[2:, 1:-1, 1:-1] + phi_pad[:-2, 1:-1, 1:-1]
        + phi_pad[1:-1, 2:, 1:-1] + phi_pad[1:-1, :-2, 1:-1]
        + phi_pad[1:-1, 1:-1, 2:] + phi_pad[1:-1, 1:-1, :-2]
    )
    return (nbr_sum - 6.0 * phi) / (dx * dx)


def residual(phi, rhs, dx, halo_fn):
    """r = rhs - ∇²Φ."""
    return rhs - laplacian_7pt(phi, dx, halo_fn)


def _make_red_mask(shape):
    """Boolean mask: True where (i+j+k) is even (red cells)."""
    Nx, Ny, Nz = shape
    i = jnp.arange(Nx)[:, None, None]
    j = jnp.arange(Ny)[None, :, None]
    k = jnp.arange(Nz)[None, None, :]
    return ((i + j + k) % 2) == 0


def _gs_full_update(phi, rhs, dx, halo_fn):
    """Gauss-Seidel target for every cell; the red/black mask picks which cells take it."""
    phi_pad = halo_fn(phi)
    nbr_sum = (
        phi_pad[2:, 1:-1, 1:-1] + phi_pad[:-2, 1:-1, 1:-1]
        + phi_pad[1:-1, 2:, 1:-1] + phi_pad[1:-1, :-2, 1:-1]
        + phi_pad[1:-1, 1:-1, 2:] + phi_pad[1:-1, 1:-1, :-2]
    )
    return (nbr_sum - dx * dx * rhs) / 6.0


def rbgs_smoother(phi, rhs, dx, halo_fn, n_iter):
    """``n_iter`` red-black Gauss-Seidel iterations: all red cells (i+j+k even), then all black."""
    red_mask = _make_red_mask(phi.shape)

    def body(_, phi):
        phi_target = _gs_full_update(phi, rhs, dx, halo_fn)
        phi = jnp.where(red_mask, phi_target, phi)
        phi_target = _gs_full_update(phi, rhs, dx, halo_fn)
        phi = jnp.where(red_mask, phi, phi_target)
        return phi

    return jax.lax.fori_loop(0, n_iter, body, phi)


def coarsest_sweeps(n_coarsest, shape, halo_fn):
    """Coarsest-grid RBGS sweeps: `n_coarsest`, or (None) 4 on a whole periodic grid of side <= 4, else 50."""
    if n_coarsest is not None:
        return int(n_coarsest)
    if getattr(halo_fn, "whole_periodic", False) and max(shape) <= 4:
        return 4
    return 50


def rbgs_smoother_ca(phi, rhs, dx, halo_w_fn, n_iter):
    """``n_iter`` RBGS iterations after ONE width-2·n_iter halo exchange, on a shrinking frame.

    Needs even local dims and shard offsets (static parity); ``halo_w_fn(field, width)`` gives the wide halo.
    """
    w = 2 * n_iter
    E = halo_w_fn(phi, w)
    R = halo_w_fn(rhs, w)
    dx2 = dx * dx
    for k in range(w):
        nbr = (E[2:, 1:-1, 1:-1] + E[:-2, 1:-1, 1:-1]
               + E[1:-1, 2:, 1:-1] + E[1:-1, :-2, 1:-1]
               + E[1:-1, 1:-1, 2:] + E[1:-1, 1:-1, :-2])
        o = k + 1                                # frame origin offset so far
        Rc = R[o:-o, o:-o, o:-o]
        target = (nbr - dx2 * Rc) / 6.0
        red = _make_red_mask(target.shape)
        if (3 * o) % 2 == 1:                     # frame origin parity flip
            red = ~red
        want_red = (k % 2 == 0)                  # red app, then black, ...
        E = jnp.where(red if want_red else ~red, target, E[1:-1, 1:-1, 1:-1])
    return E


def cheb_smoother_ca(phi, rhs, dx, halo_w_fn, n_iter, alpha=4.0):
    """Degree-2·n_iter Chebyshev-Jacobi smoothing after ONE wide halo exchange (Adams et al. 2003).

    Smooths D⁻¹A over [λmax/α, λmax], λmax = 2 for the 7-point Laplacian.
    """
    k = 2 * n_iter
    E = halo_w_fn(phi, k)
    R = halo_w_fn(rhs, k)
    lmax = 2.0
    lmin = lmax / alpha
    theta = 0.5 * (lmax + lmin)
    delta = 0.5 * (lmax - lmin)
    inv_diag = -(dx * dx) / 6.0
    sigma = theta / delta
    rho = 1.0 / sigma
    d = None
    for j in range(k):
        nbr = (E[2:, 1:-1, 1:-1] + E[:-2, 1:-1, 1:-1]
               + E[1:-1, 2:, 1:-1] + E[1:-1, :-2, 1:-1]
               + E[1:-1, 1:-1, 2:] + E[1:-1, 1:-1, :-2])
        core = E[1:-1, 1:-1, 1:-1]
        o = j + 1
        lap = (nbr - 6.0 * core) / (dx * dx)
        r = (R[o:-o, o:-o, o:-o] - lap) * inv_diag
        if d is None:
            d = r / theta
        else:
            rho_new = 1.0 / (2.0 * sigma - rho)
            d = rho_new * rho * d[1:-1, 1:-1, 1:-1] \
                + (2.0 * rho_new / delta) * r
            rho = rho_new
        E = core + d
    return E


def restrict(field):
    """8 fine cells → 1 coarse cell by plain average (even dims only)."""
    Nx, Ny, Nz = field.shape
    if Nx % 2 or Ny % 2 or Nz % 2:
        raise ValueError(f"restrict requires even dims; got {field.shape}")
    coarse = field.reshape(Nx // 2, 2, Ny // 2, 2, Nz // 2, 2).mean(axis=(1, 3, 5))
    return coarse


def prolong(field, halo_fn=None):
    """Trilinear prolongation, 1 coarse cell → 8 fine cells; ``halo_fn`` None pads with the edge value."""
    Nx, Ny, Nz = field.shape

    if halo_fn is None:
        c_full = jnp.pad(field, ((1, 1), (1, 1), (1, 1)), mode="edge")
    else:
        c_full = halo_fn(field)

    # one axis at a time; y and z are still padded here
    left_x = (3.0 * c_full[1:-1] + c_full[:-2]) * 0.25
    right_x = (3.0 * c_full[1:-1] + c_full[2:]) * 0.25
    fx = jnp.stack([left_x, right_x], axis=1).reshape(2 * Nx, Ny + 2, Nz + 2)

    left_y = (3.0 * fx[:, 1:-1] + fx[:, :-2]) * 0.25
    right_y = (3.0 * fx[:, 1:-1] + fx[:, 2:]) * 0.25
    fxy = jnp.stack([left_y, right_y], axis=2).reshape(2 * Nx, 2 * Ny, Nz + 2)

    left_z = (3.0 * fxy[:, :, 1:-1] + fxy[:, :, :-2]) * 0.25
    right_z = (3.0 * fxy[:, :, 1:-1] + fxy[:, :, 2:]) * 0.25
    return jnp.stack([left_z, right_z], axis=3).reshape(2 * Nx, 2 * Ny, 2 * Nz)


def _max_levels(shape, min_per_side=4):
    """Multigrid depth (1 = no coarsening): coarsen while every axis stays even and >= min_per_side."""
    dims = (int(shape),) if isinstance(shape, int) else tuple(int(s) for s in shape)
    n = 1
    while all((d >> n) >= min_per_side and (d >> (n - 1)) % 2 == 0 for d in dims):
        n += 1
    return n


def shard_levels(nloc, min_side=2):
    """Levels a shard holds with even sides >= min_side, so restriction and red-black colouring are the whole grid's."""
    k = 0
    while all((int(d) >> k) % 2 == 0 and (int(d) >> k) >= min_side for d in nloc):
        k += 1
    return max(k, 1)


def gather_grid(a, mesh):
    """The whole grid from this shard's block (or a tuple of blocks), on every device: one all-gather over the mesh."""
    many = isinstance(a, tuple)
    blocks = jnp.stack(a) if many else a[None]
    pd = [int(mesh.shape[name]) for name, _ in AXIS_SHARDS]
    if max(pd) > 1:
        # one collective over all devices: per-axis gathers would split sub-communicators on every run
        g = jax.lax.all_gather(blocks, tuple(name for name, _ in AXIS_SHARDS), axis=0)
        k, (nx, ny, nz) = blocks.shape[0], blocks.shape[1:]
        g = g.reshape(pd[0], pd[1], pd[2], k, nx, ny, nz).transpose(3, 0, 4, 1, 5, 2, 6)
        blocks = g.reshape(k, pd[0] * nx, pd[1] * ny, pd[2] * nz)
    return tuple(blocks) if many else blocks[0]


def own_block(a, mesh):
    """This device's block of a whole-grid array (in a shard_map body)."""
    for name, ax in AXIS_SHARDS:
        n = int(mesh.shape[name])
        if n > 1:
            loc = a.shape[ax] // n
            a = jax.lax.dynamic_slice_in_dim(a, jax.lax.axis_index(name) * loc, loc, axis=ax)
    return a


def split_levels(gshape, mesh, n_levels=None, agg_n=32, min_local=2):
    """(levels on the shards, levels in all): the shards coarsen to the root grid of about `agg_n` cells, then gather."""
    total = _max_levels(gshape) if n_levels is None else int(n_levels)
    pdims = [int(mesh.shape[name]) for name, _ in AXIS_SHARDS]
    if max(pdims) == 1:
        return total, total
    agg_eff = max(int(agg_n or 0), 2 * max(pdims))      # >= 2 cells per shard side on the root grid
    n_rule = max(1, (max(int(gshape[0]) // agg_eff, 1)).bit_length())
    nloc = [int(g) // p for g, p in zip(gshape, pdims)]
    return max(1, min(n_rule, total, shard_levels(nloc, min_local))), total


def gathered_coarse(mesh, halo_fn, depth, n_pre, n_post, n_coarsest, n_vcycles_per_level=1):
    """(coarse_fn, coarse_top_fn): a V-cycle / an FMG from the coarsest shard level down `depth` more, gathered."""
    if depth < 1:
        return None, None

    def coarse_fn(phi_l, rhs_l, dxc):
        e = vcycle(*gather_grid((phi_l, rhs_l), mesh), dxc, halo_fn, depth, n_pre, n_post, n_coarsest)
        return own_block(e, mesh)

    def coarse_top_fn(phi_l, rhs_l, dxc):
        return own_block(fmg_solve(gather_grid(rhs_l, mesh), dxc, halo_fn, depth + 1, n_pre, n_post,
                                   n_coarsest, n_vcycles_per_level), mesh)

    return coarse_fn, coarse_top_fn


def _resolve_n_levels(rho, n_levels):
    """`n_levels`, defaulting to the deepest ladder with >= 4 cells per side."""
    return _max_levels(rho.shape) if n_levels is None else n_levels


def _finish(phi, rhs, dx, halo_fn, return_residual):
    """phi, or (phi, residual ∞-norm) if `return_residual`."""
    if return_residual:
        return phi, jnp.max(jnp.abs(residual(phi, rhs, dx, halo_fn)))
    return phi


def vcycle(phi, rhs, dx, halo_fn, level, n_pre, n_post, n_coarsest, halo_inner=None,
           coarse_fn=None, smoother=None):
    """Recursive V-cycle; `level` counts down to 0 (coarsest).

    halo_inner: halo of the error subproblems and prolongation (isolated BC); None uses `halo_fn`.
    coarse_fn(phi, rhs, dx): replaces the level-0 smoothing (shard_map agglomeration).
    """
    if level == 0:
        if coarse_fn is not None:
            return coarse_fn(phi, rhs, dx)
        return rbgs_smoother(phi, rhs, dx, halo_fn, coarsest_sweeps(n_coarsest, phi.shape, halo_fn))

    smooth = (smoother if smoother is not None
              else lambda p, r, d, n: rbgs_smoother(p, r, d, halo_fn, n))
    inner = halo_fn if halo_inner is None else halo_inner
    phi = smooth(phi, rhs, dx, n_pre)
    r = residual(phi, rhs, dx, halo_fn)
    r_c = restrict(r)
    e_c = vcycle(
        jnp.zeros_like(r_c), r_c, 2.0 * dx, inner, level - 1, n_pre, n_post, n_coarsest,
        halo_inner=halo_inner, coarse_fn=coarse_fn, smoother=smoother,
    )
    phi = phi + prolong(e_c, halo_fn=inner)
    phi = smooth(phi, rhs, dx, n_post)
    return phi


def repeat_vcycles(phi, rhs, dx, halo_fn, top, n, n_pre, n_post,
                   n_coarsest, coarse_fn=None, smoother=None):
    """``n`` top-level V-cycles in a ``fori_loop``, so the body compiles once."""
    if n <= 1:
        return vcycle(phi, rhs, dx, halo_fn, top, n_pre, n_post, n_coarsest,
                      coarse_fn=coarse_fn, smoother=smoother) if n == 1 else phi
    body = lambda _, ph: vcycle(ph, rhs, dx, halo_fn, top, n_pre, n_post,
                                n_coarsest, coarse_fn=coarse_fn,
                                smoother=smoother)
    return jax.lax.fori_loop(0, n, body, phi)


def rms(x):
    """Root mean square of a whole (unsharded) grid."""
    return jnp.sqrt(jnp.mean(x * x))


def extrapolated_seed(seed, prev, q):
    """psi_n + q (psi_n - psi_n-1), q = dt_n / dt_n-1; the carried psi_n when prev is None."""
    return seed if prev is None else seed + q * (seed - prev)


def cycles_to_tol(phi, rhs, dx, halo_fn, top, n_pre, n_post, n_coarsest, tol, max_cycles,
                  mixed=False, halo_inner=None, coarse_fn=None, smoother=None, norm=rms):
    """(phi, cycles, |r| / |rhs|) after V-cycles until |r| <= tol |rhs| (1 to max_cycles; mixed: FP32 cycles)."""
    def resid(ph):
        return rhs - laplacian_7pt(ph, dx, halo_fn)

    if mixed:
        # the FP32 correction has zero face values: its top level takes the error-subproblem halo
        top_e = halo_fn if halo_inner is None else halo_inner

        def cycle(x):
            ph, r = x
            r32 = r.astype(jnp.float32)
            ph = ph + vcycle(jnp.zeros_like(r32), r32, dx, top_e, top, n_pre, n_post, n_coarsest,
                             halo_inner=halo_inner, coarse_fn=coarse_fn, smoother=smoother)
            return ph, resid(ph)
        x0, measure = (phi, resid(phi)), lambda x: norm(x[1])
    else:
        def cycle(ph):
            return vcycle(ph, rhs, dx, halo_fn, top, n_pre, n_post, n_coarsest,
                          halo_inner=halo_inner, coarse_fn=coarse_fn, smoother=smoother)
        x0, measure = phi, lambda ph: norm(resid(ph))
    f_norm = norm(rhs)

    def cond(c):
        k, _, rn = c
        return (k < max_cycles) & (rn > tol * f_norm)

    def body(c):
        k, x, _ = c
        x = cycle(x)
        return k + 1, x, measure(x)

    x1 = cycle(x0)                       # the first cycle always runs, outside the loop
    k, x, rn = jax.lax.while_loop(cond, body, (jnp.int32(1), x1, measure(x1)))
    return (x[0] if mixed else x), k, rn / f_norm


def solve_poisson(
    rho,
    dx,
    fourpiG,
    boundary,
    n_levels=None,
    n_cycles=10,
    n_pre=2,
    n_post=2,
    n_coarsest=50,
    tol=None,
    phi_init=None,
    return_residual=False,
):
    """Solve ∇²Φ = 4πGρ with a fixed `n_cycles` V-cycles from `phi_init` (zeros if None).

    tol: must be None (no early exit). return_residual: also return the residual ∞-norm.
    """
    n_levels = _resolve_n_levels(rho, n_levels)
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1; got {n_levels}")
    if tol is not None:
        raise ValueError("solve_poisson: tol-based early exit is not implemented; "
                         "the solver runs a fixed n_cycles (pass tol=None)")
    rhs = fourpiG * rho
    phi = phi_init if phi_init is not None else jnp.zeros_like(rho)
    if boundary == 3:
        # isolated: multipole-Dirichlet halo at the top, zero Dirichlet for the error subproblems
        from luna.gravity.poisson_bc import (make_linear_extrap_dirichlet_halo_3d,
                                             make_odd_dirichlet_halo_3d,
                                             multipole_dirichlet_face_values)
        top = make_linear_extrap_dirichlet_halo_3d(
            multipole_dirichlet_face_values(rho, dx, fourpiG))
        inner = make_odd_dirichlet_halo_3d()
        phi = jax.lax.fori_loop(0, n_cycles, lambda _, ph: vcycle(
            ph, rhs, dx, top, n_levels - 1, n_pre, n_post, n_coarsest,
            halo_inner=inner), phi)
        return _finish(phi, rhs, dx, top, return_residual)

    halo_fn = make_halo_3d(boundary)

    phi = repeat_vcycles(phi, rhs, dx, halo_fn, n_levels - 1, n_cycles,
                         n_pre, n_post, n_coarsest)

    return _finish(phi, rhs, dx, halo_fn, return_residual)


AXIS_SHARDS = [("x", 0), ("y", 1), ("z", 2)]


def _shardmap_ladder(gshape, boundary, mesh, n_levels, n_pre, n_post,
                     n_coarsest, agg_n, ca="auto", smoother="rbgs", n_vcycles_per_level=1):
    """The sharded ladder: (shard levels, coarse_fn, coarse_top_fn, halo_fn, smoother_fn, ca, smoother)."""

    from luna.gravity.poisson_bc import make_halo_3d
    from luna.gravity.poisson_bc_shardmap import make_scalar_halo_3d_shardmap

    max_shards = max(int(mesh.shape[a]) for a, _ in AXIS_SHARDS)
    n_sh, total = split_levels(gshape, mesh, n_levels, agg_n)
    # the gathered root grid keeps the one-pad halo: in place it was 12% slower there (CODE_NOTES, gravity)
    coarse_fn, coarse_top_fn = gathered_coarse(mesh, make_halo_3d(boundary, inplace=False), total - n_sh,
                                               n_pre, n_post, n_coarsest, n_vcycles_per_level)

    halo_fn = make_scalar_halo_3d_shardmap(mesh)(boundary)

    # Communication-avoiding smoothing only on "1": "auto" is off, it lost on every mesh measured (docs/GRAVITY.md).
    _ca, _smoo = str(ca), str(smoother)
    ca_on = _ca == "1"
    smoother = None
    if ca_on and max_shards > 1 and boundary in (0, 1):
        _ca_fn = cheb_smoother_ca if _smoo == "cheb" else rbgs_smoother_ca

        def smoother(p, r, d, n):
            if min(p.shape) < 2 * n:
                return rbgs_smoother(p, r, d, halo_fn, n)
            return _ca_fn(p, r, d, halo_fn, n)

    return n_sh, coarse_fn, coarse_top_fn, halo_fn, smoother, _ca, _smoo


def solve_poisson_fmg_shardmap(
    rho,
    dx,
    fourpiG,
    boundary,
    mesh,
    n_levels=None,
    n_vcycles_per_level=1,
    n_extra_vcycles=0,
    n_pre=2,
    n_post=2,
    n_coarsest=50,
    subtract_mean=True,
    agg_n=32,
    phi_init=None,
    warm_cycles=1,
    mixed=False,
    ca="auto",
    smoother="rbgs",
    phi_prev=None,
    q=0.0,
    tol=0.0,
    max_cycles=8,
):
    """Sharded FMG solve, or V-cycles from ``phi_init`` (+ q (phi_init - phi_prev)); tol > 0: `cycles_to_tol`'s tuple."""
    from functools import partial
    from luna.utils.shmap import shard_map
    from jax.sharding import PartitionSpec as P

    if boundary == 3:
        from luna.gravity.isolated import solve_isolated_shardmap
        nl = None if n_levels is None else int(n_levels)
        return solve_isolated_shardmap(
            rho, dx, fourpiG, mesh, n_levels=nl, n_vcycles_per_level=n_vcycles_per_level,
            n_extra_vcycles=n_extra_vcycles, n_pre=n_pre, n_post=n_post,
            n_coarsest=n_coarsest, phi_init=phi_init, warm_cycles=warm_cycles, agg_n=agg_n,
            mixed=mixed, phi_prev=phi_prev, q=q, tol=tol, max_cycles=max_cycles)

    n_sh, coarse_fn, coarse_top_fn, halo_fn, smoother, _ca, _smoo = _shardmap_ladder(
        rho.shape, boundary, mesh, n_levels, n_pre, n_post, n_coarsest, agg_n,
        ca=ca, smoother=smoother, n_vcycles_per_level=n_vcycles_per_level)

    if subtract_mean and boundary == 0:
        rho_src = rho - jnp.mean(rho)
    else:
        rho_src = rho

    _mixed = bool(mixed)
    # Program memo for eager callers only: a traced dx/fourpiG cannot key it, and the enclosing jit caches.
    if isinstance(dx, jax.core.Tracer) or isinstance(fourpiG, jax.core.Tracer):
        _key = None
    else:
        _key = (tuple(sorted((k, int(v)) for k, v in mesh.shape.items())),
                tuple(str(d) for d in mesh.devices.flat), tuple(rho.shape),
                float(dx),
                float(fourpiG), boundary, n_levels, n_sh, n_vcycles_per_level,
                n_extra_vcycles, n_pre, n_post, n_coarsest, agg_n,
                phi_init is not None, warm_cycles, _ca, _smoo, _mixed,
                str(rho_src.dtype), phi_prev is not None, float(tol), int(max_cycles))
        _hit = _SHARDMAP_PROGRAM_CACHE.get(_key)
        if _hit is not None:
            if phi_init is None:
                return _hit(rho_src)
            if phi_prev is None:
                return _hit(rho_src, phi_init)
            return _hit(rho_src, phi_init, phi_prev, jnp.asarray(q, rho_src.dtype))

    if phi_init is not None:
        # mixed: FP64 residual of the carried phi, FP32 V-cycle on the correction, FP64 accumulate
        grid = P("x", "y", "z")
        n_tot = float(rho.size)

        def norm(x):
            return jnp.sqrt(jax.lax.psum(jnp.sum(x * x), ("x", "y", "z")) / n_tot)

        @partial(jax.jit, static_argnames=())
        @partial(shard_map, mesh=mesh,
                 in_specs=(grid, grid, grid, P()) if phi_prev is not None else (grid, grid),
                 out_specs=(grid, P(), P()) if tol > 0 else grid)
        def _local_vcycles(rho_local, phi_local, *extrap):
            rhs_local = fourpiG * rho_local
            if extrap:
                phi_local = extrapolated_seed(phi_local, *extrap)
            if tol > 0:
                return cycles_to_tol(phi_local, rhs_local, dx, halo_fn, n_sh - 1, n_pre, n_post,
                                     n_coarsest, tol, max_cycles, mixed=_mixed, coarse_fn=coarse_fn,
                                     smoother=smoother, norm=norm)
            n_cyc = (max(1, n_extra_vcycles) if warm_cycles is None
                     else warm_cycles)
            if _mixed:
                def _dc(_, ph):
                    r = (rhs_local
                         - laplacian_7pt(ph, dx, halo_fn)).astype(jnp.float32)
                    e = vcycle(
                        jnp.zeros_like(r), r, dx, halo_fn, n_sh - 1,
                        n_pre, n_post, n_coarsest, coarse_fn=coarse_fn,
                        smoother=smoother,
                    )
                    return ph + e
                phi_local = jax.lax.fori_loop(0, n_cyc, _dc, phi_local)
            else:
                phi_local = repeat_vcycles(
                    phi_local, rhs_local, dx, halo_fn, n_sh - 1, n_cyc,
                    n_pre, n_post, n_coarsest, coarse_fn=coarse_fn,
                    smoother=smoother)
            return phi_local

        if _key is not None:
            _cache_put(_key, _local_vcycles)
        if phi_prev is not None:
            return _local_vcycles(rho_src, phi_init, phi_prev, jnp.asarray(q, rho_src.dtype))
        return _local_vcycles(rho_src, phi_init)

    @partial(jax.jit, static_argnames=())
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P("x", "y", "z"),),
        out_specs=P("x", "y", "z"),
    )
    def _local_fmg(rho_local):
        rhs_local = fourpiG * rho_local
        phi_local = fmg_solve(
            rhs_local, dx, halo_fn, n_sh, n_pre, n_post, n_coarsest,
            n_vcycles_per_level, coarse_fn=coarse_fn, smoother=smoother,
            coarse_top_fn=coarse_top_fn,
        )
        phi_local = repeat_vcycles(
            phi_local, rhs_local, dx, halo_fn, n_sh - 1, n_extra_vcycles,
            n_pre, n_post, n_coarsest, coarse_fn=coarse_fn, smoother=smoother)
        return phi_local

    if _key is not None:
        _cache_put(_key, _local_fmg)
    return _local_fmg(rho_src)


def fmg_solve(
    rhs,
    dx,
    halo_fn,
    n_levels,
    n_pre=2,
    n_post=2,
    n_coarsest=50,
    n_vcycles_per_level=1,
    coarse_fn=None,
    smoother=None,
    coarse_top_fn=None,
):
    """Full Multigrid solve of ∇²Φ = rhs: coarsest solve, then prolong with ``n_vcycles_per_level`` V-cycles per level."""
    rhs_levels = [rhs]
    dx_levels = [dx]
    for _ in range(n_levels - 1):
        rhs_levels.append(restrict(rhs_levels[-1]))
        dx_levels.append(2.0 * dx_levels[-1])

    return _fmg_ladder(
        rhs_levels, dx_levels, [halo_fn] * n_levels, None,
        n_pre, n_post, n_coarsest, n_vcycles_per_level, coarse_fn=coarse_fn,
        smoother=smoother, coarse_top_fn=coarse_top_fn,
    )


def _fmg_ladder(rhs_levels, dx_levels, halo_tops, halo_inner,
                n_pre, n_post, n_coarsest, n_vcycles_per_level, coarse_fn=None,
                smoother=None, halo_prolong=None, coarse_top_fn=None):
    """FMG walk-up from the coarsest level of `rhs_levels` (finest first).

    halo_tops[L] smooths level L; halo_inner (None: halo_tops[L]) the error subproblems; halo_prolong the prolongation.
    """
    n_levels = len(rhs_levels)
    phi = jnp.zeros_like(rhs_levels[-1])
    # coarse_top_fn: the coarsest level's own problem; coarse_fn: the coarsest error subproblem
    if coarse_top_fn is not None:
        phi = coarse_top_fn(phi, rhs_levels[-1], dx_levels[-1])
    elif coarse_fn is not None:
        phi = coarse_fn(phi, rhs_levels[-1], dx_levels[-1])
    else:
        phi = rbgs_smoother(phi, rhs_levels[-1], dx_levels[-1], halo_tops[-1],
                            coarsest_sweeps(n_coarsest, phi.shape, halo_tops[-1]))

    for L in range(n_levels - 2, -1, -1):
        if halo_prolong is not None:
            phi = prolong(phi, halo_fn=halo_prolong)
        else:
            phi = prolong(phi, halo_fn=halo_tops[L] if halo_inner is None else halo_inner)
        depth = (n_levels - 1) - L
        if n_vcycles_per_level == 1 or halo_inner is not None:
            for _ in range(n_vcycles_per_level):
                phi = vcycle(
                    phi, rhs_levels[L], dx_levels[L], halo_tops[L], depth,
                    n_pre, n_post, n_coarsest, halo_inner=halo_inner,
                    coarse_fn=coarse_fn, smoother=smoother,
                )
        else:
            phi = repeat_vcycles(
                phi, rhs_levels[L], dx_levels[L], halo_tops[L], depth,
                n_vcycles_per_level, n_pre, n_post, n_coarsest,
                coarse_fn=coarse_fn, smoother=smoother)

    return phi


def solve_poisson_fmg(
    rho,
    dx,
    fourpiG,
    boundary,
    n_levels=None,
    n_vcycles_per_level=1,
    n_extra_vcycles=0,
    n_pre=2,
    n_post=2,
    n_coarsest=50,
    return_residual=False,
):
    """FMG solve of ∇²Φ = 4πGρ, then `n_extra_vcycles` V-cycles at the finest level."""
    n_levels = _resolve_n_levels(rho, n_levels)

    if boundary == 3:
        return _solve_poisson_fmg_isolated(
            rho, dx, fourpiG, n_levels, n_vcycles_per_level,
            n_extra_vcycles, n_pre, n_post, n_coarsest, return_residual,
        )

    halo_fn = make_halo_3d(boundary)
    rhs = fourpiG * rho

    phi = fmg_solve(rhs, dx, halo_fn, n_levels, n_pre, n_post, n_coarsest, n_vcycles_per_level)

    phi = repeat_vcycles(phi, rhs, dx, halo_fn, n_levels - 1, n_extra_vcycles,
                         n_pre, n_post, n_coarsest)

    return _finish(phi, rhs, dx, halo_fn, return_residual)


def _solve_poisson_fmg_isolated(
    rho, dx, fourpiG, n_levels, n_vcycles_per_level, n_extra_vcycles,
    n_pre, n_post, n_coarsest, return_residual,
):
    """FMG with isolated faces: each level's multipole halo from its own ρ (Guillet & Teyssier 2011)."""
    from luna.gravity.poisson_bc import (
        make_linear_extrap_dirichlet_halo_3d,
        make_odd_dirichlet_halo_3d,
        multipole_dirichlet_face_values,
    )
    halo_inner = make_odd_dirichlet_halo_3d()

    rho_levels = [rho]
    for _ in range(n_levels - 1):
        rho_levels.append(restrict(rho_levels[-1]))
    rhs_levels = [fourpiG * r for r in rho_levels]
    dx_levels = [dx * (2 ** L) for L in range(n_levels)]

    halo_tops = [
        make_linear_extrap_dirichlet_halo_3d(
            multipole_dirichlet_face_values(rho_levels[L], dx_levels[L], fourpiG)
        )
        for L in range(n_levels)
    ]

    # prolong by extrapolation: Φ does not vanish on the faces, only the corrections do
    from luna.gravity.isolated import extrap_pad
    phi = _fmg_ladder(
        rhs_levels, dx_levels, halo_tops, halo_inner,
        n_pre, n_post, n_coarsest, n_vcycles_per_level,
        halo_prolong=lambda f: extrap_pad(f, 1),
    )

    for _ in range(n_extra_vcycles):
        phi = vcycle(
            phi, rhs_levels[0], dx_levels[0], halo_tops[0], n_levels - 1,
            n_pre, n_post, n_coarsest, halo_inner=halo_inner,
        )

    return _finish(phi, rhs_levels[0], dx, halo_tops[0], return_residual)
