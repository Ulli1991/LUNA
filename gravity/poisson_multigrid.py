"""Multigrid (V-cycle + FMG) Poisson solver for ∇²Φ = 4πGρ: uniform grid,
7-point Laplacian, red-black Gauss-Seidel smoother, 8-cell restriction,
trilinear prolongation; single-device or shard_map-sharded.

`boundary` follows the LUNA hydro convention: 0 periodic, 1 Neumann,
2 reflective (= Neumann for Φ), 3 isolated (multipole Dirichlet faces).
`n_levels` is capped so the coarsest grid is >= 4 cells per side (per shard).
"""

import jax
import jax.numpy as jnp
from functools import partial

from gravity.poisson_bc import make_halo_3d


# ---------------------------------------------------------------------------
# Stencils
# ---------------------------------------------------------------------------
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
    """r = rhs - ∇²Φ. At the top level rhs = 4πGρ; in recursion rhs is the
    restricted residual and Φ is an error correction."""
    return rhs - laplacian_7pt(phi, dx, halo_fn)


# ---------------------------------------------------------------------------
# Smoother (red-black Gauss-Seidel)
# ---------------------------------------------------------------------------
def _make_red_mask(shape):
    """Boolean mask: True where (i+j+k) is even (red cells)."""
    Nx, Ny, Nz = shape
    i = jnp.arange(Nx)[:, None, None]
    j = jnp.arange(Ny)[None, :, None]
    k = jnp.arange(Nz)[None, None, :]
    return ((i + j + k) % 2) == 0


def _gs_full_update(phi, rhs, dx, halo_fn):
    """Compute the Gauss-Seidel target for every cell (used by red & black
    sweeps; mask selects which cells actually take the new value)."""
    phi_pad = halo_fn(phi)
    nbr_sum = (
        phi_pad[2:, 1:-1, 1:-1] + phi_pad[:-2, 1:-1, 1:-1]
        + phi_pad[1:-1, 2:, 1:-1] + phi_pad[1:-1, :-2, 1:-1]
        + phi_pad[1:-1, 1:-1, 2:] + phi_pad[1:-1, 1:-1, :-2]
    )
    return (nbr_sum - dx * dx * rhs) / 6.0


def rbgs_smoother(phi, rhs, dx, halo_fn, n_iter):
    """Red-black Gauss-Seidel: parallel sweep of all red cells (i+j+k even),
    then all black cells using the freshly-updated reds; ``n_iter`` full
    iterations."""
    red_mask = _make_red_mask(phi.shape)

    def body(_, phi):
        # Red sweep: update cells where mask is True.
        phi_target = _gs_full_update(phi, rhs, dx, halo_fn)
        phi = jnp.where(red_mask, phi_target, phi)
        # Black sweep: update cells where mask is False, using freshly
        # written red values.
        phi_target = _gs_full_update(phi, rhs, dx, halo_fn)
        phi = jnp.where(red_mask, phi, phi_target)
        return phi

    return jax.lax.fori_loop(0, n_iter, body, phi)


def rbgs_smoother_ca(phi, rhs, dx, halo_w_fn, n_iter):
    """Communication-avoiding red-black Gauss-Seidel: ONE width-2·n_iter halo
    exchange, then 2·n_iter stencil applications on a shrinking extended
    frame — the same update sequence as ``n_iter`` standard RBGS iterations
    with 1 exchange instead of 2·n_iter. Periodic/Neumann shards with even
    local dims and even shard offsets (parity is then compile-time static).
    ``halo_w_fn(field, width)`` supplies the wide halo."""
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
    """Communication-avoiding Chebyshev-Jacobi smoother: ONE width-2·n_iter
    halo exchange, then a degree-2·n_iter Chebyshev iteration on the
    shrinking frame — same halo budget as ``rbgs_smoother_ca(n_iter)``, but
    every stencil application advances the polynomial (no red/black halves).
    Smooths D⁻¹A over [λmax/α, λmax]; λmax = 2 exactly for the 7-pt
    Laplacian (Adams et al. 2003; Baker et al. 2011)."""
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


# ---------------------------------------------------------------------------
# Inter-grid transfer
# ---------------------------------------------------------------------------
def restrict(field):
    """8 fine cells → 1 coarse cell, plain average. Requires every dim
    to be a multiple of 2 (uniform grid, no AMR for now)."""
    Nx, Ny, Nz = field.shape
    if Nx % 2 or Ny % 2 or Nz % 2:
        raise ValueError(f"restrict requires even dims; got {field.shape}")
    coarse = field.reshape(Nx // 2, 2, Ny // 2, 2, Nz // 2, 2).mean(axis=(1, 3, 5))
    return coarse


def prolong(field, halo_fn=None):
    """Trilinear prolongation, 1 coarse cell → 8 fine cells:
    ``fine[2I] = (3 c[I] + c[I-1])/4``, ``fine[2I+1] = (3 c[I] + c[I+1])/4``
    per axis. ``halo_fn`` (shard_map context) supplies neighbour-shard data at
    interfaces; None falls back to edge-pad (Neumann-style)."""
    Nx, Ny, Nz = field.shape

    if halo_fn is None:
        c_full = jnp.pad(field, ((1, 1), (1, 1), (1, 1)), mode="edge")
    else:
        c_full = halo_fn(field)

    # 1D interpolation along axis 0 (still padded in y/z).
    left_x = (3.0 * c_full[1:-1] + c_full[:-2]) * 0.25
    right_x = (3.0 * c_full[1:-1] + c_full[2:]) * 0.25
    fx = jnp.stack([left_x, right_x], axis=1).reshape(2 * Nx, Ny + 2, Nz + 2)

    # 1D along axis 1.
    left_y = (3.0 * fx[:, 1:-1] + fx[:, :-2]) * 0.25
    right_y = (3.0 * fx[:, 1:-1] + fx[:, 2:]) * 0.25
    fxy = jnp.stack([left_y, right_y], axis=2).reshape(2 * Nx, 2 * Ny, Nz + 2)

    # 1D along axis 2.
    left_z = (3.0 * fxy[:, :, 1:-1] + fxy[:, :, :-2]) * 0.25
    right_z = (3.0 * fxy[:, :, 1:-1] + fxy[:, :, 2:]) * 0.25
    return jnp.stack([left_z, right_z], axis=3).reshape(2 * Nx, 2 * Ny, 2 * Nz)


# ---------------------------------------------------------------------------
# V-cycle
# ---------------------------------------------------------------------------
def _max_levels(Nx, min_per_side=4):
    """Largest n such that Nx // 2^(n-1) >= min_per_side.
    n=1 means single level (no coarsening). n=2 means one restrict."""
    n = 1
    while (Nx >> n) >= min_per_side and (Nx >> n) > 0:
        n += 1
    return n


def _resolve_n_levels(rho, n_levels):
    """Shared driver preamble: default the multigrid depth so the coarsest
    grid keeps at least 4 cells per side."""
    return _max_levels(rho.shape[0]) if n_levels is None else n_levels


def _finish(phi, rhs, dx, halo_fn, return_residual):
    """Shared driver postamble: optionally append the residual ∞-norm."""
    if return_residual:
        return phi, jnp.max(jnp.abs(residual(phi, rhs, dx, halo_fn)))
    return phi


def vcycle(phi, rhs, dx, halo_fn, level, n_pre, n_post, n_coarsest, halo_inner=None,
           coarse_fn=None, smoother=None):
    """Recursive V-cycle. `level` counts down to 0 (coarsest).

    `halo_fn` is passed to both the smoother *and* the prolongation —
    using the same halo source for prolong is what keeps coarse-fine
    interfaces correct under shard_map (the smoother already uses it for
    the Laplacian stencil, so reuse for free).

    `halo_inner` (isolated-BC split, Guillet-Teyssier style): when set,
    the recursive error subproblem and the prolongation use this halo
    (zero Dirichlet) while this level's smoothing keeps `halo_fn`
    (multipole Dirichlet). None → use `halo_fn` throughout.

    `coarse_fn(phi, rhs, dx)` replaces the level-0 heavy smoothing when set
    (shard_map agglomeration: gather the small coarse grid and finish the
    ladder locally, comm-free).
    """
    if level == 0:
        if coarse_fn is not None:
            return coarse_fn(phi, rhs, dx)
        return rbgs_smoother(phi, rhs, dx, halo_fn, n_coarsest)

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


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
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
    """Solve ∇²Φ = 4πGρ on a uniform 3D grid via V-cycle multigrid.

    Parameters
    ----------
    rho : array (Nx, Ny, Nz)
        Density field.
    dx : float
        Cell size (assumed cubic, dx = dy = dz).
    fourpiG : float
        Pre-multiplied 4πG in code units.
    boundary : int
        LUNA boundary code (0=periodic, 1=Neumann, 2=reflective→Neumann).
    n_levels : int or None
        Multigrid depth. None → auto choose so the coarsest grid has
        at least 4 cells per side.
    n_cycles : int
        Maximum number of V-cycles.
    n_pre, n_post : int
        Smoother iterations on the way down / up of each V-cycle.
    n_coarsest : int
        Smoother iterations at the coarsest level.
    tol : float or None
        If set, stop early when the residual ∞-norm < tol. Adds one host
        sync per V-cycle.
    phi_init : array or None
        Warm-start. None → zeros.
    return_residual : bool
        If True, also return the final residual ∞-norm.

    Returns
    -------
    phi : array (Nx, Ny, Nz)
    [phi, r_norm] if return_residual
    """
    n_levels = _resolve_n_levels(rho, n_levels)
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1; got {n_levels}")

    halo_fn = make_halo_3d(boundary)
    rhs = fourpiG * rho
    phi = phi_init if phi_init is not None else jnp.zeros_like(rho)

    # Always run n_cycles V-cycles. JIT-compatible (no host syncs inside the
    # iteration). `tol`-based early exit was previously supported but
    # required `float(r_norm)` which is illegal under @jax.jit; for the
    # self-gravity-per-step use case fixed-cycle is the right pattern
    # anyway (cost is predictable, warm-start in Phase 2 will let us drop
    # n_cycles to 2–4 once we cache φ across steps).
    for c in range(n_cycles):
        phi = vcycle(phi, rhs, dx, halo_fn, n_levels - 1, n_pre, n_post, n_coarsest)

    return _finish(phi, rhs, dx, halo_fn, return_residual)


# ---------------------------------------------------------------------------
# Multi-process: shard_map driver for the FMG path
# ---------------------------------------------------------------------------
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
):
    """shard_map-wrapped FMG Poisson solve: each device runs FMG on its local
    shard with ppermute halos. ``subtract_mean`` applies the periodic-BC Jeans
    swindle globally before entering shard_map.

    ``agg_n``: coarse-level agglomeration — once the global grid is this
    small, all_gather it and finish the ladder locally (redundantly) on
    every device: no communication on any coarser level, and full ladder
    depth independent of slab thickness. 0 disables.

    ``phi_init``: warm start — skip FMG entirely and apply ``warm_cycles``
    V-cycles from this carried potential (the per-step production path)."""
    from functools import partial
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    from gravity.poisson_bc import make_halo_3d
    from gravity.poisson_bc_shardmap import make_scalar_halo_3d_shardmap

    N = rho.shape[0]
    axis_shards = [("x", 0), ("y", 1), ("z", 2)]
    max_shards = max(int(mesh.shape[a]) for a, _ in axis_shards)
    coarse_fn = None
    if agg_n and max_shards > 1:
        agg_eff = max(int(agg_n), 2 * max_shards)      # keep >=2 cells/shard
        # sharded levels: N -> agg_eff by factors of 2, then agglomerate
        n_levels = max(1, (max(N // agg_eff, 1)).bit_length())
        plain_halo = make_halo_3d(boundary)

        def coarse_fn(phi_l, rhs_l, dxc):
            g_phi, g_rhs = phi_l, rhs_l
            for name, ax in axis_shards:
                if int(mesh.shape[name]) > 1:
                    g_phi = jax.lax.all_gather(g_phi, name, axis=ax, tiled=True)
                    g_rhs = jax.lax.all_gather(g_rhs, name, axis=ax, tiled=True)
            depth = _max_levels(g_rhs.shape[0]) - 1
            e = vcycle(g_phi, g_rhs, dxc, plain_halo, depth,
                       n_pre, n_post, n_coarsest)
            for name, ax in axis_shards:
                nsh = int(mesh.shape[name])
                if nsh > 1:
                    i = jax.lax.axis_index(name)
                    loc = e.shape[ax] // nsh
                    e = jax.lax.dynamic_slice_in_dim(e, i * loc, loc, axis=ax)
            return e
    else:
        n_levels = _resolve_n_levels(rho, n_levels)

    halo_factory = make_scalar_halo_3d_shardmap(mesh)
    halo_fn = halo_factory(boundary)

    # Communication-avoiding smoothing: one wide halo exchange per smoothing
    # block instead of 2 per red-black iteration. Falls back to the sync
    # smoother on levels whose local shard is thinner than the wide slab.
    # Default "auto": ON for multi-node runs (measured -24%/-29% at 4/8 GPUs
    # over the network), OFF intra-node (XGMI latency is tiny; measured
    # slightly slower there). LUNA_MG_CA=0/1 forces.
    import os as _os
    _ca = _os.environ.get("LUNA_MG_CA", "auto")
    ca_on = (_ca == "1") or (_ca == "auto" and jax.process_count() > 1)
    _smoo = _os.environ.get("LUNA_MG_SMOOTHER", "rbgs")   # rbgs | cheb
    smoother = None
    if ca_on and max_shards > 1 and boundary in (0, 1):
        _ca_fn = cheb_smoother_ca if _smoo == "cheb" else rbgs_smoother_ca

        def smoother(p, r, d, n):
            if min(p.shape) < 2 * n:
                return rbgs_smoother(p, r, d, halo_fn, n)
            return _ca_fn(p, r, d, halo_fn, n)

    # Jeans-swindle subtraction. Under auto-sharding `jnp.mean` works
    # globally; do it before shard_map so each shard sees the centred
    # source.
    if subtract_mean and boundary == 0:
        rho_src = rho - jnp.mean(rho)
    else:
        rho_src = rho

    if phi_init is not None:
        # LUNA_MG_MIXED=1: FP64 residual of the carried phi, FP32 V-cycle on
        # the correction, FP64 accumulate — same converged residual at ~2/3
        # the cycle cost (measured single-GPU), and it halves every halo and
        # agglomeration-gather byte on the sharded ladder.
        mixed = _os.environ.get("LUNA_MG_MIXED", "0") == "1"

        @partial(jax.jit, static_argnames=())
        @partial(shard_map, mesh=mesh,
                 in_specs=(P("x", "y", "z"), P("x", "y", "z")),
                 out_specs=P("x", "y", "z"))
        def _local_vcycles(rho_local, phi_local):
            rhs_local = fourpiG * rho_local
            for _ in range(max(1, n_extra_vcycles) if warm_cycles is None
                           else warm_cycles):
                if mixed:
                    r = (rhs_local
                         - laplacian_7pt(phi_local, dx, halo_fn)
                         ).astype(jnp.float32)
                    e = vcycle(
                        jnp.zeros_like(r), r, dx, halo_fn, n_levels - 1,
                        n_pre, n_post, n_coarsest, coarse_fn=coarse_fn,
                        smoother=smoother,
                    )
                    phi_local = phi_local + e
                else:
                    phi_local = vcycle(
                        phi_local, rhs_local, dx, halo_fn, n_levels - 1,
                        n_pre, n_post, n_coarsest, coarse_fn=coarse_fn,
                        smoother=smoother,
                    )
            return phi_local

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
            rhs_local, dx, halo_fn, n_levels, n_pre, n_post, n_coarsest,
            n_vcycles_per_level, coarse_fn=coarse_fn, smoother=smoother,
        )
        for _ in range(n_extra_vcycles):
            phi_local = vcycle(
                phi_local, rhs_local, dx, halo_fn, n_levels - 1, n_pre, n_post,
                n_coarsest, coarse_fn=coarse_fn, smoother=smoother,
            )
        return phi_local

    return _local_fmg(rho_src)


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Full Multigrid (FMG)
# ---------------------------------------------------------------------------
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
):
    """Full Multigrid Poisson solve: restrict the RHS down the hierarchy,
    solve coarsest with heavy smoothing, then prolong upward with
    ``n_vcycles_per_level`` V-cycles per level. One FMG pass typically reaches
    discretisation-error accuracy."""
    # 1) Build the RHS hierarchy: rhs_levels[0] is finest, [-1] coarsest.
    rhs_levels = [rhs]
    dx_levels = [dx]
    for _ in range(n_levels - 1):
        rhs_levels.append(restrict(rhs_levels[-1]))
        dx_levels.append(2.0 * dx_levels[-1])

    # 2-4) Coarsest heavy smooth, then prolong + V-cycle up the ladder.
    return _fmg_ladder(
        rhs_levels, dx_levels, [halo_fn] * n_levels, None,
        n_pre, n_post, n_coarsest, n_vcycles_per_level, coarse_fn=coarse_fn,
        smoother=smoother,
    )


def _fmg_ladder(rhs_levels, dx_levels, halo_tops, halo_inner,
                n_pre, n_post, n_coarsest, n_vcycles_per_level, coarse_fn=None,
                smoother=None):
    """Shared FMG walk-up: heavy smooth at the coarsest level, then
    prolong + n_vcycles_per_level V-cycles at each finer level.

    `halo_tops[L]` smooths level L; `halo_inner` (or None → `halo_tops[L]`)
    handles prolongation and the V-cycles' error subproblems (the
    isolated-BC split; the standard path passes one halo everywhere).
    """
    n_levels = len(rhs_levels)
    phi = jnp.zeros_like(rhs_levels[-1])
    if coarse_fn is not None:
        phi = coarse_fn(phi, rhs_levels[-1], dx_levels[-1])
    else:
        phi = rbgs_smoother(phi, rhs_levels[-1], dx_levels[-1], halo_tops[-1], n_coarsest)

    for L in range(n_levels - 2, -1, -1):
        phi = prolong(phi, halo_fn=halo_tops[L] if halo_inner is None else halo_inner)
        depth = (n_levels - 1) - L  # remaining levels below = V-cycle depth
        for _ in range(n_vcycles_per_level):
            phi = vcycle(
                phi, rhs_levels[L], dx_levels[L], halo_tops[L], depth,
                n_pre, n_post, n_coarsest, halo_inner=halo_inner, coarse_fn=coarse_fn,
                smoother=smoother,
            )

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
    """Top-level FMG driver. Defaults give discretisation-accurate Φ in
    ~1 pass for smooth periodic problems.

    n_extra_vcycles: after the FMG walk-up, run this many additional
    V-cycles at the finest level (cheap insurance for harder problems).
    """
    n_levels = _resolve_n_levels(rho, n_levels)

    # Isolated BC (Guillet-Teyssier-style): top-level Dirichlet halo from a
    # multipole expansion of ρ; coarse error-correction subproblems use
    # zero-Dirichlet. Dispatch here so the periodic / Neumann path stays
    # exactly as it was.
    if boundary == 3:
        return _solve_poisson_fmg_isolated(
            rho, dx, fourpiG, n_levels, n_vcycles_per_level,
            n_extra_vcycles, n_pre, n_post, n_coarsest, return_residual,
        )

    halo_fn = make_halo_3d(boundary)
    rhs = fourpiG * rho

    phi = fmg_solve(rhs, dx, halo_fn, n_levels, n_pre, n_post, n_coarsest, n_vcycles_per_level)

    for _ in range(n_extra_vcycles):
        phi = vcycle(phi, rhs, dx, halo_fn, n_levels - 1, n_pre, n_post, n_coarsest)

    return _finish(phi, rhs, dx, halo_fn, return_residual)


# ---------------------------------------------------------------------------
# Isolated-BC FMG (Guillet & Teyssier 2011 style: multipole Dirichlet at top
# of each V-cycle, zero Dirichlet for coarse error-correction subproblems —
# the split lives in vcycle's `halo_inner` argument)
# ---------------------------------------------------------------------------
def _solve_poisson_fmg_isolated(
    rho, dx, fourpiG, n_levels, n_vcycles_per_level, n_extra_vcycles,
    n_pre, n_post, n_coarsest, return_residual,
):
    """FMG with isolated BC (Guillet-Teyssier face-centred recipe): each
    ladder level gets a multipole-Dirichlet halo from its own restricted ρ;
    coarse error subproblems use zero-Dirichlet."""
    from gravity.poisson_bc import (
        make_zero_dirichlet_halo_3d,
        make_linear_extrap_dirichlet_halo_3d,
        multipole_dirichlet_face_values,
    )
    halo_inner = make_zero_dirichlet_halo_3d()

    # Build ρ + rhs hierarchies (finest → coarsest).
    rho_levels = [rho]
    for _ in range(n_levels - 1):
        rho_levels.append(restrict(rho_levels[-1]))
    rhs_levels = [fourpiG * r for r in rho_levels]
    dx_levels = [dx * (2 ** L) for L in range(n_levels)]

    # Face-centred multipole BC at each level — same physical Φ on the
    # box faces, sampled on the level's grid.
    halo_tops = [
        make_linear_extrap_dirichlet_halo_3d(
            multipole_dirichlet_face_values(rho_levels[L], dx_levels[L], fourpiG)
        )
        for L in range(n_levels)
    ]

    # Coarsest heavy smooth + walk-up with the top/inner halo split:
    # prolong and error subproblems use halo_inner (zero Dirichlet), each
    # level's own smoothing uses its multipole-Dirichlet halo.
    phi = _fmg_ladder(
        rhs_levels, dx_levels, halo_tops, halo_inner,
        n_pre, n_post, n_coarsest, n_vcycles_per_level,
    )

    for _ in range(n_extra_vcycles):
        phi = vcycle(
            phi, rhs_levels[0], dx_levels[0], halo_tops[0], n_levels - 1,
            n_pre, n_post, n_coarsest, halo_inner=halo_inner,
        )

    return _finish(phi, rhs_levels[0], dx, halo_tops[0], return_residual)
