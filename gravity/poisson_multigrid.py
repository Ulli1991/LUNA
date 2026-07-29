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


def vcycle(phi, rhs, dx, halo_fn, level, n_pre, n_post, n_coarsest, halo_inner=None):
    """Recursive V-cycle. `level` counts down to 0 (coarsest).

    `halo_fn` is passed to both the smoother *and* the prolongation —
    using the same halo source for prolong is what keeps coarse-fine
    interfaces correct under shard_map (the smoother already uses it for
    the Laplacian stencil, so reuse for free).

    `halo_inner` (isolated-BC split, Guillet-Teyssier style): when set,
    the recursive error subproblem and the prolongation use this halo
    (zero Dirichlet) while this level's smoothing keeps `halo_fn`
    (multipole Dirichlet). None → use `halo_fn` throughout.
    """
    if level == 0:
        return rbgs_smoother(phi, rhs, dx, halo_fn, n_coarsest)

    inner = halo_fn if halo_inner is None else halo_inner
    phi = rbgs_smoother(phi, rhs, dx, halo_fn, n_pre)
    r = residual(phi, rhs, dx, halo_fn)
    r_c = restrict(r)
    e_c = vcycle(
        jnp.zeros_like(r_c), r_c, 2.0 * dx, inner, level - 1, n_pre, n_post, n_coarsest,
        halo_inner=halo_inner,
    )
    phi = phi + prolong(e_c, halo_fn=inner)
    phi = rbgs_smoother(phi, rhs, dx, halo_fn, n_post)
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
):
    """shard_map-wrapped FMG Poisson solve: each device runs FMG on its local
    shard with ppermute halos. ``subtract_mean`` applies the periodic-BC Jeans
    swindle globally before entering shard_map."""
    from functools import partial
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    from gravity.poisson_bc_shardmap import make_scalar_halo_3d_shardmap

    n_levels = _resolve_n_levels(rho, n_levels)

    halo_factory = make_scalar_halo_3d_shardmap(mesh)
    halo_fn = halo_factory(boundary)

    # Jeans-swindle subtraction. Under auto-sharding `jnp.mean` works
    # globally; do it before shard_map so each shard sees the centred
    # source.
    if subtract_mean and boundary == 0:
        rho_src = rho - jnp.mean(rho)
    else:
        rho_src = rho

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
            rhs_local, dx, halo_fn, n_levels, n_pre, n_post, n_coarsest, n_vcycles_per_level
        )
        for _ in range(n_extra_vcycles):
            phi_local = vcycle(
                phi_local, rhs_local, dx, halo_fn, n_levels - 1, n_pre, n_post, n_coarsest
            )
        return phi_local

    return _local_fmg(rho_src)


# ---------------------------------------------------------------------------
# Multigrid-preconditioned Conjugate Gradient
# ---------------------------------------------------------------------------
def solve_poisson_pcg(
    rho,
    dx,
    fourpiG,
    boundary,
    n_levels=None,
    n_pcg_iter=20,
    n_pre=2,
    n_post=2,
    n_coarsest=50,
    return_residual=False,
):
    """Preconditioned CG with one V-cycle as the preconditioner.

    Algorithm (Hestenes-Stiefel CG, M^{-1} = single V-cycle):
        r_0 = rhs - A x_0   (with x_0 = 0, so r_0 = rhs)
        z_0 = M^{-1} r_0    (one V-cycle on r_0)
        p_0 = z_0
        for k = 0, 1, ...:
            α = (r_k · z_k) / (p_k · A p_k)
            x_{k+1} = x_k + α p_k
            r_{k+1} = r_k - α A p_k
            z_{k+1} = M^{-1} r_{k+1}
            β = (r_{k+1} · z_{k+1}) / (r_k · z_k)
            p_{k+1} = z_{k+1} + β p_k

    The Laplacian is SPD (with the zero-mean constraint imposed by
    periodic BC, or modulo a constant for Neumann). With a multigrid
    preconditioner the condition number stays O(1) independent of N, so
    PCG converges in ~5-15 iterations to high accuracy regardless of
    resolution. Each iteration is one V-cycle + a handful of BLAS dots,
    so the per-step cost matches a single V-cycle.

    For pure forward (no autodiff) self-grav this is overkill vs FMG;
    most use cases will prefer FMG. PCG shines for very tight tolerances
    or for hard problems (e.g. cosmological boxes where one mode is
    millions of times stronger than the others).
    """
    n_levels = _resolve_n_levels(rho, n_levels)
    halo_fn = make_halo_3d(boundary)
    rhs = fourpiG * rho

    def apply_A(phi):
        return laplacian_7pt(phi, dx, halo_fn)

    def precond(r):
        # One V-cycle starting from zero gives z ≈ A^{-1} r
        return vcycle(jnp.zeros_like(r), r, dx, halo_fn, n_levels - 1, n_pre, n_post, n_coarsest)

    x = jnp.zeros_like(rhs)
    r = rhs - apply_A(x)
    z = precond(r)
    p = z
    rz = jnp.sum(r * z)

    for _ in range(n_pcg_iter):
        Ap = apply_A(p)
        alpha = rz / jnp.sum(p * Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        z = precond(r)
        rz_new = jnp.sum(r * z)
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    return _finish(x, rhs, dx, halo_fn, return_residual)


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
        n_pre, n_post, n_coarsest, n_vcycles_per_level,
    )


def _fmg_ladder(rhs_levels, dx_levels, halo_tops, halo_inner,
                n_pre, n_post, n_coarsest, n_vcycles_per_level):
    """Shared FMG walk-up: heavy smooth at the coarsest level, then
    prolong + n_vcycles_per_level V-cycles at each finer level.

    `halo_tops[L]` smooths level L; `halo_inner` (or None → `halo_tops[L]`)
    handles prolongation and the V-cycles' error subproblems (the
    isolated-BC split; the standard path passes one halo everywhere).
    """
    n_levels = len(rhs_levels)
    phi = jnp.zeros_like(rhs_levels[-1])
    phi = rbgs_smoother(phi, rhs_levels[-1], dx_levels[-1], halo_tops[-1], n_coarsest)

    for L in range(n_levels - 2, -1, -1):
        phi = prolong(phi, halo_fn=halo_tops[L] if halo_inner is None else halo_inner)
        depth = (n_levels - 1) - L  # remaining levels below = V-cycle depth
        for _ in range(n_vcycles_per_level):
            phi = vcycle(
                phi, rhs_levels[L], dx_levels[L], halo_tops[L], depth,
                n_pre, n_post, n_coarsest, halo_inner=halo_inner,
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
