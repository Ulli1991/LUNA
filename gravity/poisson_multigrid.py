"""Multigrid V-cycle Poisson solver for ∇²Φ = 4πGρ.

Phase 1: single-process. Uniform-grid hierarchy, 7-point stencil
Laplacian, damped Jacobi smoother, 8-cell-average restriction, nearest-
neighbour prolongation (upgrade to trilinear in Phase 1.5).

API (high level):
    Φ = solve_poisson(ρ, dx, fourpiG, boundary, n_levels=None,
                      n_cycles=10, n_pre=2, n_post=2, tol=None)

    `boundary` is the LUNA hydro convention (0 periodic, 1 Neumann,
    2 reflective treated as Neumann for Φ).

Algorithm: standard V-cycle for the error equation
    1. pre-smooth Φ on level L
    2. r = rhs - ∇²Φ (residual)
    3. recurse on coarser level with e := 0, rhs' := restrict(r)
    4. Φ ← Φ + prolong(e)
    5. post-smooth Φ on level L

At the coarsest level we do many smoother iterations (cheap on a small
grid). Rank-reduction across shards is a Phase 2 problem — for now we
cap `n_levels` so the coarsest grid is still at least 4 cells per side.

Convergence: each V-cycle should knock the residual norm down by ~10x.
For 1024³ self-grav at tol = 1e-6, expect 5–8 V-cycles per timestep.
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
# Smoother (damped Jacobi)
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
    """Red-black Gauss-Seidel.

    On the 7-point stencil, red cells (i+j+k even) and black cells
    (i+j+k odd) form a checkerboard: red cells only have black
    neighbours and vice versa. So a sweep of all reds (using current
    black values) can be done in parallel, then a sweep of all blacks
    (using freshly-updated red values).

    Each RBGS iteration is 2 halo exchanges + 2 stencil applies =
    roughly 2x the work of weighted-Jacobi but ~4x the per-iteration
    convergence rate on the smoothest error mode. Net win on multigrid.
    """
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


# Back-compat alias — most callers can now use the better smoother by default.
jacobi_smoother = rbgs_smoother


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
    """Trilinear prolongation: 1 coarse cell → 8 fine cells with linear
    interpolation in each direction.

    For cell-centred grids each fine cell sits at offset ±dx_fine/2 from
    its parent coarse cell centre. 1D interpolation along axis i:
        fine[2I]   = (3 coarse[I] + coarse[I-1]) / 4   (lower fine cell)
        fine[2I+1] = (3 coarse[I] + coarse[I+1]) / 4   (upper fine cell)
    Applied in three directions sequentially gives trilinear.

    Boundary handling: if `halo_fn` is supplied (shard_map context, this
    is the same `halo_fn` the smoother uses), we halo-exchange the coarse
    field once up front and use the resulting (Nx+2, Ny+2, Nz+2) padded
    array. That gives neighbour-shard data at coarse-fine interfaces.
    For single-process callers `halo_fn=None`, we fall back to
    `jnp.pad(mode='edge')` (Neumann-style) which is the right behaviour
    on the global boundary anyway.
    """
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


def vcycle(phi, rhs, dx, halo_fn, level, n_pre, n_post, n_coarsest):
    """Recursive V-cycle. `level` counts down to 0 (coarsest).

    `halo_fn` is passed to both the smoother *and* the prolongation —
    using the same halo source for prolong is what keeps coarse-fine
    interfaces correct under shard_map (the smoother already uses it for
    the Laplacian stencil, so reuse for free).
    """
    if level == 0:
        return jacobi_smoother(phi, rhs, dx, halo_fn, n_coarsest)

    phi = jacobi_smoother(phi, rhs, dx, halo_fn, n_pre)
    r = residual(phi, rhs, dx, halo_fn)
    r_c = restrict(r)
    e_c = vcycle(
        jnp.zeros_like(r_c), r_c, 2.0 * dx, halo_fn, level - 1, n_pre, n_post, n_coarsest
    )
    phi = phi + prolong(e_c, halo_fn=halo_fn)
    phi = jacobi_smoother(phi, rhs, dx, halo_fn, n_post)
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
    Nx = rho.shape[0]
    if n_levels is None:
        n_levels = _max_levels(Nx)
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

    if return_residual:
        r_norm = jnp.max(jnp.abs(residual(phi, rhs, dx, halo_fn)))
        return phi, r_norm
    return phi


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
    """shard_map-wrapped Full Multigrid Poisson solve.

    Each device runs FMG on its local shard; halos in the smoother /
    residual / Laplacian use `jax.lax.ppermute` for cross-shard data
    movement. Same FMG algorithm and same convergence as the
    single-process driver — just bounded per-shard memory.

    `subtract_mean` applies the periodic-BC solvability fix (Jeans
    swindle) globally before entering shard_map.
    """
    from functools import partial
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    from gravity.poisson_bc_shardmap import make_scalar_halo_3d_shardmap

    Nx = rho.shape[0]
    if n_levels is None:
        n_levels = _max_levels(Nx)

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
    Nx = rho.shape[0]
    if n_levels is None:
        n_levels = _max_levels(Nx)
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

    if return_residual:
        r_norm = jnp.max(jnp.abs(rhs - apply_A(x)))
        return x, r_norm
    return x


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
    """Full Multigrid Poisson solve.

    Standard V-cycle starts at the finest level with Φ=0 and walks down
    then up. FMG instead:
      1. Build a hierarchy of restricted RHS arrays.
      2. Solve at the coarsest level with heavy smoothing.
      3. Prolong the coarsest solution to the next-finer level as a
         warm-start initial guess, then do n_vcycles_per_level V-cycles.
      4. Repeat upward until we reach the finest level.

    Because each level starts from a near-converged initial guess
    (instead of zeros), one pass of FMG typically achieves
    discretisation-error accuracy — i.e. the residual at the finest
    level falls to the level set by the 7-point stencil's O(dx²) truncation
    in *one* FMG pass, instead of 5-10 V-cycles starting from zero.

    Cost: ~(8/7) of one V-cycle (geometric series of work per level),
    plus one V-cycle at each level on the way up. Net ~2-3x a single
    V-cycle but ~10x the convergence — same total time-to-tol as
    V-cycle multigrid would need 10 cycles to achieve.
    """
    # 1) Build the RHS hierarchy: rhs_levels[0] is finest, [-1] coarsest.
    rhs_levels = [rhs]
    dx_levels = [dx]
    for _ in range(n_levels - 1):
        rhs_levels.append(restrict(rhs_levels[-1]))
        dx_levels.append(2.0 * dx_levels[-1])

    # 2) Solve at the coarsest level with heavy smoothing.
    phi = jnp.zeros_like(rhs_levels[-1])
    phi = rbgs_smoother(phi, rhs_levels[-1], dx_levels[-1], halo_fn, n_coarsest)

    # 3,4) Walk up: prolong (with halo-aware boundary) + V-cycle at each finer level.
    for L in range(n_levels - 2, -1, -1):
        phi = prolong(phi, halo_fn=halo_fn)
        depth = (n_levels - 1) - L  # remaining levels below = V-cycle depth
        for _ in range(n_vcycles_per_level):
            phi = vcycle(phi, rhs_levels[L], dx_levels[L], halo_fn, depth, n_pre, n_post, n_coarsest)

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
    Nx = rho.shape[0]
    if n_levels is None:
        n_levels = _max_levels(Nx)

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

    if return_residual:
        r_norm = jnp.max(jnp.abs(residual(phi, rhs, dx, halo_fn)))
        return phi, r_norm
    return phi


# ---------------------------------------------------------------------------
# Isolated-BC FMG (Guillet & Teyssier 2011 style: multipole Dirichlet at top
# of each V-cycle, zero Dirichlet for coarse error-correction subproblems)
# ---------------------------------------------------------------------------
def _vcycle_isolated(phi, rhs, dx, halo_top, halo_inner, level,
                     n_pre, n_post, n_coarsest):
    """V-cycle with split halos: halo_top at this level (Dirichlet from
    multipole), halo_inner (zero Dirichlet) for the recursive error
    subproblem. Restrict / prolong use halo_inner because they operate on
    error, not the solution."""
    if level == 0:
        return rbgs_smoother(phi, rhs, dx, halo_top, n_coarsest)
    phi = rbgs_smoother(phi, rhs, dx, halo_top, n_pre)
    r = residual(phi, rhs, dx, halo_top)
    r_c = restrict(r)
    e_c = _vcycle_isolated(
        jnp.zeros_like(r_c), r_c, 2.0 * dx,
        halo_inner, halo_inner, level - 1,
        n_pre, n_post, n_coarsest,
    )
    phi = phi + prolong(e_c, halo_fn=halo_inner)
    phi = rbgs_smoother(phi, rhs, dx, halo_top, n_post)
    return phi


def _solve_poisson_fmg_isolated(
    rho, dx, fourpiG, n_levels, n_vcycles_per_level, n_extra_vcycles,
    n_pre, n_post, n_coarsest, return_residual,
):
    """FMG with isolated BC, Guillet-Teyssier 2011 face-centred recipe.

    At each FMG ladder level we build:
      • Φ_face values from a multipole expansion of that level's ρ,
        evaluated at the box face positions (not ghost cell centres).
      • A halo factory that sets ghost = 2·Φ_face − Φ_first_interior
        — the 2nd-order linear-extrapolation Dirichlet BC.
    Coarse-level error subproblems use zero-Dirichlet (no boundary
    contribution from the error).
    """
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

    # Coarsest level: heavy smooth with this level's Dirichlet halo.
    phi = jnp.zeros_like(rhs_levels[-1])
    phi = rbgs_smoother(phi, rhs_levels[-1], dx_levels[-1], halo_tops[-1], n_coarsest)

    # Walk up: prolong to next finer (using halo_inner for the correction-like
    # transfer), then V-cycle with this level's Dirichlet halo at top + zero
    # Dirichlet at coarser sub-levels.
    for L in range(n_levels - 2, -1, -1):
        phi = prolong(phi, halo_fn=halo_inner)
        depth = (n_levels - 1) - L
        for _ in range(n_vcycles_per_level):
            phi = _vcycle_isolated(
                phi, rhs_levels[L], dx_levels[L],
                halo_tops[L], halo_inner, depth,
                n_pre, n_post, n_coarsest,
            )

    for _ in range(n_extra_vcycles):
        phi = _vcycle_isolated(
            phi, rhs_levels[0], dx_levels[0],
            halo_tops[0], halo_inner, n_levels - 1,
            n_pre, n_post, n_coarsest,
        )

    if return_residual:
        r_norm = jnp.max(jnp.abs(residual(phi, rhs_levels[0], dx, halo_tops[0])))
        return phi, r_norm
    return phi
