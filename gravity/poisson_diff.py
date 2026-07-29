"""Differentiable Poisson solve via the implicit-function adjoint: the
backward pass is ONE extra FMG solve with the cotangent Φ̄ as RHS
(ρ̄ = 4πG·A⁻¹Φ̄), never V-cycle unrolling.

Caveats: periodic BC mean-projects the cotangent (adjoint of the Jeans
swindle); for boundary 3 (isolated) the ρ-dependence of the multipole face
values is FROZEN in the adjoint (gradient approximate w.r.t. the far-field BC).
"""
from __future__ import annotations

import functools
import jax
import jax.numpy as jnp

from gravity.poisson_multigrid import solve_poisson_fmg, laplacian_7pt
from gravity.poisson_bc import make_halo_3d


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9))
def solve_poisson_fmg_diff(rho, dx, fourpiG, boundary,
                            n_vcycles_per_level=1, n_extra_vcycles=1,
                            n_levels=None, n_pre=2, n_post=2, n_coarsest=50):
    """Differentiable FMG Poisson solve: forward-identical to
    ``solve_poisson_fmg``; reverse-mode uses the implicit adjoint (one extra
    FMG solve). Cycle counts are floored so both solves are tight enough for
    FD-correct gradients (the periodic null-space otherwise loosens them)."""
    nvpl = max(int(n_vcycles_per_level), 2)
    nev = max(int(n_extra_vcycles), 12 if boundary == 0 else 4)
    phi = solve_poisson_fmg(
        rho, dx, fourpiG,
        boundary=boundary,
        n_levels=n_levels,
        n_vcycles_per_level=nvpl,
        n_extra_vcycles=nev,
        n_pre=n_pre, n_post=n_post, n_coarsest=n_coarsest,
    )
    if boundary == 0:
        # Pin the gauge: under periodic BC the Laplacian is singular on
        # constants, so the additive constant in Φ is a solver artifact
        # (it depends on iteration count and initial guess, NOT smoothly
        # on ρ). Returning the zero-mean potential makes the map ρ → Φ
        # well-defined and differentiable; physics uses only ∇Φ so this
        # changes nothing downstream.
        phi = phi - jnp.mean(phi)
    return phi


def _fwd(rho, dx, fourpiG, boundary, n_vcycles_per_level, n_extra_vcycles,
         n_levels, n_pre, n_post, n_coarsest):
    phi = solve_poisson_fmg_diff(rho, dx, fourpiG, boundary,
                                  n_vcycles_per_level, n_extra_vcycles,
                                  n_levels, n_pre, n_post, n_coarsest)
    return phi, None     # nothing to save: the adjoint needs only Φ̄


def _bwd(dx, fourpiG, boundary, n_vcycles_per_level, n_extra_vcycles,
         n_levels, n_pre, n_post, n_coarsest, residuals, phi_bar):
    # Adjoint solve:  A·λ = Φ̄   →   ρ̄ = 4πG · λ
    # Reuse the same solver (same tuning) with fourpiG=1 so the RHS is Φ̄.
    if boundary == 0:
        # Gauge-pinned periodic map: Φ(ρ) = P·A⁻¹·(4πG ρ) with P = I − mean
        # (see _fwd). Its adjoint is ρ̄ = 4πG · P·A⁻¹·P · Φ̄ — project the
        # cotangent into the zero-mean subspace, solve, project again.
        phi_bar = phi_bar - jnp.mean(phi_bar)
    nvpl = max(int(n_vcycles_per_level), 2)
    nev = max(int(n_extra_vcycles), 12 if boundary == 0 else 4)
    lam = solve_poisson_fmg(
        phi_bar, dx, 1.0,
        boundary=boundary,
        n_levels=n_levels,
        n_vcycles_per_level=nvpl,
        n_extra_vcycles=nev,
        n_pre=n_pre, n_post=n_post, n_coarsest=n_coarsest,
    )
    if boundary == 0:
        lam = lam - jnp.mean(lam)
    rho_bar = fourpiG * lam
    return (rho_bar,)


solve_poisson_fmg_diff.defvjp(_fwd, _bwd)


# ---------------------------------------------------------------------------
# Both-modes variant (forward AND reverse) via jax.lax.custom_linear_solve.
# ---------------------------------------------------------------------------
def solve_poisson_fmg_bidiff(rho, dx, fourpiG, boundary,
                             n_vcycles_per_level=1, n_extra_vcycles=1):
    """Poisson solve differentiable in BOTH forward and reverse mode via
    ``jax.lax.custom_linear_solve`` (``solve_poisson_fmg_diff`` is
    reverse-only and raises on jacfwd). Periodic BC projects operator, RHS
    and Φ to zero-mean to keep the implicit-diff rule well-posed."""
    halo_fn = make_halo_3d(boundary)
    periodic = boundary == 0

    def _zm(x):
        return x - jnp.mean(x) if periodic else x

    def matvec(phi):
        return _zm(laplacian_7pt(phi, dx, halo_fn))

    def solve(_mv, b):
        phi = solve_poisson_fmg(_zm(b), dx, 1.0, boundary,
                                n_vcycles_per_level=n_vcycles_per_level,
                                n_extra_vcycles=n_extra_vcycles)
        return _zm(phi)

    phi = jax.lax.custom_linear_solve(matvec, _zm(fourpiG * rho), solve,
                                      symmetric=True)
    return _zm(phi)
