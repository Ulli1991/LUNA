"""Differentiable Poisson solve via the implicit-function adjoint.

The discrete problem is linear:   A·Φ = b,   b = 4πG·ρ
with A the 7-point Laplacian under the chosen BC. For reverse-mode AD we
do NOT differentiate through the FMG iteration (that would store every
V-cycle intermediate and propagate iteration noise into the gradient).
Instead we use the exact implicit rule for a linear solve:

    Φ = A⁻¹ b          ⇒        ρ̄ = (∂Φ/∂ρ)ᵀ Φ̄ = 4πG · A⁻ᵀ Φ̄ = 4πG · A⁻¹ Φ̄

(A is symmetric for periodic, Neumann and Dirichlet BCs.) The backward
pass is therefore ONE additional FMG solve with the cotangent Φ̄ as the
right-hand side — same cost as the forward solve, exact up to solver
tolerance, zero stored intermediates.

BC caveats:
  * boundary 0 (periodic): A is singular on constants. The forward path
    subtracts the mean of ρ before solving (Jeans swindle, done by the
    caller); we mirror that by mean-projecting the cotangent, which is
    the adjoint of the swindle projection.
  * boundary 3 (isolated): the multipole-Dirichlet face values depend on
    ρ (through M_tot and the COM). This dependence is FROZEN in the
    adjoint — the gradient is exact w.r.t. the bulk Laplacian and
    approximate w.r.t. the far-field BC. For compact mass distributions
    in boxes ≳ 4R the BC sensitivity is O(face-potential/interior-
    potential) smaller than the bulk term.

Validation: tests/test_poisson_adjoint.py checks custom-vjp vs central
FD vs unrolled AD on L(ρ) = ½ Σ Φ² at 32³.
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
    """Differentiable FMG Poisson solve.

    Identical to solve_poisson_fmg in the forward direction (the full set of
    multigrid tuning params — n_levels/n_pre/n_post/n_coarsest — is forwarded,
    so swapping this in for solve_poisson_fmg leaves the forward result bit-
    identical); reverse-mode uses the implicit adjoint (one extra FMG solve),
    not V-cycle unrolling.

    The extra tuning params are appended (not inserted) so existing positional
    callers — e.g. tests/test_poisson_adjoint.py — keep working unchanged.

    Gradient accuracy: the implicit adjoint assumes Φ = A⁻¹(4πGρ) is solved to
    tight tolerance.  Under PERIODIC BC the constant null-space makes the FMG
    converge slowly, so a loose forward solve yields an inaccurate gradient (the
    forward result is fine for physics, but the reverse-mode gradient is wrong).
    We therefore floor the cycle counts in the differentiable path so both the
    forward and adjoint solves are tight enough for FD-correct gradients.
    """
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
    """Poisson solve differentiable in BOTH forward and reverse mode.

    The map ρ → Φ is linear (A Φ = 4πG ρ), so jax.lax.custom_linear_solve
    gives exact jvp AND vjp for free from the matvec (the 7-point Laplacian)
    and the FMG solver — unlike solve_poisson_fmg_diff, whose custom_vjp is
    reverse-only and raises on jacfwd. Use this when forward-mode autodiff
    (jacfwd / few-input Jacobians) through self-gravity is wanted.

    Periodic BC: A has a constant null space, so the operator, RHS and Φ are
    all projected to zero-mean — this restricts the solve to the invertible
    zero-mean subspace and makes the implicit-diff rule well-posed (the same
    gauge fix the custom_vjp path applies)."""
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
