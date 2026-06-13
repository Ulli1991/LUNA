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

from gravity.poisson_multigrid import solve_poisson_fmg


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5))
def solve_poisson_fmg_diff(rho, dx, fourpiG, boundary,
                            n_vcycles_per_level=1, n_extra_vcycles=1):
    """Differentiable FMG Poisson solve.

    Identical to solve_poisson_fmg in the forward direction; reverse-mode
    uses the implicit adjoint (one extra FMG solve), not unrolling.

    Note: n_extra_vcycles defaults to 1 here (vs 0 in the plain solver)
    because gradient quality benefits from a tighter forward residual.
    """
    phi = solve_poisson_fmg(
        rho, dx, fourpiG,
        boundary=boundary,
        n_vcycles_per_level=n_vcycles_per_level,
        n_extra_vcycles=n_extra_vcycles,
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


def _fwd(rho, dx, fourpiG, boundary, n_vcycles_per_level, n_extra_vcycles):
    phi = solve_poisson_fmg_diff(rho, dx, fourpiG, boundary,
                                  n_vcycles_per_level, n_extra_vcycles)
    return phi, None     # nothing to save: the adjoint needs only Φ̄


def _bwd(dx, fourpiG, boundary, n_vcycles_per_level, n_extra_vcycles,
         residuals, phi_bar):
    # Adjoint solve:  A·λ = Φ̄   →   ρ̄ = 4πG · λ
    # Reuse the same solver with fourpiG=1 so the RHS is Φ̄ itself.
    if boundary == 0:
        # Gauge-pinned periodic map: Φ(ρ) = P·A⁻¹·(4πG ρ) with P = I − mean
        # (see _fwd). Its adjoint is ρ̄ = 4πG · P·A⁻¹·P · Φ̄ — project the
        # cotangent into the zero-mean subspace, solve, project again.
        phi_bar = phi_bar - jnp.mean(phi_bar)
    lam = solve_poisson_fmg(
        phi_bar, dx, 1.0,
        boundary=boundary,
        n_vcycles_per_level=n_vcycles_per_level,
        n_extra_vcycles=n_extra_vcycles,
    )
    if boundary == 0:
        lam = lam - jnp.mean(lam)
    rho_bar = fourpiG * lam
    return (rho_bar,)


solve_poisson_fmg_diff.defvjp(_fwd, _bwd)
