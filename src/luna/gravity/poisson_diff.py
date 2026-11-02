"""Differentiable FMG Poisson solve: the backward pass is one extra FMG solve, ρ̄ = 4πG·A⁻¹Φ̄.

Isolated (3): the multipole face values are frozen in the adjoint. See docs/AUTODIFF.md.
"""
from __future__ import annotations

import functools
import jax
import jax.numpy as jnp

from luna.gravity.poisson_multigrid import solve_poisson_fmg, laplacian_7pt
from luna.gravity.poisson_bc import make_halo_3d

def _singular(boundary):
    """True where the gauge is pinned: periodic only (pinning Neumann makes its adjoint worse)."""
    return boundary == 0


def _cycles(boundary, n_vcycles_per_level, n_extra_vcycles):
    """(V-cycles per level, extra V-cycles), floored so forward and adjoint give FD-correct gradients."""
    return (max(int(n_vcycles_per_level), 2),
            max(int(n_extra_vcycles), 12 if _singular(boundary) else 4))


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9))
def solve_poisson_fmg_diff(rho, dx, fourpiG, boundary,
                            n_vcycles_per_level=1, n_extra_vcycles=1,
                            n_levels=None, n_pre=2, n_post=2, n_coarsest=50):
    """``solve_poisson_fmg`` with an implicit-adjoint VJP (reverse mode only)."""
    nvpl, nev = _cycles(boundary, n_vcycles_per_level, n_extra_vcycles)
    phi = solve_poisson_fmg(
        rho, dx, fourpiG,
        boundary=boundary,
        n_levels=n_levels,
        n_vcycles_per_level=nvpl,
        n_extra_vcycles=nev,
        n_pre=n_pre, n_post=n_post, n_coarsest=n_coarsest,
    )
    if _singular(boundary):
        # Pin the gauge: Φ's additive constant is a solver artifact, not a function of ρ.
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
    # A·λ = Φ̄, ρ̄ = 4πG·λ; fourpiG=1 so the RHS is Φ̄.
    if _singular(boundary):
        # Adjoint of the pinned map P·A⁻¹·4πG: ρ̄ = 4πG·P·A⁻¹·P·Φ̄, P = I − mean.
        phi_bar = phi_bar - jnp.mean(phi_bar)
    nvpl, nev = _cycles(boundary, n_vcycles_per_level, n_extra_vcycles)
    lam = solve_poisson_fmg(
        phi_bar, dx, 1.0,
        boundary=boundary,
        n_levels=n_levels,
        n_vcycles_per_level=nvpl,
        n_extra_vcycles=nev,
        n_pre=n_pre, n_post=n_post, n_coarsest=n_coarsest,
    )
    if _singular(boundary):
        lam = lam - jnp.mean(lam)
    rho_bar = fourpiG * lam
    return (rho_bar,)


solve_poisson_fmg_diff.defvjp(_fwd, _bwd)


def solve_poisson_fmg_bidiff(rho, dx, fourpiG, boundary,
                             n_vcycles_per_level=1, n_extra_vcycles=1):
    """Poisson solve differentiable in forward and reverse mode (``jax.lax.custom_linear_solve``)."""
    if boundary == 3:
        raise NotImplementedError("solve_poisson_fmg_bidiff: isolated boundary (3) "
                                  "is only implemented in solve_poisson_fmg")
    halo_fn = make_halo_3d(boundary)
    singular = _singular(boundary)
    nvpl, nev = _cycles(boundary, n_vcycles_per_level, n_extra_vcycles)

    def _zm(x):
        return x - jnp.mean(x) if singular else x

    def matvec(phi):
        return _zm(laplacian_7pt(phi, dx, halo_fn))

    def solve(_mv, b):
        phi = solve_poisson_fmg(_zm(b), dx, 1.0, boundary,
                                n_vcycles_per_level=nvpl,
                                n_extra_vcycles=nev)
        return _zm(phi)

    phi = jax.lax.custom_linear_solve(matvec, _zm(fourpiG * rho), solve,
                                      symmetric=True)
    return _zm(phi)
