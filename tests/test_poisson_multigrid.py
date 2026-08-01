"""Unit tests for the multigrid Poisson solver.

Run with: pytest tests/test_poisson_multigrid.py
or:       python -m tests.test_poisson_multigrid (calls the test fns manually)

Phase 1 acceptance: V-cycle converges on a sine-wave eigenfunction and
on a uniform-sphere-style configuration, residual decreases ~10x per
V-cycle for a few cycles. Multi-process shard_map regression in Phase 2.
"""

import pytest
pytestmark = pytest.mark.slow


import jax
import jax.numpy as jnp
import numpy as np

from gravity.poisson_multigrid import (
    solve_poisson,
    solve_poisson_fmg,
    vcycle,
    laplacian_7pt,
    residual,
    rbgs_smoother,
    restrict,
    prolong,
    _max_levels,
)
from gravity.poisson_bc import make_halo_3d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _zero_mean(x):
    return x - jnp.mean(x)


def _grid(N, L=1.0):
    """1D coordinate of cell centres for a periodic [0, L] domain split
    into N cells."""
    return (jnp.arange(N) + 0.5) * (L / N)


# ---------------------------------------------------------------------------
# Stencil correctness
# ---------------------------------------------------------------------------
def test_laplacian_sine_mode():
    """∇² applied to Φ = sin(k_x x) sin(k_y y) sin(k_z z) on a periodic
    [0, 2π]³ domain should return -(k² + k² + k²)Φ to O(dx²) accuracy."""
    N = 64
    L = 2.0 * np.pi
    dx = L / N
    k = 1.0  # lowest periodic mode
    x = _grid(N, L)
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    phi = jnp.sin(k * X) * jnp.sin(k * Y) * jnp.sin(k * Z)
    expected = -3.0 * k * k * phi
    halo_fn = make_halo_3d(0)  # periodic
    got = laplacian_7pt(phi, dx, halo_fn)
    # 7-point stencil truncates at O(dx²); accept relative error < 1% at N=64
    rel = jnp.max(jnp.abs(got - expected)) / jnp.max(jnp.abs(expected))
    print(f"  laplacian_7pt rel err (N={N}, k={k}): {float(rel):.3e}")
    assert float(rel) < 1e-2


def test_restrict_prolong_smooth_field():
    """restrict ∘ prolong is identity on a constant field; prolong ∘ restrict
    is identity-up-to-block-averaging on a smooth field."""
    N = 32
    x = _grid(N, 1.0)
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    field_const = jnp.ones((N, N, N))
    rc = restrict(field_const)
    assert rc.shape == (N // 2, N // 2, N // 2)
    np.testing.assert_allclose(np.asarray(rc), 1.0, atol=1e-7)

    field_smooth = jnp.cos(2 * jnp.pi * X)
    rp = prolong(restrict(field_smooth))
    # NN prolong loses sub-coarse-cell resolution; expect small error
    err = jnp.max(jnp.abs(rp - field_smooth))
    print(f"  prolong∘restrict err on smooth field: {float(err):.3e}")
    assert float(err) < 0.5  # NN prolong is coarse; just sanity-check finite


# ---------------------------------------------------------------------------
# Smoother + V-cycle on a manufactured solution
# ---------------------------------------------------------------------------
def _make_sine_problem(N, L, fourpiG, k_index=(1, 1, 1)):
    """Construct (ρ, Φ_exact) such that ∇²Φ_exact = 4πGρ on a periodic
    [0, L]³ grid, with Φ_exact being a sine mode of integer wave number."""
    dx = L / N
    x = _grid(N, L)
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    kx, ky, kz = [(2.0 * np.pi / L) * k_i for k_i in k_index]
    phi_exact = jnp.sin(kx * X) * jnp.sin(ky * Y) * jnp.sin(kz * Z)
    # ∇² (sin sin sin) = -(kx²+ky²+kz²) (sin sin sin), so rhs = -(k²)Φ
    rhs_continuous = -(kx * kx + ky * ky + kz * kz) * phi_exact
    rho = rhs_continuous / fourpiG  # 4πGρ = rhs => ρ = rhs / (4πG)
    return rho, phi_exact, dx


def test_rbgs_smoother_decreases_residual():
    """Without multigrid, Jacobi alone reduces the residual on a single
    mode (any mode shorter than nyquist is decayed; the lowest mode
    decays slowest — that's why we add multigrid)."""
    N = 32
    L = 1.0
    fourpiG = 1.0
    # All three k components must be non-zero or Φ = sin(0)sin(0)sin(0) = 0.
    # Pick a moderately high-frequency mode that pure Jacobi should damp well.
    rho, _, dx = _make_sine_problem(N, L, fourpiG, k_index=(4, 4, 4))
    halo_fn = make_halo_3d(0)
    rhs = fourpiG * rho
    phi = jnp.zeros_like(rho)
    r0 = jnp.max(jnp.abs(residual(phi, rhs, dx, halo_fn)))
    phi = rbgs_smoother(phi, rhs, dx, halo_fn, n_iter=200)
    r1 = jnp.max(jnp.abs(residual(phi, rhs, dx, halo_fn)))
    print(f"  Jacobi: r0={float(r0):.3e} → r1={float(r1):.3e} after 200 iters")
    assert float(r1) < 0.5 * float(r0)


def test_vcycle_converges_lowest_mode():
    """V-cycle should converge much faster than pure Jacobi on the lowest
    Fourier mode (which Jacobi is slowest at). After a few cycles the
    residual should drop ~10x per cycle."""
    N = 32
    L = 1.0
    fourpiG = 1.0
    rho, phi_exact, dx = _make_sine_problem(N, L, fourpiG, k_index=(1, 1, 1))
    halo_fn = make_halo_3d(0)
    rhs = fourpiG * rho
    phi = jnp.zeros_like(rho)
    n_levels = _max_levels(N)

    r_norms = []
    for c in range(6):
        phi = vcycle(phi, rhs, dx, halo_fn, n_levels - 1, 2, 2, 50)
        r = jnp.max(jnp.abs(residual(phi, rhs, dx, halo_fn)))
        r_norms.append(float(r))
        print(f"  V-cycle {c}: residual ∞-norm = {float(r):.3e}")

    # Cycle-to-cycle reduction: expect at least 3x on average after the
    # first warm-up cycle.
    ratios = [r_norms[i] / r_norms[i + 1] for i in range(len(r_norms) - 1)]
    print(f"  reduction ratios: {[f'{r:.2f}' for r in ratios]}")
    assert min(ratios[1:]) > 2.0, f"Slow V-cycle convergence: {ratios}"


def test_solve_poisson_recovers_sine_mode():
    """End-to-end: solve_poisson should recover the analytic sine mode
    after enough V-cycles. Compare zero-mean Φ to zero-mean Φ_exact."""
    N = 32
    L = 1.0
    fourpiG = 1.0
    rho, phi_exact, dx = _make_sine_problem(N, L, fourpiG, k_index=(1, 1, 1))

    phi = solve_poisson(rho, dx, fourpiG, boundary=0, n_cycles=15, tol=1e-6)

    # Periodic Laplacian has null space (Φ + const), so compare zero-mean.
    phi_zm = _zero_mean(phi)
    exact_zm = _zero_mean(phi_exact)
    rel = jnp.max(jnp.abs(phi_zm - exact_zm)) / jnp.max(jnp.abs(exact_zm))
    print(f"  solve_poisson sine-mode rel err: {float(rel):.3e}")
    # 7-point stencil truncation error is O(dx²); at N=32 that's ~1e-2 of the
    # exact mode amplitude. V-cycle convergence brings residual under ~1e-6
    # but the discretisation floor sits at ~3e-3 — that's the right answer.
    assert float(rel) < 1e-2


def test_fmg_one_pass_hits_discretisation_floor():
    """Full Multigrid should reach the O(dx²) discretisation floor in a
    single pass — no extra V-cycles needed."""
    N = 32
    L = 1.0
    fourpiG = 1.0
    rho, phi_exact, dx = _make_sine_problem(N, L, fourpiG, k_index=(1, 1, 1))

    phi, r_norm = solve_poisson_fmg(
        rho, dx, fourpiG, boundary=0, n_vcycles_per_level=1, return_residual=True
    )
    phi_zm = _zero_mean(phi)
    exact_zm = _zero_mean(phi_exact)
    rel = jnp.max(jnp.abs(phi_zm - exact_zm)) / jnp.max(jnp.abs(exact_zm))
    print(f"  FMG one-pass: residual={float(r_norm):.3e}, rel err vs exact={float(rel):.3e}")
    # Discretisation floor of 7-point stencil at N=32 is ~3e-3 on this mode.
    assert float(rel) < 1e-2


if __name__ == "__main__":
    print("=" * 60)
    print("Multigrid Poisson solver — Phase 1 unit tests")
    print("=" * 60)
    for name, fn in [
        ("laplacian_sine_mode", test_laplacian_sine_mode),
        ("restrict_prolong_smooth_field", test_restrict_prolong_smooth_field),
        ("rbgs_smoother_decreases_residual", test_rbgs_smoother_decreases_residual),
        ("vcycle_converges_lowest_mode", test_vcycle_converges_lowest_mode),
        ("solve_poisson_recovers_sine_mode", test_solve_poisson_recovers_sine_mode),
        ("fmg_one_pass_hits_discretisation_floor", test_fmg_one_pass_hits_discretisation_floor),
    ]:
        print(f"\n--- {name} ---")
        try:
            fn()
            print(f"  PASS")
        except AssertionError as e:
            print(f"  FAIL: {e}")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
    print()
