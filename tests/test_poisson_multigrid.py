"""Unit tests for the multigrid Poisson solver.

Run with: pytest tests/test_poisson_multigrid.py
or:       python -m tests.test_poisson_multigrid (calls the test fns manually)

Phase 1 acceptance: V-cycle converges on a sine-wave eigenfunction and
on a uniform-sphere-style configuration; per-cycle reductions, algebraic and
discretisation errors are bounded at measured or analytic values (see each test).
"""

import pytest
pytestmark = pytest.mark.slow


import jax.numpy as jnp
import numpy as np

from luna.gravity.poisson_multigrid import (
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
from luna.gravity.poisson_bc import make_halo_3d


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
    """RBGS alone on the (4,4,4) mode at 32^3: the residual falls at the analytic rate cos(k dx)^2 per iteration."""
    N, L, fourpiG = 32, 1.0, 1.0
    rho, _, dx = _make_sine_problem(N, L, fourpiG, k_index=(4, 4, 4))
    halo_fn = make_halo_3d(0)
    rhs = fourpiG * rho
    phi = rbgs_smoother(jnp.zeros_like(rho), rhs, dx, halo_fn, n_iter=10)
    r10 = float(jnp.max(jnp.abs(residual(phi, rhs, dx, halo_fn))))
    phi = rbgs_smoother(phi, rhs, dx, halo_fn, n_iter=10)
    r20 = float(jnp.max(jnp.abs(residual(phi, rhs, dx, halo_fn))))
    # Jacobi's eigenvalue of this mode is mu = (cos kx dx + cos ky dx + cos kz dx) / 3 = cos(pi / 4); red-black GS
    # (consistently ordered) has mu^2 per iteration on the mode and its red-black partner (Young 1971)
    mu = np.cos(2 * np.pi * 4 / N)
    rate = (r20 / r10) ** 0.1
    print(f"MEASURED RBGS (4,4,4) N=32: per-iteration factor {rate:.9f}, analytic mu^2 {mu * mu:.9f}, "
          f"|rate / mu^2 - 1| = {abs(rate / (mu * mu) - 1.0):.3e}")
    assert abs(rate / (mu * mu) - 1.0) < RBGS_RATE_TOL, rate


RBGS_RATE_TOL = 1e-5     # measured rate 0.500000 against mu^2 = 0.500000 (6 digits; CPU-8 FP64, job 31654893)

# V-cycle (2 pre, 2 post RBGS, 50 coarsest) on the lowest mode at 32^3: cycle-to-cycle residual reduction after the
# first cycle, measured 9.05 .. 9.13 (CPU-8 FP64, job 31654893); bound = measured minimum 9.07 / 1.3
VCYCLE_MIN_RATIO = 6.9   # smoothing alone (teeth) measured 1.17


def _lowest_mode_ratios(cycle):
    """Cycle-to-cycle residual ratios of `cycle(phi, rhs, dx, halo)` over 6 cycles on the (1,1,1) mode at 32^3."""
    N, L, fourpiG = 32, 1.0, 1.0
    rho, _, dx = _make_sine_problem(N, L, fourpiG, k_index=(1, 1, 1))
    halo_fn = make_halo_3d(0)
    rhs = fourpiG * rho
    phi = jnp.zeros_like(rho)
    r = []
    for _ in range(6):
        phi = cycle(phi, rhs, dx, halo_fn)
        r.append(float(jnp.max(jnp.abs(residual(phi, rhs, dx, halo_fn)))))
    return [r[i] / r[i + 1] for i in range(len(r) - 1)]


def test_vcycle_converges_lowest_mode():
    """The V-cycle reduces the lowest mode's residual by at least the measured factor per cycle (cycles 2-6)."""
    n_levels = _max_levels(32)
    ratios = _lowest_mode_ratios(lambda p, r, dx, h: vcycle(p, r, dx, h, n_levels - 1, 2, 2, 50))
    print(f"MEASURED V-cycle lowest mode N=32 reduction ratios {[f'{x:.4f}' for x in ratios]}")
    assert min(ratios[1:]) > VCYCLE_MIN_RATIO, ratios


def test_vcycle_bound_catches_smoothing_alone():
    """Teeth: the same fine-grid work without the coarse-grid correction (4 RBGS sweeps per cycle) fails it."""
    ratios = _lowest_mode_ratios(lambda p, r, dx, h: rbgs_smoother(p, r, dx, h, n_iter=4))
    print(f"MEASURED 4 RBGS sweeps per cycle, lowest mode N=32, ratios {[f'{x:.4f}' for x in ratios]}")
    assert min(ratios[1:]) < VCYCLE_MIN_RATIO, ratios


def _discrete_floor(N, L, k_index):
    """|k^2 / k_h^2 - 1|: the 7-point stencil's exact relative error on a sine mode (its discrete eigenvalue)."""
    dx = L / N
    k = [2 * np.pi * i / L for i in k_index]
    kh2 = sum((2.0 / dx * np.sin(ki * dx / 2)) ** 2 for ki in k)
    return abs(sum(ki * ki for ki in k) / kh2 - 1.0), sum(ki * ki for ki in k) / kh2


def test_solve_poisson_recovers_sine_mode():
    """15 V-cycles: the solution equals the discrete solution (k^2 / k_h^2 times the exact mode) to the solve's error."""
    N, L, fourpiG = 32, 1.0, 1.0
    rho, phi_exact, dx = _make_sine_problem(N, L, fourpiG, k_index=(1, 1, 1))
    floor, scale = _discrete_floor(N, L, (1, 1, 1))
    phi = solve_poisson(rho, dx, fourpiG, boundary=0, n_cycles=15)
    ex = _zero_mean(phi_exact)
    rel = float(jnp.max(jnp.abs(_zero_mean(phi) - ex)) / jnp.max(jnp.abs(ex)))
    alg = float(jnp.max(jnp.abs(_zero_mean(phi) - scale * ex)) / jnp.max(jnp.abs(scale * ex)))
    print(f"MEASURED solve_poisson 15 V-cycles: vs continuum {rel:.4e} (analytic floor {floor:.4e}), "
          f"vs discrete solution {alg:.3e}")
    assert alg < SOLVE15_ALG, alg                       # converged: only the algebraic error is left
    assert abs(rel - floor) <= alg * scale + 1e-15, (rel, floor)   # so the continuum error is the floor, to it


SOLVE15_ALG = 1e-14      # round-off: measured 4.38e-15 (CPU-8 FP64, job 31654893); summation order moves the last ulps
# One FMG pass (1 V(2,2) per level) on the (1,1,1) mode at 32^3, CPU-8 FP64, job 31654893: algebraic error 7.48e-3,
# 2.3x the stencil's own discretisation error 3.22e-3, so one pass does NOT reach the floor (production adds V-cycles,
# n_extra_vcycles); residual (max norm, rhs max k^2 = 118) 1.74. Bounds 1.3x measured.
FMG_ALG, FMG_RES = 9.7e-3, 2.26


def test_fmg_one_pass_error():
    """Regression only: one FMG pass leaves 2.3x the discretisation error as algebraic error; it and the residual."""
    N, L, fourpiG = 32, 1.0, 1.0
    rho, phi_exact, dx = _make_sine_problem(N, L, fourpiG, k_index=(1, 1, 1))
    floor, scale = _discrete_floor(N, L, (1, 1, 1))
    phi, r_norm = solve_poisson_fmg(rho, dx, fourpiG, boundary=0, n_vcycles_per_level=1, return_residual=True)
    ex = _zero_mean(phi_exact)
    alg = float(jnp.max(jnp.abs(_zero_mean(phi) - scale * ex)) / jnp.max(jnp.abs(scale * ex)))
    print(f"MEASURED FMG one pass N=32: algebraic error {alg:.4e}, discretisation floor {floor:.4e}, "
          f"residual {float(r_norm):.4e}")
    assert alg < FMG_ALG, alg
    assert float(r_norm) < FMG_RES, float(r_norm)


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
        ("fmg_one_pass_error", test_fmg_one_pass_error),
    ]:
        print(f"\n--- {name} ---")
        try:
            fn()
            print("  PASS")
        except AssertionError as e:
            print(f"  FAIL: {e}")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
    print()


def _vcycle_factor(shape, boundary, n_coarsest):
    """Asymptotic rms residual reduction of V(1,1) from zero on a random zero-mean source."""
    import jax
    rng = np.random.default_rng(4)
    f = jnp.asarray(rng.standard_normal(shape))
    f = f - jnp.mean(f)
    dx = 1.0 / shape[1]
    halo = make_halo_3d(boundary)
    cyc = jax.jit(lambda ph: solve_poisson(f, dx, 1.0, boundary, n_cycles=1, n_pre=1, n_post=1,
                                           n_coarsest=n_coarsest, phi_init=ph))
    phi, r = jnp.zeros_like(f), []
    for _ in range(9):
        phi = cyc(phi)
        r.append(float(jnp.sqrt(jnp.mean(residual(phi, f, dx, halo) ** 2))))
    return (r[8] / r[4]) ** 0.25


@pytest.mark.parametrize("shape,boundary", [((32, 32, 32), 0), ((64, 32, 32), 0),
                                            ((32, 32, 32), 1)])
def test_auto_coarsest_sweeps_keep_the_convergence_factor(shape, boundary):
    """The default (None) coarsest sweeps converge as fast as 50 sweeps on every grid shape and boundary."""
    auto = _vcycle_factor(shape, boundary, None)
    full = _vcycle_factor(shape, boundary, 50)
    assert auto <= 1.01 * full, (auto, full)


def test_auto_coarsest_sweeps_rule():
    from luna.gravity.poisson_multigrid import coarsest_sweeps
    assert coarsest_sweeps(None, (4, 4, 4), make_halo_3d(0)) == 4
    assert coarsest_sweeps(None, (4, 2, 2), make_halo_3d(0)) == 4
    assert coarsest_sweeps(None, (8, 4, 4), make_halo_3d(0)) == 50
    assert coarsest_sweeps(None, (4, 4, 4), make_halo_3d(1)) == 50
    assert coarsest_sweeps(7, (4, 4, 4), make_halo_3d(0)) == 7
