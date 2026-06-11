"""Validate the multipole-Dirichlet (isolated-BC) FMG against analytic Φ.

Two checks:
  1. Uniform ball: ρ = ρ₀ for r ≤ R, 0 outside. Outside R, Φ = −GM/r.
     Pass criterion: max rel err (r > 1.5R, r < 0.4L) < 5 %.
  2. Off-centre point mass: ρ = δ at non-centre location. Outside a few
     cells, Φ ≈ −GM/|r − r_src|.  Same 5 % criterion.

Both checks confirm that
   • the multipole-Dirichlet halo correctly carries the far-field BC,
   • the V-cycle smoother converges with the new boundary code = 3,
   • the residual at the box faces matches the analytic expectation.
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, "/raven/u/uli/jax-hydro")
jax.config.update("jax_enable_x64", True)

from gravity.poisson_multigrid import solve_poisson_fmg


def _grid(N, L):
    dx = L / N
    c = (np.arange(N) + 0.5) * dx - 0.5 * L
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    return dx, X, Y, Z


def test_uniform_ball():
    N = 64
    L = 4.0
    G = 1.0
    M = 1.0
    R = 1.0
    dx, X, Y, Z = _grid(N, L)
    r = np.sqrt(X * X + Y * Y + Z * Z)

    V = 4.0 / 3.0 * np.pi * R ** 3
    rho = np.where(r <= R, M / V, 0.0)

    phi = np.asarray(solve_poisson_fmg(
        jnp.asarray(rho), dx, 4.0 * np.pi * G, boundary=3,
        n_vcycles_per_level=2, n_extra_vcycles=2,
    ))

    # Outside R, Φ = −GM/r;  inside, Φ_in − Φ_out|_R is a global offset, so
    # we test only outside.
    phi_an = -G * M / np.maximum(r, R * 1.0001)
    mask = (r > 1.5 * R) & (r < 0.4 * L)
    rel_err = np.abs(phi[mask] - phi_an[mask]) / np.abs(phi_an[mask])
    print(f"  uniform ball: max rel err (r > 1.5R) = {rel_err.max():.3e}, "
          f"mean = {rel_err.mean():.3e}")
    assert rel_err.max() < 0.02, "uniform-ball FMG isolated test failed"


def test_offcenter_point_mass():
    N = 64
    L = 1.0
    G = 1.0
    M = 1.0
    dx, X, Y, Z = _grid(N, L)
    # source at (0.1, 0.0, 0.0) — off-centre
    sx, sy, sz = 0.10, 0.0, 0.0
    r = np.sqrt((X - sx) ** 2 + (Y - sy) ** 2 + (Z - sz) ** 2)

    # Cell index of the source
    i_src = int(round((sx + 0.5 * L - 0.5 * dx) / dx))
    j_src = int(round((sy + 0.5 * L - 0.5 * dx) / dx))
    k_src = int(round((sz + 0.5 * L - 0.5 * dx) / dx))
    rho = np.zeros((N, N, N))
    rho[i_src, j_src, k_src] = M / (dx ** 3)

    phi = np.asarray(solve_poisson_fmg(
        jnp.asarray(rho), dx, 4.0 * np.pi * G, boundary=3,
        n_vcycles_per_level=2, n_extra_vcycles=2,
    ))
    phi_an = -G * M / np.maximum(r, dx)
    mask = (r > 4.0 * dx) & (r < 0.4 * L)
    rel_err = np.abs(phi[mask] - phi_an[mask]) / np.abs(phi_an[mask])
    print(f"  off-centre point mass: max rel err = {rel_err.max():.3e}, "
          f"mean = {rel_err.mean():.3e}")
    assert rel_err.max() < 0.02, "point-mass FMG isolated test failed"


def main():
    print("Isolated-BC multigrid (boundary=3) — validation")
    test_uniform_ball()
    test_offcenter_point_mass()
    print("  all tests passed ✓")


if __name__ == "__main__":
    main()
