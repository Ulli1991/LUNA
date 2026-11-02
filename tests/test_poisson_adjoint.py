"""Three-way validation of the implicit Poisson adjoint.

On L(ρ) = ½ Σ Φ(ρ)²  at 32³ (periodic and Neumann BCs), compare:

  1. custom-vjp gradient  (gravity/poisson_diff.py — one adjoint solve)
  2. central finite differences at 5 random voxels
  3. unrolled AD          (jax.grad straight through the FMG iterations)

Pass criteria (measured x 1.3, job 31654893, see BOUNDS): custom-vjp vs FD 3.5e-6 (Neumann), 1.5e-5 (periodic);
custom-vjp vs unrolled 5.1e-4 (Neumann), 1.9e-8 (periodic): the unrolled map carries the solver's own residual.
"""

import pytest
pytestmark = pytest.mark.slow

import numpy as np
import jax
import jax.numpy as jnp


from luna.gravity.poisson_multigrid import solve_poisson_fmg
from luna.gravity.poisson_diff import solve_poisson_fmg_diff


def _make_rho(N, key, boundary):
    k = jnp.fft.fftfreq(N) * N
    kx, ky, kz = jnp.meshgrid(k, k, k, indexing="ij")
    mask = (jnp.abs(kx) <= 3) & (jnp.abs(ky) <= 3) & (jnp.abs(kz) <= 3)
    mask = mask & ~((kx == 0) & (ky == 0) & (kz == 0))
    kr, ki = jax.random.split(key)
    c = (jax.random.normal(kr, (N, N, N)) + 1j * jax.random.normal(ki, (N, N, N))) * mask
    f = jnp.fft.ifftn(c).real
    f = f / jnp.std(f)
    if boundary in (0, 1):   # both BCs are singular up to a constant
        return f                  # zero-mean field (periodic needs it)
    return f + 2.0                # positive offset fine for Neumann


def _check(boundary, name, scale=1.0):
    """(max rel err custom vs FD, rel err custom vs unrolled); scale != 1 breaks the custom gradient (teeth)."""
    N = 32
    dx = 1.0 / N
    fourpiG = 4.0 * jnp.pi
    rho = _make_rho(N, jax.random.PRNGKey(3), boundary)

    def loss_customvjp(r):
        phi = solve_poisson_fmg_diff(r, dx, fourpiG, boundary, 1, 6)
        if boundary in (0, 1):
            # GAUGE. Both singular boundaries leave Phi defined up to a
            # constant, and the solver's constant is an iteration artifact.
            # The custom op pins it internally for periodic only, so under
            # NEUMANN this loss used to read the drifting constant: the
            # adjoint returned the gauge-free derivative and the finite
            # difference measured the drift, which is the whole of the
            # 2.8e-3 discrepancy this gate reported for days (audit sec.26).
            # Measured elsewhere in the same campaign: the implicit adjoint
            # agrees with an explicitly built pseudo-inverse under Neumann
            # (tests/test_poisson_neumann_matrix.py), the discrete operator
            # IS symmetric, and the two differentiable routes agree to
            # 1.5e-6 -- so the rule was never the problem, the functional
            # was. Physics only ever reads grad Phi, which is gauge-free.
            phi = phi - jnp.mean(phi)
        return 0.5 * jnp.sum(phi * phi)

    def loss_unrolled(r):
        phi = solve_poisson_fmg(r, dx, fourpiG, boundary=boundary,
                                 n_vcycles_per_level=1, n_extra_vcycles=6)
        if boundary in (0, 1):   # both BCs are singular up to a constant
            phi = phi - jnp.mean(phi)   # same gauge pinning as the custom op
        return 0.5 * jnp.sum(phi * phi)

    g_custom = scale * np.asarray(jax.grad(loss_customvjp)(rho))
    g_unroll = np.asarray(jax.grad(loss_unrolled)(rho))

    # FD along 5 random COMPATIBLE directions.
    #
    # Not a bare one-voxel bump: both singular boundaries require the source
    # to sum to zero, and a bump breaks that. Measured when it did (Neumann,
    # 2026-09-21): the custom gradient and the finite difference differed by
    # the SAME absolute 7.74e-5 at every voxel -- a constant field, i.e. the
    # operator's null space, appearing because the probe left the solvable
    # subspace. Subtracting the mean of the bump keeps it inside, and the
    # directional derivative <grad, delta> is then the right comparison.
    rng = np.random.default_rng(0)
    idx = rng.choice(N ** 3, size=5, replace=False)
    h = 1e-5
    max_fd_err = 0.0
    singular = boundary in (0, 1)
    print(f"  [{name}]")
    for flat in idx:
        i, j, k = np.unravel_index(flat, (N, N, N))
        delta = np.zeros((N, N, N))
        delta[i, j, k] = 1.0
        if singular:
            delta -= delta.mean()              # stay in the solvable subspace
        d = jnp.asarray(delta)
        gfd = (float(loss_customvjp(rho + h * d))
               - float(loss_customvjp(rho - h * d))) / (2 * h)
        gcv = float(np.sum(g_custom * delta))
        rel = abs(gcv - gfd) / max(abs(gfd), 1e-30)
        max_fd_err = max(max_fd_err, rel)
        print(f"    voxel ({i:2d},{j:2d},{k:2d}): custom={gcv:+.6e} "
              f"FD={gfd:+.6e}  rel={rel:.2e}")

    rel_unroll = np.max(np.abs(g_custom - g_unroll)) / (np.max(np.abs(g_unroll)) + 1e-30)
    print(f"MEASURED poisson adjoint {name} scale={scale}: custom vs FD {max_fd_err:.3e}, "
          f"custom vs unrolled {rel_unroll:.3e}")
    return max_fd_err, rel_unroll


# The unrolled gradient differentiates the finitely converged iteration map (1 + 6 V-cycles), whose Jacobian only
# approaches the exact adjoint as the solve tightens, so it carries an O(solver residual) error of its own; FD on the
# quadratic loss is the ground truth. Bounds ~1.3x measured, CPU-8 FP64, job 31654893: Neumann 2.658e-6 (FD),
# 3.958e-4 (unrolled); periodic 1.128e-5 (FD), 1.416e-8 (unrolled). The 1.01-scaled teeth gradient measured 1.000e-2.
BOUNDS = {1: (3.5e-6, 5.1e-4), 0: (1.5e-5, 1.9e-8)}   # boundary: (custom vs FD, custom vs unrolled)
NAMES = {1: "Neumann BC", 0: "periodic BC"}


@pytest.mark.parametrize("boundary", [1, 0], ids=["neumann", "periodic"])
def test_poisson_adjoint_vjp(boundary):
    """Custom adjoint against central FD (5 compatible directions) and against unrolled AD, measured bounds."""
    fd, unroll = _check(boundary, NAMES[boundary])
    assert fd < BOUNDS[boundary][0], f"custom-vjp vs FD {fd:.3e}"
    assert unroll < BOUNDS[boundary][1], f"custom-vjp vs unrolled {unroll:.3e}"


def test_fd_gate_catches_a_one_percent_wrong_adjoint():
    """Teeth: the Neumann custom gradient scaled by 1.01 fails the FD bound."""
    fd, _ = _check(1, NAMES[1], scale=1.01)
    assert fd > BOUNDS[1][0], fd
