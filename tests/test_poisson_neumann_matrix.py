"""Settle the Neumann adjoint against an explicitly built operator.

The story so far (audit sec.26, and tests/test_autodiff_identities.py):
  * the Neumann adjoint gate is off by 2.8e-3, identically on two jax
    versions, so it is not the stack;
  * pinning the null-space gauge the way the periodic case does makes it
    WORSE (4.06e-2), so it is not the gauge;
  * the reverse-only rule and the both-modes rule agree with each other to
    1.5e-6, so it is not a transpose slip between them;
  * but the implicit adjoint disagrees with plain unrolled autodiff by 42%
    under Neumann, while periodic agrees to 1e-4.

Two rules that agree with each other and disagree with unrolling share an
assumption. The implicit adjoint solves A lambda = Phi_bar with the SAME
operator as the forward solve, which is only the transpose when A is
symmetric. So: build A on a box small enough to write down, and ask.

8^3 = 512 unknowns, one 512x512 matrix, a second of numpy.
"""

import numpy as np
import pytest


import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402

pytestmark = pytest.mark.fast

M = 8                      # cells per side: 512 unknowns
DX = 1.0 / M


def _operator_matrix(boundary):
    """A as an explicit matrix: apply the code's own 7-point Laplacian, with
    the code's own halo for this boundary, to every unit vector."""
    from luna.gravity.poisson_bc import make_halo_3d
    from luna.gravity.poisson_multigrid import laplacian_7pt
    halo = make_halo_3d(boundary)
    apply = jax.jit(lambda u: laplacian_7pt(u.reshape(M, M, M), DX, halo).reshape(-1))
    eye = jnp.eye(M ** 3)
    return np.asarray(jax.vmap(apply)(eye)).T          # columns = A e_i


@pytest.mark.parametrize("bc,name", [(0, "periodic"), (1, "Neumann")])
def test_the_discrete_operator_is_symmetric(bc, name):
    """The implicit adjoint solves with the same operator as the forward, so
    it is the transpose ONLY if the discrete Laplacian is symmetric. The
    periodic stencil is symmetric by construction. If the Neumann one is not,
    every adjoint through it is wrong by exactly that asymmetry -- and that
    is the whole of the sec.26 discrepancy."""
    A = _operator_matrix(bc)
    asym = np.max(np.abs(A - A.T)) / max(np.max(np.abs(A)), 1e-300)
    assert asym < 1e-12, (
        f"{name}: the discrete Laplacian is NOT symmetric (max |A - A^T| / "
        f"max|A| = {asym:.2e}). A solve with A is then not the adjoint of a "
        "solve with A, so the implicit rule needs the transposed operator.")


@pytest.mark.parametrize("bc,name", [(0, "periodic"), (1, "Neumann")])
def test_the_gradient_matches_exact_linear_algebra(bc, name):
    """d/db of 1/2 ||A^+ b||^2 against the pseudo-inverse, which needs no
    solver and no autodiff: the answer is (A^+)^T A^+ b. Compares the
    implicit adjoint and plain unrolled autodiff against it, so whichever is
    wrong is named rather than inferred."""
    from luna.gravity.poisson_diff import solve_poisson_fmg_diff
    from luna.gravity.poisson_multigrid import solve_poisson_fmg
    A = _operator_matrix(bc)
    Ap = np.linalg.pinv(A, rcond=1e-10)                # handles the null space
    rng = np.random.default_rng(11)
    b = rng.normal(size=M ** 3)
    b -= b.mean()                                      # compatible RHS
    exact = Ap.T @ (Ap @ b)                            # d/db 1/2||A^+ b||^2

    rho = jnp.asarray(b.reshape(M, M, M))

    def loss_custom(r):
        phi = solve_poisson_fmg_diff(r, DX, 1.0, bc)
        return 0.5 * jnp.sum(phi * phi)

    def loss_unrolled(r):
        phi = solve_poisson_fmg(r, DX, 1.0, boundary=bc, n_vcycles_per_level=2,
                                n_extra_vcycles=12)
        if bc == 0:
            phi = phi - jnp.mean(phi)
        return 0.5 * jnp.sum(phi * phi)

    gc = np.asarray(jax.jit(jax.grad(loss_custom))(rho)).ravel()
    gu = np.asarray(jax.jit(jax.grad(loss_unrolled))(rho)).ravel()
    # the null-space component of the gradient is gauge, not physics: compare
    # the part that acts on compatible sources
    def rel(g):
        d = g - exact
        return float(np.linalg.norm(d - d.mean()) / np.linalg.norm(exact - exact.mean()))

    r_custom, r_unrolled = rel(gc), rel(gu)
    assert r_custom < 1e-2, (
        f"{name}: the implicit adjoint differs from exact linear algebra by "
        f"{r_custom:.2e} (unrolled autodiff differs by {r_unrolled:.2e})")
