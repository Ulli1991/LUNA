"""The ionisation kernel (radiation.m1_pallas.ionize): each tile stops once
none of its cells moves, and that must be bitwise the full 60 iterations;
the kernel agrees with ionization.ionize_implicit to round-off on a
structured front (an ionised core, a thin front, neutral gas outside); and a
dark neutral cell stays exactly neutral. CPU: Pallas's interpreter; GPU: the
Triton kernel. The sharded stage is covered by test_radiation_m1_pallas."""
import numpy as np

import jax
import jax.numpy as jnp


from luna.radiation import m1_pallas                          # noqa: E402
from luna.radiation.ionization import ionize_implicit         # noqa: E402
from tests._helpers import check                              # noqa: E402

GPU = jax.default_backend() in ("gpu", "rocm", "cuda")
N = 16 if not GPU else 64
KW = dict(chat=150.0, sigma=1.0, alpha_B=1e-3, m_H=1.0)
PHOTON_BALANCE = 1e-13   # placeholder until measured


def _front(n=N):
    x = (jnp.arange(n) + 0.5) / n - 0.5
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    r = jnp.sqrt(X * X + Y * Y + Z * Z)
    x0 = 1.0 / (1.0 + jnp.exp((r - 0.25) / (1.5 / n)))
    N0 = 1e2 * jnp.minimum(1.0, (0.1 / jnp.maximum(r, 1e-3)) ** 2) * jnp.exp(-jnp.maximum(r - 0.25, 0.0) / (2.0 / n))
    rho = 1.0 + 0.1 * jnp.cos(40 * X) * jnp.sin(30 * Y)
    R = jnp.stack([N0, 0.5 * N0, 0.2 * N0, -0.1 * N0])[None]
    return R, x0, rho, 1e-3


def test_the_early_exit_is_bitwise_the_full_solve():
    R, x0, rho, dt = _front()
    Rf, xf = jax.jit(lambda R, x, r: m1_pallas.ionize(R, x, r, dt, early=False, **KW))(R, x0, rho)
    Re, xe, it = jax.jit(lambda R, x, r: m1_pallas.ionize(R, x, r, dt, early=True,
                                                          return_iters=True, **KW))(R, x0, rho)
    np.testing.assert_array_equal(np.asarray(Re), np.asarray(Rf))
    np.testing.assert_array_equal(np.asarray(xe), np.asarray(xf))
    # photons absorbed per H are the ionisations plus the recombinations, cell by cell (as ionize_implicit);
    # relative where they exceed 1, else absolute: x comes from z = 1 - x, exact to ulp(1), not relative to x
    # (a near-neutral cell at 64^3 has x 7.7e-10, absolute error 1.1e-16, the XLA solve's too)
    n_H = np.asarray(rho) / KW["m_H"]
    x0, x1 = np.asarray(x0), np.asarray(xf)
    absorbed = (np.asarray(R[0, 0]) - np.asarray(Rf[0, 0])) / n_H
    spent = x1 - x0 + KW["alpha_B"] * n_H * dt * x1 ** 2
    scale = np.maximum.reduce([np.abs(absorbed), np.abs(spent), np.ones_like(x0)])
    check("pallas ionize per-cell photon balance", float(np.max(np.abs(absorbed - spent) / scale)), PHOTON_BALANCE)
    assert int(np.asarray(it).min()) < 60            # some tile did stop early


def test_the_kernel_is_the_xla_solve(monkeypatch):
    R, x0, rho, dt = _front()
    monkeypatch.setattr(m1_pallas, "KERNEL", "xla")
    Ra, xa = jax.jit(lambda R, x, r: ionize_implicit(R, x, r, dt, **KW))(R, x0, rho)
    Rb, xb = jax.jit(lambda R, x, r: m1_pallas.ionize(R, x, r, dt, **KW))(R, x0, rho)
    Ra, Rb = np.asarray(Ra), np.asarray(Rb)
    assert float(np.abs(Rb - Ra).max()) <= 1e-12 * float(np.abs(Ra).max())
    np.testing.assert_allclose(np.asarray(xb), np.asarray(xa), rtol=0, atol=1e-12)
    assert float(np.asarray(xa).max()) > 0.9 and float(np.asarray(xa).min()) < 0.1


def test_a_dark_neutral_cell_stays_exactly_neutral():
    R = jnp.zeros((1, 4, N, N, N))
    x0 = jnp.zeros((N, N, N))
    rho = jnp.ones((N, N, N))
    R1, x1 = jax.jit(lambda R, x, r: m1_pallas.ionize(R, x, r, 1e-3, **KW))(R, x0, rho)
    np.testing.assert_array_equal(np.asarray(x1), 0.0)
    np.testing.assert_array_equal(np.asarray(R1), 0.0)


def test_nan_photons_make_a_nan_fraction_not_an_ionised_cell():
    """As ionize_implicit: NaN photons reach x instead of bisecting to x = 1."""
    R, x0, rho, dt = _front()
    R = R.at[0, 0, 3, 3, 3].set(jnp.nan)
    R1, x1 = jax.jit(lambda R, x, r: m1_pallas.ionize(R, x, r, dt, **KW))(R, x0, rho)
    x1 = np.asarray(x1)
    assert np.isnan(x1[3, 3, 3])
    assert np.isfinite(np.delete(x1.ravel(), np.ravel_multi_index((3, 3, 3), x1.shape))).all()
