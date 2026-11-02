"""`sink_gas_moments` on each sink's window, one device and per block of a
sharded grid: the same moments as a whole-grid reference (nearest image in a
periodic box), and no whole-grid temporaries -- the whole-grid version asked
for 36 GiB on one device at 1024^3."""
from types import SimpleNamespace

import numpy as np
import pytest

import jax

import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from luna.species.sinks import _moment_kernel, sink_gas_moments  # noqa: E402

N, L, GAMMA = 16, 1.0, 5.0 / 3.0
R_ACC = 2.5 / N
CFG = SimpleNamespace(Lx=L, Ly=L, Lz=L, Nx=N, Ny=N, Nz=N, gamma=GAMMA, BOUNDARY=0,
                      sink_accretion_radius=R_ACC)
POS = np.array([[0.5, 0.5, 0.5], [0.52, 0.31, 0.77], [0.01, 0.97, 0.5], [0.3, 0.3, 0.02]])


def _gas(seed=0):
    rng = np.random.default_rng(seed)
    rho = 1.0 + rng.random((N, N, N))
    v = 0.3 * rng.standard_normal((3, N, N, N))
    P_ = 1.0 + 0.5 * rng.random((N, N, N))
    E = P_ / (GAMMA - 1.0) + 0.5 * rho * (v ** 2).sum(0)
    return np.concatenate([rho[None], rho * v, E[None]])


def _reference(U, pos):
    """Whole-grid numpy, nearest image, as the moments are defined."""
    x = (np.arange(N) + 0.5) / N
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    rho = U[0]
    v = U[1:4] / rho
    P_ = (GAMMA - 1.0) * (U[4] - 0.5 * rho * (v ** 2).sum(0))
    cs = np.sqrt(P_ / rho)
    out = []
    for p in pos:
        d = [np.mod(Q - q + 0.5 * L, L) - 0.5 * L for Q, q in zip((X, Y, Z), p)]
        w = np.where(d[0] ** 2 + d[1] ** 2 + d[2] ** 2 < R_ACC ** 2, rho, 0.0)
        M = w.sum()
        mv = np.array([(w * v[k]).sum() for k in range(3)]) / M
        rv = [v[k] - mv[k] for k in range(3)]
        var = np.array([(w * rv[k] ** 2).sum() for k in range(3)]) / M
        Lz = [(w * (d[1] * rv[2] - d[2] * rv[1])).sum(), (w * (d[2] * rv[0] - d[0] * rv[2])).sum(),
              (w * (d[0] * rv[1] - d[1] * rv[0])).sum()]
        out.append(((w * cs).sum() / M, mv, var, np.array(Lz) / M))
    return out


def _check(got, U):
    for i, (cs, mv, var, am) in enumerate(_reference(U, POS)):
        np.testing.assert_allclose(got["mean_cs"][i], cs, rtol=1e-12)
        np.testing.assert_allclose(got["mean_velocity"][i], mv, rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(got["var_velocity"][i], var, rtol=1e-10, atol=1e-13)
        np.testing.assert_allclose(got["angular_momentum"][i], am, rtol=1e-10, atol=1e-13)


def test_the_window_gives_the_whole_grid_moments():
    U = _gas()
    _check(sink_gas_moments(CFG, jnp.asarray(U), POS), U)


def test_on_a_mesh_the_blocks_add_up_to_the_same_moments():
    if jax.device_count() < 8:
        pytest.skip("a (2,2,2) mesh needs 8 devices")
    mesh = Mesh(np.array(jax.devices()[:8]).reshape(2, 2, 2), ("x", "y", "z"))
    U = _gas(1)
    Us = jax.device_put(jnp.asarray(U), NamedSharding(mesh, P(None, "x", "y", "z")))
    _check(sink_gas_moments(CFG, Us, POS), U)


def test_no_whole_grid_temporaries():
    """The compiled kernel's scratch stays far below one field of the grid."""
    n = 64
    k = _moment_kernel(4, (n, n, n), (L, L, L), GAMMA, 2.5 / n, True)
    U = jax.ShapeDtypeStruct((5, n, n, n), jnp.float64)
    ma = k.lower(U, jax.ShapeDtypeStruct((4, 3), jnp.float64),
                 jax.ShapeDtypeStruct((4,), jnp.float64)).compile().memory_analysis()
    one_field = n ** 3 * 8
    assert ma.temp_size_in_bytes < one_field / 10, (ma.temp_size_in_bytes, one_field)
