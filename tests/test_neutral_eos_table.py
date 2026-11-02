"""The neutral sound speed from a table (rad_neutral_eos_table): P = rho
cs_o^2(rho) where x = 0, log-log between the rows, held at the ends; the
ionized limit and the constant-cs_o path unchanged."""
import numpy as np
import pytest

import jax.numpy as jnp


from luna.radiation.ionization import load_neutral_table, two_temperature   # noqa: E402

G = 5.0 / 3.0


def _U(rho, v=0.3):
    rho = jnp.asarray(rho, jnp.float64)
    return jnp.stack([rho, rho * v, 0 * rho, 0 * rho, rho])


def _pressure(U):
    kin = 0.5 * jnp.sum(U[1:4] ** 2, axis=0) / U[0]
    return np.asarray((U[4] - kin) * (G - 1.0))


@pytest.fixture
def table(tmp_path):
    f = tmp_path / "eos.txt"
    np.savetxt(f, np.array([[1.0, 4.0], [10.0, 1.0], [100.0, 0.25]]))
    return load_neutral_table(str(f))


def test_rows_between_and_beyond(table):
    rho = np.array([1.0, 10.0, 100.0, np.sqrt(10.0), 0.1, 1e3])
    U = two_temperature(_U(rho), jnp.zeros(6), cs2_neutral=99.0, cs2_ionized=500.0,
                        gamma=G, neutral_table=table)
    cs2 = _pressure(U) / rho
    want = np.array([4.0, 1.0, 0.25, 2.0, 4.0, 0.25])   # sqrt(10): the log-log midpoint
    assert np.max(np.abs(cs2 / want - 1.0)) < 1e-12


def test_ionized_and_constant_paths(table):
    rho = np.array([3.0, 30.0])
    Ui = two_temperature(_U(rho), jnp.ones(2), cs2_neutral=99.0, cs2_ionized=500.0,
                         gamma=G, neutral_table=table)
    assert np.allclose(_pressure(Ui) / rho, 500.0, rtol=1e-12)
    U0 = two_temperature(_U(rho), jnp.zeros(2), cs2_neutral=7.0, cs2_ionized=500.0, gamma=G)
    assert np.allclose(_pressure(U0) / rho, 7.0, rtol=1e-12)


def test_bad_table_refused(tmp_path):
    f = tmp_path / "bad.txt"
    np.savetxt(f, np.array([[10.0, 1.0], [1.0, 4.0]]))
    with pytest.raises(ValueError):
        load_neutral_table(str(f))
