"""Cooling tables against published thermal equilibria, and the isothermal projection's precision."""
import numpy as np
import pytest

from luna.sources.cooling import combine, koyama_inutsuka_2002, sutherland_dopita_cie
from tests._helpers import check

# Published equilibria (n Lambda = Gamma) of the KI 2002 fit with its corrected 1.4e-2: Koyama &
# Inutsuka 2004 (n 2.93, T 756 K), Jennings & Li 2021 (n 2, T 1500 K); the misprinted 14.0 gives ~1000.
EQUILIBRIA = ((2.93, 756.0, 0.9986), (2.0, 1500.0, 1.0199))


def balance(table, n, T):
    """n Lambda(T) / Gamma on the table's log-log interpolant (as apply_cooling uses it)."""
    return n * 10.0 ** np.interp(np.log10(T), table.log_T, table.log_Lambda) / table.heating


@pytest.mark.parametrize("name", ["koyama_inutsuka", "combined"])
def test_published_equilibria(name):
    """Both tables hold the published two-phase states in balance, to the interpolation (0.5 %)."""
    ki = koyama_inutsuka_2002()
    table = ki if name == "koyama_inutsuka" else combine(ki, sutherland_dopita_cie())
    for n, T, exact in EQUILIBRIA:
        check(f"{name} n={n} T={T:g} |n Lambda/Gamma - exact|", abs(balance(table, n, T) / exact - 1.0), 5e-3)


@pytest.mark.parametrize("mode", ["fp64", pytest.param("fp32", marks=pytest.mark.precision("fp32"))])
def test_isothermal_projection_keeps_the_run_precision(mode):
    """After the projection the step's own pressure, (gamma - 1)(E - KE), is cs2 rho to the run's round-off."""
    import jax.numpy as jnp
    from luna.sources.eos_projection import isothermal_project
    dtype = jnp.float64 if mode == "fp64" else jnp.float32
    rng = np.random.default_rng(5)
    rho = 1.0 + rng.random((8, 8))
    U = jnp.asarray(np.stack([rho, rng.standard_normal((8, 8)), rng.standard_normal((8, 8)), 10 + rho]), dtype)
    cs2, gamma = 0.1, 5.0 / 3.0
    V = isothermal_project(U, cs2, gamma)
    assert V.dtype == dtype
    V = np.asarray(V, np.float64)
    p = (gamma - 1.0) * (V[-1] - 0.5 * (V[1] ** 2 + V[2] ** 2) / V[0])
    tol = 1e-14 if mode == "fp64" else 1e-5       # fp32: E - KE cancellation, ~10 ulp
    check(f"{mode} isothermal projection max |p / (cs2 rho) - 1|", float(np.abs(p / (cs2 * V[0]) - 1.0).max()), tol)
