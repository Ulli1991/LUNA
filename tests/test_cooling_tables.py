"""Cooling: the KI fit against published equilibria, the tables against the fit, and the isothermal projection's precision."""
import dataclasses

import numpy as np
import pytest

from luna.sources.cooling import combine, koyama_inutsuka_2002, sutherland_dopita_cie
from tests._helpers import check

# Published two-phase equilibria (n [cm^-3], T [K], half a unit in the last published digit of n and of T):
# Koyama & Inutsuka 2004 (n 2.93, T 756 K) and Jennings & Li 2021 (n 2, T 1500 K), both for the KI 2002 fit.
EQUILIBRIA = ((2.93, 756.0, 0.005, 0.5), (2.0, 1500.0, 0.5, 50.0))


def ki_fit(T, c=1.4e-2):
    """Lambda / Gamma of the KI 2002 fit, written here from the paper (the low-T coefficient 1.4e-2; printed 14)."""
    return 1.0e7 * np.exp(-1.184e5 / (T + 1000.0)) + c * np.sqrt(T) * np.exp(-92.0 / T)


def balance(table, n, T):
    """n Lambda(T) / Gamma on the table's log-log interpolant (as apply_cooling uses it)."""
    return n * 10.0 ** np.interp(np.log10(T), table.log_T, table.log_Lambda) / table.heating


def rounding_bound(n, T, dn, dT):
    """|ln(n Lambda / Gamma)| that the published rounding of (n, T) allows: dn/n + |dln Lambda/dln T| dT/T."""
    slope = (np.log(ki_fit(T * 1.0001)) - np.log(ki_fit(T / 1.0001))) / (2.0 * np.log(1.0001))
    return dn / n + abs(slope) * dT / T


def interp_bound(spacings, x):
    """A-priori max relative error of log-log linear interpolation near log10 T = x: ln(10) sum h^2/8 max|g''|."""
    # g = log10 Lambda(10^x). A sum over spacings: the combined table re-interpolates the KI table, and linear
    # interpolation does not grow a max-norm error.
    h = max(spacings)
    xs = np.linspace(x - 2 * h, x + 2 * h, 4001)
    g = np.log10(ki_fit(10.0 ** xs))
    g2 = np.abs(np.diff(g, 2)).max() / (xs[1] - xs[0]) ** 2
    return np.log(10.0) * sum(s * s for s in spacings) / 8.0 * g2


def test_fit_balances_the_published_states():
    """The KI fit (1.4e-2) balances both published states to their published rounding; the printed 14 does not."""
    # Bound from the published digits, no LUNA number. Measured 2026-11-02: 756 K is 0.14 % off balance (bound
    # 0.21 %); 1500 K is 2.0 % off (bound 27 %, n published to one digit, so a weak check). 14 is ~1000x off.
    for n, T, dn, dT in EQUILIBRIA:
        bound = rounding_bound(n, T, dn, dT)
        check(f"KI fit n={n} T={T:g} |ln(n Lambda/Gamma)|", abs(np.log(n * ki_fit(T))), bound)
        assert abs(np.log(n * ki_fit(T, c=14.0))) > bound      # teeth: the misprint is rejected


@pytest.mark.parametrize("name", ["koyama_inutsuka", "combined"])
def test_tables_reproduce_the_fit(name):
    """Both tables give the fit's n Lambda/Gamma at the published states to the a-priori interpolation bound."""
    # Measured 2026-11-02: 3.6e-5 at 756 K, 2.6e-5 at 1500 K (bounds 5.9e-5 / 3.0e-5 KI, 7.7e-5 / 3.9e-5 combined).
    # Teeth: a 1 % error in the table's heating rate fails.
    ki = koyama_inutsuka_2002()
    table = ki if name == "koyama_inutsuka" else combine(ki, sutherland_dopita_cie())
    spacings = [np.diff(ki.log_T).max()] + ([np.diff(table.log_T).max()] if table is not ki else [])
    for n, T, _, _ in EQUILIBRIA:
        bound = interp_bound(spacings, np.log10(T))
        check(f"{name} n={n} T={T:g} |table / fit - 1|", abs(balance(table, n, T) / (n * ki_fit(T)) - 1.0), bound)
        bad = dataclasses.replace(table, heating=1.01 * table.heating)
        assert abs(balance(bad, n, T) / (n * ki_fit(T)) - 1.0) > bound      # teeth


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
