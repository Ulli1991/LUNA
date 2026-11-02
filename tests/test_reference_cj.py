"""The prescribed-ash CJ solver (`luna.nuclear.detonation.prescribed_ash_cj`) against an independent CJ, and the ZND
complete-burn endpoint (`make_znd(...).recover`) against that CJ state.

Independent method: the CJ detonation is the minimum detonation speed over the reactive Hugoniot (Jouguet's rule):
on each rho the Hugoniot temperature (vectorised Newton), D^2 = (p - p0) / (rho0^2 (1/rho0 - 1/rho)), minimised
over ln rho by four zooms of a 2001-node grid (one jitted batch each).
LUNA instead solves Hugoniot + frozen-sonic (u = c_s) jointly by least squares. Same EOS for both: the analytic
ions + radiation Helmholtz table of tests/test_aprox13.py (zero electron free energy; it runs everywhere). The
Timmes-table CJ is checked against AMReX Microphysics by tools/check_detonation_reference.py (60 MB table, not in
the repo). Published CJ speeds (NSE ash) are not like-for-like with a prescribed pure-Ni56 ash: printed only.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from luna.reference_data import reference_path
from tests._helpers import check

jax.config.update("jax_enable_x64", True)

NI, SI = 12, 5                                     # aprox13 order: he4 c12 o16 ... si28 (5) ... ni56 (12)
CASES = {"he90c10_1e6": (1e6, 1e8, {0: 0.9, 1: 0.1}), "co_1e7": (1e7, 1e8, {1: 0.5, 2: 0.5})}


def table():
    """The analytic ions + radiation table of tests/test_aprox13.py."""
    from luna.eos.helmholtz import HelmholtzTable
    return HelmholtzTable(jnp.array([1e-12, 1e15]), jnp.array([1e3, 1e13]), jnp.zeros((2, 2, 3, 3)),
                          jnp.zeros((2, 2, 2, 2)))


def comp(d):
    """13 mass fractions from {index: X}."""
    x = np.zeros(13)
    for i, v in d.items():
        x[i] = v
    return x


def heat_independent(fuel, ash):
    """q = sum (X_ash - X_fuel) B_i / A_i N_A MeV, from the aprox13 data tables."""
    from luna.nuclear.aprox13 import BINDING_MEV, MASS_NUMBERS, MEV_TO_ERG, NA
    return float(np.sum((ash - fuel) * np.array(BINDING_MEV) / np.array(MASS_NUMBERS)) * NA * MEV_TO_ERG)


@jax.jit
def _hugoniot(tab, lnr, c):
    """(D^2, T, scaled residual) on the reactive Hugoniot at rho = exp(lnr); c = (rho0, T0, p0, e0, q, A, Z) of
    fuel and ash, passed as data so one compilation serves every case."""
    from luna.eos.helmholtz import evaluate
    rho0, T0, p0, e0, q, ma, za = (c[k] for k in range(7))
    rho = jnp.exp(lnr)
    dv = 1 / rho0 - 1 / rho

    def step(_, lnT):
        s = evaluate(tab, rho, jnp.exp(lnT), ma, za)
        g = s.energy - e0 - q - 0.5 * (s.pressure + p0) * dv
        dg = (s.cv - 0.5 * s.dpdt * dv) * jnp.exp(lnT)
        return jnp.clip(lnT - jnp.clip(g / dg, -1.0, 1.0), jnp.log(T0), np.log(9.9e12))
    lnT = jax.lax.fori_loop(0, 40, step, jnp.full_like(lnr, np.log(5e9)))
    s = evaluate(tab, rho, jnp.exp(lnT), ma, za)
    res = (s.energy - e0 - q - 0.5 * (s.pressure + p0) * dv) / q
    return (s.pressure - p0) / (rho0 ** 2 * dv), jnp.exp(lnT), res


def cj_min_speed(tab, rho0, T0, fuel, ash):
    """(D, rho, T) at the minimum of D over the reactive Hugoniot: one jitted batch per zoom over a grid of
    ln rho, the Hugoniot temperature of every node by vectorised Newton in ln T (40 iterations, step clipped)."""
    from luna.eos.helmholtz import evaluate
    from luna.nuclear.aprox13 import mean_composition
    mf, zf = (float(v) for v in mean_composition(jnp.asarray(fuel)))
    ma, za = (float(v) for v in mean_composition(jnp.asarray(ash)))
    s0 = evaluate(tab, rho0, T0, mf, zf)
    c = jnp.array([rho0, T0, float(s0.pressure), float(s0.energy), heat_independent(fuel, ash), ma, za])
    lo, hi = np.log(1.05 * rho0), np.log(8 * rho0)
    for _ in range(4):
        g = np.linspace(lo, hi, 2001)
        D2, T, res = (np.asarray(a) for a in _hugoniot(tab, jnp.asarray(g), c))
        ok = np.abs(res) < 1e-13                    # no Hugoniot T inside the table at the dense end: excluded
        i = int(np.argmin(np.where(ok, D2, np.inf)))
        assert 2 <= i <= len(g) - 3 and ok[i - 2:i + 3].all(), "minimum not interior to converged nodes"
        lo, hi = g[max(i - 2, 0)], g[min(i + 2, len(g) - 1)]
    return float(np.sqrt(D2[i])), float(np.exp(g[i])), float(T[i])


@pytest.fixture(scope="module")
def tab():
    return table()


@pytest.fixture(scope="module")
def solved(tab):
    """{case: (LUNA Jump, independent (D, rho, T))}."""
    from luna.nuclear.detonation import prescribed_ash_cj
    out = {}
    for name, (rho0, T0, f) in CASES.items():
        fuel, ash = comp(f), comp({NI: 1.0})
        out[name] = (prescribed_ash_cj(tab, rho0, T0, fuel, ash), cj_min_speed(tab, rho0, T0, fuel, ash))
    return out


@pytest.mark.parametrize("name", list(CASES))
def test_cj_against_minimum_speed(solved, name):
    """D_CJ of LUNA equals the minimum Hugoniot speed (1e-9: LUNA accepts scaled residuals <= 1e-10, D is first
    order in them; the minimum is second order in its rho error); rho_CJ, T_CJ to 1e-4 (at tangency rho is fixed
    only to ~ sqrt(residual) = 1e-5 times O(1)); the heat release equals the independent q to 4 u."""
    jump, (D, rho, T) = solved[name]
    rho0, T0, f = CASES[name]
    q = heat_independent(comp(f), comp({NI: 1.0}))
    print(f"MEASURED CJ {name}: LUNA D={jump.speed:.10e} rho={jump.density:.8e} T={jump.temperature:.8e}; "
          f"min-speed D={D:.10e} rho={rho:.8e} T={T:.8e}; residual {jump.residual:.2e}")
    check(f"CJ {name} |q / q_ind - 1|", abs(jump.heat / q - 1), 4 * np.finfo(float).eps)
    check(f"CJ {name} |D / D_minspeed - 1|", abs(jump.speed / D - 1), 1e-9)
    check(f"CJ {name} |rho / rho_minspeed - 1|", abs(jump.density / rho - 1), 1e-4)
    check(f"CJ {name} |T / T_minspeed - 1|", abs(jump.temperature / T - 1), 1e-4)


def test_cj_teeth_wrong_composition(tab, solved):
    """Teeth: LUNA with 1 % Si28 left in the ash misses the pure-Ni56 minimum speed by far more than 1e-9."""
    from luna.nuclear.detonation import prescribed_ash_cj
    rho0, T0, f = CASES["co_1e7"]
    wrong = prescribed_ash_cj(tab, rho0, T0, comp(f), comp({NI: 0.99, SI: 0.01}))
    miss = abs(wrong.speed / solved["co_1e7"][1][0] - 1)
    print(f"MEASURED teeth CJ 1 % Si28 ash: |D / D_minspeed - 1| = {miss:.3e}")
    assert miss > 1e-9


@pytest.fixture(scope="module")
def znd_flows(tab, solved):
    """make_znd for C/O at 1e7 at D_CJ (1 + delta), delta = 1e-4 and 1e-6 (D_CJ from the minimum speed)."""
    from luna.nuclear.detonation import make_znd
    rho0, T0, f = CASES["co_1e7"]
    D = solved["co_1e7"][1][0]
    return [make_znd(tab, rho0, T0, jnp.asarray(comp(f)), D * (1 + d)) for d in (1e-4, 1e-6)]


def znd_ratio(flows, solved, ash):
    """(miss at delta = 1e-4, miss at 1e-6): |rho_end / rho_CJ - 1| of the complete burn to `ash`."""
    rho_cj = solved["co_1e7"][1][1]
    return [abs(float(fl.recover(jnp.asarray(ash))[0]) / rho_cj - 1) for fl in flows]


def test_znd_endpoint_converges_to_cj(znd_flows, solved):
    """At D = D_CJ (1 + delta) the complete-burn ZND endpoint (strong branch) is off the CJ rho by C sqrt(delta)
    (Rayleigh line near tangency): the miss falls by 10 from delta = 1e-4 to 1e-6, within 10 % (the O(sqrt delta)
    correction plus the 1e-8 rho_CJ uncertainty of the minimum)."""
    e1, e2 = znd_ratio(znd_flows, solved, comp({NI: 1.0}))
    print(f"MEASURED ZND endpoint miss {e1:.4e} (delta 1e-4), {e2:.4e} (1e-6), ratio {e1 / e2:.4f}")
    check("ZND endpoint |ratio / 10 - 1|", abs(e1 / e2 / 10 - 1), 0.1)


def test_znd_endpoint_teeth(znd_flows, solved):
    """Teeth: with 1 % Si28 left unburnt the endpoint does not converge to the pure-Ni56 CJ (ratio far from 10)."""
    e1, e2 = znd_ratio(znd_flows, solved, comp({NI: 0.99, SI: 0.01}))
    print(f"MEASURED teeth ZND 1 % Si28: miss {e1:.3e}, {e2:.3e}, ratio {e1 / e2:.3f}")
    assert abs(e1 / e2 / 10 - 1) > 0.1


def test_cj_published_context():
    """Printed, not gated: the Timmes-table prescribed-Ni56 CJ speed (AMReX Microphysics, luna-reference-solutions) for C/O at
    1e7 g/cc. Published CJ speeds (Gamezo+1999, Kushnir & Katz 2020) use NSE / equilibrium ash, a smaller q than
    pure Ni56 at these densities, so they are not comparable; no published value was transcribed here."""
    ref = json.loads(Path(reference_path("microphysics/detonation_cj/detonation_cj_reference.json")).read_text())
    for row, val in zip(ref["inputs"], ref["values"]):
        if row["case"] == "co" and row["density"] == 1e7:
            print(f"MEASURED AMReX prescribed-Ni56 CJ C/O rho=1e7 coulomb={row['coulomb']}: D = {val[3]:.6e} cm/s")
    assert ref["scope"].startswith("prescribed pure Ni56 ash")
