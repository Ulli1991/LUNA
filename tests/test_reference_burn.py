"""LUNA's one-zone He burn (aprox13, screen5, sneut5, Rodas3) against the AMReX Microphysics references in
luna-reference-solutions (Microphysics 99b9b47, compiled outside LUNA by tools/helmholtz_reference.py,
tools/detonation_reference.py; burn_cell CI benchmark), on the Timmes table cut to the nodes these states read
when the full table is absent (tables/helmholtz/helm_table_subset.npz in luna-reference-solutions, tools/make_helm_subset.py; source helm_table.dat
sha256 c9a57c26..., md5 b1f27809...; bitwise equal to the full table on every kept cell, checked by
`make_helm_subset.py --verify`). The full table is used where it exists ($LUNA_HELM_TABLE or the Raven path).
"""
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from luna.reference_data import reference_path

from tests._helpers import check

jax.config.update("jax_enable_x64", True)

SHA_FULL = "c9a57c26c6fd2b2b378b9d5295ca1214022f6fec6289d038b47bf8c8938881a1"


FULL = Path(os.environ.get("LUNA_HELM_TABLE", ""))   # optional full table; no machine default


@pytest.fixture(scope="module")
def table():
    """The full Timmes table where it exists (sha256 checked), else the cut table on the same grid (unkept
    nodes NaN, bitwise equal on every kept cell), built as luna.eos.helmholtz.load_table does."""
    from luna.eos.helmholtz import HelmholtzTable, load_table
    if os.environ.get("LUNA_HELM_TABLE") and FULL.is_file() and hashlib.sha256(FULL.read_bytes()).hexdigest() == SHA_FULL:
        print(f"MEASURED table: full {FULL}")
        return load_table(FULL)
    d = np.load(reference_path("tables/helmholtz/helm_table_subset.npz"))
    assert json.loads(str(d["provenance"]))["sha256"] == SHA_FULL
    print("MEASURED table: tables/helmholtz/helm_table_subset.npz (luna-reference-solutions)")
    return HelmholtzTable(jnp.asarray(np.logspace(-12, 15, 541)), jnp.asarray(np.logspace(3, 13, 201)),
                          jnp.asarray(d["free"]), jnp.asarray(d["dpdr"]))


# Microphysics burn_cell CI benchmark (aprox13_Rosenbrock_unit_test.out, run at rtol = atol = 1e-6).
def helium_burn(table, screening="screen5"):
    """(T_final, added energy, final X, reference) of the one-zone He burn rho = 1e6, T0 = 1e9, 0.01 s (Rodas3,
    rtol 1e-7)."""
    from luna.eos.inversion import temperature_from_energy_checked
    from luna.nuclear.aprox13 import mean_composition
    from luna.nuclear.burn import BurnParams, initial_state, make_network
    from luna.ode.rodas3 import Options, integrate
    ref = json.loads(Path(reference_path("microphysics/helium_burn/helium_burn_reference.json")).read_text())
    rho, T0, dt = ref["density"], ref["initial_temperature"], ref["duration"]
    net = make_network(table, screening=screening, neutrinos="sneut5")
    start = np.asarray(initial_state(table, rho, T0, np.eye(13)[0]))
    res = integrate(jnp.asarray(start[None]), BurnParams(jnp.asarray([rho]), jnp.asarray([T0])), dt, net=net,
                    opt=Options(max_steps=20000, tiers=((None, 1),), diff_fields=(), rtol=1e-7))
    assert not bool(res.failed[0])
    y = np.asarray(res.y[0])
    m, z = mean_composition(y[:13])
    T = float(temperature_from_energy_checked(table, rho, y[13], m, z, temperature_guess=T0).temperature)
    return T, y[13] - start[13], y[:13], ref


def helium_misses(table, screening="screen5"):
    """Relative misses / the reference's comparison tolerances (T, added energy, worst abundance)."""
    T, de, X, ref = helium_burn(table, screening)
    tol = ref["comparison_tolerances"]
    Xr = np.asarray(ref["final_mass_fractions"])
    return (abs(T / ref["final_temperature"] - 1) / tol["temperature_rtol"],
            abs(de / ref["added_energy"] - 1) / tol["energy_rtol"],
            float((np.abs(X - Xr) / (tol["abundance_atol"] + tol["abundance_rtol"] * np.abs(Xr))).max()))


def test_helium_burn_against_microphysics(table):
    """Final T, added energy, 13 abundances against Microphysics burn_cell (aprox13, its run rtol = atol = 1e-6):
    within the reference's comparison tolerances (T, energy 1e-6; X: 1e-5 relative + 1e-8)."""
    for name, m in zip(("T_final", "added energy", "abundances"), helium_misses(table)):
        check(f"He burn {name} miss / reference tolerance", m, 1.0)


def test_helium_burn_teeth_no_screening(table):
    """Teeth: without screening the burn misses the reference."""
    m = helium_misses(table, screening="none")
    print(f"MEASURED teeth He burn no screening: {m}")
    assert max(m) > 1.0
