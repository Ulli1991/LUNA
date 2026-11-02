"""LUNA's Helmholtz EOS and prescribed-ash CJ against the AMReX Microphysics references in
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


# ------------------------------------------------------------------------------------------------------ Helmholtz
# Bound per state and column: MARGIN x A x u, A the measured round-off amplification of LUNA's evaluation there:
# the largest |d out / out| / u over DRAWS random +-1 ulp perturbations of every table entry and of rho, T
# (the free-energy cancellations amplify exactly these). MARGIN = 10: x2 for two independent FP64 codes, x5 for
# the sampled maximum under-reading the worst sign pattern. Columns in NOT_DERIVED keep the tolerance of
# tools/check_helmholtz.py:39-49 (not derived) with the measured miss printed next to it.
U = np.finfo(float).eps / 2
MARGIN, DRAWS = 10.0, 8
TOOL_RTOL = dict(pressure=3e-12, energy=3e-12, entropy=1e-7, dpdr=3e-14, dpdt=3e-8, dedr=1e-10, cv=4e-7,
                 gamma1=3e-12, sound_speed=3e-12)
NOT_DERIVED = set()


def helm_eval(table, inp, flip_coulomb=False):
    """{column: values} over the reference states, Coulomb per row (flipped if asked)."""
    from luna.eos.helmholtz import evaluate
    out = {}
    for coulomb in (0, 1):
        rows = inp[:, -1] == coulomb
        s = evaluate(table, *inp[rows, :4].T, coulomb=bool(coulomb) ^ flip_coulomb)
        assert bool(np.asarray(s.valid).all())
        for c in TOOL_RTOL:
            out.setdefault(c, np.empty(len(inp)))[rows] = np.asarray(getattr(s, c))
    return out


def helm_ratios(table, flip_coulomb=False):
    """{column: (max miss / derived bound, max miss / tool bound, max A)} over the 50 reference states."""
    ref = json.loads(Path(reference_path("microphysics/helmholtz_eos/helmholtz_reference.json")).read_text())
    assert ref["table_sha256"] == SHA_FULL
    inp, val = np.asarray(ref["inputs"]), np.asarray(ref["values"])
    base = helm_eval(table, inp)
    rng = np.random.default_rng(1)
    amp = {c: np.zeros(len(inp)) for c in TOOL_RTOL}
    for _ in range(DRAWS):
        sign = lambda a: 1.0 + U * rng.choice([-1.0, 1.0], size=np.shape(a))  # noqa: E731
        t = table._replace(free=table.free * sign(table.free), dpdr=table.dpdr * sign(table.dpdr))
        pin = inp.copy()
        pin[:, :2] *= sign(pin[:, :2])
        pert = helm_eval(t, pin)
        for c in TOOL_RTOL:
            amp[c] = np.maximum(amp[c], np.abs(pert[c] / base[c] - 1) / U)
    got = helm_eval(table, inp, flip_coulomb)
    out = {}
    for k, c in enumerate(ref["columns"]):
        miss = np.abs(got[c] / val[:, k] - 1)
        out[c] = (float((miss / (MARGIN * np.maximum(amp[c], 1.0) * U)).max()), float(miss.max() / TOOL_RTOL[c]),
                  float(amp[c].max()))
    return out


def test_helmholtz_against_microphysics(table):
    """All 9 EOS outputs of the 50 Microphysics states (Coulomb on and off): per state within MARGIN A u (derived
    above); columns in NOT_DERIVED within the tool tolerance, labelled as such."""
    for c, (rd, rt, a) in helm_ratios(table).items():
        if c in NOT_DERIVED:
            check(f"Helmholtz {c} max miss / {TOOL_RTOL[c]:g} (from tools/check_helmholtz.py, not derived; "
                  f"miss / derived bound {rd:.3g})", rt, 1.0)
        else:
            check(f"Helmholtz {c} max miss / (10 A u), max A = {a:.3g} (tool bound ratio {rt:.3g})", rd, 1.0)


def test_helmholtz_teeth_coulomb_flipped(table):
    """Teeth: the Coulomb flag flipped misses the pressure bound."""
    m = helm_ratios(table, flip_coulomb=True)["pressure"][0]
    print(f"MEASURED teeth Helmholtz Coulomb flipped: pressure miss / derived bound = {m:.3e}")
    assert m > 1.0


# ------------------------------------------------------------------------------------------------------------- CJ
CJ = json.loads(Path(reference_path("microphysics/detonation_cj/detonation_cj_reference.json")).read_text())
CJ_IDS = [f"{r['case']}-{r['density']:.0e}-coul{int(r['coulomb'])}" for r in CJ["inputs"]]
CJ_TOL = 1e-8          # Microphysics cj_det.H (99b9b47) lines 9/37 and 60/95: Newton stops at |dT| < 1e-8 T, |drho| < 1e-8 rho


def cj_misses(table, k, ash=None):
    """(|rho/ref - 1|, |T/ref - 1|, |p/ref - 1|, |D/ref - 1|) and their bounds derived from CJ_TOL: p through
    |dln p/dln rho| + |dln p/dln T| at the state, D from D^2 = (p - p0) / (rho0^2 (1/rho0 - 1/rho))."""
    from luna.eos.helmholtz import evaluate
    from luna.nuclear.aprox13 import mean_composition
    from luna.nuclear.detonation import prescribed_ash_cj
    row, ref = CJ["inputs"][k], np.asarray(CJ["values"][k])
    ash = row["ash"] if ash is None else ash
    j = prescribed_ash_cj(table, row["density"], row["temperature"], row["fuel"], ash, coulomb=row["coulomb"])
    got = np.array([j.density, j.temperature, j.pressure, j.speed])
    ma, za = mean_composition(jnp.asarray(row["ash"]))
    s = evaluate(table, ref[0], ref[1], ma, za, coulomb=row["coulomb"])
    mf, zf = mean_composition(jnp.asarray(row["fuel"]))
    p0 = float(evaluate(table, row["density"], row["temperature"], mf, zf, coulomb=row["coulomb"]).pressure)
    bp = (abs(float(s.dpdr)) * ref[0] + abs(float(s.dpdt)) * ref[1]) / ref[2] * CJ_TOL
    rho0, rho = row["density"], ref[0]
    bD = 0.5 * (ref[2] / (ref[2] - p0) * bp + (1 / rho) / (1 / rho0 - 1 / rho) * CJ_TOL)
    return np.abs(got / ref - 1), np.array([CJ_TOL, CJ_TOL, bp, bD])


@pytest.mark.parametrize("k", range(len(CJ_IDS)), ids=CJ_IDS)
def test_cj_against_microphysics(table, k):
    """rho, T within the Microphysics Newton tolerance 1e-8, p and D within the bounds it implies; all ten
    fixture states (He and C/O pure-Ni56 ash, rho0 = 1e6 to 3e8, Coulomb on and off)."""
    miss, bound = cj_misses(table, k)
    for name, m, b in zip(("rho", "T", "p", "D"), miss, bound):
        check(f"CJ {CJ_IDS[k]} |{name} / Microphysics - 1|", m, b)


def test_cj_teeth_wrong_ash(table):
    """Teeth: 1 % Si28 left in the ash misses the Microphysics pure-Ni56 speed by far more than its bound."""
    k = CJ_IDS.index("co-1e+07-coul1")
    ash = np.asarray(CJ["inputs"][k]["ash"], float)
    ash[12], ash[5] = 0.99, 0.01
    miss, bound = cj_misses(table, k, ash=ash)
    print(f"MEASURED teeth CJ 1 % Si28: |D/ref - 1| = {miss[3]:.3e} (bound {bound[3]:.3e})")
    assert miss[3] > bound[3]
