"""The stiff integrator driving network 17: the 288 DVODE one-zone trajectories."""
import pytest


from luna.ode import rodas3 as ig  # noqa: E402
from tools.chem_integrator import validate_net17 as val  # noqa: E402

# vs DVODE at rtol 1e-8 with converged dust ($LUNA_DATA/chemistry, run_variants.sbatch): every kept
# output outside thermal runaways (T changing by > 25 % within one output interval, where a
# phase lag dominates) / the equilibrium end states
TARGET, EQ_TARGET = 2e-3, 1e-4


@pytest.fixture(scope="module")
def reference():
    return val.load()


def test_trajectories_match_dvode(reference):
    names, cp, t, y0, ref, keep, steady, fixed = reference
    res = val.run(cp, t, y0, ig.Options(), fixed)
    assert not res["failed"].any()
    rows, eq = val.table(names, cp, res["got"], ref, keep, steady, fixed)
    for q, (worst, med, case, k, st) in rows.items():
        print(f"{q}: max {worst:.2e} (median {med:.1e}) at {case} output {k}; off runaways "
              f"{st:.2e}; eq {eq[q]:.2e}")
    for q in val.QTY:
        assert rows[q][4] < TARGET, q
        assert eq[q] < EQ_TARGET, q
