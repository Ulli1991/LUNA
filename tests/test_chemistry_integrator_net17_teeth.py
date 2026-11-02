"""The stiff integrator driving network 17: teeth on the DVODE trajectories."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


from luna.ode import rodas3 as ig  # noqa: E402
from tools.chem_integrator import validate_net17 as val  # noqa: E402
from tests.test_chemistry_integrator_net17 import TARGET  # noqa: E402

TEETH_CASES = [9, 60, 121, 170, 230, 281]


@pytest.fixture(scope="module")
def reference():
    return val.load()


def _broken_euler(net, y, f0, jac, p, h, w):
    y1 = y + h * f0
    f1, aux = net.rhs(y1, p)
    return y1, 0.5 * h * (f1 - f0), aux["invalid"]


@pytest.mark.parametrize("brk", ["explicit_euler", "no_error_estimate"])
def test_teeth_trajectories(brk, reference, monkeypatch):
    names, cp, t, y0, ref, keep, steady, fixed = reference
    sel = np.asarray(TEETH_CASES)
    if brk == "explicit_euler":
        monkeypatch.setattr(ig, "_rodas3", _broken_euler)
    else:
        monkeypatch.setattr(ig, "_error_norm", lambda err, w: jnp.zeros(()))
    jax.clear_caches()
    res = val.run(cp[sel], t, y0[sel], ig.Options(max_steps=300, tiers=((None, 1),)), fixed)
    jax.clear_caches()
    rows, _ = val.table(names[sel], cp[sel], res["got"], ref[sel], keep, steady[sel], fixed)
    worst = max(r[4] for r in rows.values())
    print(f"{brk}: failed {int(res['failed'].sum())}, worst rel err {worst:.2e}")
    assert res["failed"].any() or not worst < TARGET
