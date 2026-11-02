"""Regression only: luna.stellar.run (the standalone stellar runner) against its own frozen output, bitwise.

The reference luna/stellar/stellar_golden.npz (luna-reference-solutions) is this runner's output (written by this file's __main__, commit 07e1d08),
not an external or analytic solution: a pass says the runner did not change, nothing about whether it is right.
Three cases (planar He burn, periodic smooth flow without burning, spherical C/O burn with self-gravity) on an
analytic ions+radiation EOS (a zero Helmholtz table), compared at the final checkpoint: steps, time, state,
temperature and the diagnostics JSON. No main.py path is run here.

Regenerate (only on purpose, from the reference commit): python tests/test_stellar_golden.py
"""
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from luna import stellar  # noqa: E402
from luna.eos.helmholtz import HelmholtzTable  # noqa: E402
from luna.reference_data import reference_path, reference_root  # noqa: E402

GOLDEN = "luna/stellar/stellar_golden.npz"   # in luna-reference-solutions

# (Config fields, stop time, dt_max): planar burn, periodic smooth flow, spherical self-gravity burn
CASES = {
    "he_planar": (dict(case="he", cells=64, length=1e8, density=1e6, temperature=1e8,
                       hot_temperature=2e9), 1e-6, 1e-7),
    "smooth_periodic": (dict(case="smooth", cells=32, length=1e8, density=1e6, temperature=1e9,
                             burn=False), 2e-3, 2e-4),
    "co_spherical": (dict(case="co", cells=32, length=1e8, density=1e7, temperature=1e8,
                          hot_temperature=3e9, geometry="spherical", gravity=True), 1e-6, 1e-7),
}


def ion_radiation_table():
    """An analytic ions+radiation EOS, as test_aprox13's fixture."""
    return HelmholtzTable(jnp.array([1e-12, 1e15]), jnp.array([1e3, 1e13]),
                          jnp.zeros((2, 2, 3, 3)), jnp.zeros((2, 2, 2, 2)))


def run_case(name, workdir, monkeypatch=None):
    """One case's final checkpoint arrays and counters."""
    fields, stop, dt_max = CASES[name]
    table = ion_radiation_table()
    if monkeypatch is None:
        stellar.load_table = lambda path: table
    else:
        monkeypatch.setattr(stellar, "load_table", lambda path: table)
    path = Path(workdir) / "analytic.table"
    path.write_bytes(b"analytic ions+radiation")
    result = stellar.run(path, stellar.Config(**fields), stop, Path(workdir) / name, dt_max=dt_max)
    state, temperature, time, steps, diagnostics = stellar.read_checkpoint(
        result["checkpoint"], stellar.Config(**fields), hashlib.sha256(path.read_bytes()).hexdigest()
    )
    return dict(state=np.asarray(state), temperature=np.asarray(temperature), time=time, steps=steps,
                diagnostics=json.dumps(diagnostics, sort_keys=True))


@pytest.mark.parametrize("name", sorted(CASES))
def test_stellar_runner_matches_golden(name, tmp_path, monkeypatch):
    """The runner's final checkpoint equals its frozen one bitwise (regression only)."""
    golden = np.load(reference_path(GOLDEN))
    result = run_case(name, tmp_path, monkeypatch)
    assert result["steps"] == int(golden[f"{name}/steps"]) and result["steps"] > 3
    assert result["time"] == float(golden[f"{name}/time"])
    np.testing.assert_array_equal(result["state"], golden[f"{name}/state"])
    np.testing.assert_array_equal(result["temperature"], golden[f"{name}/temperature"])
    assert result["diagnostics"] == str(golden[f"{name}/diagnostics"])


if __name__ == "__main__":
    import tempfile

    arrays = {}
    with tempfile.TemporaryDirectory() as scratch:
        for name in sorted(CASES):
            result = run_case(name, scratch)
            print(name, "steps", result["steps"], "t", result["time"],
                  "rejected", json.loads(result["diagnostics"])["rejected_steps"])
            arrays.update({f"{name}/{k}": np.asarray(v) for k, v in result.items()})
    out = reference_root() / GOLDEN
    np.savez_compressed(out, **arrays)
    print("wrote", out, "- now run make_manifest.sh in luna-reference-solutions and commit there")
