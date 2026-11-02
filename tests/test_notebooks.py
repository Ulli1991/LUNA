"""Every notebook in examples/notebooks executes end to end (jupyter nbconvert --execute), JAX on the CPU.

Needs nbconvert and a Python kernel (`pip install jupyter`), scipy, and luna-reference-solutions for the
McNally part of 08 (luna.reference_data); a missing one fails the notebook, it is not skipped."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

DIR = Path(__file__).resolve().parents[1] / "examples" / "notebooks"
NOTEBOOKS = sorted(DIR.glob("*.ipynb"))


def test_notebooks_found():
    """The eleven notebooks are where this file looks (a moved directory must not pass as none run)."""
    assert len(NOTEBOOKS) >= 11, f"{len(NOTEBOOKS)} notebooks in {DIR}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_notebook_executes(path, tmp_path):
    """Every cell runs without an error; the executed copy goes to tmp_path, the committed file is untouched."""
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": ""}   # one CPU device, as the notebooks are sized
    cmd = [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--output-dir", str(tmp_path),
           "--ExecutePreprocessor.timeout=3000", str(path)]
    r = subprocess.run(cmd, cwd=path.parent, env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"{path.name} failed to execute:\n{r.stderr[-4000:]}"
