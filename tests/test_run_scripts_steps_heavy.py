"""The cosmology, refinement and radiation run scripts: IC, two steps, finite, mass."""
import pytest

from tests._run_script_cases import check, params


@pytest.mark.parametrize("name", params("heavy"))
def test_script_runs_two_steps(name, tmp_path):
    """Run as `python -m <script>` would: the IC, two steps and the checks of its case."""
    check(name, tmp_path)
