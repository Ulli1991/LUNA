"""The autodiff run scripts: IC, a two-step rollout, one optimiser iteration, finite results."""
import pytest

from tests._run_script_cases import check, params


@pytest.mark.parametrize("name", params("autodiff"))
def test_script_runs_two_steps(name, tmp_path):
    """Run as `python -m <script>` would: the IC, two steps and the checks of its case."""
    check(name, tmp_path)
