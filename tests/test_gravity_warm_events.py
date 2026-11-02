"""The warm-started gravity solve through collapse (periodic and isolated), sink formation and a feedback burst."""
import os

import pytest

from tests import _gravity_warm as gw
from tests.test_gravity_warm import N_CPU, N_GPU, check

CASES = {
    "collapse_periodic": (lambda N: gw.collapse(N), 0, (), ("skip", 20)),
    "collapse_isolated": (lambda N: gw.collapse(N), 3, (), ("skip", 20)),
    "sinks": (lambda N: gw.sinks(N), 0, (), ("skip", 6), gw.SINK_EVENTS),
    "feedback": (lambda N: gw.feedback(N), 0, (), ("skip", 5)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_warm_solve_through_events(name):
    check(name, N_CPU, CASES)


@pytest.mark.gpu
@pytest.mark.parametrize("name", list(CASES))
def test_warm_solve_through_events_large(name):
    if os.environ.get("JAX_PLATFORMS", "") == "cpu":
        pytest.skip("the large cases run on a GPU")
    check(name, N_GPU, CASES)
