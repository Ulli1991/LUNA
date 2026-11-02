"""The warm-started gravity solve, step by step, against a converged solve: smooth evolution, dt changes, restarts.

Each case also runs a deliberately broken solve that the same bound must catch (docs/GRAVITY.md).
"""
import os

import pytest

from tests import _gravity_warm as gw

N_CPU, N_GPU = 32, 128

# case: (sequence(N), boundary, restart steps, broken solve the bound must catch)
CASES = {
    "jeans": (lambda N: gw.jeans(N), 0, (), ("skip", 8)),
    "dt_jump": (lambda N: gw.moving_blob(N, [0.005] * 6 + [0.02] * 4 + [0.002] * 4), 0, (),
                ("skip", 6)),
    "restart": (lambda N: gw.turbulence(N, n=12), 0, (6,), ("skip", 6)),
    "turbulence_fast": (lambda N: gw.turbulence(N, n=12, dt=0.05), 0, (), ("skip", 6)),
    "turbulence_slow": (lambda N: gw.turbulence(N, n=12, dt=0.005, t0=0.5), 0, (), ("skip", 6)),
}


def check(name, N, cases=CASES):
    make, boundary, restarts, broken = cases[name][:4]
    events = cases[name][4] if len(cases[name]) > 4 else ()
    steps = make(N)
    be = gw.backend(gw.config(N, boundary))
    err = gw.run(be, steps, boundary, restart_at=restarts, events=events)
    bad = gw.run(be, steps, boundary, restart_at=restarts, broken=broken, events=events)
    gw.assert_tracks(name, N, err, bad, restarts)
    return err


@pytest.mark.parametrize("name", list(CASES))
def test_warm_solve_tracks_the_converged_solve(name):
    check(name, N_CPU)


@pytest.mark.gpu
@pytest.mark.parametrize("name", list(CASES))
def test_warm_solve_tracks_the_converged_solve_large(name):
    if os.environ.get("JAX_PLATFORMS", "") == "cpu":
        pytest.skip("the large cases run on a GPU")
    check(name, N_GPU)
