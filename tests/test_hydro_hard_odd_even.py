"""Odd-even decoupling, Liska & Wendroff 2003 sec. 4.9: the Woodward-Colella blast on a thin 2D strip.

LW run reflective x, periodic y; for y-uniform data reflective y is the same, so BOUNDARY=2 throughout."""
import numpy as np
import pytest

from luna.ic_generation import woodward_colella as wc
from tests._helpers import _make_step_fns, accuracy_cfg, check, check_conserved

N, NY, T = 1024, 16, 0.038


def strip(n, ny, solver):
    """Config and y-uniform IC of the WC blast on (0, 1) x (0, ny/n): square cells."""
    cfg = accuracy_cfg(test_problem="Woodward-Colella", SPATIAL_DIMENSION=2, Nx=n, Ny=ny, Lx=1.0,
                       Ly=ny / n, BOUNDARY=2, gamma=1.4, riemann_solver=solver)
    one = accuracy_cfg(test_problem="Woodward-Colella", SPATIAL_DIMENSION=1, Nx=n, Lx=1.0,
                       BOUNDARY=2, gamma=1.4, riemann_solver=solver)
    U1 = np.asarray(wc.woodward_colella(one))
    U2 = np.stack([np.repeat(U1[0][:, None], ny, 1), np.repeat(U1[1][:, None], ny, 1),
                   np.zeros((n, ny)), np.repeat(U1[2][:, None], ny, 1)])
    return cfg, one, U1, U2


@pytest.mark.parametrize("solver", ("hllc", "hll"))
def test_odd_even(solver):
    """v_y stays exactly 0, every row equal, and the strip equals the 1D run step for step (same dt); mass and
    energy to round-off in the closed box (own loop, so not via evolve)."""
    cfg, one, U1, U2 = strip(N, NY, solver)
    U1_0, U2_0 = U1, U2
    (dt1, step1), (dt2, step2) = _make_step_fns(one), _make_step_fns(cfg)
    t = 0.0
    while t < T * (1 - 1e-14):
        dt = min(float(dt1(U1)), float(dt2(U2)), T - t)
        U1, U2, t = step1(U1, dt), step2(U2, dt), t + dt
    U1, U2 = np.asarray(U1), np.asarray(U2)
    check(f"odd-even {solver} max |v_y|", float(np.abs(U2[2] / U2[0]).max()), 0.0)
    check(f"odd-even {solver} row spread", float(np.abs(U2 - U2[:, :, :1]).max()), 0.0)
    check(f"odd-even {solver} |strip - 1D|", float(np.abs(U2[[0, 1, 3], :, 0] - U1).max()), 0.0)
    check_conserved(f"odd-even {solver} 1D", U1_0, U1, momentum=False)
    check_conserved(f"odd-even {solver} strip", U2_0, U2, momentum=False)


def test_check_sees_a_seeded_asymmetry():
    """Teeth: a 1e-6 pressure bump in one cell gives v_y != 0."""
    cfg, _, _, U2 = strip(N // 8, NY, "hllc")
    U2[3, N // 16, 0] *= 1.0 + 1e-6
    _, step = _make_step_fns(cfg)
    for _ in range(20):
        U2 = step(U2, 1e-5)
    assert float(np.abs(np.asarray(U2)[2]).max()) > 0.0
