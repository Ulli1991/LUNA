"""Planar-Riemann-2D on a strip: exact solution, uniform in y, and the 2D kernel equal to the 1D one."""
import numpy as np
import pytest

from luna.hydro import exact_riemann as er
from tests._helpers import _make_step_fns, accuracy_cfg, check, check_tube_budget, evolve

G, T, NS = 1.4, 0.1, (256, 512, 1024)
WL, WR = (1.0, 0.0, 1.0), (0.125, 2.0, 0.1)      # two rarefactions about a contact
# {solver: {N: (rho, u, p)}} ~1.3x measured (minmod, FP64, default dt); orders measured -0.1 .. +0.3
L1_BOUND = {"hllc": {256: (0.003, 0.0072, 0.0025), 512: (0.0016, 0.0037, 0.0013), 1024: (0.00083, 0.0019, 0.00063)},
            # measured 2.28e-3 5.51e-3 1.90e-3 | 1.22e-3 2.85e-3 9.63e-4 | 6.36e-4 1.46e-3 4.85e-4
            "hll": {256: (0.003, 0.0071, 0.0025), 512: (0.0016, 0.0037, 0.0013), 1024: (0.00085, 0.0019, 0.00064)}}
            # measured 2.33e-3 5.48e-3 1.92e-3 | 1.25e-3 2.85e-3 9.73e-4 | 6.53e-4 1.45e-3 4.89e-4
L2_BOUND = {"hllc": {256: (0.0069, 0.014, 0.0066), 512: (0.0045, 0.0087, 0.0038), 1024: (0.003, 0.0053, 0.0022)},
            # measured 5.31e-3 1.08e-2 5.04e-3 | 3.44e-3 6.71e-3 2.92e-3 | 2.28e-3 4.11e-3 1.66e-3
            "hll": {256: (0.0069, 0.013, 0.0066), 512: (0.0045, 0.0082, 0.0038), 1024: (0.003, 0.005, 0.0022)}}
            # measured 5.29e-3 1.03e-2 5.09e-3 | 3.43e-3 6.28e-3 2.93e-3 | 2.29e-3 3.85e-3 1.67e-3
L1_ORDER = {"hllc": ((0.82, 1.22), (0.86, 1.26), (0.89, 1.29)),     # measured 0.922, 0.959, 0.985
            "hll": ((0.82, 1.22), (0.86, 1.26), (0.89, 1.29))}      # measured 0.919, 0.958, 0.985
L2_ORDER = {"hllc": ((0.51, 0.91), (0.60, 1.00), (0.70, 1.10)),     # measured 0.611, 0.697, 0.801
            "hll": ((0.51, 0.91), (0.61, 1.01), (0.70, 1.10))}      # measured 0.606, 0.707, 0.804



def strip_cfg(n, solver="hllc"):
    """The registered IC on n x 4 square cells."""
    return accuracy_cfg(test_problem="Planar-Riemann-2D", SPATIAL_DIMENSION=2, Nx=n, Ny=4, Lx=1.0,
                        Ly=4.0 / n, BOUNDARY=1, gamma=G, riemann_solver=solver)


def run(n, t=T, solver="hllc"):
    """The strip at t: (U, (rho, u, p) along the first row, exact cell averages); the budget checked."""
    from luna.ic_generation.ic import initial_conditions
    cfg = strip_cfg(n, solver)
    U0 = np.asarray(initial_conditions(cfg))
    U, _ = evolve(cfg, U0, t)
    check_tube_budget(f"planar {solver} N={n}", U0[[0, 1, 3], :, 0], U[[0, 1, 3], :, 0], WL, WR, t, 1.0 / n, G)
    rho, u = U[0, :, 0], U[1, :, 0] / U[0, :, 0]
    num = (rho, u, (G - 1.0) * (U[3, :, 0] - 0.5 * rho * u * u))
    return U, num, er.cell_average(WL, WR, G, np.linspace(0.0, 1.0, n + 1), 0.5, t)


@pytest.mark.parametrize("solver", ("hllc", "hll"))
def test_planar_riemann_ladder(solver):
    """L1 and L2 of rho, u, p at every N, their orders, the budget; the strip stays uniform in y, v = 0."""
    errs = {}
    for n in NS:
        U, num, ex = run(n, solver=solver)
        h = 1.0 / n
        l1 = [h * float(np.abs(a - b).sum()) for a, b in zip(num, ex)]
        l2 = [float(np.sqrt(h * ((a - b) ** 2).sum())) for a, b in zip(num, ex)]
        errs[n] = (l1, l2)
        for k, name in enumerate("rup"):
            check(f"planar {solver} N={n} L1 {name}", l1[k], L1_BOUND[solver][n][k])
            check(f"planar {solver} N={n} L2 {name}", l2[k], L2_BOUND[solver][n][k])
        check(f"planar {solver} N={n} spread in y", float(np.abs(U - U[:, :, :1]).max()), 0.0)
        check(f"planar {solver} N={n} max |v|", float(np.abs(U[2]).max()), 0.0)
    for j, (norm, bands) in enumerate((("L1", L1_ORDER[solver]), ("L2", L2_ORDER[solver]))):
        for k, name in enumerate("rup"):
            order = -np.polyfit(np.log(NS), np.log([errs[n][j][k] for n in NS]), 1)[0]
            check(f"planar {solver} {norm} {name} order", order, bands[k][0], lower=True)
            check(f"planar {solver} {norm} {name} order", order, bands[k][1])


def test_2d_kernel_equals_1d_kernel():
    """With the same dt sequence the 2D step on a y-uniform strip is bitwise the 1D step (the RAMSES
    dt rule alone differs: in 2D it also sums the y term)."""
    from luna.ic_generation.ic import initial_conditions
    n = 256
    c2 = strip_cfg(n)
    U2 = np.asarray(initial_conditions(c2))
    U1 = U2[[0, 1, 3], :, 0]
    c1 = accuracy_cfg(test_problem="Sod-Shock", SPATIAL_DIMENSION=1, Nx=n, Lx=1.0, BOUNDARY=1, gamma=G)
    (dt1, step1), (dt2, step2) = _make_step_fns(c1), _make_step_fns(c2)
    t = 0.0
    while t < T * (1 - 1e-14):
        dt = min(float(dt1(U1)), float(dt2(U2)), T - t)
        U1, U2, t = step1(U1, dt), step2(U2, dt), t + dt
    check("|2D strip - 1D| with equal dt", float(np.abs(np.asarray(U1) - np.asarray(U2)[[0, 1, 3], :, 0]).max()), 0.0)


def test_bound_catches_a_short_run():
    """Teeth: stopped at 0.9 t, the density L1 at 512 misses its bound (measured 6.76e-3)."""
    _, num, _ = run(512, t=0.9 * T)
    ex = er.cell_average(WL, WR, G, np.linspace(0.0, 1.0, 513), 0.5, T)
    l1 = float(np.abs(num[0] - ex[0]).sum()) / 512
    assert l1 > L1_BOUND["hllc"][512][0], f"L1 rho {l1:.4e} of a short run passes the bound"
