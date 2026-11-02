"""LeBlanc shock tube (LANL LA-14379) against the exact Riemann solution: errors, orders, positivity,
the contact overshoot, and second order in the smooth fan."""
import numpy as np
import pytest

from luna.hydro import exact_riemann as er
from tests._helpers import accuracy_cfg, check, check_tube_budget, evolve

G = 5.0 / 3.0
WL, WR, X0, T = (1.0, 0.0, (2.0 / 3.0) * 1e-1), (1e-3, 0.0, (2.0 / 3.0) * 1e-10), 0.3, 0.5
NS = (256, 512, 1024)
# Per (solver, limiter) (0 minmod, 1 MC, 2 van Leer; HLL with minmod), CPU, FP64, default dt; ~1.3x measured:
#   L1, L2: {N: (rho, u, p)}; orders: fitted over NS, measured -0.1 .. +0.3, (rho, u, p);
#   overshoot: max e / max e_exact - 1 at the contact, the classic LeBlanc symptom.
BOUNDS = {
    ("hllc", 0): dict(L1={256: (0.0037, 0.041, 1.75e-4), 512: (0.0021, 0.025, 9.3e-5), 1024: (0.001, 0.013, 4.5e-5)},
            L2={256: (0.0072, 0.172, 4.2e-4), 512: (0.0048, 0.137, 2.6e-4), 1024: (0.0031, 0.098, 1.5e-4)},
            L1_order=((0.84, 1.24), (0.74, 1.14), (0.88, 1.28)),
            L2_order=((0.52, 0.92), (0.31, 0.71), (0.64, 1.04)),
            overshoot={256: 0.28, 512: 0.22, 1024: 0.19}),
    # minmod measured L1 2.82e-3 3.11e-2 1.35e-4 | 1.56e-3 1.91e-2 7.14e-5 | 7.61e-4 9.73e-3 3.48e-5;
    # L2 5.48e-3 0.132 3.24e-4 | 3.67e-3 0.105 1.97e-4 | 2.32e-3 0.0748 1.16e-4; orders L1 0.944 0.839
    # 0.976, L2 0.621 0.410 0.742; overshoot 0.213 0.171 0.143
    ("hllc", 1): dict(L1={256: (0.0028, 0.0312, 1.17e-4), 512: (0.00155, 0.0191, 6.5e-5), 1024: (0.0008, 0.00985, 3.56e-5)},
            L2={256: (0.0058, 0.15, 2.61e-4), 512: (0.0039, 0.119, 1.63e-4), 1024: (0.0025, 0.0854, 1.01e-4)},
            L1_order=((0.81, 1.21), (0.73, 1.13), (0.76, 1.16)),
            L2_order=((0.50, 0.90), (0.30, 0.70), (0.59, 0.99)),
            overshoot={256: 0.34, 512: 0.32, 1024: 0.32}),
    # MC measured L1 2.17e-3 2.40e-2 8.98e-5 | 1.19e-3 1.47e-2 4.99e-5 | 6.15e-4 7.58e-3 2.74e-5;
    # L2 4.44e-3 0.115 2.01e-4 | 2.99e-3 0.0916 1.25e-4 | 1.93e-3 0.0657 7.75e-5; orders L1 0.908 0.832
    # 0.857, L2 0.601 0.402 0.687; overshoot 0.260 0.248 0.248
    ("hllc", 2): dict(L1={256: (0.0029, 0.0337, 1.26e-4), 512: (0.0016, 0.0203, 6.8e-5), 1024: (0.00074, 0.0097, 3.33e-5)},
            L2={256: (0.0059, 0.156, 2.93e-4), 512: (0.004, 0.123, 1.79e-4), 1024: (0.0024, 0.0849, 1.05e-4)},
            L1_order=((0.89, 1.29), (0.80, 1.20), (0.86, 1.26)),
            L2_order=((0.55, 0.95), (0.34, 0.74), (0.64, 1.04)),
            overshoot={256: 0.30, 512: 0.27, 1024: 0.25}),
    # van Leer measured L1 2.25e-3 2.59e-2 9.66e-5 | 1.22e-3 1.56e-2 5.23e-5 | 5.67e-4 7.46e-3 2.56e-5;
    # L2 4.57e-3 0.120 2.25e-4 | 3.04e-3 0.0947 1.38e-4 | 1.86e-3 0.0653 8.04e-5; orders L1 0.992 0.899
    # 0.959, L2 0.650 0.438 0.743; overshoot 0.230 0.208 0.189
    ("hll", 0): dict(L1={256: (0.0038, 0.039, 0.00018), 512: (0.0021, 0.024, 9.4e-05), 1024: (0.0012, 0.013, 4.8e-05)},
            L2={256: (0.0072, 0.17, 0.00043), 512: (0.0049, 0.13, 0.00026), 1024: (0.0033, 0.1, 0.00016)},
            L1_order=((0.76, 1.16), (0.67, 1.07), (0.84, 1.24)),
            L2_order=((0.46, 0.86), (0.27, 0.67), (0.63, 1.03)),
            overshoot={256: 0.26, 512: 0.2, 1024: 0.15}),
    # HLL minmod measured L1 0.00293 0.03 0.000137 | 0.00161 0.0187 7.22e-05 | 0.000885 0.0103 3.73e-05;
    # L2 0.00557 0.128 0.000331 | 0.00377 0.103 0.000198 | 0.00257 0.0767 0.00012; orders L1 0.864 0.771 0.939,
    # L2 0.557 0.369 0.733; overshoot 0.203 0.156 0.116
}
# Fan interior (central 60 %) restarted from the exact solution at t = 0.1 is second order; from t = 0
# it is first order (start-up error of the centred fan). {(solver, limiter): norms of rho, orders}
FAN_START = 0.1
FAN_BOUNDS = {
    ("hllc", 0): dict(L1={256: 3.4e-05, 512: 9.4e-06, 1024: 2.5e-06},
        L1_order=(1.78, 2.18),
        L2={256: 0.0001, 512: 2.7e-05, 1024: 7.1e-06},
        L2_order=(1.81, 2.21),
        Linf={256: 0.00054, 512: 0.00016, 1024: 4.1e-05},
        Linf_order=(1.76, 2.16)),
    # measured L1 2.64e-05, 7.24e-06, 1.94e-06; 1.884 | L2 7.69e-05, 2.09e-05, 5.46e-06; 1.908 | Linf 0.000419, 0.000121, 3.16e-05; 1.863
    ("hllc", 1): dict(L1={256: 1.8e-05, 512: 4.5e-06, 1024: 1.1e-06},
        L1_order=(1.9, 2.3),
        L2={256: 4e-05, 512: 1e-05, 1024: 2.5e-06},
        L2_order=(1.92, 2.32),
        Linf={256: 0.00015, 512: 4.1e-05, 1024: 9.9e-06},
        Linf_order=(1.88, 2.28)),
    # measured L1 1.37e-05, 3.48e-06, 8.59e-07; 1.998 | L2 3.09e-05, 7.71e-06, 1.89e-06; 2.017 | Linf 0.000118, 3.16e-05, 7.58e-06; 1.983
    ("hllc", 2): dict(L1={256: 2e-05, 512: 4.9e-06, 1024: 1.2e-06},
        L1_order=(1.93, 2.33),
        L2={256: 4.4e-05, 512: 1.1e-05, 1024: 2.6e-06},
        L2_order=(1.94, 2.34),
        Linf={256: 0.00016, 512: 4.3e-05, 1024: 1e-05},
        Linf_order=(1.85, 2.25)),
    # measured L1 1.53e-05, 3.81e-06, 9.19e-07; 2.027 | L2 3.38e-05, 8.42e-06, 2.01e-06; 2.035 | Linf 0.00012, 3.32e-05, 7.99e-06; 1.955
    ("hll", 0): dict(L1={256: 3.1e-05, 512: 8.3e-06, 1024: 2.2e-06},
        L1_order=(1.79, 2.19),
        L2={256: 8.1e-05, 512: 2.2e-05, 1024: 5.9e-06},
        L2_order=(1.78, 2.18),
        Linf={256: 0.00037, 512: 0.0001, 1024: 2.8e-05},
        Linf_order=(1.77, 2.17)),
    # measured L1 2.35e-05, 6.36e-06, 1.71e-06; 1.892 | L2 6.2e-05, 1.69e-05, 4.56e-06; 1.883 | Linf 0.000286, 7.99e-05, 2.13e-05; 1.873
}



def _prim(U):
    """(rho, u, p) from 1D conserved."""
    rho, u = U[0], U[1] / U[0]
    return rho, u, (G - 1.0) * (U[2] - 0.5 * rho * u * u)


def cfg_for(n, limiter, solver="hllc"):
    """The accuracy Config of LeBlanc on n cells of [0, 1]."""
    return accuracy_cfg(test_problem="LeBlanc", SPATIAL_DIMENSION=1, Nx=n, Lx=1.0, BOUNDARY=1, gamma=G,
                        LIMITER=limiter, riemann_solver=solver)


def run(n, limiter=0, t=T, solver="hllc"):
    """Numerical and exact (rho, u, p) on n cells of [0, 1] at t, from the registered IC; U at 0 and t."""
    from luna.ic_generation.ic import initial_conditions
    cfg = cfg_for(n, limiter, solver)
    U0 = np.asarray(initial_conditions(cfg))
    U, _ = evolve(cfg, U0, t)
    return _prim(U), er.cell_average(WL, WR, G, np.linspace(0.0, 1.0, n + 1), X0, t), U0, U


def norms(num, ex, n):
    """L1 and L2 (integrals over [0, 1]) of rho, u, p."""
    h = 1.0 / n
    l1 = [h * float(np.abs(a - b).sum()) for a, b in zip(num, ex)]
    l2 = [float(np.sqrt(h * ((a - b) ** 2).sum())) for a, b in zip(num, ex)]
    return l1, l2


IDS = {("hllc", 0): "minmod", ("hllc", 1): "mc", ("hllc", 2): "vanleer", ("hll", 0): "hll-minmod"}


@pytest.mark.parametrize("solver,limiter", list(BOUNDS), ids=[IDS[k] for k in BOUNDS])
def test_leblanc_ladder(solver, limiter):
    """L1 and L2 of rho, u, p at every N, their orders, positivity, the contact overshoot and the exact
    budget."""
    B, errs = BOUNDS[solver, limiter], {}
    for n in NS:
        num, ex, U0, U = run(n, limiter, solver=solver)
        check_tube_budget(f"LeBlanc {solver} lim{limiter} N={n}", U0, U, WL, WR, T, 1.0 / n, G)
        l1, l2 = norms(num, ex, n)
        errs[n] = (l1, l2)
        for k, name in enumerate("rup"):
            check(f"LeBlanc {solver} lim{limiter} N={n} L1 {name}", l1[k], B["L1"][n][k])
            check(f"LeBlanc {solver} lim{limiter} N={n} L2 {name}", l2[k], B["L2"][n][k])
        check(f"LeBlanc {solver} lim{limiter} N={n} min rho / rho_R", float(num[0].min()) / WR[0], 1.0 - 1e-9, lower=True)
        check(f"LeBlanc {solver} lim{limiter} N={n} min p / p_R", float(num[2].min()) / WR[2], 1.0 - 1e-6, lower=True)
        e_num, e_ex = num[2] / ((G - 1.0) * num[0]), ex[2] / ((G - 1.0) * ex[0])
        check(f"LeBlanc {solver} lim{limiter} N={n} e overshoot", float(e_num.max() / e_ex.max()) - 1.0,
              B["overshoot"][n])
    for j, norm in enumerate(("L1", "L2")):
        for k, name in enumerate("rup"):
            order = -np.polyfit(np.log(NS), np.log([errs[n][j][k] for n in NS]), 1)[0]
            lo, hi = B[f"{norm}_order"][k]
            check(f"LeBlanc {solver} lim{limiter} {norm} {name} order", order, lo, lower=True)
            check(f"LeBlanc {solver} lim{limiter} {norm} {name} order", order, hi)


@pytest.mark.parametrize("solver,limiter", list(FAN_BOUNDS), ids=[IDS[k] for k in FAN_BOUNDS])
def test_fan_is_second_order(solver, limiter):
    """The smooth fan interior, from the exact solution at t = 0.1 to t_f: L1 rho and its order ~2."""
    bnd = FAN_BOUNDS[solver, limiter]
    w = er.waves(WL, WR, G)
    head, tail = (X0 + s * T for s in w["left"])
    a, b = head + 0.2 * (tail - head), tail - 0.2 * (tail - head)
    errs = []
    for n in NS:
        edges = np.linspace(0.0, 1.0, n + 1)
        rho, u, p = er.cell_average(WL, WR, G, edges, X0, FAN_START)
        U0 = np.stack([rho, rho * u, p / (G - 1.0) + 0.5 * rho * u * u])
        U, _ = evolve(cfg_for(n, limiter, solver), U0, T - FAN_START)
        check_tube_budget(f"LeBlanc {solver} lim{limiter} N={n} fan", U0, U, WL, WR, T - FAN_START, 1.0 / n, G)
        ex = er.cell_average(WL, WR, G, edges, X0, T)
        inside = (edges[:-1] >= a) & (edges[1:] <= b)
        e = (U[0] - ex[0])[inside]
        errs.append((float(np.abs(e).sum()) / n, float(np.sqrt((e * e).sum() / n)), float(np.abs(e).max())))
        for k, norm in enumerate(("L1", "L2", "Linf")):
            if norm in bnd:
                check(f"LeBlanc {solver} lim{limiter} N={n} fan {norm} rho", errs[-1][k], bnd[norm][n])
    for k, norm in enumerate(("L1", "L2", "Linf")):
        if f"{norm}_order" in bnd:
            order = -np.polyfit(np.log(NS), np.log([e[k] for e in errs]), 1)[0]
            check(f"LeBlanc {solver} lim{limiter} fan {norm} order", order, bnd[f"{norm}_order"][0], lower=True)
            check(f"LeBlanc {solver} lim{limiter} fan {norm} order", order, bnd[f"{norm}_order"][1])


def test_bound_catches_a_short_run():
    """Teeth: stopped at 0.9 t_f, the density L1 at 512 misses its bound."""
    num = run(512, t=0.9 * T)[0]
    ex = er.cell_average(WL, WR, G, np.linspace(0.0, 1.0, 513), X0, T)
    l1 = norms(num, ex, 512)[0][0]
    assert l1 > BOUNDS["hllc", 0]["L1"][512][0], f"L1 rho {l1:.4e} of a short run passes the bound"


def test_refuses_another_gamma():
    """The problem is defined with gamma = 5/3."""
    from luna.ic_generation.ic import initial_conditions
    cfg = accuracy_cfg(test_problem="LeBlanc", SPATIAL_DIMENSION=1, Nx=64, Lx=1.0, BOUNDARY=1, gamma=1.4)
    with pytest.raises(ValueError, match="gamma = 5/3"):
        initial_conditions(cfg)
