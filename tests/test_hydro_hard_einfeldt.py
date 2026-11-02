"""Einfeldt rarefactions (Athena 1-2-0-3, 1-1-0-5, 1-1-2-5) against the exact Riemann solution."""
import numpy as np
import pytest

from luna.hydro import exact_riemann as er
from luna.ic_generation import einfeldt as ef
from tests._helpers import accuracy_cfg, at_n, check, check_tube_budget, evolve

G, T, NS = 1.4, 0.15, (256, 512, 1024)
PROBLEMS = {"strong": ("Einfeldt-Strong", ef.STRONG), "shear": ("Einfeldt-Shear", ef.SHEAR),
            "transverse": ("Einfeldt-Transverse", ef.TRANSVERSE)}
# Per (problem, solver) (minmod, FP64, default dt), ~1.3x measured: L1, L2 {N: (rho, u, p)}; orders
# measured -0.1 .. +0.3. Transverse runs as a 2D strip (4 cells in y): HLLC keeps v exact (< 1e-14),
# HLL has no contact wave and smears the shear (max |v - v_exact| measured 1.73, 1.77, 1.81).
SOLVERS = ("hllc", "hll")
HLL_SHEAR_SMEAR = (1.5, 2.4)
BOUNDS = {
    ("strong", "hllc"): dict(L1={256: (0.005, 0.024, 0.0027), 512: (0.003, 0.018, 0.0013), 1024: (0.0021, 0.012, 0.00065)},
                L2={256: (0.0085, 0.04, 0.0046), 512: (0.0051, 0.036, 0.0026), 1024: (0.0037, 0.027, 0.0015)},
                L1_order=((0.54, 0.94), (0.37, 0.77), (0.91, 1.31)),
                L2_order=((0.51, 0.91), (0.16, 0.56), (0.73, 1.13))),
    # measured L1 [[0.00388, 0.0185, 0.00205], [0.00228, 0.014, 0.00101], [0.00159, 0.00959, 0.000503]]; L2 [[0.00656, 0.0304, 0.00353], [0.00396, 0.0277, 0.002], [0.00282, 0.0211, 0.00112]]; orders L1 [0.64, 0.47, 1.01], L2 [0.61, 0.26, 0.83]
    ("shear", "hllc"): dict(L1={256: (0.0034, 0.0062, 0.0063), 512: (0.0017, 0.0031, 0.0032), 1024: (0.00087, 0.0015, 0.0016)},
                L2={256: (0.0067, 0.012, 0.013), 512: (0.004, 0.0069, 0.0073), 1024: (0.0024, 0.0038, 0.004)},
                L1_order=((0.89, 1.29), (0.9, 1.3), (0.9, 1.3)),
                L2_order=((0.63, 1.03), (0.76, 1.16), (0.75, 1.15))),
    # measured L1 [[0.00265, 0.00474, 0.00486], [0.00134, 0.00238, 0.00243], [0.000671, 0.00119, 0.00122]]; L2 [[0.00512, 0.00954, 0.01], [0.00306, 0.00529, 0.00559], [0.00187, 0.0029, 0.00309]]; orders L1 [0.99, 1.0, 1.0], L2 [0.73, 0.86, 0.85]
    ("transverse", "hllc"): dict(L1={256: (0.0065, 0.0093, 0.0062), 512: (0.0033, 0.0047, 0.0031), 1024: (0.0017, 0.0023, 0.0016)},
                L2={256: (0.011, 0.017, 0.012), 512: (0.007, 0.0092, 0.0066), 1024: (0.0044, 0.005, 0.0037)},
                L1_order=((0.89, 1.29), (0.89, 1.29), (0.9, 1.3)),
                L2_order=((0.59, 0.99), (0.76, 1.16), (0.73, 1.13))),
    # measured L1 [[0.005, 0.00712, 0.0048], [0.00254, 0.00358, 0.0024], [0.00128, 0.00179, 0.0012]]; L2 [[0.00882, 0.0128, 0.00902], [0.00541, 0.00707, 0.00511], [0.00338, 0.00387, 0.00286]]; orders L1 [0.99, 0.99, 1.0], L2 [0.69, 0.86, 0.83]
    ("strong", "hll"): dict(L1={256: (0.005, 0.024, 0.0027), 512: (0.003, 0.018, 0.0013), 1024: (0.0021, 0.012, 0.00065)},
                L2={256: (0.0085, 0.04, 0.0046), 512: (0.0051, 0.036, 0.0026), 1024: (0.0037, 0.027, 0.0015)},
                L1_order=((0.54, 0.94), (0.37, 0.77), (0.91, 1.31)),
                L2_order=((0.51, 0.91), (0.16, 0.56), (0.73, 1.13))),
    # measured L1 [[0.00387, 0.0185, 0.00205], [0.00227, 0.014, 0.00101], [0.00159, 0.00959, 0.000503]]; L2 [[0.00656, 0.0304, 0.00353], [0.00395, 0.0278, 0.002], [0.00282, 0.0212, 0.00112]]; orders L1 [0.64, 0.473, 1.014], L2 [0.61, 0.262, 0.826]
    ("shear", "hll"): dict(L1={256: (0.0036, 0.0064, 0.0066), 512: (0.0018, 0.0032, 0.0033), 1024: (0.0009, 0.0016, 0.0016)},
                L2={256: (0.0063, 0.013, 0.013), 512: (0.0036, 0.0069, 0.0074), 1024: (0.002, 0.0038, 0.0041)},
                L1_order=((0.89, 1.29), (0.9, 1.3), (0.9, 1.3)),
                L2_order=((0.72, 1.12), (0.77, 1.17), (0.76, 1.16))),
    # measured L1 [[0.00275, 0.0049, 0.00505], [0.00138, 0.00246, 0.00253], [0.000695, 0.00123, 0.00126]]; L2 [[0.00488, 0.00966, 0.0102], [0.00276, 0.00531, 0.00567], [0.00157, 0.0029, 0.00312]]; orders L1 [0.991, 0.996, 0.999], L2 [0.82, 0.867, 0.856]
    ("transverse", "hll"): dict(L1={256: (0.019, 0.023, 0.013), 512: (0.011, 0.014, 0.0073), 1024: (0.0069, 0.0084, 0.0042)},
                L2={256: (0.038, 0.034, 0.017), 512: (0.029, 0.021, 0.01), 1024: (0.023, 0.013, 0.0062)},
                L1_order=((0.62, 1.02), (0.63, 1.03), (0.7, 1.1)),
                L2_order=((0.28, 0.68), (0.58, 0.98), (0.64, 1.04))),
    # measured L1 [[0.0145, 0.0179, 0.0098], [0.00866, 0.0107, 0.0056], [0.0053, 0.0065, 0.00325]]; L2 [[0.0294, 0.026, 0.0133], [0.0224, 0.0162, 0.00793], [0.0173, 0.0102, 0.00478]]; orders L1 [0.724, 0.729, 0.795], L2 [0.381, 0.678, 0.739]
}
# Liska & Wendroff 2003: rel. L1 of e at N 100, t 0.15 is 6.3 % (PPM) ... 31.3 % (LL); LUNA 18.1 %.
# Guo et al.: min rho 1.835e-2 at N 400, t 0.1; LUNA 1.886e-2 (exact cell-averaged 2.185e-2).
LW_E_ERROR = 23.5          # percent, 1.3x measured
GUO_MIN_RHO = (0.017, 0.0205)



def run(problem, n, t=T, solver="hllc"):
    """Numerical and exact primitives on n cells of [0, 1] at t, from the registered IC, and the
    (rho, rho u, E) rows at 0 and t for the budget.

    (rho, u, p) in 1D; (rho, u, p, v) on the transverse strip, along its first y row."""
    from luna.ic_generation.ic import initial_conditions
    name, ((rl, ul, vl, pl), (rr, ur, vr, pr)) = PROBLEMS[problem]
    two_d = problem == "transverse"
    kw = dict(SPATIAL_DIMENSION=2, Ny=4, Ly=4.0 / n) if two_d else dict(SPATIAL_DIMENSION=1)
    cfg = accuracy_cfg(test_problem=name, Nx=n, Lx=1.0, BOUNDARY=1, gamma=G, riemann_solver=solver, **kw)
    U0 = np.asarray(initial_conditions(cfg))
    U, _ = evolve(cfg, U0, t)
    U0, U = (U0[:, :, 0], U[:, :, 0]) if two_d else (U0, U)
    rho, u = U[0], U[1] / U[0]
    v = U[2] / U[0] if two_d else 0.0
    num = [rho, u, (G - 1.0) * (U[-1] - 0.5 * rho * (u * u + v * v))]
    edges = np.linspace(0.0, 1.0, n + 1)
    ex = list(er.cell_average((rl, ul, pl), (rr, ur, pr), G, edges, 0.5, t))
    if two_d:
        contact = 0.5 + er.waves((rl, ul, pl), (rr, ur, pr), G)["u"] * t
        num.append(v)
        ex.append(np.where(0.5 * (edges[:-1] + edges[1:]) < contact, vl, vr))
    return num, ex, U0[[0, 1, -1]], U[[0, 1, -1]]


@pytest.mark.parametrize("problem", list(PROBLEMS))
@pytest.mark.parametrize("solver", SOLVERS)
def test_einfeldt_ladder(problem, solver):
    """L1 and L2 of rho, u, p at every N, their orders, positivity, the exact budget; v exact (HLLC)
    or smeared (HLL) on the strip."""
    B, errs = BOUNDS[problem, solver], {}
    (_, ul, vl, pl), (_, ur, vr, pr) = PROBLEMS[problem][1]
    for n in NS:
        num, ex, U0, U = run(problem, n, solver=solver)
        check_tube_budget(f"{problem} {solver} N={n}", U0, U, (1.0, ul, pl, vl), (1.0, ur, pr, vr), T, 1.0 / n, G)
        h = 1.0 / n
        l1 = [h * float(np.abs(a - b).sum()) for a, b in zip(num, ex)]
        l2 = [float(np.sqrt(h * ((a - b) ** 2).sum())) for a, b in zip(num, ex)]
        errs[n] = (l1, l2)
        for k, name in enumerate("rup"):
            check(f"{problem} {solver} N={n} L1 {name}", l1[k], B["L1"][n][k])
            check(f"{problem} {solver} N={n} L2 {name}", l2[k], B["L2"][n][k])
        check(f"{problem} {solver} N={n} min rho", float(num[0].min()), 0.0, lower=True)
        check(f"{problem} {solver} N={n} min p", float(num[2].min()), 0.0, lower=True)
        if problem == "transverse":
            dv = float(np.abs(num[3] - ex[3]).max())
            if solver == "hllc":
                check(f"{problem} {solver} N={n} max |v - v_exact|", dv, 1e-12)
            else:
                check(f"{problem} {solver} N={n} max |v - v_exact|", dv, HLL_SHEAR_SMEAR[0], lower=True)
                check(f"{problem} {solver} N={n} max |v - v_exact|", dv, HLL_SHEAR_SMEAR[1])
    for j, norm in enumerate(("L1", "L2")):
        for k, name in enumerate("rup"):
            order = -np.polyfit(np.log(NS), np.log([errs[n][j][k] for n in NS]), 1)[0]
            lo, hi = B[f"{norm}_order"][k]
            check(f"{problem} {solver} {norm} {name} order", order, lo, lower=True)
            check(f"{problem} {solver} {norm} {name} order", order, hi)


def test_near_vacuum_against_the_literature():
    """Einfeldt-Strong: the LW internal-energy error at N = 100 and Guo's min rho at N = 400, t = 0.1,
    each from the neighbouring power-of-2 runs."""
    def e_err(n):
        num, ex, _, _ = run("strong", n)
        e, e_ex = num[2] / ((G - 1.0) * num[0]), ex[2] / ((G - 1.0) * ex[0])
        return 100.0 * float(np.abs(e - e_ex).sum() / np.abs(e_ex).sum())
    check("Einfeldt-Strong N=100 L1 rel e error (%)", at_n(100, 64, e_err(64), 128, e_err(128)), LW_E_ERROR)
    lo, hi = (float(run("strong", n, t=0.1)[0][0].min()) for n in (256, 512))
    rho_min = at_n(400, 256, lo, 512, hi)
    check("Einfeldt-Strong N=400 t=0.1 min rho", rho_min, GUO_MIN_RHO[0], lower=True)
    check("Einfeldt-Strong N=400 t=0.1 min rho", rho_min, GUO_MIN_RHO[1])


def test_bound_catches_a_short_run():
    """Teeth: Einfeldt-Shear stopped at 0.9 t misses its density L1 bound at 512."""
    num = run("shear", 512, t=0.9 * T)[0]
    ex = run("shear", 512)[1]
    l1 = float(np.abs(num[0] - ex[0]).sum()) / 512
    assert l1 > BOUNDS["shear", "hllc"]["L1"][512][0], f"L1 rho {l1:.4e} of a short run passes the bound"
