"""Shock tubes against the exact Riemann solution: Toro 1-5 in 1D, Sod along x, y, z and diagonals in 3D."""
import numpy as np
import pytest

from luna.hydro import exact_riemann as er
from tests._helpers import accuracy_cfg, check, check_conserved, check_tube_budget, evolve, gpu_only

G = 1.4
NS = (128, 256, 512)
SOLVERS = ("hllc", "hll")
# Bounds ~1.3x measured (CPU, FP64, minmod, default dt); {solver: {case: ...}}, measured values as comments.
L1_BOUND, L2_BOUND = {}, {}   # {solver: {case: {N: (rho, u, p)}}}
L1_BOUND["hllc"] = {
    1: {128: (0.0063, 0.0097, 0.0047), 256: (0.0034, 0.0046, 0.0023), 512: (0.002, 0.002, 0.0011)},
    # measured 0.00483, 0.00749, 0.00365; 0.00265, 0.00351, 0.0018; 0.00154, 0.00151, 0.00085
    2: {128: (0.0091, 0.033, 0.0054), 256: (0.005, 0.024, 0.0027), 512: (0.003, 0.018, 0.0013)},
    # measured 0.00701, 0.0257, 0.00415; 0.00388, 0.0185, 0.00205; 0.00228, 0.014, 0.00101
    3: {128: (0.15, 0.33, 7.4), 256: (0.086, 0.16, 3.6), 512: (0.049, 0.08, 1.8)},
    # measured 0.116, 0.256, 5.66; 0.0664, 0.122, 2.8; 0.0374, 0.0614, 1.35
    4: {128: (0.13, 0.098, 0.7), 256: (0.077, 0.053, 0.35), 512: (0.052, 0.025, 0.18)},
    # measured 0.102, 0.0752, 0.537; 0.059, 0.0411, 0.271; 0.04, 0.0191, 0.138
    5: {128: (0.45, 0.13, 9.5), 256: (0.27, 0.06, 4.5), 512: (0.16, 0.023, 2.4)},
    # measured 0.349, 0.0999, 7.33; 0.211, 0.0465, 3.47; 0.127, 0.0174, 1.88
}
L2_BOUND["hllc"] = {
    1: {128: (0.013, 0.034, 0.01), 256: (0.0096, 0.019, 0.006), 512: (0.0076, 0.008, 0.0031)},
    # measured 0.00991, 0.026, 0.00799; 0.00742, 0.0147, 0.00464; 0.00586, 0.00612, 0.00239
    2: {128: (0.015, 0.045, 0.008), 256: (0.0085, 0.04, 0.0046), 512: (0.0051, 0.036, 0.0026)},
    # measured 0.0112, 0.0343, 0.00614; 0.00656, 0.0304, 0.00353; 0.00396, 0.0277, 0.002
    3: {128: (0.56, 1.2, 25.0), 256: (0.4, 0.81, 16.0), 512: (0.29, 0.68, 10.0)},
    # measured 0.429, 0.922, 19.5; 0.308, 0.622, 12.4; 0.223, 0.526, 7.97
    4: {128: (0.49, 0.36, 2.1), 256: (0.36, 0.34, 1.6), 512: (0.32, 0.18, 1.1)},
    # measured 0.379, 0.279, 1.6; 0.281, 0.265, 1.22; 0.247, 0.141, 0.846
    5: {128: (1.7, 1.1, 61.0), 256: (1.2, 0.65, 37.0), 512: (0.93, 0.32, 21.0)},
    # measured 1.33, 0.81, 46.6; 0.952, 0.5, 28.8; 0.712, 0.244, 16.2
}
L1_BOUND["hll"] = {
    1: {128: (0.0069, 0.01, 0.005), 256: (0.0037, 0.0047, 0.0024), 512: (0.0022, 0.0021, 0.0012)},
    # measured 0.00529, 0.00789, 0.00381; 0.00286, 0.00362, 0.00184; 0.00168, 0.00162, 0.000892
    2: {128: (0.0091, 0.033, 0.0054), 256: (0.005, 0.024, 0.0027), 512: (0.003, 0.018, 0.0013)},
    # measured 0.00703, 0.0257, 0.00415; 0.00387, 0.0185, 0.00205; 0.00227, 0.014, 0.00101
    3: {128: (0.15, 0.38, 8.2), 256: (0.088, 0.18, 4.1), 512: (0.05, 0.091, 2.0)},
    # measured 0.118, 0.289, 6.34; 0.0678, 0.139, 3.13; 0.0383, 0.0697, 1.52
    4: {128: (0.13, 0.11, 0.78), 256: (0.079, 0.06, 0.39), 512: (0.053, 0.028, 0.2)},
    # measured 0.104, 0.0854, 0.603; 0.0605, 0.0463, 0.303; 0.0408, 0.0216, 0.155
    5: {128: (0.5, 0.14, 12.0), 256: (0.3, 0.062, 5.3), 512: (0.18, 0.025, 3.0)},
    # measured 0.384, 0.11, 8.98; 0.228, 0.0477, 4.06; 0.138, 0.0194, 2.34
}
L2_BOUND["hll"] = {
    1: {128: (0.014, 0.036, 0.011), 256: (0.011, 0.019, 0.0063), 512: (0.0082, 0.0089, 0.0034)},
    # measured 0.011, 0.0275, 0.00857; 0.00809, 0.0149, 0.00486; 0.00634, 0.00686, 0.00264
    2: {128: (0.015, 0.044, 0.008), 256: (0.0085, 0.04, 0.0046), 512: (0.0051, 0.036, 0.0026)},
    # measured 0.0112, 0.0342, 0.00614; 0.00656, 0.0304, 0.00353; 0.00395, 0.0278, 0.002
    3: {128: (0.56, 1.2, 26.0), 256: (0.4, 0.81, 16.0), 512: (0.29, 0.69, 10.0)},
    # measured 0.431, 0.935, 19.8; 0.31, 0.627, 12.5; 0.224, 0.529, 8.01
    4: {128: (0.5, 0.37, 2.1), 256: (0.37, 0.35, 1.6), 512: (0.32, 0.18, 1.1)},
    # measured 0.381, 0.283, 1.64; 0.282, 0.266, 1.23; 0.248, 0.141, 0.847
    5: {128: (1.8, 1.1, 73.0), 256: (1.3, 0.67, 46.0), 512: (0.98, 0.36, 27.0)},
    # measured 1.42, 0.834, 56; 1.02, 0.518, 35.4; 0.754, 0.274, 20.7
}
# fitted order of L1(rho) and L2(rho), measured -0.1 .. +0.3
RATE_BOUND = {
    "hllc": {1: ((0.73, 1.13), (0.28, 0.68)), 2: ((0.71, 1.11), (0.65, 1.05)), 3: ((0.71, 1.11), (0.37, 0.77)), 4: ((0.57, 0.97), (0.21, 0.61)), 5: ((0.63, 1.03), (0.35, 0.75))},
    # measured L1 0.825, 0.811, 0.813, 0.674, 0.732; L2 0.379, 0.753, 0.472, 0.310, 0.449
    "hll": {1: ((0.73, 1.13), (0.3, 0.7)), 2: ((0.71, 1.11), (0.65, 1.05)), 3: ((0.71, 1.11), (0.37, 0.77)), 4: ((0.57, 0.97), (0.21, 0.61)), 5: ((0.64, 1.04), (0.36, 0.76))},
    # measured L1 0.828, 0.814, 0.811, 0.674, 0.737; L2 0.396, 0.755, 0.471, 0.310, 0.458
}
# max |front - exact| in cells at N = 512 (no density jump in test 2)
POS_BOUND = {
    "hllc": {1: 0.4, 2: 0.0, 3: 0.92, 4: 0.94, 5: 0.73},
    # measured 0.305, 0, 0.705, 0.722, 0.562
    "hll": {1: 0.71, 2: 0.0, 3: 0.94, 4: 0.94, 5: 0.93},
    # measured 0.545, 0, 0.725, 0.726, 0.717
}
# Budget: round-off, except where a strong fan's smeared head reaches the end cells; there ~5x its
# measured tail (HLLC | HLL), which also pins its decay with N.
BUDGET_TOL = {(2, 128): 1e-9,                                 # measured 1.4e-10 | 1.4e-10
              (3, 128): 1e-4, (3, 256): 2e-6, (3, 512): 3e-9,  # 2.1e-5, 4.4e-7, 5.2e-10 | 2.2e-5, 4.7e-7, 5.6e-10
              (4, 128): 1.5e-6, (4, 256): 5e-10}               # 2.5e-7, 9.2e-11 | 2.7e-7, 1.0e-10


def _cons(rho, u, p):
    """1D conserved state from primitives."""
    return np.stack([rho, rho * u, p / (G - 1.0) + 0.5 * rho * u * u])


def _prim(U):
    """1D primitives (rho, u, p) from conserved."""
    rho, u = U[0], U[1] / U[0]
    return rho, u, (G - 1.0) * (U[2] - 0.5 * rho * u * u)


def run_1d(case, n, solver="hllc"):
    """Numerical and exact (rho, u, p) of Toro test `case` on n cells at its t_end."""
    WL, WR, x0, t = er.TORO[case]
    edges = np.linspace(0.0, 1.0, n + 1)
    U0 = _cons(*er.cell_average(WL, WR, G, edges, x0, 1e-12))
    cfg = accuracy_cfg(test_problem="Sod-Shock", SPATIAL_DIMENSION=1, Nx=n, Lx=1.0,
                       BOUNDARY=1, gamma=G, riemann_solver=solver)
    U, _ = evolve(cfg, U0, t)
    return _prim(U), er.cell_average(WL, WR, G, edges, x0, t), U0, U


def fronts(case):
    """[(x, q_left, q_right, half window)] of the density jumps of Toro test `case`."""
    WL, WR, x0, t = er.TORO[case]
    w = er.waves(WL, WR, G)
    edges = [x0 + s * t for s in (*w["left"], w["u"], *w["right"])]
    out = []
    if w["left"][0] == w["left"][1]:
        out.append((edges[0], WL[0], w["rho_L"]))
    if abs(w["rho_L"] - w["rho_R"]) > 0.05 * max(w["rho_L"], w["rho_R"]):
        out.append((edges[2], w["rho_L"], w["rho_R"]))
    if w["right"][0] == w["right"][1]:
        out.append((edges[3], w["rho_R"], WR[0]))
    return [(x, a, b, 0.5 * min(abs(x - e) for e in edges if abs(x - e) > 1e-12)) for x, a, b in out]


def locate(x, q, xs, a, b, half):
    """Where q crosses (a + b)/2 nearest xs within xs +- half (linear interpolation)."""
    m = 0.5 * (a + b)
    sel = np.nonzero(np.abs(x - xs) <= max(half, 2 * (x[1] - x[0])))[0]
    i = sel[:-1][(q[sel[:-1]] - m) * (q[sel[:-1] + 1] - m) <= 0]
    if i.size == 0:
        return np.inf
    xc = x[i] + (m - q[i]) / (q[i + 1] - q[i]) * (x[i + 1] - x[i])
    return xc[np.argmin(np.abs(xc - xs))]


def _toro(case, solver, ns, l1_bound, l2_bound, rates, pos_bound):
    """L1 (and L2) of rho, u, p at every N, the rho orders, the front positions, the exact budget."""
    WL, WR, x0, t = er.TORO[case]
    errs = []
    for n in ns:
        num, ex, U0, U = run_1d(case, n, solver)
        check_tube_budget(f"toro{case} {solver} N={n}", U0, U, WL, WR, t, 1.0 / n, G,
                          tol=BUDGET_TOL.get((case, n), 1e-12))
        l1 = [float(np.mean(np.abs(a - b))) for a, b in zip(num, ex)]
        l2 = [float(np.sqrt(np.mean((a - b) ** 2))) for a, b in zip(num, ex)]
        errs.append((l1, l2))
        for k, name in enumerate("rup"):
            check(f"toro{case} {solver} N={n} L1 {name}", l1[k], l1_bound[n][k])
            if l2_bound:
                check(f"toro{case} {solver} N={n} L2 {name}", l2[k], l2_bound[n][k])
    for j, (norm, band) in enumerate(zip(("L1", "L2"), rates)):
        if band:
            rate = -np.polyfit(np.log(ns), np.log([e[j][0] for e in errs]), 1)[0]
            check(f"toro{case} {solver} {norm} rho order", rate, band[0], lower=True)
            check(f"toro{case} {solver} {norm} rho order", rate, band[1])
    x = (np.arange(ns[-1]) + 0.5) / ns[-1]
    worst = max([abs(locate(x, num[0], *f) - f[0]) * ns[-1] for f in fronts(case)], default=0.0)
    check(f"toro{case} {solver} N={ns[-1]} front offset (cells)", worst, pos_bound)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("case", list(er.TORO))
def test_toro_1d(case, solver):
    """L1 and L2 of rho, u, p at three resolutions, the orders, the fronts and the exact budget."""
    _toro(case, solver, NS, L1_BOUND[solver][case], L2_BOUND[solver][case], RATE_BOUND[solver][case],
          POS_BOUND[solver][case])


# --- 3D: a periodic double Sod tube, R | L | R along the phase (n . x) / L mod 1 ---
SOD = er.TORO[1][:2]
T3 = 0.08
DIRS = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1), "xy": (1, 1, 0), "xyz": (1, 1, 1)}
# L1 (rho, u_n, p) at n = 128 (xy: 64 x 64); measured x, y, z 0.00877, 0.0186, 0.00812; xy 0.0172, 0.0286, 0.0161
L1_3D_BOUND = {"x": (0.012, 0.024, 0.011), "y": (0.012, 0.024, 0.011), "z": (0.012, 0.024, 0.011),
               "xy": (0.022, 0.035, 0.02)}


def tube_grid(d, n, nt=4):
    """Grid shape for direction d: n cells along every axis the normal uses, nt across."""
    return tuple(n if c else nt for c in DIRS[d])


def tube_exact(d, shape, t, sub=128):
    """Cell-averaged exact (rho, u_n, p) of the double tube at time t (cells are cubes of side 1/n)."""
    used = [k for k in range(3) if DIRS[d][k]]
    m, n = len(used), shape[used[0]]
    w = np.ones(1)                                  # n . x over a cell: m uniforms summed
    for _ in range(m):
        w = np.convolve(w, np.ones(sub))
    off = (np.arange(w.size) - 0.5 * (w.size - 1)) / (sub * n)
    s = (np.arange(m * (n - 1) + 1) + 0.5 * m) / n  # n . x at the cell centres, by index sum
    phase = np.mod(s[:, None] + off[None, :], 1.0)
    rho, u, p = er.double_tube(*SOD, G, phase, t, scale=np.sqrt(m))
    avg = lambda q: (q * w).sum(1) / w.sum()
    M, MU = avg(rho), avg(rho * u)
    E = avg(p / (G - 1.0) + 0.5 * rho * u * u)
    idx = sum(np.arange(shape[k]).reshape([-1 if j == k else 1 for j in range(3)]) for k in used)
    idx = np.broadcast_to(idx, shape)
    return M[idx], (MU / M)[idx], ((G - 1.0) * (E - 0.5 * MU * MU / M))[idx]


def run_tube(d, n, mesh_shape=None, solver="hllc", nt=4):
    """L1 (rho, u_n, p) of the 3D double tube along d at T3; mass, momentum, energy to round-off."""
    shape = tube_grid(d, n, nt)
    nv = np.array(DIRS[d], dtype=float)
    nhat = nv / np.linalg.norm(nv)
    rho, u, p = tube_exact(d, shape, 1e-12)
    v = u[None] * nhat[:, None, None, None]
    U0 = np.stack([rho, *(rho * v), p / (G - 1.0) + 0.5 * rho * u * u])
    cfg = accuracy_cfg(test_problem="Sod-Shock-3D-AMR", SPATIAL_DIMENSION=3, Nx=shape[0],
                       Ny=shape[1], Nz=shape[2], Lx=shape[0] / n, Ly=shape[1] / n,
                       Lz=shape[2] / n, BOUNDARY=0, gamma=G, riemann_solver=solver)
    U, _ = evolve(cfg, U0, T3, mesh_shape)
    check_conserved(f"sod3d {d} {solver} N={n} mesh {mesh_shape}", U0, U)
    un = np.einsum("k...,k->...", U[1:4], nhat) / U[0]
    num = (U[0], un, (G - 1.0) * (U[4] - 0.5 * (U[1:4] ** 2).sum(0) / U[0]))
    return [float(np.mean(np.abs(a - b))) for a, b in zip(num, tube_exact(d, shape, T3))]


SPLIT = {"x": (2, 1, 1), "y": (1, 2, 1), "z": (1, 1, 2), "xy": (2, 1, 1), "xyz": (1, 1, 2)}


def test_sod_3d_directions(dirs=("x", "y", "z", "xy"), n=128, bound=L1_3D_BOUND):
    """3D Sod along x, y, z and a diagonal: L1 bound, axes alike, 1 and 2 devices alike."""
    L1 = {}
    for d in dirs:
        e1 = run_tube(d, n if len(d) == 1 else n // 2)
        e2 = run_tube(d, n if len(d) == 1 else n // 2, SPLIT[d])
        print(f"MEASURED sod3d {d} L1 1 dev {e1} 2 dev {e2}")
        assert np.allclose(e1, e2, rtol=1e-12, atol=0), f"{d}: 1 vs 2 devices {e1} {e2}"
        for name, v, b in zip(("rho", "u_n", "p"), e1, bound[d]):
            check(f"sod3d {d} N={n} L1 {name}", v, b)
        L1[d] = e1
    axes = [L1[d] for d in ("x", "y", "z") if d in L1]
    assert all(np.allclose(a, axes[0], rtol=1e-12, atol=0) for a in axes), f"axes differ {axes}"


HLL_OVER_HLLC = 1.02       # lower bound on L1 rho (HLL) / L1 rho (HLLC), 3D Sod along x; measured 1.047


def test_sod_3d_hll_more_diffusive():
    """HLL, without the contact wave, has the larger density error on Sod."""
    ratio = run_tube("x", 128, solver="hll")[0] / run_tube("x", 128)[0]
    check("sod3d x L1 rho HLL/HLLC", ratio, HLL_OVER_HLLC, lower=True)


NS_GPU = (800, 1600, 3200)
# GPU tier, measured on one MI300A (ROCm, jax_env2), HLLC, L1 only: to re-measure on viper at powers of
# 2 with L2, HLL and the budget.
L1_BOUND_GPU = {
    1: {800: (0.0015, 0.0012, 0.00069), 1600: (0.00094, 0.00078, 0.00037), 3200: (0.00055, 0.00034, 0.00017)},
    # measured 0.00104, 0.000924, 0.00053; 0.000661, 0.000601, 0.000285; 0.000381, 0.00026, 0.000134
    2: {800: (0.0023, 0.016, 0.00084), 1600: (0.0016, 0.0092, 0.00042), 3200: (0.001, 0.0052, 0.00021)},
    # measured 0.00178, 0.0111, 0.000646; 0.00126, 0.00708, 0.000321; 0.000793, 0.00402, 0.000161
    3: {800: (0.041, 0.057, 1.1), 1600: (0.022, 0.026, 0.56), 3200: (0.013, 0.014, 0.3)},
    # measured 0.0284, 0.0436, 0.857; 0.017, 0.0197, 0.427; 0.0102, 0.0107, 0.232
    4: {800: (0.034, 0.016, 0.12), 1600: (0.025, 0.0081, 0.056), 3200: (0.014, 0.005, 0.031)},
    # measured 0.0263, 0.0122, 0.0885; 0.0174, 0.0062, 0.0428; 0.0107, 0.00387, 0.0242
    5: {800: (0.13, 0.023, 2.3), 1600: (0.084, 0.011, 1.3), 3200: (0.053, 0.0056, 0.69)},
    # measured 0.0989, 0.0174, 1.78; 0.0597, 0.00836, 0.994; 0.0381, 0.00433, 0.534
}
# measured orders 0.723, 0.585, 0.736, 0.646, 0.689
RATE_BOUND_GPU = {1: (0.63, 1.03), 2: (0.50, 0.90), 3: (0.64, 1.04), 4: (0.56, 0.96), 5: (0.61, 1.01)}
# measured front offsets at N = 3200: 0.263, -, 0.841, 0.828, 0.770 cells
POS_BOUND_GPU = {1: 0.34, 2: 0.0, 3: 1.1, 4: 1.1, 5: 1.0}
# n = 256 (diagonals 128); measured x, y, z 0.00522, 0.00927, 0.00411; xy 0.00979, 0.0165, 0.00845;
# xyz 0.0106, 0.0128, 0.00936
L1_3D_BOUND_GPU = {"x": (0.0068, 0.012, 0.0053), "y": (0.0068, 0.012, 0.0053),
                   "z": (0.0068, 0.012, 0.0053), "xy": (0.013, 0.02, 0.011), "xyz": (0.014, 0.017, 0.012)}


@gpu_only
@pytest.mark.parametrize("case", list(er.TORO))
def test_toro_1d_gpu(case):
    """Toro 1-5 at 800, 1600, 3200 cells, on a GPU."""
    _toro(case, "hllc", NS_GPU, L1_BOUND_GPU[case], None, (RATE_BOUND_GPU[case], None), POS_BOUND_GPU[case])


@gpu_only
def test_sod_3d_directions_gpu():
    """3D Sod along x, y, z, xy and xyz at 256 (diagonals 128) cells, 1 and 2 GPUs."""
    test_sod_3d_directions(("x", "y", "z", "xy", "xyz"), 256, L1_3D_BOUND_GPU)
