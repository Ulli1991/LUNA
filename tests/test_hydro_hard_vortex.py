"""Isentropic vortex (Yee, Sandham & Djomehri 1999) advected 10 and 100 periods along x and the diagonal, against the exact solution.

Measures L1, L2 and Linf of rho, the peak (density-deficit) loss from minmod's extremum clipping, the phase
error and the one-period orders, for HLLC and HLL."""
import numpy as np
import pytest

from tests._helpers import accuracy_cfg, check, evolve, gpu_only

G, BETA, L = 1.4, 5.0, 10.0
DIRS = {"x": (1.0, 0.0), "diag": (1.0, 1.0)}



def vortex_avg(n, centre, vel, q=4):
    """Cell averages (Gauss-Legendre q x q) of the conserved vortex at `centre` on the periodic [0, L]^2."""
    gx, gw = np.polynomial.legendre.leggauss(q)
    h = L / n
    pts = (np.arange(n)[:, None] + 0.5 + 0.5 * gx[None, :]) * h       # (n, q)
    U = np.zeros((4, n, n))
    for i, wi in enumerate(gw):
        for j, wj in enumerate(gw):
            X, Y = np.meshgrid(pts[:, i], pts[:, j], indexing="ij")
            dx = (X - centre[0] + 0.5 * L) % L - 0.5 * L
            dy = (Y - centre[1] + 0.5 * L) % L - 0.5 * L
            f = np.exp(0.5 * (1.0 - dx * dx - dy * dy))
            T = 1.0 - (G - 1.0) * BETA ** 2 / (8.0 * G * np.pi ** 2) * f * f
            rho = T ** (1.0 / (G - 1.0))
            u = vel[0] - BETA / (2 * np.pi) * f * dy
            v = vel[1] + BETA / (2 * np.pi) * f * dx
            p = rho * T
            W = np.stack([rho, rho * u, rho * v, p / (G - 1.0) + 0.5 * rho * (u * u + v * v)])
            U += 0.25 * wi * wj * W
    return U


def metrics(U, n, centre):
    """(L1, L2, Linf of rho, peak deficit ratio, phase error in cells) against the exact vortex at `centre`."""
    ex = vortex_avg(n, centre, (0.0, 0.0))[0]
    e = U[0] - ex
    l1, l2, linf = float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e * e))), float(np.abs(e).max())
    peak = float((1.0 - U[0].min()) / (1.0 - ex.min()))
    h = L / n
    x = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(x, x, indexing="ij")
    w = np.clip(1.0 - U[0], 0.0, None)
    w = w * ((((X - centre[0] + 0.5 * L) % L - 0.5 * L) ** 2
              + ((Y - centre[1] + 0.5 * L) % L - 0.5 * L) ** 2) < 9.0)
    off = [float(np.sum(w * ((C - c + 0.5 * L) % L - 0.5 * L)) / np.sum(w)) for C, c in ((X, centre[0]), (Y, centre[1]))]
    return l1, l2, linf, peak, float(np.hypot(*off) / h)


def vortex(n, d, periods, order=1, solver="hllc"):
    """[(L1, L2, Linf, peak ratio, phase cells)] after each of `periods` periods of advection along `d`."""
    vel = DIRS[d]
    U0 = vortex_avg(n, (0.5 * L, 0.5 * L), vel)
    cfg = accuracy_cfg(test_problem="Square-Advected", SPATIAL_DIMENSION=2, Nx=n, Ny=n, Lx=L, Ly=L,
                       BOUNDARY=0, gamma=G, ORDER=order, riemann_solver=solver)
    states, _ = evolve(cfg, U0, [L * k for k in periods])
    return [metrics(U, n, (0.5 * L, 0.5 * L)) for U in states]


# {(solver, dir, N): {periods: bounds}}: L1/L2/Linf of rho and phase (cells) ~1.3x measured, peak ratio
# measured / 1.3 (CPU, FP64, minmod, default dt); measured (L1, L2, Linf, peak, phase) as comments.
BOUND = {
    ("hllc", "x", 32): {1: dict(L1=0.0147, L2=0.0371, Linf=0.292, peak=0.431, phase=0.171), 10: dict(L1=0.0323, L2=0.0751, Linf=0.558, peak=0.0835, phase=0.158), 100: dict(L1=0.0381, L2=0.0833, Linf=0.596, peak=0.0355, phase=0.0234)},
    # 1p 0.01132 0.02856 0.2244 0.5607 0.1315; 10p 0.02488 0.05775 0.4292 0.1086 0.1217; 100p 0.02928 0.06411 0.4586 0.04612 0.01801
    ("hllc", "x", 64): {1: dict(L1=0.00507, L2=0.0124, Linf=0.0786, peak=0.698, phase=0.0868), 10: dict(L1=0.0224, L2=0.0555, Linf=0.45, peak=0.24, phase=0.462)},
    # 1p 0.003902 0.009524 0.0605 0.9073 0.06678; 10p 0.01726 0.04266 0.3465 0.3121 0.3553
    ("hllc", "x", 128): {1: dict(L1=0.0013, L2=0.00326, Linf=0.0229, peak=0.76, phase=0.0648), 10: dict(L1=0.0131, L2=0.0294, Linf=0.2, peak=0.549, phase=0.397)},
    # 1p 0.000997 0.002509 0.01761 0.9874 0.04981; 10p 0.01011 0.02261 0.1541 0.7138 0.3055
    ("hllc", "diag", 32): {1: dict(L1=0.018, L2=0.0452, Linf=0.357, peak=0.349, phase=0.256), 10: dict(L1=0.0351, L2=0.0793, Linf=0.579, peak=0.0567, phase=0.183), 100: dict(L1=0.0392, L2=0.0847, Linf=0.602, peak=0.029, phase=0.0882)},
    # 1p 0.01387 0.03477 0.2743 0.4539 0.197; 10p 0.02698 0.06104 0.4455 0.07377 0.1408; 100p 0.03013 0.06518 0.4633 0.03773 0.06787
    ("hllc", "diag", 64): {1: dict(L1=0.00667, L2=0.0163, Linf=0.105, peak=0.669, phase=0.14), 10: dict(L1=0.0259, L2=0.063, Linf=0.504, peak=0.176, phase=0.62)},
    # 1p 0.005128 0.01251 0.08093 0.8701 0.1073; 10p 0.01995 0.04849 0.3877 0.2285 0.4772
    ("hllc", "diag", 128): {1: dict(L1=0.00175, L2=0.00452, Linf=0.0334, peak=0.77, phase=0.105), 10: dict(L1=0.0156, L2=0.0356, Linf=0.258, peak=0.478, phase=0.968)},
    # 1p 0.001344 0.00348 0.02568 1.001 0.08071; 10p 0.01201 0.02741 0.1986 0.6214 0.7444
    ("hll", "x", 32): {1: dict(L1=0.0199, L2=0.0498, Linf=0.387, peak=0.3, phase=0.125), 10: dict(L1=0.0363, L2=0.0811, Linf=0.587, peak=0.0468, phase=0.0912), 100: dict(L1=0.0392, L2=0.0847, Linf=0.602, peak=0.0288, phase=0.0919)},
    # 1p 0.01531 0.03828 0.2974 0.3901 0.0958; 10p 0.02792 0.06237 0.4515 0.06088 0.07012; 100p 0.03013 0.06517 0.4633 0.03749 0.07068
    ("hll", "x", 64): {1: dict(L1=0.00784, L2=0.019, Linf=0.118, peak=0.647, phase=0.0637), 10: dict(L1=0.0277, L2=0.0669, Linf=0.529, peak=0.144, phase=0.198)},
    # 1p 0.00603 0.01459 0.09073 0.8417 0.04901; 10p 0.02134 0.05143 0.4067 0.1871 0.1522
    ("hll", "x", 128): {1: dict(L1=0.00208, L2=0.00538, Linf=0.0372, peak=0.781, phase=0.0618), 10: dict(L1=0.0169, L2=0.0398, Linf=0.294, peak=0.427, phase=0.213)},
    # 1p 0.001603 0.004138 0.0286 1.015 0.04754; 10p 0.01298 0.0306 0.2261 0.5552 0.1635
    ("hll", "diag", 32): {1: dict(L1=0.0198, L2=0.0495, Linf=0.387, peak=0.304, phase=0.169), 10: dict(L1=0.0361, L2=0.0808, Linf=0.586, peak=0.0486, phase=0.193), 100: dict(L1=0.0392, L2=0.0848, Linf=0.603, peak=0.0294, phase=0.124)},
    # 1p 0.01519 0.03804 0.2976 0.3953 0.13; 10p 0.02774 0.06214 0.4504 0.06314 0.1481; 100p 0.03017 0.06522 0.4635 0.03818 0.09531
    ("hll", "diag", 64): {1: dict(L1=0.00778, L2=0.0189, Linf=0.122, peak=0.642, phase=0.117), 10: dict(L1=0.0274, L2=0.0662, Linf=0.525, peak=0.149, phase=0.514)},
    # 1p 0.005985 0.01455 0.09364 0.835 0.08987; 10p 0.02111 0.05095 0.404 0.1943 0.3953
    ("hll", "diag", 128): {1: dict(L1=0.00207, L2=0.00534, Linf=0.0389, peak=0.775, phase=0.12), 10: dict(L1=0.0157, L2=0.0385, Linf=0.291, peak=0.436, phase=0.842)},
    # 1p 0.001595 0.004104 0.02996 1.008 0.09244; 10p 0.01211 0.0296 0.2241 0.5666 0.6478
}
# one-period orders 64 -> 128 of (L1, L2, Linf), measured -0.1 .. +0.3
ORDER = {("hllc", "x"): ((1.87, 2.27), (1.82, 2.22), (1.68, 2.08)), ("hllc", "diag"): ((1.83, 2.23), (1.75, 2.15), (1.56, 1.96)), ("hll", "x"): ((1.81, 2.21), (1.72, 2.12), (1.57, 1.97)), ("hll", "diag"): ((1.81, 2.21), (1.73, 2.13), (1.54, 1.94))}
# measured hllc x 1.969 1.925 1.780; hllc diag 1.932 1.846 1.656; hll x 1.912 1.818 1.666; hll diag 1.908 1.826 1.644


NORMS = ("L1", "L2", "Linf")


def _vortex(solver, d, bound, order):
    """Norms, peak loss and phase after each period count, and the one-period orders of the top two N."""
    one = {}
    for (sv, dd, n), per in sorted(bound.items()):
        if (sv, dd) != (solver, d):
            continue
        res = vortex(n, d, tuple(per), solver=solver)
        for k, m in zip(per, res):
            b = per[k]
            for j, nm in enumerate(NORMS):
                if nm in b:
                    check(f"vortex {solver} {d} N={n} {k}p {nm} rho", m[j], b[nm])
            check(f"vortex {solver} {d} N={n} {k}p peak ratio", m[3], b["peak"], lower=True)
            check(f"vortex {solver} {d} N={n} {k}p phase (cells)", m[4], b["phase"])
        one[n] = res[0]
    lo, hi = sorted(one)[-2:]
    for j, band in enumerate(order[solver, d]):
        rate = float(np.log2(one[lo][j] / one[hi][j]))
        check(f"vortex {solver} {d} one-period {NORMS[j]} order {lo}->{hi}", rate, band[0], lower=True)
        check(f"vortex {solver} {d} one-period {NORMS[j]} order {lo}->{hi}", rate, band[1])


@pytest.mark.parametrize("solver", ("hllc", "hll"))
@pytest.mark.parametrize("d", list(DIRS))
def test_vortex(d, solver):
    """L1, L2, Linf, peak loss and phase after 1, 10 (and 100 at N=32) periods, and the one-period orders."""
    _vortex(solver, d, BOUND, ORDER)


# GPU tier, one MI300A; N = 256 measured (L1, peak, phase): x 1p (0.000246, 0.9967, 0.0264),
# 10p (0.00408, 0.976, 0.364), 100p (0.0181, 0.382, 1.45); diag 1p (0.000327, 0.9997, 0.0458),
# 10p (0.00397, 0.977, 0.577), 100p (0.0206, 0.276, 3.10)
_g = lambda t: dict(L1=t[0], peak=t[1], phase=t[2])                                      # noqa: E731
BOUND_GPU = {   # HLLC, L1 only until re-measured on viper (L2, Linf, HLL)
    ("hllc", "x", 128): {1: _g((0.0013, 0.760, 0.065)), 10: _g((0.0131, 0.549, 0.40))},
    ("hllc", "x", 256): {1: _g((0.00032, 0.767, 0.034)), 10: _g((0.0053, 0.751, 0.47)), 100: _g((0.0235, 0.294, 1.89))},
    ("hllc", "diag", 128): {1: _g((0.00175, 0.770, 0.105)), 10: _g((0.0156, 0.478, 0.97))},
    ("hllc", "diag", 256): {1: _g((0.00043, 0.769, 0.060)), 10: _g((0.0052, 0.752, 0.75)), 100: _g((0.0268, 0.212, 4.0))},
}
ORDER_GPU = {("hllc", "x"): ((1.92, 9.0),), ("hllc", "diag"): ((1.94, 9.0),)}   # L1 one-period 128 -> 256, measured 2.02, 2.04


@gpu_only
@pytest.mark.parametrize("d", list(DIRS))
def test_vortex_gpu(d):
    """The same at 128^2 and 256^2 up to 100 periods, on a GPU."""
    _vortex("hllc", d, BOUND_GPU, ORDER_GPU)
