"""Low-Mach Gresho vortex (Gresho & Chan 1990; M = 0.01 as Miczek, Roepke & Edelmann 2015, A&A 576, A50) advected at
1.07 c along (1, 0.5), periodic, 30 x 15 box crossings in a quarter revolution: low Mach, Galilean and grid orientation
together, against the exact state (the IC) and the static run at the same N.

Finding: the moving error is 1.8-2.0x the static one for minmod, 3-4.6x for MC and 2.5-3.5x for van Leer, and the ratio
grows with N (the moving ladder converges slower); bounds pin it (~1.3x measured), they do not call it good."""
import functools

import numpy as np
import pytest

from luna.ic_generation import gresho_vortex
from tests._helpers import accuracy_cfg, check, evolve, gpu_only

GAMMA, MACH = 5.0 / 3.0, 0.01                        # p(0) = rho / (gamma M^2), gresho_vortex.py:23; c = 100.0
T = 0.1 * np.pi                                      # a quarter revolution: peak speed 1 at r = 0.2
CROSS = (30, 15)                                     # box crossings in x, y over T: direction (1, 0.5)
BULK = tuple(k / T for k in CROSS)                   # |u| = 106.8 = 1.07 c
LIMITERS = {"minmod": 0, "mc": 1, "vanleer": 2}
LZ_REDISTRIBUTED = 0.8   # |dL_z| of the box / the core's loss, as tests/test_hydro_accuracy_smooth.py


def _ic(n):
    from luna.ic_generation.ic import initial_conditions
    cfg = accuracy_cfg(test_problem="Gresho-Vortex-2D", SPATIAL_DIMENSION=2, Nx=n, Ny=n, Lx=1.0, Ly=1.0,
                       BOUNDARY=0, gamma=GAMMA, problem=gresho_vortex.Params(Mmax=MACH))
    return np.asarray(initial_conditions(cfg))


def _boost(U, u):
    """The state seen moving at +u = (ux, uy)."""
    U = U.copy()
    U[3] += u[0] * U[1] + u[1] * U[2] + 0.5 * (u[0] ** 2 + u[1] ** 2) * U[0]
    U[1] += u[0] * U[0]
    U[2] += u[1] * U[0]
    return U


def metrics(U0, U, u):
    """(L1 |v - v_exact|, KE loss, core L_z loss, |dL_z| / L_z(0) of the box), comoving, about the vortex centre."""
    n = U0.shape[1]
    x, y = np.meshgrid((np.arange(n) + 0.5) / n - 0.5, (np.arange(n) + 0.5) / n - 0.5, indexing="ij")
    rel = lambda W: (W[1] / W[0] - u[0], W[2] / W[0] - u[1])          # noqa: E731
    (a0, b0), (a, b) = rel(U0), rel(U)
    ke = lambda W, vx, vy: float((0.5 * W[0] * (vx * vx + vy * vy)).sum())   # noqa: E731
    lz = lambda W, vx, vy: W[0] * (x * vy - y * vx)                  # noqa: E731
    L0, dL = lz(U0, a0, b0).sum(), lz(U, a, b) - lz(U0, a0, b0)
    return (float(np.hypot(a - a0, b - b0).mean()), 1.0 - ke(U, a, b) / ke(U0, a0, b0),
            float(-dL[np.hypot(x, y) < 0.4].sum() / L0), float(abs(dL.sum()) / L0))


@functools.lru_cache(maxsize=None)
def run(n, limiter, moving):
    """(U0, U at T, bulk velocity); the exact state at T is U0 (integer box crossings)."""
    u = BULK if moving else (0.0, 0.0)
    U0 = _boost(_ic(n), u)
    cfg = accuracy_cfg(test_problem="Gresho-Vortex-2D", SPATIAL_DIMENSION=2, Nx=n, Ny=n, Lx=1.0, Ly=1.0,
                       BOUNDARY=0, gamma=GAMMA, LIMITER=LIMITERS[limiter])
    U, _ = evolve(cfg, U0, T)
    return U0, U, u


def gresho(n, limiter, moving):
    """metrics() of the static or moving run against the exact state."""
    return metrics(*run(n, limiter, moving))


NAMES = ("v L1", "KE loss", "core L_z loss")


def _check(n, limiter, bound):
    """Static and moving (L1, KE loss, core L_z loss) at N, the L_z remainder, and moving / static L1; the moving L1."""
    res = {mv: gresho(n, limiter, mv) for mv in (False, True)}
    for mv, r in res.items():
        tag = f"gresho M=0.01 {limiter} N={n} {'moving' if mv else 'static'}"
        for name, v, b in zip(NAMES, r, bound[mv]):
            check(f"{tag} {name}", v, b)
        check(f"{tag} |dL_z| / core loss", r[3] / r[2], LZ_REDISTRIBUTED)
    check(f"gresho M=0.01 {limiter} N={n} moving / static v L1", res[True][0] / res[False][0], bound["ratio"])
    return res[False][0], res[True][0]


# CPU: MC at 32^2, {moving: (v L1, KE loss, core L_z loss)} ~1.3x measured (CPU-8 and A100 agree, FP64).
N_CPU = 32
BOUND_CPU = {False: (0.11, 0.34, 0.08), True: (0.22, 0.77, 0.28), "ratio": 2.6}
# static 0.0824, 0.254, 0.0614; moving 0.1625, 0.588, 0.2157; ratio 1.97 (64^2: 0.0189 / 0.0616, ratio 3.27)
TEETH_SHIFT = (6, 3)     # cells: the vortex 0.21 off, a 0.6 % error in the advection speed; measured 0.304


def test_gresho_moving():
    """MC at 32^2: static and moving errors, KE and core L_z loss, the moving / static ratio."""
    _check(N_CPU, "mc", BOUND_CPU)


def test_bound_catches_a_displaced_vortex():
    """Teeth: the moving MC end state against the exact vortex moved by TEETH_SHIFT cells misses its L1 bound."""
    U0, U, u = run(N_CPU, "mc", True)
    check(f"gresho M=0.01 mc N={N_CPU} moving v L1 vs a displaced vortex",
          metrics(np.roll(U0, TEETH_SHIFT, axis=(1, 2)), U, u)[0], BOUND_CPU[True][0], lower=True)


# GPU, one A100 (Raven), FP64, HLLC, CFL 0.8: {limiter: {N: {moving: (v L1, KE loss, core L_z loss)}, "ratio", orders}}
NS_GPU = (128, 256, 512, 1024)
BOUND_GPU = {
    "minmod": {128: {False: (0.064, 0.22, 0.082), True: (0.12, 0.49, 0.18)},
               256: {False: (0.027, 0.062, 0.03), True: (0.047, 0.16, 0.059)},
               512: {False: (0.011, 0.019, 0.012), True: (0.022, 0.041, 0.024)},
               1024: {False: (0.0043, 0.0049, 0.0046), True: (0.0086, 0.011, 0.0098)},
               "ratio": 2.7, "order": {False: (1.20, 1.60), True: (1.13, 1.53)}},
    # static L1 0.04904, 0.02075, 0.008292, 0.003273; KE loss 0.165, 0.0476, 0.0138, 0.00377; L_z 0.0623, 0.0229, 0.0089, 0.00351
    # moving L1 0.08763, 0.03557, 0.01658, 0.006597; KE loss 0.371, 0.120, 0.0311, 0.00837; L_z 0.138, 0.0454, 0.0184, 0.00752
    # ratio 1.79, 1.71, 2.00, 2.02; L1 orders static 1.304, moving 1.230 (bands -0.1 .. +0.3)
    "mc": {128: {False: (0.0101, 0.016, 0.0061), True: (0.031, 0.052, 0.018)},
           256: {False: (0.0034, 0.0032, 0.0022), True: (0.013, 0.01, 0.0073)},
           512: {False: (0.0012, 0.00063, 0.00077), True: (0.0048, 0.0021, 0.0029)},
           1024: {False: (0.0004, 0.00013, 0.00027), True: (0.0019, 0.00045, 0.0011)},
           "ratio": 6.0, "order": {False: (1.46, 1.86), True: (1.27, 1.67)}},
    # static L1 0.007724, 0.002609, 0.0008838, 0.0003024; KE loss 0.0118, 0.00239, 0.000484, 9.9e-5; L_z 0.00468, 0.00167, 0.000586, 0.000207
    # moving L1 0.02373, 0.009637, 0.003647, 0.001393; KE loss 0.0396, 0.00766, 0.00159, 0.000341; L_z 0.0134, 0.00561, 0.00216, 0.000825
    # ratio 3.07, 3.69, 4.13, 4.61; L1 orders static 1.559, moving 1.367
    "vanleer": {128: {False: (0.017, 0.027, 0.016), True: (0.042, 0.092, 0.03)},
                256: {False: (0.0056, 0.0054, 0.0053), True: (0.017, 0.018, 0.012)},
                512: {False: (0.0019, 0.0011, 0.0018), True: (0.0062, 0.0036, 0.0044)},
                1024: {False: (0.00065, 0.00023, 0.00062), True: (0.0023, 0.00074, 0.0017)},
                "ratio": 4.6, "order": {False: (1.46, 1.86), True: (1.31, 1.71)}},
    # static L1 0.01261, 0.004265, 0.001445, 0.0004942; KE loss 0.0205, 0.00412, 0.000833, 0.000171; L_z 0.0122, 0.00404, 0.00137, 0.000475
    # moving L1 0.03194, 0.01304, 0.004714, 0.001732; KE loss 0.0700, 0.0131, 0.0027, 0.000568; L_z 0.023, 0.00903, 0.00337, 0.00128
    # ratio 2.53, 3.06, 3.26, 3.50; L1 orders static 1.558, moving 1.408
}


@gpu_only
@pytest.mark.parametrize("limiter", list(BOUND_GPU))
def test_gresho_moving_gpu(limiter):
    """The 128-1024 ladder per limiter: static and moving bounds at every N and the fitted velocity L1 orders."""
    b = BOUND_GPU[limiter]
    errs = np.array([_check(n, limiter, {**b[n], "ratio": b["ratio"]}) for n in NS_GPU])
    for j, mv in enumerate((False, True)):
        order = -np.polyfit(np.log(NS_GPU), np.log(errs[:, j]), 1)[0]
        name = f"gresho M=0.01 {limiter} {'moving' if mv else 'static'} v L1 order"
        check(name, order, b["order"][mv][0], lower=True)
        check(name, order, b["order"][mv][1])
