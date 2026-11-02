"""Low-Mach Gresho sweep (as Miczek, Roepke & Edelmann 2015): KE loss and velocity L1, L2, Linf after one
revolution (t = 0.4 pi) vs peak Mach and N.

Plain upwind fluxes dissipate like 1/M at low Mach; this pins how LUNA's minmod HLLC error scales as M -> 0."""
import functools

import numpy as np
import pytest

from luna.ic_generation import gresho_vortex
from tests._helpers import accuracy_cfg, check, evolve, gpu_only

GAMMA = 5.0 / 3.0
MACHS = (0.1, 0.03, 0.01, 0.003, 0.001)
REV = 0.4 * np.pi                      # one revolution: peak speed 1 at r = 0.2
SCHEMES = {"hllc-minmod": ("hllc", 0), "hll-minmod": ("hll", 0), "hllc-mc": ("hllc", 1)}



@functools.lru_cache(maxsize=None)
def gresho(n, mach, scheme="hllc-minmod", t=REV):
    """(1 - KE(t)/KE(0), L1, L2, Linf of |v - v_0|) of the registered (balanced) Gresho vortex."""
    from luna.ic_generation.ic import initial_conditions
    solver, limiter = SCHEMES[scheme]
    cfg = accuracy_cfg(test_problem="Gresho-Vortex-2D", SPATIAL_DIMENSION=2, Nx=n, Ny=n, Lx=1.0, Ly=1.0, BOUNDARY=0,
                       gamma=GAMMA, problem=gresho_vortex.Params(Mmax=mach), riemann_solver=solver, LIMITER=limiter)
    U0 = np.asarray(initial_conditions(cfg))
    U, _ = evolve(cfg, U0, t)
    ke = lambda W: float(np.sum(0.5 * (W[1] ** 2 + W[2] ** 2) / W[0]))  # noqa: E731
    e = np.hypot(*(U[1:3] / U[0] - U0[1:3] / U0[0]))
    return 1.0 - ke(U) / ke(U0), float(e.mean()), float(np.sqrt((e * e).mean())), float(e.max())


def exponent(machs, vals):
    """Fitted slope of log(vals) against log(machs)."""
    return float(np.polyfit(np.log(machs), np.log(vals), 1)[0])


# {(scheme, N): [(KE loss, v L1, v L2, v Linf) per Mach]} ~1.3x measured (KE loss capped at 1: HLL loses
# the whole vortex within one revolution at M <= 0.01, as an upwind scheme's c/M dissipation predicts).
BOUND = {
    ("hllc-minmod", 32): [(0.65, 0.2, 0.24, 0.63), (0.74, 0.31, 0.36, 0.86), (0.7, 0.33, 0.37, 0.88), (0.68, 0.32, 0.37, 0.86), (0.67, 0.32, 0.37, 0.86)],
    # 0.502 0.152 0.184 0.483; 0.57 0.238 0.276 0.663; 0.542 0.251 0.287 0.675; 0.523 0.249 0.286 0.663; 0.516 0.249 0.286 0.659
    ("hllc-minmod", 64): [(0.32, 0.084, 0.11, 0.37), (0.6, 0.18, 0.22, 0.6), (0.72, 0.3, 0.34, 0.83), (0.69, 0.33, 0.37, 0.87), (0.66, 0.32, 0.37, 0.84)],
    # 0.244 0.0645 0.0811 0.281; 0.46 0.139 0.169 0.465; 0.553 0.227 0.263 0.64; 0.527 0.251 0.286 0.67; 0.505 0.249 0.285 0.649
    ("hll-minmod", 32): [(1.0, 0.28, 0.42, 1.1), (1.0, 0.32, 0.52, 1.3), (1.0, 0.33, 0.53, 1.3), (1.0, 0.33, 0.53, 1.3), (1.0, 0.33, 0.53, 1.3)],
    # 0.932 0.214 0.325 0.836; 1 0.249 0.404 0.984; 1 0.251 0.409 0.994; 1 0.251 0.409 0.994; 1 0.251 0.409 0.994
    ("hll-minmod", 64): [(0.8, 0.18, 0.26, 0.74), (1.0, 0.27, 0.41, 1.1), (1.0, 0.32, 0.51, 1.3), (1.0, 0.33, 0.53, 1.3), (1.0, 0.33, 0.53, 1.3)],
    # 0.613 0.14 0.199 0.573; 0.909 0.208 0.312 0.813; 0.998 0.246 0.396 0.973; 1 0.251 0.409 0.996; 1 0.251 0.409 0.996
    ("hllc-mc", 32): [(0.21, 0.053, 0.079, 0.3), (0.42, 0.11, 0.14, 0.45), (0.65, 0.2, 0.25, 0.66), (0.71, 0.31, 0.36, 0.85), (0.67, 0.32, 0.37, 0.85)],
    # 0.165 0.0411 0.0604 0.228; 0.325 0.0827 0.111 0.347; 0.501 0.153 0.191 0.51; 0.548 0.238 0.274 0.655; 0.515 0.25 0.285 0.656
    ("hllc-mc", 64): [(0.042, 0.019, 0.029, 0.15), (0.1, 0.029, 0.044, 0.21), (0.22, 0.058, 0.082, 0.31), (0.45, 0.12, 0.16, 0.5), (0.67, 0.22, 0.27, 0.72)],
    # 0.032 0.0149 0.0224 0.115; 0.0779 0.022 0.0339 0.16; 0.169 0.0444 0.0628 0.238; 0.349 0.0943 0.123 0.382; 0.517 0.167 0.205 0.551
}
# KE-loss exponent in M over 0.1 .. 0.01 at N = 64, measured -0.1 .. +0.3: hllc-minmod -0.357, hll-minmod -0.214, hllc-mc -0.723
SLOPE = {"hllc-minmod": (-0.46, -0.06), "hll-minmod": (-0.31, 0.09), "hllc-mc": (-0.82, -0.42)}


# Where 1.3x measured KE loss reaches 1 (HLL: the vortex is all but gone) an upper bound cannot fail; there the bound
# is the lower one, measured / 1.3: the loss stays near total, as the upwind c/M dissipation predicts (re-measured
# CPU-8 FP64, job 31654894: 0.932, 0.9997, 1, 1, 1 at 32; 0.909, 0.998, 1, 1 at 64)
KE_SATURATED = {("hll-minmod", 32): (0.932, 1.0, 1.0, 1.0, 1.0), ("hll-minmod", 64): (None, 0.909, 0.998, 1.0, 1.0)}


@pytest.mark.parametrize("scheme,n", list(BOUND), ids=[f"{sc}-{n}" for sc, n in BOUND])
def test_gresho_low_mach(scheme, n):
    """Regression only: KE loss and v L1, L2, Linf after one revolution, M = 0.1 .. 0.001; at N = 64 the M-exponent."""
    res = [gresho(n, m, scheme) for m in MACHS]
    for i, (m, r, b) in enumerate(zip(MACHS, res, BOUND[scheme, n])):
        for name, v, bound in zip(("KE loss", "v L1", "v L2", "v Linf"), r, b):
            if name == "KE loss" and bound >= 1.0:
                check(f"gresho {scheme} N={n} M={m} {name} (saturated)", v, KE_SATURATED[scheme, n][i] / 1.3,
                      lower=True)
            else:
                check(f"gresho {scheme} N={n} M={m} {name}", v, bound)
    if n == 64:
        e = exponent(MACHS[:3], [r[0] for r in res[:3]])
        check(f"gresho {scheme} N=64 KE-loss exponent in M", e, SLOPE[scheme][0], lower=True)
        check(f"gresho {scheme} N=64 KE-loss exponent in M", e, SLOPE[scheme][1])


# GPU tier, one MI300A, HLLC + minmod at t = 1 (0.8 revolution) until re-measured on viper.
BOUND_GPU = {
    128: [(0.079, 0.030), (0.226, 0.063), (0.47, 0.138), (0.67, 0.27), (0.68, 0.33)],
    # measured (0.0602, 0.0229), (0.174, 0.0487), (0.364, 0.106), (0.517, 0.206), (0.520, 0.249)
    256: [(0.0228, 0.0125), (0.066, 0.0275), (0.177, 0.055), (0.43, 0.122), (0.64, 0.24)],
    # measured (0.0175, 0.00955), (0.0505, 0.0211), (0.136, 0.0420), (0.331, 0.0936), (0.496, 0.184)
}
SLOPE_FLOOR_GPU = {128: -0.88, 256: -0.99}   # measured -0.783, -0.890


def sweep(n, bound, floor=None):
    """The GPU sweep at t = 1: (KE loss, v L1) per Mach, and (if `floor`) the KE-loss exponent over 0.1 .. 0.01."""
    res = [gresho(n, m, t=1.0) for m in MACHS]
    for m, r, (bk, bl) in zip(MACHS, res, bound):
        check(f"gresho N={n} M={m} KE loss", r[0], bk)
        check(f"gresho N={n} M={m} L1 v", r[1], bl)
    if floor is not None:
        check(f"gresho N={n} KE-loss exponent in M (0.1..0.01)", exponent(MACHS[:3], [r[0] for r in res[:3]]),
              floor, lower=True)


@gpu_only
@pytest.mark.parametrize("n", list(BOUND_GPU))
def test_gresho_low_mach_gpu(n):
    """The sweep at 128^2 and 256^2, on a GPU."""
    sweep(n, BOUND_GPU[n], SLOPE_FLOOR_GPU[n])
