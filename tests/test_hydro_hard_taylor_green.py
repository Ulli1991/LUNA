"""Inviscid Taylor-Green vortex at low Mach against a pseudo-spectral incompressible Euler reference.

Measures LUNA's numerical dissipation (KE loss), velocity L2 error and enstrophy deficit in the reference's resolved window."""
import numpy as np

from tests import _tg_reference as tg
from tests._helpers import check, gpu_only

N_SPEC = 128                       # reference; within 6e-5 (rel. L2) of the 256^3 spectral run up to t = 2
TIMES = (1.0, 2.0)
# n = 32, M = 0.1, default dt, CPU-8: (bound ~1.3x measured, measured) per metric.
BOUND = {1.0: dict(ke_loss=(0.21, 0.162), l2=(0.155, 0.1195), z_deficit=(0.23, 0.179)),
         2.0: dict(ke_loss=(0.47, 0.358), l2=(0.37, 0.283), z_deficit=(0.62, 0.477))}
# M = 0.1 over M = 0.05 at n = 32, t = 1: rms(rho - 1) (~M^2: 4) and KE loss (~1/M: 0.5); measured 3.20, 0.536
RATIO_BAND = {"drho": (2.46, 4.16), "ke_loss": (0.41, 0.70)}



def measure(n, mach, times, mesh_shape=None):
    """{t: metrics} of LUNA's TG at n^3 against the spectral reference."""
    ref = tg.reference(N_SPEC)
    U0 = tg.luna_ic(n, mach)
    out = {}
    for t, U in zip(times, tg.luna_run(n, mach, times, mesh_shape=mesh_shape)):
        m = tg.metrics(U, U0, ref, t)
        m["k_eff"] = tg.k_eff(m.pop("ek_ratio"), m.pop("e_ref"))
        m["z_deficit"] = 1.0 - m.pop("z_ratio")
        print(f"MEASURED tg n={n} M={mach} t={t}: " + ", ".join(f"{k}={v:.5g}" for k, v in m.items()))
        out[t] = m
    return out


def assert_bounds(n, res, bound):
    """KE loss, velocity L2 error and enstrophy deficit at most their bounds."""
    for t, m in res.items():
        for k, (b, _) in bound[t].items():
            check(f"tg n={n} t={t} {k}", m[k], b)


def test_tg_dissipation():
    """n = 32, M = 0.1: dissipation, L2 error and enstrophy deficit against the reference."""
    assert_bounds(32, measure(32, 0.1, TIMES), BOUND)


def mach_ratios(n, band):
    """rms(rho - 1) and KE loss at M = 0.1 over M = 0.05: the compressible part ~M^2, the dissipation ~1/M."""
    a, b = (measure(n, m, (1.0,))[1.0] for m in (0.1, 0.05))
    for k, (lo, hi) in band.items():
        check(f"tg n={n} {k} M=0.1/M=0.05", a[k] / b[k], lo, lower=True)
        check(f"tg n={n} {k} M=0.1/M=0.05", a[k] / b[k], hi)


def test_tg_mach_scaling():
    """Density fluctuation falls as M^2, but the numerical KE loss grows as 1/M (HLLC at low Mach)."""
    mach_ratios(32, RATIO_BAND)


GPU_BOUND = {  # one MI300A (256^3 on two); (bound, measured) as above
    64: {1.0: dict(ke_loss=(0.053, 0.0406), l2=(0.044, 0.0338), z_deficit=(0.059, 0.0453)),
         2.0: dict(ke_loss=(0.148, 0.1139), l2=(0.124, 0.0950), z_deficit=(0.22, 0.1696))},
    128: {1.0: dict(ke_loss=(0.0127, 0.00974), l2=(0.012, 0.00921), z_deficit=(0.0139, 0.01068)),
          2.0: dict(ke_loss=(0.0385, 0.0296), l2=(0.0345, 0.0265), z_deficit=(0.0557, 0.0428))},
    256: {1.0: dict(ke_loss=(0.003, 0.00229), l2=(0.0032, 0.00250), z_deficit=(0.0031, 0.00241)),
          2.0: dict(ke_loss=(0.0092, 0.00707), l2=(0.0093, 0.00718), z_deficit=(0.0122, 0.00937))},
}
GPU_ORDER = (1.97, 2.37)           # KE loss at t = 1 over 64..256, measured 2.07 (-0.1, +0.3)
GPU_RATIO_BAND = {"drho": (3.07, 5.2), "ke_loss": (0.39, 0.66)}   # n = 64; measured 3.996, 0.510


@gpu_only
def test_tg_dissipation_gpu():
    """64^3 to 256^3 on a GPU: the bounds and the second-order convergence of the KE loss."""
    loss = []
    for n in sorted(GPU_BOUND):
        ms = (2, 1, 1) if n >= 256 else None             # 256^3 split over 2 GPUs; evolve raises with fewer
        res = measure(n, 0.1, TIMES, ms)
        assert_bounds(n, res, GPU_BOUND[n])
        loss.append(res[1.0]["ke_loss"])
    order = -np.polyfit(np.log(sorted(GPU_BOUND)), np.log(loss), 1)[0]
    check("tg KE loss order", order, GPU_ORDER[0], lower=True)
    check("tg KE loss order", order, GPU_ORDER[1])


@gpu_only
def test_tg_mach_scaling_gpu():
    """The Mach scalings at 64^3, on a GPU."""
    mach_ratios(64, GPU_RATIO_BAND)
