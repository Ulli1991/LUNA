"""Taylor-Green effective Reynolds number (A15): LUNA's dissipation rate eps(t) = -dK/dt to t = 20 matched to
viscous pseudo-spectral Taylor-Green runs at known Re (spectral/taylor_green_viscous/tg_viscous_spectral.npz in luna-reference-solutions, tools/tg_viscous_reference.py:
euler_3d with viscosity, Re = 1600 checked against the HiOCFD C3.5 512^3 reference to 4e-4 of the peak). Gated: the matched
Re_eff and the distance D of eps(t) from the resolved Re = 1600 curve, at each N and against N."""
import functools

import numpy as np
import pytest

from luna.reference_data import reference_path
from tests._helpers import accuracy_cfg, check, evolve, gpu_only
from tests._tg_reference import GAMMA, L, luna_ic

MACH, T_END, DT = 0.1, 20.0, 0.1
WINDOW = (4.0, 14.0)                             # the transition and the dissipation peak (t ~ 9 at Re 1600)
REF = reference_path("spectral/taylor_green_viscous/tg_viscous_spectral.npz")   # tools/tg_viscous_reference.py
SCHEMES = {"minmod": 0, "mc": 1}
NS, NS_GPU = (32,), (64, 128)
NAMES = ("Re_eff", "D")
D_GATED = ("mc",)        # minmod's D is within 1.3x of a first-order run's (0.66 at 32-64^3): printed, not gated
# Re_eff >= measured / 1.3 (rounded down), D <= 1.3 x measured (rounded up); rates d ln Re_eff / d ln N and
# -d ln D / d ln N measured -0.1 .. +0.3. Minmod at 32^3 (Re_eff 26) is indistinguishable from first order (<= 25).
BOUNDS = {
    "minmod": dict(Re_eff={32: 20.0}, D={32: 0.72}),
    # measured Re_eff 26.1 | D 0.548
    "mc": dict(Re_eff={32: 42.0}, D={32: 0.63}),
    # measured Re_eff 55.2 | D 0.480
}
BOUNDS_GPU = {
    "minmod": dict(Re_eff={64: 46.0, 128: 82.0}, D={64: 0.75, 128: 0.61}, Re_eff_rate=(0.71, 1.11)),
    # measured Re_eff 61.0, 107 (rate 0.81) | D 0.572, 0.466
    "mc": dict(Re_eff={64: 140.0, 128: 620.0}, D={64: 0.49, 128: 0.17}, Re_eff_rate=(2.01, 2.41),
               D_rate=(1.50, 1.90)),
    # measured Re_eff 189, 813 (rate 2.11) | D 0.376, 0.124 (rate 1.60)
}


@functools.lru_cache(maxsize=None)
def references():
    """(Re list, t grid over WINDOW, eps(t) per Re) from the stored viscous runs, eps = 2 Z / Re."""
    with np.load(REF) as z:
        res = sorted(int(k[2:-2]) for k in z.files if k.startswith("Re") and k.endswith("_t"))
        tg = np.arange(WINDOW[0], WINDOW[1] + 1e-9, 0.05)
        eps = np.array([np.interp(tg, z[f"Re{r}_t"], 2.0 / r * z[f"Re{r}_Z"]) for r in res])
    return res, tg, eps


def eps_ref(re):
    """Reference eps(t) on the window at any Re in the stored range: linear in log Re, pointwise in t."""
    res, _, eps = references()
    lg = np.log(np.array(res, float))
    x = np.log(re)
    j = int(np.clip(np.searchsorted(lg, x) - 1, 0, len(res) - 2))
    w = (x - lg[j]) / (lg[j + 1] - lg[j])
    return (1 - w) * eps[j] + w * eps[j + 1]


@functools.lru_cache(maxsize=None)
def dissipation(n, scheme, order=1):
    """LUNA eps(t) = -dK/dt on the window (K per unit mass every DT; mass and energy checked by evolve)."""
    cfg = accuracy_cfg(test_problem="Sod-Shock-3D-AMR", SPATIAL_DIMENSION=3, Nx=n, Ny=n, Nz=n, Lx=L, Ly=L, Lz=L,
                       BOUNDARY=0, gamma=GAMMA, LIMITER=SCHEMES[scheme], ORDER=order)
    times = np.arange(1, int(round(T_END / DT)) + 1) * DT
    kin = lambda U: float((0.5 * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / U[0]).mean() / U[0].mean())  # noqa: E731
    U0 = luna_ic(n, MACH)
    K, _ = evolve(cfg, U0, times, observe=kin)
    t, K = np.r_[0.0, times], np.r_[kin(U0), K]
    return np.interp(references()[1], t, -np.gradient(K, t))


def measure(n, scheme, **kw):
    """(Re_eff, D): the Re whose reference eps(t) is closest in rms over the window, and the rms distance of
    LUNA's eps(t) from the Re = 1600 curve, both relative to the reference peak."""
    from scipy.optimize import minimize_scalar
    res = references()[0]
    eps = dissipation(n, scheme, **kw)
    lg = np.log(np.array(res, float))
    r = minimize_scalar(lambda x: np.sqrt(np.mean((eps - eps_ref(np.exp(x))) ** 2)), bounds=(lg[0], lg[-1]),
                        method="bounded")
    hi = eps_ref(1600.0)
    return float(np.exp(r.x)), float(np.sqrt(np.mean((eps - hi) ** 2)) / hi.max())


def gate(scheme, ns, b):
    """Re_eff at least its bound and D at most its bound at every N; with two or more N, the rates."""
    res = [dict(zip(NAMES, measure(n, scheme))) for n in ns]
    print(f"MEASURED tg-Re {scheme} {','.join(map(str, ns))} = {res}")
    names = NAMES if scheme in D_GATED else ("Re_eff",)
    for n, r in zip(ns, res):
        check(f"tg-Re {scheme} N={n} Re_eff", r["Re_eff"], b[scheme]["Re_eff"][n], lower=True)
        if scheme in D_GATED:
            check(f"tg-Re {scheme} N={n} D", r["D"], b[scheme]["D"][n])
    if len(ns) > 1:
        for nm in names:
            sign = 1.0 if nm == "Re_eff" else -1.0                  # Re_eff rises with N, D falls
            rate = sign * np.polyfit(np.log(ns), np.log([r[nm] for r in res]), 1)[0]
            check(f"tg-Re {scheme} {nm} rate", rate, b[scheme][f"{nm}_rate"][0], lower=True)
            check(f"tg-Re {scheme} {nm} rate", rate, b[scheme][f"{nm}_rate"][1])


@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_taylor_green_reynolds(scheme):
    """Re_eff and the distance to the resolved Re = 1600 dissipation at 32^3 (M = 0.1)."""
    gate(scheme, NS, BOUNDS)


def test_bound_catches_a_first_order_run():
    """Teeth: the first-order scheme at 32^3 (Re_eff <= 25) falls below MC's Re_eff bound."""
    assert measure(32, "minmod", order=0)[0] < BOUNDS["mc"]["Re_eff"][32]


@gpu_only
@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_taylor_green_reynolds_gpu(scheme):
    """The same at 64^3 and 128^3 on a GPU, with Re_eff rising and D falling at the measured rates."""
    gate(scheme, NS_GPU, BOUNDS_GPU)


@gpu_only
def test_bound_catches_a_first_order_run_gpu():
    """Teeth: the first-order scheme at 64^3 (Re_eff <= 25) falls below minmod's Re_eff bound (46)."""
    assert measure(64, "minmod", order=0)[0] < BOUNDS_GPU["minmod"]["Re_eff"][64]
