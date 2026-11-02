"""Gresho vortex at Liska & Wendroff's p(0) = 5 (peak Mach ~0.38): static and moving against the exact
state, and their Tables 4.5-4.6 (sec. 4.10). Assumed (not stated by LW): gamma 1.4, moving centre (0.5, 0.5)."""
import functools

import numpy as np
import pytest

from luna.ic_generation import gresho_vortex
from tests._helpers import accuracy_cfg, at_n, check, evolve

G = 1.4
MMAX = 1.0 / np.sqrt(G * 5.0)                        # LUNA's IC sets p(0) = rho / (gamma Mmax^2) = 5
REV = 0.4 * np.pi                                    # one revolution at peak speed 1, r = 0.2
SCHEMES = {"hllc-minmod": ("hllc", 0), "hll-minmod": ("hll", 0), "hllc-mc": ("hllc", 1)}
NS, NORMS, FIELDS = (32, 64, 128, 256), ("L1", "L2", "Linf"), ("v", "rho", "p")
# {(scheme, moving): {field: norms per N, orders}} ~1.3x measured, orders measured -0.1 .. +0.3.
# Velocity Linf ~0.8: the kinks of the profile at r = 0.2, 0.4.
BOUNDS = {
    ("hllc-minmod", False): {
        "v": dict(L1={32: 0.089, 64: 0.037, 128: 0.016, 256: 0.0063}, L1_order=(1.17, 1.57),
            L2={32: 0.12, 64: 0.049, 128: 0.024, 256: 0.011}, L2_order=(1.01, 1.41),
            Linf={32: 0.37, 64: 0.21, 128: 0.12, 256: 0.074}, Linf_order=(0.68, 1.08)),
        # L1 0.0688, 0.0283, 0.0121, 0.00484; 1.27 | L2 0.0893, 0.038, 0.0183, 0.00884; 1.11 | Linf 0.285, 0.16, 0.0896, 0.0565; 0.78
        "rho": dict(L1={32: 0.0048, 64: 0.0018, 128: 0.00056, 256: 0.00019}, L1_order=(1.47, 1.87),
            L2={32: 0.0072, 64: 0.0026, 128: 0.00086, 256: 0.00031}, L2_order=(1.42, 1.82),
            Linf={32: 0.024, 64: 0.0078, 128: 0.0026, 256: 0.0013}, Linf_order=(1.33, 1.73)),
        # L1 0.00369, 0.00136, 0.000434, 0.000145; 1.57 | L2 0.00551, 0.00199, 0.000663, 0.000239; 1.52 | Linf 0.0184, 0.006, 0.00199, 0.000978; 1.43
        "p": dict(L1={32: 0.044, 64: 0.015, 128: 0.0045, 256: 0.0015}, L1_order=(1.55, 1.95),
            L2={32: 0.075, 64: 0.025, 128: 0.0076, 256: 0.0024}, L2_order=(1.57, 1.97),
            Linf={32: 0.24, 64: 0.074, 128: 0.022, 256: 0.0066}, Linf_order=(1.62, 2.02)),
        # L1 0.0335, 0.0118, 0.00345, 0.00112; 1.65 | L2 0.0579, 0.0192, 0.00584, 0.00181; 1.67 | Linf 0.181, 0.057, 0.0173, 0.00505; 1.72
    },
    ("hllc-minmod", True): {
        "v": dict(L1={32: 0.11, 64: 0.038, 128: 0.018, 256: 0.0072}, L1_order=(1.17, 1.57),
            L2={32: 0.14, 64: 0.055, 128: 0.028, 256: 0.013}, L2_order=(1.02, 1.42),
            Linf={32: 0.47, 64: 0.25, 128: 0.15, 256: 0.089}, Linf_order=(0.69, 1.09)),
        # L1 0.0819, 0.0293, 0.0142, 0.00558; 1.27 | L2 0.108, 0.042, 0.0213, 0.0102; 1.12 | Linf 0.358, 0.193, 0.114, 0.0685; 0.79
        "rho": dict(L1={32: 0.0051, 64: 0.0017, 128: 0.00059, 256: 0.0002}, L1_order=(1.46, 1.86),
            L2={32: 0.0072, 64: 0.0026, 128: 0.00092, 256: 0.00034}, L2_order=(1.37, 1.77),
            Linf={32: 0.024, 64: 0.0094, 128: 0.0041, 256: 0.0016}, Linf_order=(1.20, 1.60)),
        # L1 0.00392, 0.00134, 0.000457, 0.000152; 1.56 | L2 0.00557, 0.00201, 0.000709, 0.000263; 1.47 | Linf 0.0185, 0.00727, 0.00317, 0.00122; 1.30
        "p": dict(L1={32: 0.05, 64: 0.016, 128: 0.005, 256: 0.0015}, L1_order=(1.58, 1.98),
            L2={32: 0.085, 64: 0.028, 128: 0.0085, 256: 0.0026}, L2_order=(1.58, 1.98),
            Linf={32: 0.28, 64: 0.095, 128: 0.033, 256: 0.0094}, Linf_order=(1.53, 1.93)),
        # L1 0.0382, 0.0122, 0.00385, 0.00116; 1.68 | L2 0.0653, 0.0214, 0.00652, 0.00198; 1.68 | Linf 0.218, 0.0729, 0.0252, 0.00725; 1.63
    },
    ("hll-minmod", False): {
        "v": dict(L1={32: 0.19, 64: 0.082, 128: 0.038, 256: 0.017}, L1_order=(1.05, 1.45),
            L2={32: 0.26, 64: 0.12, 128: 0.056, 256: 0.028}, L2_order=(0.98, 1.38),
            Linf={32: 0.74, 64: 0.42, 128: 0.25, 256: 0.15}, Linf_order=(0.66, 1.06)),
        # L1 0.144, 0.0635, 0.0289, 0.0131; 1.15 | L2 0.203, 0.0941, 0.0431, 0.0217; 1.08 | Linf 0.571, 0.32, 0.19, 0.118; 0.76
        "rho": dict(L1={32: 0.0052, 64: 0.0028, 128: 0.0012, 256: 0.00039}, L1_order=(1.14, 1.54),
            L2={32: 0.0066, 64: 0.0039, 128: 0.002, 256: 0.00063}, L2_order=(1.01, 1.41),
            Linf={32: 0.017, 64: 0.015, 128: 0.01, 256: 0.0035}, Linf_order=(0.65, 1.05)),
        # L1 0.00402, 0.00212, 0.000952, 0.000296; 1.24 | L2 0.00506, 0.00302, 0.0015, 0.000487; 1.11 | Linf 0.0132, 0.0117, 0.00768, 0.0027; 0.75
        "p": dict(L1={32: 0.095, 64: 0.039, 128: 0.012, 256: 0.0035}, L1_order=(1.50, 1.90),
            L2={32: 0.17, 64: 0.068, 128: 0.021, 256: 0.0065}, L2_order=(1.47, 1.87),
            Linf={32: 0.59, 64: 0.23, 128: 0.082, 256: 0.029}, Linf_order=(1.34, 1.74)),
        # L1 0.0727, 0.0301, 0.00904, 0.00268; 1.60 | L2 0.127, 0.052, 0.0165, 0.00497; 1.57 | Linf 0.452, 0.175, 0.0629, 0.0226; 1.44
    },
    ("hll-minmod", True): {
        "v": dict(L1={32: 0.19, 64: 0.082, 128: 0.037, 256: 0.017}, L1_order=(1.05, 1.45),
            L2={32: 0.26, 64: 0.12, 128: 0.056, 256: 0.028}, L2_order=(0.98, 1.38),
            Linf={32: 0.74, 64: 0.42, 128: 0.25, 256: 0.15}, Linf_order=(0.66, 1.06)),
        # L1 0.143, 0.063, 0.0287, 0.013; 1.15 | L2 0.201, 0.0935, 0.0428, 0.0215; 1.08 | Linf 0.569, 0.32, 0.189, 0.117; 0.76
        "rho": dict(L1={32: 0.0052, 64: 0.0027, 128: 0.0012, 256: 0.00038}, L1_order=(1.15, 1.55),
            L2={32: 0.0066, 64: 0.0039, 128: 0.0019, 256: 0.00063}, L2_order=(1.02, 1.42),
            Linf={32: 0.018, 64: 0.016, 128: 0.0099, 256: 0.0035}, Linf_order=(0.68, 1.08)),
        # L1 0.00402, 0.00211, 0.000938, 0.000294; 1.25 | L2 0.00508, 0.00301, 0.00148, 0.000482; 1.12 | Linf 0.0138, 0.0119, 0.0076, 0.00268; 0.78
        "p": dict(L1={32: 0.094, 64: 0.039, 128: 0.012, 256: 0.0034}, L1_order=(1.51, 1.91),
            L2={32: 0.16, 64: 0.067, 128: 0.021, 256: 0.0064}, L2_order=(1.47, 1.87),
            Linf={32: 0.59, 64: 0.23, 128: 0.081, 256: 0.029}, Linf_order=(1.35, 1.75)),
        # L1 0.0724, 0.0298, 0.00893, 0.00265; 1.61 | L2 0.126, 0.0515, 0.0164, 0.00492; 1.57 | Linf 0.452, 0.175, 0.0627, 0.0226; 1.45
    },
    ("hllc-mc", False): {
        "v": dict(L1={32: 0.028, 64: 0.013, 128: 0.0045, 256: 0.0015}, L1_order=(1.33, 1.73),
            L2={32: 0.046, 64: 0.021, 128: 0.0091, 256: 0.004}, L2_order=(1.08, 1.48),
            Linf={32: 0.21, 64: 0.12, 128: 0.069, 256: 0.041}, Linf_order=(0.68, 1.08)),
        # L1 0.0216, 0.0101, 0.00346, 0.00114; 1.43 | L2 0.0355, 0.0161, 0.00698, 0.0031; 1.18 | Linf 0.158, 0.0912, 0.0527, 0.0313; 0.78
        "rho": dict(L1={32: 0.0013, 64: 0.00032, 128: 7.7e-05, 256: 2.5e-05}, L1_order=(1.82, 2.22),
            L2={32: 0.0018, 64: 0.00043, 128: 0.00011, 256: 4.2e-05}, L2_order=(1.74, 2.14),
            Linf={32: 0.0062, 64: 0.0014, 128: 0.00041, 256: 0.00029}, Linf_order=(1.40, 1.80)),
        # L1 0.00103, 0.000246, 5.92e-05, 1.95e-05; 1.92 | L2 0.00142, 0.000328, 8.2e-05, 3.23e-05; 1.84 | Linf 0.00477, 0.00104, 0.000318, 0.00022; 1.50
        "p": dict(L1={32: 0.012, 64: 0.0033, 128: 0.00074, 256: 0.00019}, L1_order=(1.92, 2.32),
            L2={32: 0.022, 64: 0.0051, 128: 0.0012, 256: 0.00033}, L2_order=(1.92, 2.32),
            Linf={32: 0.077, 64: 0.02, 128: 0.0061, 256: 0.0019}, Linf_order=(1.68, 2.08)),
        # L1 0.00951, 0.00251, 0.000567, 0.000148; 2.02 | L2 0.0169, 0.00395, 0.000958, 0.000256; 2.02 | Linf 0.0589, 0.0156, 0.00472, 0.00143; 1.78
    },
    ("hllc-mc", True): {
        "v": dict(L1={32: 0.038, 64: 0.017, 128: 0.0058, 256: 0.0019}, L1_order=(1.35, 1.75),
            L2={32: 0.058, 64: 0.026, 128: 0.011, 256: 0.005}, L2_order=(1.09, 1.49),
            Linf={32: 0.26, 64: 0.15, 128: 0.083, 256: 0.049}, Linf_order=(0.70, 1.10)),
        # L1 0.0291, 0.0131, 0.00443, 0.00147; 1.45 | L2 0.0444, 0.0204, 0.00856, 0.00382; 1.19 | Linf 0.197, 0.112, 0.0638, 0.0374; 0.80
        "rho": dict(L1={32: 0.0018, 64: 0.00048, 128: 0.00013, 256: 3e-05}, L1_order=(1.87, 2.27),
            L2={32: 0.0024, 64: 0.00066, 128: 0.00019, 256: 5.4e-05}, L2_order=(1.72, 2.12),
            Linf={32: 0.0081, 64: 0.0028, 128: 0.00091, 256: 0.00037}, Linf_order=(1.39, 1.79)),
        # L1 0.00137, 0.00037, 9.62e-05, 2.29e-05; 1.97 | L2 0.00185, 0.000508, 0.000144, 4.17e-05; 1.82 | Linf 0.0062, 0.00215, 0.000703, 0.000288; 1.49
        "p": dict(L1={32: 0.015, 64: 0.0043, 128: 0.001, 256: 0.0002}, L1_order=(1.98, 2.38),
            L2={32: 0.027, 64: 0.0066, 128: 0.0016, 256: 0.00041}, L2_order=(1.91, 2.31),
            Linf={32: 0.096, 64: 0.035, 128: 0.0092, 256: 0.0029}, Linf_order=(1.60, 2.00)),
        # L1 0.0117, 0.00331, 0.000773, 0.000157; 2.08 | L2 0.0205, 0.00511, 0.00126, 0.000315; 2.01 | Linf 0.0739, 0.0266, 0.00706, 0.00224; 1.70
    },
}
# moving / static velocity L1, max over N, x1.3; measured hllc-minmod 1.19, hll-minmod 0.99, hllc-mc 1.35
GALILEAN = {"hllc-minmod": 1.5, "hll-minmod": 1.3, "hllc-mc": 1.7}
# LW Tables 4.5 (40^2) and 4.6 (160x40): worst of 8 schemes for (rel. L1 vorticity, L1 density, rel. TKE) %
# (static JT/LL/LL; moving CFLFh/WAFT/LL). CLAW, LW's MUSCL-type scheme: static (28, 0.10, 6.1), moving (37, 0.29, 8.3).
LW_WORST = {False: (45.0, 0.23, 26.1), True: (83.0, 0.77, 30.9)}


def _ic(nx, ny, lx, boundary):
    from luna.ic_generation.ic import initial_conditions
    cfg = accuracy_cfg(test_problem="Gresho-Vortex-2D", SPATIAL_DIMENSION=2, Nx=ny, Ny=ny, Lx=1.0, Ly=1.0,
                       BOUNDARY=boundary, gamma=G, problem=gresho_vortex.Params(Mmax=MMAX))
    V = np.asarray(initial_conditions(cfg))                      # vortex at (0.5, 0.5) of a unit square
    return np.concatenate([V, np.repeat(V[:, -1:, :], nx - ny, 1)], axis=1) if nx > ny else V


def _boost(U, u):
    """The state seen moving at +u in x."""
    U = U.copy()
    U[3] += u * U[1] + 0.5 * u * u * U[0]
    U[1] += u * U[0]
    return U


def _cfg(nx, ny, lx, boundary, scheme):
    solver, limiter = SCHEMES[scheme]
    return accuracy_cfg(test_problem="Gresho-Vortex-2D", SPATIAL_DIMENSION=2, Nx=nx, Ny=ny, Lx=lx, Ly=1.0,
                        BOUNDARY=boundary, gamma=G, riemann_solver=solver, LIMITER=limiter)


def _prim(U, u):
    rho = U[0]
    vx, vy = U[1] / rho - u, U[2] / rho
    return rho, vx, vy, (G - 1.0) * (U[3] - 0.5 * rho * ((vx + u) ** 2 + vy ** 2))


@functools.lru_cache(maxsize=None)
def periodic(n, scheme, moving, frac=1.0):
    """{field: (L1, L2, Linf)} after frac revolutions vs the exact state after one; moving: bulk 1/REV."""
    u = 1.0 / REV if moving else 0.0
    U0 = _boost(_ic(n, n, 1.0, 0), u)
    U, _ = evolve(_cfg(n, n, 1.0, 0, scheme), U0, frac * REV)
    r0, x0, y0, p0 = _prim(U0, u)
    r, x, y, p = _prim(U, u)
    out = {}
    for name, e in (("v", np.hypot(x - x0, y - y0)), ("rho", r - r0), ("p", p - p0)):
        out[name] = (float(np.abs(e).mean()), float(np.sqrt((e * e).mean())), float(np.abs(e).max()))
    return out


def _vorticity(vx, vy, h):
    return np.gradient(vy, h, axis=0) - np.gradient(vx, h, axis=1)


@functools.lru_cache(maxsize=None)
def lw_metrics(nx, ny, scheme, moving):
    """LW's (rel. L1 vorticity %, L1 density %, rel. TKE error %) at T = 3, transmissive edges."""
    u = 1.0 if moving else 0.0
    U0 = _boost(_ic(nx, ny, nx / ny, 1), u)
    U, _ = evolve(_cfg(nx, ny, nx / ny, 1, scheme), U0, 3.0)
    ex = np.roll(U0, int(round(3.0 * ny)) if moving else 0, axis=1)
    (r, x, y, _), (re, xe, ye, _) = _prim(U, u), _prim(ex, u)
    h = 1.0 / ny
    w, we = _vorticity(x, y, h), _vorticity(xe, ye, h)
    ke = lambda rr, a, b: float((0.5 * rr * (a * a + b * b)).sum())          # noqa: E731
    return (100 * float(np.abs(w - we).sum() / np.abs(we).sum()), 100 * float(np.abs(r - re).sum() / re.sum()),
            100 * abs(ke(r, x, y) / ke(re, xe, ye) - 1.0))


CASES = [(sc, mv) for sc in SCHEMES for mv in (False, True)]


@pytest.mark.parametrize("scheme,moving", CASES, ids=[f"{sc}-{'moving' if mv else 'static'}" for sc, mv in CASES])
def test_gresho(scheme, moving):
    """L1, L2, Linf of v, rho, p at every N and their orders."""
    b = BOUNDS[scheme, moving]
    res = [periodic(n, scheme, moving) for n in NS]
    for fl in FIELDS:
        for j, nm in enumerate(NORMS):
            vals = [r[fl][j] for r in res]
            for n, v in zip(NS, vals):
                check(f"gresho {scheme} {'moving' if moving else 'static'} N={n} {fl} {nm}", v, b[fl][nm][n])
            order = -np.polyfit(np.log(NS), np.log(vals), 1)[0]
            check(f"gresho {scheme} {fl} {nm} order", order, b[fl][f"{nm}_order"][0], lower=True)
            check(f"gresho {scheme} {fl} {nm} order", order, b[fl][f"{nm}_order"][1])


@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_galilean(scheme):
    """Moving / static velocity L1 at every N stays below its bound."""
    for n in NS:
        check(f"gresho {scheme} N={n} moving/static v L1", periodic(n, scheme, True)["v"][0] / periodic(n, scheme, False)["v"][0],
              GALILEAN[scheme])


@pytest.mark.parametrize("moving", (False, True), ids=("static", "moving"))
def test_mc_no_worse_than_liska_wendroff(moving):
    """HLLC + MC at LW's grids (from the neighbouring power-of-2 runs) is no worse than their worst scheme."""
    lo, hi = ((128, 32), (256, 64)) if moving else ((32, 32), (64, 64))
    a, b = lw_metrics(*lo, "hllc-mc", moving), lw_metrics(*hi, "hllc-mc", moving)
    for j, name in enumerate(("rel L1 vorticity", "L1 density", "rel TKE")):
        check(f"LW gresho {'moving' if moving else 'static'} {name} %", at_n(40, lo[1], a[j], hi[1], b[j]), LW_WORST[moving][j])


def test_bound_catches_a_short_run():
    """Teeth: the moving vortex stopped at 0.98 revolution misses its velocity L1 bound."""
    assert periodic(64, "hllc-mc", True, frac=0.98)["v"][0] > BOUNDS["hllc-mc", True]["v"]["L1"][64]
