"""Liska & Wendroff 2003 sec. 4.1 smooth periodic problem: rho = 1 + 0.2 sin(pi(x + y - t(u + v))).

u = 1, v = -1/2, p = 1, gamma = 1.4, periodic on [0, 2]^2 (the period of the solution), T = 4."""
import functools

import numpy as np
import pytest

from tests._helpers import accuracy_cfg, at_n, check, evolve

G, U, V, P, T, L = 1.4, 1.0, -0.5, 1.0, 4.0, 2.0
SCHEMES = {"hllc-minmod": ("hllc", 0), "hll-minmod": ("hll", 0), "hllc-mc": ("hllc", 1)}
NS, NORMS = (32, 64, 128, 256), ("L1", "L2", "Linf")
# {scheme: bounds} on [0, 2]^2, ~1.3x measured; orders measured -0.1 .. +0.3 (CPU, FP64).
BOUNDS = {
    "hllc-minmod": dict(L1={32: 0.029, 64: 0.013, 128: 0.0039, 256: 0.0011}, L1_order=(1.50, 1.90),
        L2={32: 0.039, 64: 0.016, 128: 0.0052, 256: 0.0017}, L2_order=(1.42, 1.82),
        Linf={32: 0.07, 64: 0.03, 128: 0.013, 256: 0.0053}, Linf_order=(1.14, 1.54)),
    # measured L1 0.0226, 0.0101, 0.003, 0.000843; 1.599 | L2 0.0298, 0.0119, 0.00402, 0.00129; 1.517 | Linf 0.0536, 0.0232, 0.00983, 0.00404; 1.243
    "hll-minmod": dict(L1={32: 0.047, 64: 0.017, 128: 0.0063, 256: 0.0018}, L1_order=(1.47, 1.87),
        L2={32: 0.058, 64: 0.021, 128: 0.008, 256: 0.0025}, L2_order=(1.39, 1.79),
        Linf={32: 0.099, 64: 0.043, 128: 0.019, 256: 0.0077}, Linf_order=(1.12, 1.52)),
    # measured L1 0.0362, 0.0135, 0.00487, 0.00136; 1.567 | L2 0.0444, 0.0163, 0.00613, 0.00196; 1.493 | Linf 0.076, 0.0331, 0.0143, 0.00595; 1.223
    "hllc-mc": dict(L1={32: 0.0033, 64: 0.00068, 128: 0.00015, 256: 3.3e-05}, L1_order=(2.10, 2.50),
        L2={32: 0.0052, 64: 0.0011, 128: 0.00028, 256: 7.1e-05}, L2_order=(1.96, 2.36),
        Linf={32: 0.013, 64: 0.0038, 128: 0.0012, 256: 0.00036}, Linf_order=(1.63, 2.03)),
    # measured L1 0.00252, 0.00052, 0.000119, 2.58e-05; 2.197 | L2 0.00402, 0.000873, 0.000216, 5.49e-05; 2.060 | Linf 0.0103, 0.00295, 0.000898, 0.00028; 1.731
}
# LW Table 4.1, 200^2, relative L1 % of rho: second-order schemes 3.7e-3 (CLAW) ... 6.7e-2 (WAFT).
# LUNA: hllc-mc 0.00445; hllc-minmod 0.136, hll-minmod 0.212 (above every LW scheme).
LW_SECOND_ORDER_200 = (3.7e-3, 6.7e-2)


def initial(n):
    """Exact cell averages of the conserved state (only rho varies)."""
    h = L / n
    x = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(x, x, indexing="ij")
    s = np.sinc(h / 2.0) ** 2                       # cell average of sin(pi (x + y)): sinc(pi h / 2) per axis
    rho = 1.0 + 0.2 * s * np.sin(np.pi * (X + Y))
    E = P / (G - 1.0) + 0.5 * rho * (U * U + V * V)
    return np.stack([rho, rho * U, rho * V, E])


@functools.lru_cache(maxsize=None)
def errors(n, scheme, t=T):
    """(relative L1 in percent as LW, L1, L2, Linf) of rho at t against the exact cell averages at T."""
    solver, limiter = SCHEMES[scheme]
    cfg = accuracy_cfg(test_problem="Soundwave-Advected-2D", SPATIAL_DIMENSION=2, Nx=n, Ny=n, Lx=L, Ly=L,
                       BOUNDARY=0, gamma=G, riemann_solver=solver, LIMITER=limiter)
    U0 = initial(n)
    Ut, _ = evolve(cfg, U0, t)
    ex = _exact(n, T)
    e = Ut[0] - ex
    return (100.0 * float(np.abs(e).sum() / np.abs(ex).sum()), float(np.abs(e).mean()),
            float(np.sqrt((e * e).mean())), float(np.abs(e).max()))


def _exact(n, t):
    """Exact cell-averaged rho at t."""
    h = L / n
    x = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 1.0 + 0.2 * np.sinc(h / 2.0) ** 2 * np.sin(np.pi * (X + Y - t * (U + V)))


@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_lw_smooth_ladder(scheme):
    """L1, L2, Linf of rho at every N and their orders."""
    b = BOUNDS[scheme]
    errs = [errors(n, scheme)[1:] for n in NS]
    for n, e in zip(NS, errs):
        for j, norm in enumerate(NORMS):
            check(f"LW smooth {scheme} N={n} {norm}", e[j], b[norm][n])
    for j, norm in enumerate(NORMS):
        order = -np.polyfit(np.log(NS), np.log([e[j] for e in errs]), 1)[0]
        check(f"LW smooth {scheme} {norm} order", order, b[f"{norm}_order"][0], lower=True)
        check(f"LW smooth {scheme} {norm} order", order, b[f"{norm}_order"][1])


def test_mc_within_liska_wendroff():
    """At LW's 200^2 (between our 128 and 256), HLLC + MC lies within their second-order range."""
    rel = at_n(200, 128, errors(128, "hllc-mc")[0], 256, errors(256, "hllc-mc")[0])
    check("LW smooth 200^2 hllc-mc rel L1 %", rel, LW_SECOND_ORDER_200[0], lower=True)
    check("LW smooth 200^2 hllc-mc rel L1 %", rel, LW_SECOND_ORDER_200[1])


def test_bound_catches_a_short_run():
    """Teeth: stopped at 0.98 T and compared with the solution at T, the MC L1 at 128 misses its bound."""
    assert errors(128, "hllc-mc", t=0.98 * T)[1] > BOUNDS["hllc-mc"]["L1"][128]
