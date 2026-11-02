"""HLLC and HLL face fluxes against the exact Riemann solution: Toro 1-5, exact contacts, consistency, positivity."""
import numpy as np
import pytest

from luna.hydro import exact_riemann as er
from tests._helpers import check

G = 1.4
SOLVERS = ("hllc", "hll")
# Toro 1-5 (Table 4.1) and Toro's modified Sod with a sonic rarefaction at x/t = 0.
CASES = {**{k: v[:2] for k, v in er.TORO.items()}, "1s": ((1.0, 0.75, 1.0), (0.125, 0.0, 0.1))}
# Component k of the error is normalised by max over L, R of |F_k(W)| + S |U_k(W)|. Bounds ~1.3x measured
# (mass, normal momentum, energy); round-off 1e-14 where the measured error is 0 (Toro 2 is symmetric).
FLUX_ERR = {
    ("hllc", 1): dict(m=0.038, n=0.23, e=0.0034),     # measured 0.0295, 0.179, 0.00259
    ("hllc", 2): dict(m=1e-14, n=0.145, e=1e-14),     # 5.9e-18, 0.111, 3.6e-17
    ("hllc", 3): dict(m=0.049, n=0.065, e=0.059),     # 0.0380, 0.0502, 0.0453
    ("hllc", 4): dict(m=0.049, n=0.065, e=0.059),     # 0.0380, 0.0502, 0.0453
    ("hllc", 5): dict(m=0.21, n=0.093, e=0.14),       # 0.160, 0.0716, 0.106
    ("hllc", "1s"): dict(m=0.054, n=0.071, e=0.022),  # 0.0413, 0.0544, 0.0171
    ("hll", 1): dict(m=0.13, n=0.16, e=0.078),        # 0.103, 0.120, 0.0599
    ("hll", 2): dict(m=1e-14, n=0.145, e=1e-14),      # 0, 0.111, 1.9e-17
    ("hll", 3): dict(m=0.39, n=0.24, e=0.18),         # 0.301, 0.182, 0.139
    ("hll", 4): dict(m=0.39, n=0.24, e=0.18),         # 0.301, 0.182, 0.139
    ("hll", 5): dict(m=0.16, n=0.11, e=0.016),        # 0.125, 0.0820, 0.0122
    ("hll", "1s"): dict(m=0.13, n=0.0058, e=0.089),   # 0.101, 0.00449, 0.0683
}



def _w3(rho, un, p, vt=(0.0, 0.0), axis=0):
    """3D primitive column (5, 1) with normal velocity un along `axis` and transverse vt."""
    v = [0.0, 0.0, 0.0]
    v[axis] = un
    for k, t in zip([d for d in range(3) if d != axis], vt):
        v[k] = t
    return np.array([[rho], *[[x] for x in v], [p]])


def _flux(solver, WL, WR, axis=0):
    """Solver flux for column states, permuted into the normal frame (F_rho, F_n, F_t1, F_t2, F_E)."""
    from luna.hydro.fluxes import hll_flux_primitive, hllc_flux_primitive
    fn = hllc_flux_primitive if solver == "hllc" else hll_flux_primitive
    F = np.asarray(fn(WL, WR, axis, G))[:, 0]
    order = [axis] + [d for d in range(3) if d != axis]
    return np.array([F[0], *[F[1 + d] for d in order], F[4]])


def _phys(W):
    """Physical normal-frame flux and conserved state of a 1D primitive (rho, u, p)."""
    rho, u, p = W
    E = p / (G - 1.0) + 0.5 * rho * u * u
    return np.array([rho * u, rho * u * u + p, 0.0, 0.0, u * (E + p)]), \
        np.array([rho, rho * u, 0.0, 0.0, E])


def _err(F, Fx, WL, WR):
    """Normalised flux error per component (see FLUX_ERR)."""
    S = max(abs(W[1]) + np.sqrt(G * W[2] / W[0]) for W in (WL, WR))
    ref = np.maximum.reduce([np.abs(f) + S * np.abs(u) for f, u in map(_phys, (WL, WR))])
    ref = np.where(ref > 0, ref, 1.0)
    return np.abs(F - Fx) / ref


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("case", list(CASES))
def test_flux_vs_exact_toro(solver, case):
    """Face flux of Toro's tests against the exact Godunov flux, per component, on every axis."""
    WL, WR = CASES[case]
    Fx = er.godunov_flux(WL, WR, G)
    errs = [_err(_flux(solver, _w3(*WL, axis=a), _w3(*WR, axis=a), a), Fx, WL, WR) for a in range(3)]
    spread = max(float(np.max(np.abs(e - errs[0]))) for e in errs)
    assert spread <= 1e-15 * max(1.0, float(np.max(errs[0]))), f"axis dependence {errs}"
    e = errs[0]
    for k, name in zip((0, 1, 4), ("mass", "normal momentum", "energy")):
        check(f"riemann {solver} toro{case} flux err {name}", float(e[k]), FLUX_ERR[(solver, case)][name[0]])
    check(f"riemann {solver} toro{case} flux err transverse", float(max(e[2], e[3])), 1e-14)


def _textbook(solver, WL, WR, vl=(0.0, 0.0), vr=(0.0, 0.0)):
    """Toro (2009) HLL (10.21) / HLLC (10.37-10.39) flux with Davis speeds (10.48), numpy."""
    def UF(W, vt):
        rho, u, p = W
        E = p / (G - 1.0) + 0.5 * rho * (u * u + vt[0] ** 2 + vt[1] ** 2)
        U = np.array([rho, rho * u, rho * vt[0], rho * vt[1], E])
        return U, u * U + np.array([0.0, p, 0.0, 0.0, p * u])
    (UL, FL), (UR, FR) = UF(WL, vl), UF(WR, vr)
    cl, cr = np.sqrt(G * WL[2] / WL[0]), np.sqrt(G * WR[2] / WR[0])
    SL, SR = min(WL[1] - cl, WR[1] - cr), max(WL[1] + cl, WR[1] + cr)
    if SL >= 0:
        return FL
    if SR <= 0:
        return FR
    if solver == "hll":
        return (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL)
    (rl, ul, pl), (rr, ur, pr) = WL, WR
    Sm = (pr - pl + rl * ul * (SL - ul) - rr * ur * (SR - ur)) / (rl * (SL - ul) - rr * (SR - ur))
    S, U, F, W, vt = (SL, UL, FL, WL, vl) if Sm >= 0 else (SR, UR, FR, WR, vr)
    rho, u, p = W
    f = rho * (S - u) / (S - Sm)
    Us = f * np.array([1.0, Sm, vt[0], vt[1], U[4] / rho + (Sm - u) * (Sm + p / (rho * (S - u)))])
    return F + S * (Us - U)


@pytest.mark.parametrize("solver", SOLVERS)
def test_flux_is_the_textbook_solver(solver):
    """The flux equals Toro's HLL/HLLC with Davis wave speeds, on Toro's tests and random states."""
    rng = np.random.default_rng(3)
    pairs = [(WL, WR, (0.0, 0.0), (0.0, 0.0)) for WL, WR in CASES.values()]
    for _ in range(40):
        WL, WR = (tuple(rng.uniform([0.1, -3, 0.01], [10, 3, 100])) for _ in range(2))
        pairs.append((WL, WR, tuple(rng.uniform(-2, 2, 2)), tuple(rng.uniform(-2, 2, 2))))
    worst = 0.0
    for WL, WR, vl, vr in pairs:
        F = _flux(solver, _w3(*WL, vl), _w3(*WR, vr))
        Fr = _textbook(solver, WL, WR, vl, vr)
        worst = max(worst, float(np.max(np.abs(F - Fr) / (np.abs(Fr) + 1.0))))
    check(f"riemann {solver} vs textbook formula", worst, 1e-14)  # round-off; measured 2.0e-15 (hll), 3.0e-15 (hllc)


CONTACTS = {"stationary": 0.0, "moving right": 0.7, "moving left": -0.4}
# lower bound on HLL's normalised flux error on these contacts; measured 2.55, 1.20, 2.20
HLL_CONTACT_ERR = {0.0: 2.0, 0.7: 0.92, -0.4: 1.7}


@pytest.mark.parametrize("u0", list(CONTACTS.values()), ids=list(CONTACTS))
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_hllc_exact_on_contact_and_shear(u0, axis):
    """HLLC reproduces the exact flux of an isolated contact with shear; HLL smears it."""
    WL, WR = (1.0, u0, 2.5), (0.2, u0, 2.5)
    vl, vr = (0.3, -1.1), (-0.8, 0.6)
    Fx = er.godunov_flux(WL, WR, G, vl, vr)
    ref = np.abs(Fx) + 1.0
    e_c = np.max(np.abs(_flux("hllc", _w3(*WL, vl, axis), _w3(*WR, vr, axis), axis) - Fx) / ref)
    e_l = np.max(np.abs(_flux("hll", _w3(*WL, vl, axis), _w3(*WR, vr, axis), axis) - Fx) / ref)
    check(f"riemann hllc contact u0={u0} axis={axis}", e_c, 1e-14)  # measured <= 1.4e-15
    check(f"riemann hll contact u0={u0} axis={axis}", e_l, HLL_CONTACT_ERR[u0], lower=True)


@pytest.mark.parametrize("solver", SOLVERS)
def test_consistency_uniform_state(solver):
    """F(U, U) is the physical flux F(U) for random states, every axis."""
    rng = np.random.default_rng(7)
    n = 64
    W = np.stack([rng.uniform(0.1, 10, n), *rng.uniform(-3, 3, (3, n)), rng.uniform(0.01, 100, n)])
    from luna.hydro.fluxes import hll_flux_primitive, hllc_flux_primitive
    fn = hllc_flux_primitive if solver == "hllc" else hll_flux_primitive
    rho, v, p = W[0], W[1:4], W[4]
    E = p / (G - 1.0) + 0.5 * rho * (v * v).sum(0)
    worst = 0.0
    for a in range(3):
        F = np.asarray(fn(W, W, a, G))
        Fx = np.stack([rho * v[a], *[rho * v[a] * v[k] + (p if k == a else 0.0) for k in range(3)],
                       v[a] * (E + p)])
        worst = max(worst, float(np.max(np.abs(F - Fx) / (np.abs(Fx) + rho + p))))
    check(f"riemann {solver} consistency F(U,U)=F(U)", worst, 1e-14)  # measured 3.9e-15


def _godunov_update(solver, WL, WR, n=8, cfl=0.5):
    """One first-order update of an n-cell Riemann problem: min density and pressure."""
    W = np.array([WL] * (n // 2) + [WR] * (n // 2)).T
    W3 = np.stack([W[0], W[1], 0 * W[0], 0 * W[0], W[2]])
    F = np.stack([_flux(solver, W3[:, i:i + 1], W3[:, i + 1:i + 2]) for i in range(n - 1)], 1)
    U = np.stack([W[0], W[0] * W[1], 0 * W[0], 0 * W[0], W[2] / (G - 1) + 0.5 * W[0] * W[1] ** 2])
    S = np.max(np.abs(W[1]) + np.sqrt(G * W[2] / W[0]))
    U = U[:, 1:-1] - cfl / S * (F[:, 1:] - F[:, :-1])
    return float(U[0].min()), float(((G - 1) * (U[4] - 0.5 * U[1] ** 2 / U[0])).min())


@pytest.mark.parametrize("solver", SOLVERS)
def test_positivity_strong_rarefactions(solver):
    """A first-order update at CFL 0.5 keeps density and pressure positive up to near vacuum."""
    worst = np.inf
    for u in (2.0, 3.0, 4.0, 4.5, 4.7):          # vacuum forms at u = 2 c / (gamma - 1) = 4.73
        rmin, pmin = _godunov_update(solver, (1.0, -u, 0.4), (1.0, u, 0.4))
        worst = min(worst, rmin / 1.0, pmin / 0.4)
    for case in (3, 5):
        WL, WR = er.TORO[case][:2]
        rmin, pmin = _godunov_update(solver, WL, WR)
        worst = min(worst, rmin / min(WL[0], WR[0]), pmin / min(WL[2], WR[2]))
    check(f"riemann {solver} positivity min(rho,p)/ambient", worst, 0.44, lower=True)  # measured 0.569


@pytest.mark.parametrize("solver", SOLVERS)
def test_1d_2d_flux_is_the_3d_flux(solver):
    """1D and 2D fluxes equal the 3D flux with the missing velocities zero, bitwise, every axis."""
    from luna.hydro.fluxes import hll_flux_primitive, hllc_flux_primitive
    fn = hllc_flux_primitive if solver == "hllc" else hll_flux_primitive
    rng = np.random.default_rng(11)
    n = 256
    W = [np.stack([rng.uniform(0.1, 10, n), *rng.uniform(-3, 3, (3, n)), rng.uniform(0.01, 100, n)])
         for _ in range(2)]
    for ndim in (1, 2):
        keep = list(range(1 + ndim)) + [4]
        WL, WR = (w.copy() for w in W)
        WL[1 + ndim:4], WR[1 + ndim:4] = 0.0, 0.0
        for axis in range(ndim):
            F3 = np.asarray(fn(WL, WR, axis, G))[keep]
            Fd = np.asarray(fn(WL[keep], WR[keep], axis, G))
            assert np.array_equal(Fd, F3), (solver, ndim, axis, float(np.abs(Fd - F3).max()))
