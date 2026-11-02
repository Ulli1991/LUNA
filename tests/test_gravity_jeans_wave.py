"""Linear Jeans wave and instability (as the Athena++ jeans tests): an oblique wave in a periodic 1 x 1/2 x 1/2
box, k = 2 pi (1, 2, 2), G from njeans = lambda / lambda_J; omega^2 = c^2 k^2 (1 - njeans^2), exact linear answer."""
import functools

import numpy as np
import pytest

from tests._helpers import accuracy_cfg, check, evolve_staged, self_gravity_energies

GAM, AMP = 5.0 / 3.0, 1e-6
RHO0, P0 = 1.0, 1.0 / GAM                    # c = 1
BOX = (1.0, 0.5, 0.5)
KVEC = 2.0 * np.pi * np.array([1.0, 2.0, 2.0])
K = float(np.linalg.norm(KVEC))
NS = (32, 64, 128)                          # cells along x; y, z get half
SCHEMES = ("hllc-minmod", "hll-minmod", "hllc-mc")
# {(njeans, scheme): bounds} of (rho - exact)/AMP and of |mode amplitude / exact - 1|, ~1.3x measured;
# orders -0.1 .. +0.3. Athena++ gates (stable abs. error < 1e-7, unstable ratio <= 1.5/4 per doubling) hold.
BOUNDS = {
    (0.5, "hllc-minmod"): dict(L1={32: 0.11, 64: 0.026, 128: 0.0059}, L1_order=(1.98, 2.38),
        L2={32: 0.12, 64: 0.029, 128: 0.0069}, L2_order=(1.95, 2.35),
        Linf={32: 0.17, 64: 0.042, 128: 0.01}, Linf_order=(1.91, 2.31),
        amp={32: 0.17, 64: 0.041, 128: 0.0098}, amp_order=(1.96, 2.36)),
    # L1 0.0821, 0.0197, 0.00457; 2.08 | L2 0.0913, 0.0222, 0.00533; 2.05 | Linf 0.128, 0.0325, 0.00784; 2.01 | amp 0.131, 0.0315, 0.00751; 2.06
    (0.5, "hll-minmod"): dict(L1={32: 0.17, 64: 0.043, 128: 0.01}, L1_order=(1.94, 2.34),
        L2={32: 0.19, 64: 0.048, 128: 0.012}, L2_order=(1.93, 2.33),
        Linf={32: 0.28, 64: 0.073, 128: 0.017}, Linf_order=(1.90, 2.30),
        amp={32: 0.28, 64: 0.069, 128: 0.016}, amp_order=(1.94, 2.34)),
    # L1 0.133, 0.0331, 0.00784; 2.04 | L2 0.148, 0.0372, 0.00889; 2.03 | Linf 0.213, 0.0558, 0.0133; 2.00 | amp 0.213, 0.0527, 0.0126; 2.04
    (0.5, "hllc-mc"): dict(L1={32: 0.029, 64: 0.0033, 128: 0.0004}, L1_order=(3.00, 3.40),
        L2={32: 0.032, 64: 0.0038, 128: 0.00045}, L2_order=(2.99, 3.39),
        Linf={32: 0.045, 64: 0.0054, 128: 0.00064}, Linf_order=(2.97, 3.37),
        amp={32: 0.046, 64: 0.0054, 128: 0.00063}, amp_order=(3.00, 3.40)),
    # L1 0.0223, 0.00257, 0.000305; 3.10 | L2 0.0248, 0.00291, 0.000344; 3.09 | Linf 0.0343, 0.00416, 0.00049; 3.07 | amp 0.0356, 0.00413, 0.000486; 3.10
    (1.5, "hllc-minmod"): dict(L1={32: 0.089, 64: 0.021, 128: 0.005}, L1_order=(1.97, 2.37),
        L2={32: 0.099, 64: 0.025, 128: 0.0062}, L2_order=(1.90, 2.30),
        Linf={32: 0.15, 64: 0.042, 128: 0.011}, Linf_order=(1.74, 2.14),
        amp={32: 0.052, 64: 0.013, 128: 0.0031}, amp_order=(1.94, 2.34)),
    # L1 0.0683, 0.016, 0.00387; 2.07 | L2 0.0759, 0.019, 0.00475; 2.00 | Linf 0.113, 0.0325, 0.00875; 1.84 | amp 0.04, 0.00972, 0.00237; 2.04
    (1.5, "hll-minmod"): dict(L1={32: 0.1, 64: 0.028, 128: 0.0072}, L1_order=(1.82, 2.22),
        L2={32: 0.11, 64: 0.028, 128: 0.0073}, L2_order=(1.84, 2.24),
        Linf={32: 0.13, 64: 0.036, 128: 0.0094}, Linf_order=(1.80, 2.20),
        amp={32: 0.056, 64: 0.014, 128: 0.0033}, amp_order=(1.94, 2.34)),
    # L1 0.0797, 0.0213, 0.00553; 1.92 | L2 0.0831, 0.0217, 0.00561; 1.94 | Linf 0.101, 0.0276, 0.00722; 1.90 | amp 0.043, 0.0104, 0.00255; 2.04
    (1.5, "hllc-mc"): dict(L1={32: 0.056, 64: 0.013, 128: 0.0029}, L1_order=(2.03, 2.43),
        L2={32: 0.06, 64: 0.014, 128: 0.0032}, L2_order=(2.01, 2.41),
        Linf={32: 0.087, 64: 0.02, 128: 0.0046}, Linf_order=(2.02, 2.42),
        amp={32: 0.031, 64: 0.0071, 128: 0.0017}, amp_order=(2.02, 2.42)),
    # L1 0.043, 0.00964, 0.00225; 2.13 | L2 0.0462, 0.0105, 0.00246; 2.11 | Linf 0.0667, 0.0151, 0.00354; 2.12 | amp 0.0242, 0.00543, 0.00128; 2.12
}


def omega2(nj):
    return K * K * (1.0 - nj * nj)           # c = 1


def initial(n, nj, amp=AMP):
    """Exact cell averages: stable, the standing wave at rest; unstable, the growing eigenmode."""
    shape = (n, n // 2, n // 2)
    h = [b / s for b, s in zip(BOX, shape)]
    X = np.meshgrid(*[(np.arange(s) + 0.5) * hh for s, hh in zip(shape, h)], indexing="ij")
    avg = np.prod([np.sinc(k * hh / (2 * np.pi)) for k, hh in zip(KVEC, h)])
    th = sum(k * x for k, x in zip(KVEC, X))
    s, c = avg * np.sin(th), avg * np.cos(th)
    rho = RHO0 * (1.0 + amp * s)
    p = P0 * (1.0 + GAM * amp * s)
    v = np.zeros((3,) + shape)
    if omega2(nj) < 0:                       # growing mode: v = (sigma / k) AMP cos(th) k-hat
        sig = np.sqrt(-omega2(nj))
        v = (sig / K) * amp * c * (KVEC / K)[:, None, None, None]
    return np.stack([rho, *(RHO0 * v), p / (GAM - 1.0) + 0.5 * RHO0 * (v * v).sum(0)])


def cfg_for(n, nj, scheme, g_scale=1.0):
    solver, limiter = {"hllc-minmod": ("hllc", 0), "hll-minmod": ("hll", 0), "hllc-mc": ("hllc", 1)}[scheme]
    lam = 2.0 * np.pi / K
    G = np.pi * nj * nj / (RHO0 * lam * lam)                 # 4 pi G rho0 = c^2 k^2 nj^2
    return accuracy_cfg(test_problem="Jeans-Instability-3D", SPATIAL_DIMENSION=3, Nx=n, Ny=n // 2, Nz=n // 2,
                        Lx=BOX[0], Ly=BOX[1], Lz=BOX[2], BOUNDARY=0, gamma=GAM, riemann_solver=solver,
                        LIMITER=limiter, SELF_GRAVITY=True, gravity_energy="kick", GRAVITATIONAL_G=float(G * g_scale), gravity_warm_tol=0.0,
                        gravity_warm_vcycles=8)


@functools.lru_cache(maxsize=None)
def errors(n, nj, scheme, frac=1.0):
    """(L1, L2, Linf) of (rho - exact)/AMP at t = frac x (one period | one e-fold), and the mode amplitude
    over its initial value (exact: 1 | e)."""
    w2 = omega2(nj)
    t_end = 2.0 * np.pi / np.sqrt(w2) if w2 > 0 else 1.0 / np.sqrt(-w2)
    U0 = initial(n, nj)
    U = evolve_staged(cfg_for(n, nj, scheme), U0, frac * t_end)
    growth = 1.0 if w2 > 0 else np.e
    e = (U[0] - (RHO0 + (U0[0] - RHO0) * growth)) / AMP
    ratio = float(((U[0] - RHO0) * (U0[0] - RHO0)).sum() / ((U0[0] - RHO0) ** 2).sum())   # mode amplitude / initial
    return float(np.abs(e).mean()), float(np.sqrt((e * e).mean())), float(np.abs(e).max()), ratio


CASES = [(nj, sc) for nj in (0.5, 1.5) for sc in SCHEMES]


@pytest.mark.parametrize("njeans,scheme", CASES, ids=[f"{'stable' if nj < 1 else 'unstable'}-{sc}" for nj, sc in CASES])
def test_jeans(njeans, scheme):
    """L1, L2, Linf of rho and the mode amplitude (1 after a period | e after an e-fold), with orders."""
    b, exact = BOUNDS[njeans, scheme], (1.0 if njeans < 1 else np.e)
    res = [errors(n, njeans, scheme) for n in NS]
    series = [(nm, [r[j] for r in res]) for j, nm in enumerate(("L1", "L2", "Linf"))]
    series.append(("amp", [abs(r[3] / exact - 1.0) for r in res]))
    for nm, vals in series:
        for n, v in zip(NS, vals):
            check(f"jeans nj={njeans} {scheme} N={n} {nm}", v, b[nm][n])
        order = -np.polyfit(np.log(NS), np.log(vals), 1)[0]
        check(f"jeans nj={njeans} {scheme} {nm} order", order, b[f"{nm}_order"][0], lower=True)
        check(f"jeans nj={njeans} {scheme} {nm} order", order, b[f"{nm}_order"][1])


@pytest.mark.parametrize("njeans", (0.5, 1.5))
def test_bound_catches_a_short_run(njeans):
    """Teeth: stopped at 0.9 of the time the L1 at 64 misses its bound (measured 0.133, 0.177)."""
    assert errors(64, njeans, "hllc-minmod", frac=0.9)[0] > BOUNDS[njeans, "hllc-minmod"]["L1"][64]


# Energy and momentum: E_kin + E_th + W (W from the 7-point potential, as the multigrid solves) at a larger
# amplitude, so the O(AMP^2) wave energy stands well above the round-off of the background's thermal energy.
AMP_E, NS_E, FRACS_E = 1e-3, (16, 32, 64), (0.25, 0.5, 0.75, 1.0)
# |dE| / |W0|, max over the period (e-fold), ~1.3x measured (A100; CPU agrees). Truncation: the kick's work
# rho v . g with g = -D phi (central difference) is not the discrete -phi div(rho v) of the hydro fluxes, a
# second-order mismatch: orders 1.81 (stable), 2.02 (unstable), bounds -0.1 .. +0.3.
E_BOUNDS = {0.5: {16: 0.82, 32: 0.27, 64: 0.067}, 1.5: {16: 0.46, 32: 0.115, 64: 0.028}}
# 0.5: 0.633, 0.206, 0.0512 | 1.5: 0.350, 0.0883, 0.0213
E_ORDER = {0.5: (1.71, 2.11), 1.5: (1.92, 2.32)}
# |dP_d| / sqrt(M E): sum rho D L^-1 rho = 0 for the exact 7-point inverse (D antisymmetric, commuting with L),
# so the drift is the 8 fixed warm V-cycles' residual, not round-off: measured 1.2e-10 (16) .. 1e-12 (64).
P_TOL = 1.5e-10


def jeans_g(nj):
    return np.pi * nj * nj / (RHO0 * (2.0 * np.pi / K) ** 2)


@functools.lru_cache(maxsize=None)
def budget(n, nj, g_scale=1.0):
    """(max over FRACS_E of |E - E0| / |W0|, max of |dP_d| / sqrt(M E)) over one period | e-fold, hllc-minmod;
    the run's G is g_scale x the diagnostic's."""
    w2 = omega2(nj)
    t_end = 2.0 * np.pi / np.sqrt(w2) if w2 > 0 else 1.0 / np.sqrt(-w2)
    U0 = initial(n, nj, AMP_E)
    h, G = BOX[0] / n, jeans_g(nj)
    e0 = self_gravity_energies(U0, h, G)
    Us = evolve_staged(cfg_for(n, nj, "hllc-minmod", g_scale), U0, [f * t_end for f in FRACS_E])
    de = max(abs(sum(self_gravity_energies(U, h, G)) - sum(e0)) / abs(e0[2]) for U in Us)
    dp = max(abs(U[d].sum() - U0[d].sum()) for U in Us for d in (1, 2, 3)) / np.sqrt(U0[0].sum() * U0[4].sum())
    return float(de), float(dp)


@pytest.mark.parametrize("njeans", (0.5, 1.5), ids=("stable", "unstable"))
def test_jeans_energy_momentum(njeans):
    """|E_kin + E_th + W - E0| / |W0| at every N and its order (the gas kick is not exactly energy conserving);
    momentum to the solve's tolerance (the central-difference force on the 7-point potential is antisymmetric)."""
    res = [budget(n, njeans) for n in NS_E]
    for n, (de, dp) in zip(NS_E, res):
        check(f"jeans energy nj={njeans} N={n} |dE|/|W0|", de, E_BOUNDS[njeans][n])
        check(f"jeans energy nj={njeans} N={n} |dP|/sqrt(ME)", dp, P_TOL)
    order = -np.polyfit(np.log(NS_E), np.log([r[0] for r in res]), 1)[0]
    check(f"jeans energy nj={njeans} order", order, E_ORDER[njeans][0], lower=True)
    check(f"jeans energy nj={njeans} order", order, E_ORDER[njeans][1])


def test_energy_bound_catches_a_doubled_g():
    """Teeth: run with 2 G, diagnose with G: the potential no longer pays for the kick's work."""
    de = budget(32, 0.5, g_scale=2.0)[0]
    print(f"MEASURED jeans energy doubled G N=32 |dE|/|W0| = {de:.6e}")
    assert de > E_BOUNDS[0.5][32]
