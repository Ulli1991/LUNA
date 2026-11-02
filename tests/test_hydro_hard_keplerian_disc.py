"""A17 Keplerian disc: a pressure-supported 2D disc in a softened point-mass potential (an exact steady state,
luna.ic_generation.keplerian_disc_2d), run for orbits at R = 1; drift of rho, v_phi and L_z from it."""
import functools

import numpy as np
import pytest

from luna.ic_generation import keplerian_disc_2d as kd
from luna.ic_generation.ic import initial_conditions
from tests._helpers import accuracy_cfg, check, evolve_staged, gpu_only

# Box [-4, 4]^2: the disc's R^-20 tail is ~1e-8 at the walls, whose ram-pressure noise otherwise drives
# N-independent waves (rho L1 ~2e-3 at L = 5, ~5e-3 with a hot floor P0 = 1e-7, measured)
L, GM, EPS, GAMMA, BC = 8.0, 1.0, 0.2, 5.0 / 3.0, 1
WARM = kd.Params()                                       # cs0 = 0.05: H/R = 0.05 at R = 1
COLD = kd.Params(cs0=0.01)                               # Mach ~100 at R = 1
T_ORB = 2 * np.pi / float(kd.exact(np.array([1.0]), GM, EPS, WARM)[2][0])   # orbit at R = 1
ANNULUS, DISC = (0.8, 1.3), (0.6, 1.5)
LIMITERS = {"minmod": 0, "mc": 1, "vanleer": 2}
BREAK = 0.1                                              # break-up: annulus rho L1 above 10 %


def cfg_for(n, limiter, P=WARM):
    return accuracy_cfg(test_problem="Keplerian-Disc-2D", SPATIAL_DIMENSION=2, Nx=n, Ny=n, Lx=L, Ly=L, BOUNDARY=BC,
                        gamma=GAMMA, LIMITER=LIMITERS[limiter], external_potential="point_mass", external_GM=GM,
                        external_softening=EPS, problem=P)


def metrics(U, U0, n):
    """Annulus L1 of rho and v_phi, L1 of v_R, and the drifts of L_z (box, disc band) and box mass against U0."""
    x = (np.arange(n) + 0.5) * (L / n) - 0.5 * L
    X, Y = np.meshgrid(x, x, indexing="ij")
    R = np.hypot(X, Y)
    a, d = (R > ANNULUS[0]) & (R < ANNULUS[1]), (R > DISC[0]) & (R < DISC[1])
    lz, lz0 = X * U[2] - Y * U[1], X * U0[2] - Y * U0[1]
    vp, vp0 = lz / (R * U[0]), lz0 / (R * U0[0])
    vr = (X * U[1] + Y * U[2]) / (R * U[0])
    return dict(rho=float(np.abs(U[0] - U0[0])[a].sum() / U0[0][a].sum()),
                vphi=float(np.abs(vp - vp0)[a].sum() / vp0[a].sum()),
                vr=float(np.abs(vr)[a].sum() / vp0[a].sum()),
                Lz=float(lz.sum() / lz0.sum() - 1.0), Lz_disc=float(lz[d].sum() / lz0[d].sum() - 1.0),
                M=float(U[0].sum() / U0[0].sum() - 1.0))


@functools.lru_cache(maxsize=None)
def history(n, limiter, orbits, cold=False, per_orbit=2):
    """[(t / T_orb, metrics)] every 1/per_orbit orbits up to `orbits` (open box: zero-gradient walls)."""
    cfg = cfg_for(n, limiter, COLD if cold else WARM)
    U0 = np.asarray(initial_conditions(cfg))
    times = T_ORB * np.arange(1, int(orbits * per_orbit) + 1) / per_orbit
    return evolve_staged(cfg, U0, times, conserve=False, observe=lambda U, t: (t / T_ORB, metrics(U, U0, n)))


def t_break(rows):
    """First sampled time (orbits) where the annulus rho L1 exceeds BREAK, inf if never."""
    return next((t for t, m in rows if m["rho"] > BREAK), np.inf)


def balance_residual(npts, P, pressure_term=True):
    """max |v_phi^2 / R - dPhi/dR - p'/rho| on npts points of [0.05, 3.6], p' by np.gradient (centred, O(h^2))."""
    R = np.linspace(0.05, 3.6, npts)
    rho, p, v = kd.exact(R, GM, EPS, P)
    if not pressure_term:                                   # teeth: v_phi of the bare softened point mass
        v = np.sqrt(GM * R * R / (R * R + EPS ** 2) ** 1.5)
    res = v ** 2 / R - GM * R / (R * R + EPS ** 2) ** 1.5 - np.gradient(p, R) / rho
    return float(np.abs(res[1:-1]).max())


# The exact state balances analytically, so the residual is the centred difference's own O(h^2) error: it must fall
# at order 2 (1.9 .. 2.1 allowed; measured 2.0000) as the points double, and at 40001 points stay below 1.3x measured
# (2.69e-6 warm, 1.08e-7 cold; CPU-8 FP64, job 31655039). The bare point-mass v_phi leaves 0.23 (warm).
BALANCE_NPTS = (10001, 20001, 40001)
BALANCE_BOUND = {WARM.cs0: 3.5e-6, COLD.cs0: 1.4e-7}


def test_exact_state_is_balanced():
    """The IC is the steady state: radial residual at O(h^2) of the difference; Rayleigh stable (d(R v_phi)/dR >= 0)."""
    for P in (WARM, COLD):
        res = [balance_residual(m, P) for m in BALANCE_NPTS]
        orders = -np.diff(np.log2(res))
        print(f"MEASURED kepler balance cs0={P.cs0} max|res| {res} orders {orders}")
        check(f"kepler exact radial balance max|res| cs0={P.cs0}", res[-1], BALANCE_BOUND[P.cs0])
        for o in orders:
            check(f"kepler balance residual order cs0={P.cs0}", float(o), 1.9, lower=True)
            check(f"kepler balance residual order cs0={P.cs0}", float(o), 2.1)
        R = np.linspace(0.05, 3.6, BALANCE_NPTS[-1])
        v = kd.exact(R, GM, EPS, P)[2]
        check(f"kepler min d(R v_phi)/dR cs0={P.cs0}", float(np.diff(R * v).min() / (R[1] - R[0])), 0.0, lower=True)


def test_balance_catches_a_missing_pressure_term():
    """Teeth: v_phi without the pressure-gradient term fails the residual bound (warm and cold disc)."""
    for P in (WARM, COLD):
        bad = balance_residual(BALANCE_NPTS[-1], P, pressure_term=False)
        print(f"MEASURED kepler balance without the pressure term cs0={P.cs0}: max|res| {bad:.4g}")
        check(f"kepler balance without pressure term cs0={P.cs0}", bad, BALANCE_BOUND[P.cs0], lower=True)


# CPU: one orbit at N = 64, 128, minmod; annulus L1 of rho, v_phi and |dL_z| of the disc band, ~1.3x measured.
CPU_NS, CPU_ORBITS = (64, 128), 1
CPU_BOUNDS = {64: dict(rho=0.175, vphi=0.061, Lz_disc=0.25), 128: dict(rho=0.051, vphi=0.011, Lz_disc=0.099)}
# measured (CPU-8, FP64, job 31387579): 64: 0.1347, 0.04713, 0.1938 | 128: 0.03952, 0.008437, 0.07647


def test_kepler_cpu_one_orbit():
    """One orbit on CPU: the drift from the steady state at N = 64, 128, and that it falls with N."""
    last = {n: history(n, "minmod", CPU_ORBITS)[-1][1] for n in CPU_NS}
    for n in CPU_NS:
        for f, b in CPU_BOUNDS[n].items():
            check(f"kepler cpu N={n} minmod {CPU_ORBITS} orbits {f}", abs(last[n][f]), b)
    assert last[128]["rho"] < last[64]["rho"] and last[128]["vphi"] < last[64]["vphi"]


TEETH_SPIN = 0.9                                        # v_phi factor of the broken IC: 19 % less centrifugal support


def test_bound_catches_a_sub_keplerian_disc():
    """Teeth: the 64^2 IC spun at 0.9 v_phi (p kept), one orbit, misses the v_phi and L_z bounds (exact state)."""
    n = 64
    cfg = cfg_for(n, "minmod")
    U0 = np.asarray(initial_conditions(cfg))
    assert U0.shape[0] == 4                             # rho, m_x, m_y, E
    U = U0.copy()
    ek = 0.5 * (U0[1] ** 2 + U0[2] ** 2) / U0[0]
    U[1:3] *= TEETH_SPIN
    U[3] -= ek * (1.0 - TEETH_SPIN ** 2)
    m = evolve_staged(cfg, U, [T_ORB], conserve=False, observe=lambda V, t: metrics(V, U0, n))[-1]
    print(f"MEASURED kepler teeth spin {TEETH_SPIN} N={n} one orbit {m}")
    for f in ("vphi", "Lz_disc"):      # at 0.95 v_phi these missed by only 3 % and 8 %, rho passed (job 31655039)
        check(f"kepler teeth spin {TEETH_SPIN} N={n} {f}", abs(m[f]), CPU_BOUNDS[n][f], lower=True)


# GPU: 10 orbits, annulus L1 of rho and v_phi, |dL_z| of the disc band and the box, per N and fitted order;
# ~1.3x measured, orders -0.1 .. +0.3 (A100, FP64, jobs 31387574/5/6). MC aborts (NaN dt) at 256, 512, 1024 and
# van Leer at 256, also with the C8 face fix (job 31402991): p < 0 after the hydro update in the fast, high-Mach
# gas inside the softening radius (test_kepler_mc_survives_one_orbit).
LADDER = {
    "minmod": dict(N=(256, 512, 1024, 2048),
                   rho=((0.10, 0.044, 0.0148, 0.0020), (1.75, 2.15)),
                   vphi=((0.0286, 0.0060, 0.00143, 0.000353), (2.0, 2.4)),
                   Lz_disc=((0.155, 0.066, 0.0227, 0.0065), (1.42, 1.82)),
                   Lz=((0.0150, 0.0071, 0.0031, 0.00114), (1.13, 1.53))),
    # rho 0.07805, 0.03383, 0.01135, 0.001551; 1.85 | vphi 0.02201, 0.004605, 0.001098, 0.0002714; 2.11
    # Lz_disc -0.1188, -0.05046, -0.01746, -0.005026; 1.52 | Lz -0.01155, -0.005442, -0.002420, -0.0008746; 1.23
    "vanleer": dict(N=(512, 1024, 2048),
                    rho=((0.0080, 0.00028, 4.6e-5), (3.6, 4.0)),
                    vphi=((2.9e-4, 4.4e-5, 6.9e-6), (2.6, 3.0)),
                    Lz_disc=((0.0081, 0.0016, 0.00027), (2.35, 2.75)),
                    Lz=((0.0020, 0.00039, 5.3e-5), (2.5, 2.9))),
    # rho 0.006160, 0.0002146, 3.569e-05; 3.72 | vphi 0.0002241, 3.356e-05, 5.302e-06; 2.70
    # Lz_disc -0.006220, -0.001222, -0.0002089; 2.45 | Lz -0.001515, -0.0003017, -4.055e-05; 2.61
}


@pytest.mark.xfail(strict=True, raises=RuntimeError,
                   reason="MC: the hydro update (faces physical since C8, the kick keeps p) makes p < 0 in the Mach "
                          "15-400 tail gas inside the softening radius (R 0.12-0.15) within ~20 steps, NaN dt")
def test_kepler_mc_survives_one_orbit():
    """MC at N = 256 for one orbit; aborts at step ~20 (A100 job 31402991; step split, job 31404345)."""
    assert t_break(history(256, "mc", 1)) == np.inf


@gpu_only
@pytest.mark.parametrize("limiter", list(LADDER))
def test_kepler_ladder(limiter):
    """Ten orbits at N = 256 .. 2048: drift from the steady state per N and its order; no break-up."""
    spec = LADDER[limiter]
    rows = {n: history(n, limiter, 10) for n in spec["N"]}
    for f in ("rho", "vphi", "Lz_disc", "Lz"):
        bounds, (lo, hi) = spec[f]
        vals = [abs(rows[n][-1][1][f]) for n in spec["N"]]
        for n, v, b in zip(spec["N"], vals, bounds):
            check(f"kepler N={n} {limiter} 10 orbits {f}", v, b)
        order = -np.polyfit(np.log(spec["N"]), np.log(vals), 1)[0]
        check(f"kepler {limiter} {f} order", order, lo, lower=True)
        check(f"kepler {limiter} {f} order", order, hi)
    for n in spec["N"]:
        assert t_break(rows[n]) == np.inf, f"{limiter} N={n} broke up within 10 orbits"
