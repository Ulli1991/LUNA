"""Planar Noh (Noh 1987, JCP 72, 78) against the exact solution: errors, orders, shock position, the
wall-heating defect and the closed-box budget.

Cold gas (rho0 = 1, u0 = -1, p0 = 1e-6, gamma = 5/3) runs into a reflecting wall at x = 0 (BOUNDARY=2).
Exact, from ExactPack `exactpack/solvers/noh/noh1.py` (github.com/lanl/ExactPack, BSD-3; read 2026-10-21,
geometry = 1, lines 53-68): shock at x_s = |u0| t (gamma - 1) / 2, behind it rho = rho0 (gamma + 1) /
(gamma - 1) = 4, u = 0, p = (gamma - 1) / 2 rho_s u0^2 = 4/3 (ExactPack writes rho0 u0^2 4 / 3, the
gamma = 5/3 value); ahead of it the inflow state. p0 > 0 changes the state by O(p0 / rho0 u0^2), checked
against LUNA's exact Riemann solver below.

The domain: BOUNDARY is one code for every face (`src/luna/template_config.py:26`,
`src/luna/utils/halo_local.py:16-31`), so the far end is a wall too. Inflow at u0 into that wall would
leave a vacuum behind (measured: MC goes to negative pressure there after 38 steps). So the gas is
brought to rest towards the far wall by a linear ramp, u = u0 (L - x) / (L - A) on [A, L]: a cold
ballistic expansion, rho = 1 / (1 + t / (L - A)) > 0, no vacuum. The inflow left of the ramp is exact
until the ramp's edge, a weak discontinuity riding at u0 - c0, arrives: at T it is at A - (1 + c0) T =
0.4. The measured region [0, X_MAX = 0.3] keeps 0.1 (25 cells at the coarsest N) from it; the
undisturbed inflow there is checked. Mass and energy then close to round-off; the momentum gains the wall
impulse p_s T.
"""
import numpy as np
import pytest

from luna.hydro import exact_riemann as er
from tests._helpers import accuracy_cfg, check, evolve, gpu_only

G, RHO0, U0, P0 = 5.0 / 3.0, 1.0, -1.0, 1e-6
L, A, T, X_MAX = 2.0, 1.0, 0.6, 0.3
D = (G - 1.0) / 2.0 * abs(U0)                       # shock speed, 1/3
RHO_S = RHO0 * (G + 1.0) / (G - 1.0)                # 4
P_S = (G - 1.0) / 2.0 * RHO_S * U0 * U0             # 4/3
NS = (256, 512, 1024, 2048)                         # cells per unit length; the box [0, 2] has 2 N
NS_GPU = (8192, 32768)
SKIP = 4            # cells next to the wall left out of L1: the HLLC defect (|rho / 4 - 1| > 2 %) is cells 0, 1
SOLVERS = {("hllc", 0): "minmod", ("hllc", 1): "mc", ("hllc", 2): "vanleer", ("hll", 0): "hll-minmod"}
# Per (solver, limiter), CPU, FP64, default dt; ~1.3x measured:
#   L1 {N: (rho, u, p)} on [SKIP h, X_MAX]; orders fitted over NS, measured -0.1 .. +0.3;
#   wall: (lo, hi) of min rho over the first SKIP cells (+-0.5 % of the measured range, CPU and GPU N);
#   extent: cells with |rho / 4 - 1| > 2 %, every N (HLLC 2, HLL 0: its defect is < 2 % and shrinks with N).
# GPU_L1: L1 at NS_GPU, A100 FP64, 1.3x measured (job 31365058).
BOUNDS = {
    ('hllc', 0): dict(L1={256: (0.0048, 0.0023, 0.002), 512: (0.0021, 0.00092, 0.00078), 1024: (0.00094, 0.00049, 0.00033), 2048: (0.00057, 0.00022, 0.00018)},
            L1_order=((0.94, 1.34), (1.01, 1.41), (1.06, 1.46)), wall=(3.3876, 3.4273), extent=2),
    # measured L1 0.00372 0.00173 0.00151 | 0.00163 0.000709 0.000597 | 0.000725 0.00038 0.000255 | 0.000441 0.000166 0.000138; orders 1.040 1.106 1.160; wall min rho 3.4046 3.4051 3.4053 3.4055 3.4068 3.4103
    ('hllc', 1): dict(L1={256: (0.0025, 0.0019, 0.00098), 512: (0.0012, 0.00075, 0.00046), 1024: (0.00072, 0.00029, 0.00026), 2048: (0.00035, 0.00014, 0.00012)},
            L1_order=((0.83, 1.23), (1.15, 1.55), (0.89, 1.29)), wall=(3.5407, 3.5768), extent=2),
    # measured L1 0.00191 0.00144 0.000756 | 0.000955 0.000579 0.000354 | 0.000554 0.000225 0.000203 | 0.000269 0.00011 9.34e-05; orders 0.926 1.250 0.985; wall min rho 3.5590 3.5585 3.5585 3.5585 3.5585 3.5585
    ('hllc', 2): dict(L1={256: (0.0031, 0.002, 0.0012), 512: (0.0015, 0.00081, 0.00056), 1024: (0.00075, 0.00037, 0.00029), 2048: (0.00041, 0.00016, 0.00015)},
            L1_order=((0.86, 1.26), (1.10, 1.50), (0.92, 1.32)), wall=(3.4645, 3.4993), extent=2),
    # measured L1 0.00235 0.00156 0.000961 | 0.00113 0.000624 0.00043 | 0.000577 0.000286 0.000223 | 0.000318 0.000125 0.000115; orders 0.961 1.203 1.015; wall min rho 3.4819 3.4819 3.4819 3.4819 3.4819 3.4819
    ('hll', 0): dict(L1={256: (0.0068, 0.0023, 0.0024), 512: (0.0029, 0.00089, 0.00084), 1024: (0.0011, 0.00052, 0.00037), 2048: (0.00069, 0.00022, 0.00017)},
            L1_order=((1.02, 1.42), (0.98, 1.38), (1.16, 1.56)), wall=(3.9120, 4.0064), extent=0),
    # measured L1 0.00524 0.00176 0.00184 | 0.00221 0.000683 0.000646 | 0.000862 0.000399 0.000282 | 0.000534 0.000173 0.000132; orders 1.124 1.081 1.260; wall min rho 3.9317 3.9447 3.9571 3.9654 3.9783 3.9865
}
SHOCK_POS = 0.38         # |x_s - D t| / h, every scheme and N; measured max 0.29
UNDISTURBED = 1e-10       # max |rho - rho0|, |u - u0| on [x_s + 0.05, X_MAX]; measured max 3.3e-11 (round-off)
IMPULSE = 0.0017         # |dP / (p_s T) - 1| at the coarsest N; first order after; measured max 0.00131
WALL_SPREAD = 0.0012       # HLLC: max - min over NS of the wall min rho (fixed in cells); measured max 0.000894

GPU_L1 = {
    ('hllc', 0): {8192: (0.00023, 6e-05, 5.8e-05), 32768: (0.00013, 1.7e-05, 2e-05)},
    ('hllc', 1): {8192: (8.3e-05, 4.8e-05, 3e-05), 32768: (2.8e-05, 9.6e-06, 9.7e-06)},
    ('hllc', 2): {8192: (9.7e-05, 5.1e-05, 3.7e-05), 32768: (3.2e-05, 1.1e-05, 1.1e-05)},
    ('hll', 0): {8192: (0.00025, 5.8e-05, 7.7e-05), 32768: (7.3e-05, 1.8e-05, 2.5e-05)},
}



def initial(n):
    """(rho, rho u, E) on 2 n cells of [0, L]: inflow u0, ramped to rest on [A, L]."""
    x = (np.arange(2 * n) + 0.5) / n
    u = U0 * np.clip((L - x) / (L - A), 0.0, 1.0)
    rho = np.full_like(x, RHO0)
    return np.stack([rho, rho * u, P0 / (G - 1.0) + 0.5 * rho * u * u])


def exact(n, t=T):
    """Exact cell averages of (rho, u, p) on the cells of [0, X_MAX] at t."""
    edges = np.arange(int(round(X_MAX * n)) + 1) / n
    f = np.clip((D * t - edges[:-1]) * n, 0.0, 1.0)        # shocked fraction of each cell
    return f * RHO_S + (1 - f) * RHO0, (1 - f) * U0, f * P_S + (1 - f) * P0


# MC's and van Leer's predicted faces at the wall go non-physical in the first steps (audit C8): those runs allow
# the first-order face state and assert its count, FACES[(limiter, Nx)] = (lo, hi), exact (CPU-8 job 31665570 and
# A100 job 31665569 agree; Nx = 2 n, both halves of the reflection-symmetric box).
FACES = {(1, 512): (118, 118), (1, 1024): (272, 272), (1, 2048): (579, 579), (1, 4096): (1193, 1193),
         (2, 512): (97, 97), (2, 1024): (199, 199), (2, 2048): (404, 404), (2, 4096): (813, 813),
         (1, 16384): (4880, 4880), (1, 65536): (19625, 19625), (2, 16384): (3271, 3271), (2, 65536): (13102, 13102)}


def run(n, limiter=0, solver="hllc", t=T, mesh_shape=None):
    """Numerical (rho, u, p) on [0, L] at t, and U at 0 and t."""
    cfg = accuracy_cfg(test_problem="Sod-Shock", SPATIAL_DIMENSION=1, Nx=2 * n, Lx=L, BOUNDARY=2, gamma=G,
                       LIMITER=limiter, riemann_solver=solver, allow_first_order_faces=limiter in (1, 2))
    U0_ = initial(n)
    U, _ = evolve(cfg, U0_, t, mesh_shape=mesh_shape,      # BOUNDARY=2: mass and energy to round-off
                  faces=FACES.get((limiter, 2 * n), (0, 0)))
    rho, u = U[0], U[1] / U[0]
    return (rho, u, (G - 1.0) * (U[2] - 0.5 * rho * u * u)), U0_, U


def shock_position(rho, n):
    """x where rho falls through (rho0 + rho_s) / 2, linear between cell centres."""
    half = 0.5 * (RHO0 + RHO_S)
    i = int(np.where((rho[:-1] >= half) & (rho[1:] < half))[0][-1])
    return (i + 0.5 + (rho[i] - half) / (rho[i] - rho[i + 1])) / n


def measure(n, limiter, solver, t=T, mesh_shape=None):
    """L1 (rho, u, p) on [SKIP h, X_MAX], wall min rho, extent, shock offset / h, undisturbed max, dP."""
    num, U0_, U = run(n, limiter, solver, t, mesh_shape)
    ex = exact(n, t)
    m = len(ex[0])
    l1 = [float(np.abs(a[SKIP:m] - b[SKIP:]).sum()) / n for a, b in zip(num, ex)]
    rho, u = num[0], num[1]
    xs = D * t
    inner = int(xs * n) - 10                               # post-shock cells clear of the shock
    extent = int(np.sum(np.abs(rho[:inner] / RHO_S - 1.0) > 0.02))
    x = (np.arange(len(rho)) + 0.5) / n
    ahead = (x > xs + 0.05) & (x < X_MAX)
    undisturbed = float(max(np.abs(rho[ahead] - RHO0).max(), np.abs(u[ahead] - U0).max()))
    dP = (U[1].sum() - U0_[1].sum()) / n
    s_wall = num[2][0] / rho[0] ** G / (P_S / RHO_S ** G) - 1.0
    return dict(l1=l1, wall=float(rho[:SKIP].min()), s_wall=float(s_wall), extent=extent,
                shock=(shock_position(rho, n) - xs) * n, undisturbed=undisturbed, impulse=dP / (P_S * t) - 1.0,
                profile=rho[:12])


def test_exact_solution():
    """The ExactPack state satisfies Rankine-Hugoniot across x_s = D t and is the p0 -> 0 limit of LUNA's
    exact Riemann solution of the mirrored collision (1, -u0, p0) | (1, u0, p0)."""
    up, down = (RHO0, U0, 0.0), (RHO_S, 0.0, P_S)

    def flux(rho, u, p):
        return np.array([rho * u, rho * u * u + p, u * (p / (G - 1.0) + 0.5 * rho * u * u + p)])

    def cons(rho, u, p):
        return np.array([rho, rho * u, p / (G - 1.0) + 0.5 * rho * u * u])

    jump = flux(*down) - flux(*up) - D * (cons(*down) - cons(*up))        # [F] = D [U], p0 = 0
    check("Noh Rankine-Hugoniot max |[F] - D [U]|", float(np.abs(jump).max()), 1e-15)
    for p0 in (1e-6, 1e-8):
        w = er.waves((RHO0, -U0, p0), (RHO0, U0, p0), G)
        rel = max(abs(w["rho_R"] / RHO_S - 1), abs(w["p"] / P_S - 1), abs(w["right"][0] / D - 1), abs(w["u"]))
        check(f"Noh exact Riemann p0={p0:g} vs Noh 1987 max rel / p0", rel / p0, 4.9)   # 3.75: O(p0)


@pytest.mark.parametrize("solver,limiter", list(BOUNDS), ids=[SOLVERS[k] for k in BOUNDS])
def test_noh_ladder(solver, limiter):
    """L1 of rho, u, p away from the wall at every N and their orders; the shock position; the inflow
    undisturbed by the far end; mass and energy to round-off (in evolve), the wall impulse in momentum;
    the wall-heating defect bounded and fixed in cells."""
    B, out = BOUNDS[solver, limiter], {}
    tag = f"Noh {solver} lim{limiter}"
    for n in NS:
        r = out[n] = measure(n, limiter, solver)
        print(f"PROFILE {tag} N={n} rho[:12] " + " ".join(f"{v:.4f}" for v in r["profile"]))
        for k, name in enumerate("rup"):
            check(f"{tag} N={n} L1 {name}", r["l1"][k], B["L1"][n][k])
        check(f"{tag} N={n} |x_s - D t| / h", abs(r["shock"]), SHOCK_POS)
        check(f"{tag} N={n} inflow max deviation on [x_s + 0.05, X_MAX]", r["undisturbed"], UNDISTURBED)
        check(f"{tag} N={n} wall min rho", r["wall"], B["wall"][0], lower=True)
        check(f"{tag} N={n} wall min rho", r["wall"], B["wall"][1])
        print(f"MEASURED {tag} N={n} wall entropy excess = {r['s_wall']:.6e}; extent (2 %) = {r['extent']}; "
              f"impulse error = {r['impulse']:.6e}")
        if B["extent"] is not None:
            assert r["extent"] == B["extent"], f"{tag} N={n}: wall defect over {r['extent']} cells, not {B['extent']}"
    check(f"{tag} N={NS[0]} |dP / (p_s T) - 1|", abs(out[NS[0]]["impulse"]), IMPULSE)
    for n in NS[1:]:
        check(f"{tag} N={n} |dP / (p_s T) - 1| * N / N0", abs(out[n]["impulse"]) * n / NS[0], IMPULSE)
    for k, name in enumerate("rup"):
        order = -np.polyfit(np.log(NS), np.log([out[n]["l1"][k] for n in NS]), 1)[0]
        lo, hi = B["L1_order"][k]
        check(f"{tag} L1 {name} order", order, lo, lower=True)
        check(f"{tag} L1 {name} order", order, hi)
    if solver == "hllc":
        walls = [out[n]["wall"] for n in NS]
        check(f"{tag} wall min rho spread over N", max(walls) - min(walls), WALL_SPREAD)


@gpu_only
@pytest.mark.parametrize("solver,limiter", list(BOUNDS), ids=[SOLVERS[k] for k in BOUNDS])
def test_noh_wall_heating_gpu(solver, limiter):
    """GPU: N to 32768 per unit; the HLLC wall defect stays the same in cells, L1 keeps converging."""
    tag = f"Noh {solver} lim{limiter}"
    for n in NS_GPU:
        r = measure(n, limiter, solver)
        print(f"PROFILE {tag} N={n} rho[:12] " + " ".join(f"{v:.4f}" for v in r["profile"]))
        for k, name in enumerate("rup"):
            check(f"{tag} N={n} L1 {name}", r["l1"][k], GPU_L1[solver, limiter][n][k])
        check(f"{tag} N={n} wall min rho", r["wall"], BOUNDS[solver, limiter]["wall"][0], lower=True)
        check(f"{tag} N={n} wall min rho", r["wall"], BOUNDS[solver, limiter]["wall"][1])
        check(f"{tag} N={n} |x_s - D t| / h", abs(r["shock"]), SHOCK_POS)
        print(f"MEASURED {tag} N={n} wall entropy excess = {r['s_wall']:.6e}; extent (2 %) = {r['extent']}; "
              f"impulse error = {r['impulse']:.6e}")


def test_bound_catches_a_short_run():
    """Teeth: stopped at 0.95 T, the density L1 at 1024 misses its bound."""
    num = run(1024, t=0.95 * T)[0]
    ex = exact(1024)
    l1 = float(np.abs(num[0][SKIP:len(ex[0])] - ex[0][SKIP:]).sum()) / 1024
    assert l1 > BOUNDS["hllc", 0]["L1"][1024][0], f"L1 rho {l1:.4e} of a short run passes the bound"
