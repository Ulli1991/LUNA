"""Kurganov & Tadmor 2002 2D Riemann problems: configuration 3 (four backward shocks) and 6 (four slip lines).

KT02 pp. 594, 596 (data as printed), gamma 1.4, (0, 1)^2 split at 1/2, T = 0.3, zero-gradient walls
(BOUNDARY=1; the papers state no boundary, lit_a11/lit.md). No exact 2D solution, so: L1/L2 of rho against a
stored LUNA MC + HLLC 8192^2 reference (`tools/kt_riemann.py`, `luna/kt_riemann/kt_riemann_c*_mc.npz` in luna-reference-solutions) after block
averaging both onto M^2, split into a core box (config 3's jet, config 6's vortex) that does not converge
pointwise and the rest, whose order is bounded; the exact 1D solution in the strips along the walls (where the
reference holds it); config 3's diagonal symmetry; the exact wall-flux budget while the walls see 1D states
(config 3 to T, config 6 to 0.25) and the code's own boundary-flux balance to round-off; config 6's pressure in
its strips (exactly 1); FP32; 2 x 2 shards. Config 6's slip lines are Kelvin-Helmholtz unstable: the target is
the laminar self-similar solution (code-independent: AthenaK PLM 4096 is 2.75e-4 from LUNA MC 4096 outside the
core), which the unseeded 8192 reference holds but a 1e-15 seed would roll up; test_reference guards it with a
seeded 4096 run. Rungs <= 2048 are seed-insensitive (a 1e-6 seed moves MC 2048 from 1.23e-3 to 1.28e-3).
Bounds are ~1.3x measured (TEST_PLAN convention). Not a teeth run: Nx != Ny (no transpose to compare).
External gate (config 3, outside the core): athenak/kt_riemann/a11_c3_ppmx_rk4_8192.npz (luna-reference-solutions),
AthenaK 886dd2a hydro kernels + luna_ic pgen (source c52f88f0), binary build_gpu_lunaic sha256 e40b2abd, PPMX + RK4
+ HLLC, CFL 0.4, dfloor = pfloor = 1e-8, FP64, 4 A100, from the test's IC, block-averaged onto 256^2
(tools/athenak_refs.py prep/reduce a11; the fixture asserts the scheme). Its own error outside the core EPS_REF =
max(4096 vs 8192, distance to LUNA's 8192 MC). Bounds ~1.3x LUNA's measured L1; matched method: LUNA van Leer
(MUSCL-Hancock) vs AthenaK PLM (van Leer harmonic limiter + RK2, athenak src/reconstruct/plm.hpp:28-31) at the same N,
L1(LUNA) <= L1(AthenaK PLM) + 2 EPS_REF (triangle inequality; no LUNA number in it). Inside the core neither code
converges (AthenaK 4096 vs 8192 there 2.75e-3), so no gate there. Config 6 has no external gate: AthenaK PPMX 8192
rolls its slip lines up (4096 vs 8192 L1 0.044, larger than every LUNA rung), so config 6 stays against the LUNA
reference (regression only) and the exact 1D strips."""
import functools

import numpy as np
import pytest

from luna.reference_data import reference_path
from luna.ic_generation import kt_riemann_2d as ktic
from tests import _helpers
from tests._athenak import load as load_athenak
from tests._helpers import _make_step_fns, accuracy_cfg, check, evolve, gpu_only
from tools import kt_riemann as kt

T, G = ktic.T_END, ktic.GAMMA
M = 128                                          # coarse grid of the reference metric
SCHEMES = {"minmod": 0, "mc": 1}
LIMITERS = {**SCHEMES, "vl": 2}                  # van Leer: only the AthenaK matched-method check
CPU_NS, GPU_NS = (256, 512), (256, 512, 1024, 2048)
# strips that hold the 1D solution (interface: width from its wall), confirmed on the reference
STRIPS = {3: {"21": 0.1, "34": 0.1, "32": 0.1, "41": 0.1}, 6: {"21": 0.07, "34": 0.1}}
# (lo, hi)^2: the jet (3) and vortex (6), no pointwise convergence. Config 3's box is wide: with (0.15, 0.55)
# the 8192 reference's own error outside it (3.3e-4, in the shock and post-shock bands of 21/41) was 0.83 of the
# 2048 rung's, and the order fit hung on the box (0.68-0.97, critic_a11 O2); with (0.05, 0.65) it is 1.1e-4.
CORE = {3: (0.05, 0.65), 6: (0.2, 0.7)}
T_BUDGET = {3: T, 6: 0.25}                       # config 6: 2D signals reach the x walls from t = 0.2586

# ---------------------------------------------------------------- measured, and the bounds from them
# Measured FP64, HLLC, CFL 0.8; GPU A100 job 31110810 (256-2048), CPU (256, 512) job 31110811 the same to the
# printed digits; config 3's core split re-measured for the (0.05, 0.65) box (job 31169083, the same runs). Rho on M^2 blocks against the reference: {(config, scheme): {N: (L1, L2, L1 core, L1 outside
# the core)}}. The core does not converge (the config-3 jet roll-up, the config-6 vortex): its L1 stalls at
# ~2e-3 / ~4e-3 from 1024 on, as the reference's own 4096 vs 8192 difference (2.35e-3 / 4.53e-3). Outside the core
# that difference is 1.09e-4 (3), 6.11e-4 (6): the 2048 rungs sit 1.8x / 2.0x above it.
MEASURED = {
    (3, "minmod"): {256: (7.872e-3, 3.697e-2, 6.443e-3, 1.429e-3), 512: (4.785e-3, 2.366e-2, 4.087e-3, 6.987e-4),
                    1024: (2.543e-3, 1.181e-2, 2.177e-3, 3.663e-4), 2048: (2.527e-3, 1.421e-2, 2.325e-3, 2.026e-4)},
    (3, "mc"): {256: (5.720e-3, 2.357e-2, 4.304e-3, 1.417e-3), 512: (3.131e-3, 1.366e-2, 2.442e-3, 6.889e-4),
                1024: (3.650e-3, 2.211e-2, 3.310e-3, 3.401e-4), 2048: (2.427e-3, 1.349e-2, 2.236e-3, 1.918e-4)},
    (6, "minmod"): {256: (2.985e-2, 7.844e-2, 1.359e-2, 1.626e-2), 512: (1.844e-2, 5.530e-2, 8.536e-3, 9.904e-3),
                    1024: (1.152e-2, 3.848e-2, 5.771e-3, 5.752e-3), 2048: (7.683e-3, 2.901e-2, 4.515e-3, 3.168e-3)},
    (6, "mc"): {256: (1.860e-2, 5.801e-2, 8.861e-3, 9.742e-3), 512: (1.082e-2, 3.803e-2, 5.634e-3, 5.182e-3),
                1024: (6.845e-3, 2.764e-2, 4.298e-3, 2.547e-3), 2048: (5.225e-3, 2.579e-2, 3.997e-3, 1.229e-3)},
}
BOUNDS = {k: {n: tuple(1.3 * x for x in v) for n, v in d.items()} for k, d in MEASURED.items()}
# L1 outside the core, fitted order: CPU (256, 512) and GPU (256..2048), measured -0.15 .. +0.15
# (config 3 pairwise 1.03, 0.93, 0.86 minmod; 1.04, 1.02, 0.83 MC)
ORDER_MEASURED = {"cpu": {(3, "minmod"): 1.032, (3, "mc"): 1.040, (6, "minmod"): 0.715, (6, "mc"): 0.911},
                  "gpu": {(3, "minmod"): 0.939, (3, "mc"): 0.967, (6, "minmod"): 0.786, (6, "mc"): 0.999}}
ORDERS = {t: {k: (v - 0.15, v + 0.15) for k, v in d.items()} for t, d in ORDER_MEASURED.items()}
# strip L1 against the exact 1D cell averages {(config, scheme): {N: {interface: L1}}}; 32/41 equal 34/21 (config 3)
STRIP_MEASURED = {
    (3, "minmod"): {256: (2.065e-3, 7.534e-4), 512: (9.161e-4, 3.242e-4), 1024: (3.850e-4, 1.719e-4),
                    2048: (2.223e-4, 1.050e-4)},
    (3, "mc"): {256: (1.810e-3, 7.251e-4), 512: (8.102e-4, 3.840e-4), 1024: (4.113e-4, 2.026e-4),
                2048: (2.298e-4, 1.278e-4)},
    (6, "minmod"): {256: (9.634e-3, 1.818e-2), 512: (6.454e-3, 1.253e-2), 1024: (3.965e-3, 7.706e-3),
                    2048: (2.701e-3, 5.226e-3)},
    (6, "mc"): {256: (4.973e-3, 9.685e-3), 512: (3.402e-3, 6.757e-3), 1024: (1.931e-3, 3.847e-3),
                2048: (1.337e-3, 2.642e-3)},
}
STRIP_BOUNDS = {k: {n: {"21": 1.3 * a, "41": 1.3 * a, "34": 1.3 * b, "32": 1.3 * b} for n, (a, b) in d.items()}
                for k, d in STRIP_MEASURED.items()}
STRIP_ORDER = {3: 0.7, 6: 0.45}     # lower bounds on the strip L1 orders; measured minima 0.84 (3), 0.52 (6)
FRONT_CELLS = {3: 0.4, 6: 0.9}      # |front - exact| in cells; measured max 0.305 (3), 0.698 (6, minmod 2048)
# config 6: L1 of the strip-mean pressure - 1, {scheme: {N: (strip 21, strip 34)}} (job 31169083, the saved runs);
# p = 1 exactly across the slip lines; a spike of ~1e-2 rides each moving contact (8192: 5.6e-3 / 7.1e-3)
P_STRIP_MEASURED = {"minmod": {256: (1.392e-3, 1.457e-3), 512: (8.068e-4, 9.184e-4), 1024: (5.069e-4, 5.824e-4),
                               2048: (3.210e-4, 3.689e-4)},
                    "mc": {256: (7.489e-4, 8.960e-4), 512: (4.639e-4, 5.605e-4), 1024: (2.895e-4, 3.516e-4),
                           2048: (1.816e-4, 2.215e-4)}}
P_STRIP_BOUNDS = {s: {n: {"21": 1.3 * a, "34": 1.3 * b} for n, (a, b) in d.items()}
                  for s, d in P_STRIP_MEASURED.items()}
# config 3 diagonal asymmetry: exactly 0 on the GPU (FP64 and FP32, every N); XLA's CPU arithmetic is not
# axis-symmetric: max 2.9e-13 (FP64, MC 512), 9.9e-6 (FP32, minmod 512)
SYM = {"fp64": 1e-12, "fp32": 3e-5}
# exact wall budget, max relative error over (M, P_x, P_y, E); measured 3.51e-5, 1.65e-5 (3); 8.79e-4, 5.76e-4 (6)
BUDGET = {3: {256: 4.6e-5, 512: 2.2e-5}, 6: {256: 1.15e-3, 512: 7.5e-4}}
REF_SELF_L1 = {3: 3.1e-3, 6: 5.9e-3}   # measured 2.350e-3, 4.532e-3 (8192 vs 4096 on M^2)
# config 6 reference vs the stored 4096 run seeded 1e-12, outside the core: measured 6.16e-4 (unseeded 4096
# 6.11e-4, seed 1e-15 6.11e-4; job 31169083); a rolled-up reference sits ~4e-3 away (seed 1e-15 at 8192: 4.1e-3)
LAMINAR_GUARD = 8.0e-4
BALANCE = {3: 1e-14, 6: 1e-14}         # measured <= 4.1e-16
FP64_L1_512 = {3: 4.785205e-3, 6: 1.844067e-2}   # FP64 minmod L1 at 512 (GPU = CPU to 7 digits)
FP32 = {3: 2e-4, 6: 2e-4}              # |FP32 L1 / FP64 L1 - 1| at 512
SHARD = {3: 0.0, 6: 0.0}               # bitwise (CPU, 2 x 2 of 8 virtual devices)


def _states(config, states):
    return ktic.STATES[config] if states is None else dict(states)


@functools.lru_cache(maxsize=None)
def _run(config, n, scheme, order, states, xc, gamma, boundary, t, mesh, precision):
    """(U0, U) of the four-quadrant problem on n^2 cells at t (precision only keys the cache)."""
    cfg = accuracy_cfg(test_problem="KT-Riemann-2D", SPATIAL_DIMENSION=2, Nx=n, Ny=n, Lx=1.0, Ly=1.0,
                       BOUNDARY=boundary, gamma=gamma, LIMITER=LIMITERS[scheme], ORDER=order,
                       problem=ktic.Params(config=config, xc=xc))
    U0 = np.asarray(ktic.quadrants(cfg, _states(config, states), xc, 0.5))
    U, _ = evolve(cfg, U0, t, mesh_shape=mesh)
    return U0, U


def run(config, n, scheme="minmod", order=1, states=None, xc=0.5, gamma=G, boundary=1, t=T, mesh=None):
    """(U0, U); `states` as a tuple of (quadrant, (rho, u, v, p)) pairs replaces KT's data."""
    return _run(config, n, scheme, order, states, xc, gamma, boundary, t, mesh, _helpers.PRECISION)


@functools.lru_cache(maxsize=None)
def reference(config):
    """The stored reference density on 256^2, after checking what it is and how good."""
    with np.load(reference_path(f"luna/kt_riemann/kt_riemann_c{config}_mc.npz")) as z:
        assert int(z["config"]) == config and float(z["t"]) == T and float(z["gamma"]) == G
        assert int(z["boundary"]) == 1 and int(z["limiter"]) == 1 and str(z["riemann_solver"]) == "hllc"
        assert str(z["precision"]) == "fp64"
        return np.asarray(z["rho"], np.float64), dict(zip(z["self_ms"].tolist(), z["self_l1_rho"].tolist()))


def ref_errors(rho, config, m=M, ref=None):
    """(L1, L2, L1 inside CORE, L1 outside CORE) of rho - the reference (default LUNA's), both on m^2 blocks."""
    ref = reference(config)[0] if ref is None else ref
    e = kt.block_mean(rho, m) - kt.block_mean(ref, m)
    x = (np.arange(m) + 0.5) / m
    lo, hi = CORE[config]
    inside = ((x > lo) & (x < hi))[:, None] & ((x > lo) & (x < hi))[None, :]
    a = np.abs(e)
    return (float(a.mean()), float(np.sqrt((e * e).mean())), float((a * inside).sum() / m / m),
            float((a * ~inside).sum() / m / m))


def front(config, name):
    """(exact position at T, density on the wall side, density behind) of the strip's main wave."""
    w = kt.waves(config, name)
    WL, WR = kt.problem(config, name)[:2]
    if config == 3:                              # the backward shock
        return 0.5 + w["left"][0] * T, WL[0], w["rho_L"]
    return 0.5 + w["u"] * T, WL[0], WR[0]        # the slip line


def strip_errors(rho, config):
    """{interface: (L1 of the strip mean - exact 1D cell averages, front position error in cells)}."""
    n = rho.shape[0]
    out = {}
    for name, width in STRIPS[config].items():
        prof = kt.strip_profile(rho, name, width)
        x_s, a, b = front(config, name)
        out[name] = (float(np.abs(prof - kt.strip_exact(config, name, n)[0]).mean()),
                     abs(kt.crossing(prof, a, b) - x_s) * n)
    return out


def pressure_strips(U):
    """Config 6: {interface: L1 of the strip-mean pressure - 1}; p = 1 exactly across every slip line, so this
    sees spurious pressure at the moving contacts, which the density strips cannot."""
    U = np.asarray(U, np.float64)
    p = (G - 1.0) * (U[3] - 0.5 * (U[1] ** 2 + U[2] ** 2) / U[0])
    return {name: float(np.abs(kt.strip_profile(p, name, w) - 1.0).mean()) for name, w in STRIPS[6].items()}


def asymmetry(U):
    """max |U - U mirrored about x = y (momenta swapped)| / max |U|."""
    R = np.asarray(U).transpose(0, 2, 1)[[0, 2, 1, 3]]
    return float(np.abs(U - R).max() / np.abs(U).max())


def budget_errors(U0, U, config, t):
    """|change of the totals - exact wall budget| per (mass, x-mom, y-mom, energy), relative to (M, sqrt(ME),
    sqrt(ME), E) at t = 0."""
    n = U.shape[1]
    got = (np.asarray(U, np.float64).sum((1, 2)) - U0.sum((1, 2))) / (n * n)
    M0, E0 = U0[0].mean(), U0[3].mean()
    return np.abs(got - kt.wall_budget(config, t)) / np.array([M0, np.sqrt(M0 * E0), np.sqrt(M0 * E0), E0])


def flux_balance(config, n, scheme="minmod"):
    """max over (mass, x-mom, y-mom, energy) of |change of the totals - sum_steps dt (boundary inflow)| relative
    to (M, sqrt(ME), sqrt(ME), E), the boundary fluxes rebuilt each step from LUNA's own 2D wide face fluxes."""
    import jax
    import jax.numpy as jnp
    from luna.hydro.conversions import cons_to_prim
    from luna.hydro.muscl_hancock import _muscl_hancock_2d_fluxes
    from luna.hydro.slopes import compute_slopes_from_padded
    from luna.utils.halo_local import pad_halo_local
    cfg = accuracy_cfg(test_problem="KT-Riemann-2D", SPATIAL_DIMENSION=2, Nx=n, Ny=n, Lx=1.0, Ly=1.0, BOUNDARY=1,
                       gamma=G, LIMITER=SCHEMES[scheme], problem=ktic.Params(config=config))
    inv = float(n)

    @jax.jit
    def inflow(U, dt):
        W_pad = pad_halo_local(cons_to_prim(U, G), 1, 2, width=2)
        dWx, dWy, _ = compute_slopes_from_padded(W_pad, cfg.LIMITER)
        Fx, Fy = _muscl_hancock_2d_fluxes(W_pad, dt, dWx, dWy, gamma=G, riemann_solver=cfg.riemann_solver,
                                          inv_dx=inv, inv_dy=inv, boundary=1, wide=True)
        return (jnp.sum(Fx[:, 0] - Fx[:, -1], 1) + jnp.sum(Fy[:, :, 0] - Fy[:, :, -1], 1)) / n

    dt_fn, step = _make_step_fns(cfg)
    U = U0 = jnp.asarray(ktic.kt_riemann_2d(cfg))
    t, acc = 0.0, np.zeros(4)
    while t < T * (1 - 1e-12):
        dt = min(float(dt_fn(U)), T - t)
        acc += dt * np.asarray(inflow(U, dt))
        U, t = step(U, dt), t + dt
    U0, U = np.asarray(U0), np.asarray(U)
    got = (U.sum((1, 2)) - U0.sum((1, 2))) / (n * n)
    M0, E0 = U0[0].mean(), U0[3].mean()
    return float((np.abs(got - acc) / np.array([M0, np.sqrt(M0 * E0), np.sqrt(M0 * E0), E0])).max())


def _b(d, *keys):
    """The bound at d[keys...]."""
    for k in keys:
        d = d[k]
    return d


def _order(ns, vals):
    return float(-np.polyfit(np.log(ns), np.log(vals), 1)[0])


CASES = [(c, s) for c in (3, 6) for s in SCHEMES]
IDS = [f"c{c}-{s}" for c, s in CASES]


def _ladder(config, scheme, ns, tier):
    """Every per-N metric against its bound, and the fitted L1 orders over ns."""
    key, l1s, sl1s = (config, scheme), [], {name: [] for name in STRIPS[config]}
    for n in ns:
        tag = f"KT{config} {scheme} N={n}"
        U = run(config, n, scheme)[1]
        errs = ref_errors(U[0], config)
        l1s.append(errs[3])
        if config == 3:
            check(f"{tag} L1 outside the core rho vs AthenaK (M={M})", ref_errors(U[0], 3, ref=athenak()["rho"])[3],
                  AK_OUT[scheme][n])
        for k, what in enumerate(("L1", "L2", "L1 core", "L1 outside the core")):
            check(f"{tag} {what} rho (M={M})", errs[k], _b(BOUNDS, key, n, k))
        for name, (sl1, cells) in strip_errors(U[0], config).items():
            sl1s[name].append(sl1)
            check(f"{tag} strip {name} L1 rho vs exact 1D", sl1, _b(STRIP_BOUNDS, key, n, name))
            check(f"{tag} strip {name} front offset (cells)", cells, _b(FRONT_CELLS, config))
        if config == 3:
            check(f"{tag} diagonal asymmetry", asymmetry(U), _b(SYM, "fp64"))
        else:
            for name, pl1 in pressure_strips(U).items():
                check(f"{tag} strip {name} L1 |p - 1|", pl1, _b(P_STRIP_BOUNDS, scheme, n, name))
    o = _order(ns, l1s)
    check(f"KT{config} {scheme} L1 outside the core order {ns}", o, _b(ORDERS, tier, key, 0), lower=True)
    check(f"KT{config} {scheme} L1 outside the core order {ns}", o, _b(ORDERS, tier, key, 1))
    for name, v in sl1s.items():
        check(f"KT{config} {scheme} strip {name} L1 order {ns}", _order(ns, v), STRIP_ORDER[config],
              lower=True)


def test_reference():
    """The stored references: what they are, their self-convergence (8192^2 vs 4096^2) on M^2 blocks."""
    for config in (3, 6):
        check(f"KT{config} reference self-convergence L1 rho (M={M})", reference(config)[1][M],
              _b(REF_SELF_L1, config))
    # Config 6's slip lines are Kelvin-Helmholtz unstable: the 8192 reference is the laminar self-similar
    # solution only because the unperturbed run adds no seed (a 1e-15 seed rolls it up, 4.1e-3 outside the
    # core; critic_a11). A 4096 run seeded at 1e-12 stays laminar; a rolled-up reference would sit ~4e-3 from it.
    with np.load(reference_path("luna/kt_riemann/kt_riemann_c6_mc.npz")) as z:
        seeded = np.asarray(z["seeded_rho"], np.float64)
        eps, n = float(z["seeded_eps"]), int(z["seeded_n"])
    check(f"KT6 reference vs {n}^2 seeded {eps:g}, L1 rho outside the core (M={M}) [laminar branch]",
          ref_errors(seeded, 6)[3], LAMINAR_GUARD)


@pytest.mark.parametrize("config,scheme", CASES, ids=IDS)
def test_ladder(config, scheme):
    """L1/L2 of rho vs the reference, strips vs exact 1D, config 3's symmetry; CPU rungs 256, 512."""
    _ladder(config, scheme, CPU_NS, "cpu")


@gpu_only
@pytest.mark.parametrize("config,scheme", CASES, ids=IDS)
def test_ladder_gpu(config, scheme):
    """As test_ladder on 256..2048, orders fitted over the four rungs."""
    _ladder(config, scheme, GPU_NS, "gpu")


@pytest.mark.parametrize("config", (3, 6))
def test_wall_budget(config):
    """Totals change by the exact wall fluxes of the four 1D interface solutions, to O(h)."""
    for n in CPU_NS:
        U0, U = run(config, n, t=T_BUDGET[config])
        e = budget_errors(U0, U, config, T_BUDGET[config]).max()
        check(f"KT{config} N={n} t={T_BUDGET[config]} wall budget max rel error", e, _b(BUDGET, config, n))


@pytest.mark.parametrize("config", (3, 6))
def test_flux_balance(config):
    """The totals change by exactly the code's own boundary face fluxes (round-off)."""
    check(f"KT{config} N=256 boundary-flux balance", flux_balance(config, 256), _b(BALANCE, config))


@pytest.mark.parametrize("prec", ["fp64", pytest.param("fp32", marks=pytest.mark.precision("fp32"))])
def test_symmetry(prec):
    """Config 3 is symmetric about x = y; LUNA keeps it to the bound in both precisions."""
    for n in CPU_NS:
        check(f"KT3 {prec} N={n} diagonal asymmetry", asymmetry(run(3, n)[1]), _b(SYM, prec))


@pytest.mark.precision("fp32")
@pytest.mark.parametrize("config", (3, 6))
def test_fp32(config):
    """FP32 (the production default) gives FP64's reference L1 at 512 to the bound."""
    l1_32 = ref_errors(np.asarray(run(config, 512)[1][0], np.float64), config)[0]
    check(f"KT{config} N=512 FP32 |L1 rho / FP64 - 1|", abs(l1_32 / _b(FP64_L1_512, config) - 1.0),
          _b(FP32, config))


@pytest.mark.parametrize("config", (3, 6))
def test_sharded_equals_one_device(config):
    """A 2 x 2 mesh gives the one-device run."""
    import jax
    if jax.device_count() < 4:
        pytest.skip("needs 4 devices")
    one = run(config, 256)[1]
    sh = run(config, 256, mesh=(2, 2))[1]
    check(f"KT{config} N=256 2x2 shards vs one device", float(np.abs(sh - one).max() / np.abs(one).max()),
          _b(SHARD, config))


def test_ic_refuses():
    """Refuse, don't fall back: gamma != 1.4, a split inside a cell, an unknown configuration."""
    base = dict(test_problem="KT-Riemann-2D", SPATIAL_DIMENSION=2, Lx=1.0, Ly=1.0, BOUNDARY=1)
    for kw, msg in ((dict(Nx=64, Ny=64, gamma=5 / 3, problem=ktic.Params()), "gamma"),
                    (dict(Nx=63, Ny=64, gamma=G, problem=ktic.Params()), "face"),
                    (dict(Nx=64, Ny=64, gamma=G, problem=ktic.Params(config=4)), "config")):
        with pytest.raises(ValueError, match=msg):
            ktic.kt_riemann_2d(accuracy_cfg(**base, **kw))


# ---------------------------------------------------------------- teeth: a broken run fails each bound
def _mod(config, q, k, value):
    """KT's states with quadrant q's variable k set to value, as run()'s `states`."""
    s = {n: list(v) for n, v in ktic.STATES[config].items()}
    s[q][k] = value
    return tuple((n, tuple(v)) for n, v in s.items())


def test_teeth_first_order():
    """ORDER=0 fails the minmod L1 bound at 512 (both configs)."""
    for config in (3, 6):
        l1 = ref_errors(run(config, 512, order=0)[1][0], config)[0]
        print(f"MEASURED KT{config} N=512 first order L1 rho = {l1:.6e}")
        assert l1 > _b(BOUNDS, (config, "minmod"), 512, 0)


def test_teeth_states():
    """A digit typo (config 3 rho3 = 0.183) fails L1 and the 34/32 strips; config 6 with q2 and q4 swapped, or
    v -> -v (counter-clockwise), fails L1; config 6 with p4 = 1.01 fails the strip-34 pressure bound."""
    n = 512
    U = run(3, n, states=_mod(3, 3, 0, 0.183))[1]
    l1 = ref_errors(U[0], 3)[0]
    print(f"MEASURED KT3 rho3 typo L1 rho = {l1:.6e}")
    assert l1 > _b(BOUNDS, (3, "minmod"), n, 0)
    s = strip_errors(U[0], 3)
    for name in ("34", "32"):
        print(f"MEASURED KT3 rho3 typo strip {name} L1 = {s[name][0]:.6e}")
        assert s[name][0] > _b(STRIP_BOUNDS, (3, "minmod"), n, name)
    q = ktic.STATES[6]
    swapped = tuple((k, q[{2: 4, 4: 2}.get(k, k)]) for k in q)
    mirrored = tuple((k, (r, u, -v, p)) for k, (r, u, v, p) in q.items())
    for name, st in (("q2 <-> q4", swapped), ("v -> -v", mirrored)):
        l1 = ref_errors(run(6, n, states=st)[1][0], 6)[0]
        print(f"MEASURED KT6 {name} L1 rho = {l1:.6e}")
        assert l1 > _b(BOUNDS, (6, "minmod"), n, 0)
    ps = pressure_strips(run(6, n, states=_mod(6, 4, 3, 1.01))[1])
    print(f"MEASURED KT6 p4 = 1.01 strip 34 L1 |p - 1| = {ps['34']:.6e}")
    assert ps["34"] > _b(P_STRIP_BOUNDS, "minmod", n, "34")


def test_teeth_geometry():
    """The split one cell off x = 1/2 fails config 3's symmetry and the 21/34 front positions; gamma 5/3 fails
    the strips; diode walls (BOUNDARY=4) fail the wall budget."""
    n = 512
    U = run(3, n, xc=0.5 + 1.0 / n)[1]
    print(f"MEASURED KT3 split +1 cell asymmetry = {asymmetry(U):.6e}")
    assert asymmetry(U) > _b(SYM, "fp64")
    s = strip_errors(U[0], 3)
    for name in ("21", "34"):
        print(f"MEASURED KT3 split +1 cell strip {name} front (cells) = {s[name][1]:.4f}")
        assert s[name][1] > _b(FRONT_CELLS, 3)
    s = strip_errors(run(3, n, gamma=5.0 / 3.0)[1][0], 3)
    for name, (sl1, cells) in s.items():
        print(f"MEASURED KT3 gamma 5/3 strip {name} L1 = {sl1:.6e}, front (cells) = {cells:.4f}")
        assert sl1 > _b(STRIP_BOUNDS, (3, "minmod"), n, name) and cells > _b(FRONT_CELLS, 3)
    for config in (3, 6):
        U0, U = run(config, 256, boundary=4, t=T_BUDGET[config])
        e = budget_errors(U0, U, config, T_BUDGET[config]).max()
        print(f"MEASURED KT{config} BOUNDARY=4 wall budget error = {e:.6e}")
        assert e > _b(BUDGET, config, 256)


# ---------------------------------------------------------------- external gate: AthenaK PPMX 8192^2, config 3
# {scheme: {N: L1 of rho outside the core}} against AthenaK PPMX 8192 on M^2, ~1.3x measured (A100 FP64, job
# 31699124; the CPU rungs 256, 512 check the same bounds). For scale, AthenaK's own L1 there at 256..2048: PLM 1.62e-3,
# 7.38e-4, 3.66e-4, 2.25e-4; PPMX 1.36e-3, 5.79e-4, 3.03e-4, 1.85e-4; EPS_REF 1.59e-4 (so at 2048 the reference's own
# error is ~70 % of every rung's distance: the 2048 bound mostly bounds the two codes' common limit).
AK_OUT = {"minmod": {256: 0.0019, 512: 0.00096, 1024: 0.00053, 2048: 0.00032},
          # measured 1.455e-3, 7.323e-4, 4.046e-4, 2.434e-4
          "mc": {256: 0.0019, 512: 0.00095, 1024: 0.0005, 2048: 0.00031}}
          # measured 1.445e-3, 7.240e-4, 3.808e-4, 2.346e-4
EPS_REF_BOUND = 2.1e-4      # max(AthenaK 4096 vs 8192 1.59e-4, vs LUNA 8192 1.52e-4) outside the core, on 128^2


@functools.lru_cache(maxsize=None)
def athenak():
    """AthenaK PPMX 8192^2 config-3 rho on 256^2, its own error outside the core, AthenaK PLM's L1 there per N."""
    kw = dict(reconstruct="ppmx", integrator="rk4", rsolver="hllc", cfl_number=0.4, nx1=8192, nx2=8192, tlim=T,
              gamma=G, dfloor="1e-08", pfloor="1e-08")
    z, _ = load_athenak("kt_riemann/a11_c3_ppmx_rk4_8192.npz", "a11_ppmx_rk4_c3_N8192", **kw)
    plm, _ = load_athenak("kt_riemann/a11_c3_plm_rk2_2048.npz", "a11_plm_rk2_c3_N2048", **(kw | dict(
        reconstruct="plm", integrator="rk2", nx1=2048, nx2=2048)))
    assert float(z["c3_ref_N"]) == 8192
    eps = max(float(z["c3_ref_self_L1_out_4096_8192"]), float(z["c3_ref_vs_luna_L1_out"]))
    check("KT3 AthenaK reference own error EPS_REF (L1 rho outside the core, M=128)", eps, EPS_REF_BOUND)
    return dict(rho=np.asarray(z["c3_ref_rho256"], np.float64), eps=eps,
                plm={n: float(plm[f"c3_N{n}_L1_out_ak"]) for n in GPU_NS})


@gpu_only
def test_vanleer_matches_athenak_plm():
    """Matched method, config 3 outside the core: LUNA van Leer at N within AthenaK PLM's L1 at N + 2 EPS_REF
    (measured 1.36e-3, 6.98e-4, 3.77e-4, 2.29e-4 against 1.94e-3, 1.06e-3, 6.85e-4, 5.44e-4; at 1024 and 2048 LUNA
    sits 3 % and 2 % above AthenaK PLM, inside the reference's uncertainty)."""
    ak = athenak()
    for n in GPU_NS:
        l1 = ref_errors(run(3, n, "vl")[1][0], 3, ref=ak["rho"])[3]
        check(f"KT3 vl N={n} L1 outside the core rho vs AthenaK (AthenaK PLM {ak['plm'][n]:.4e} + 2 EPS_REF)", l1,
              ak["plm"][n] + 2 * ak["eps"])


def test_athenak_bounds_catch_first_order():
    """Teeth: first order (ORDER=0) config 3 at 512 misses the matched-method and the MC bound vs AthenaK (measured
    1.18e-3 > 1.06e-3, 9.5e-4)."""
    ak = athenak()
    l1 = ref_errors(run(3, 512, "vl", order=0)[1][0], 3, ref=ak["rho"])[3]
    print(f"MEASURED KT3 first order N=512 L1 outside the core rho vs AthenaK = {l1:.6e}")
    assert l1 > ak["plm"][512] + 2 * ak["eps"], l1
    assert l1 > AK_OUT["mc"][512], l1
