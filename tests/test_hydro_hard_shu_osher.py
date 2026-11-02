"""Shu-Osher: rho against AthenaK PPMX 16384 cells (the external gate), LUNA's van Leer scheme against AthenaK's
van Leer PLM at the same N, and LUNA's own fine run (regression only).

External gate. Reference athenak/shu_osher/so_ppmx_rk4_16384.npz (luna-reference-solutions): AthenaK 886dd2a hydro
kernels + luna_ic pgen (source c52f88f0), binary build_gpu_lunaic sha256 e40b2abd, PPMX + RK4 + HLLC, CFL 0.4,
dfloor = pfloor = 1e-8, FP64, A100, SLURM 31699054, from the test's own IC (tools/athenak_refs.py prep/reduce so;
the fixture asserts the scheme from the stored athinput). Its own error EPS_REF is the larger of its distance to the
AthenaK run at half its N and to LUNA MC 65536 (on 8192 cells). Bounds below are ~1.3x LUNA's measured L1/L2
against it, with EPS_REF and AthenaK's own numbers at the same N alongside.
Matched method. AthenaK PLM is van Leer's harmonic limiter on primitives + RK2 + HLLC (athenak src/reconstruct/
plm.hpp:28-31); LUNA LIMITER=2 is the same limiter in MUSCL-Hancock. "LUNA van Leer is no less accurate than
AthenaK PLM at the same N" gives, by the triangle inequality, L1(LUNA, ref) <= L1(AthenaK PLM, ref) + 2 EPS_REF;
AthenaK PLM's runs at the test's N are athenak/shu_osher/so_plm_rk2_2048.npz. No number in that bound is LUNA's.
Regression only: luna/shu_osher/shu_osher_mc.npz (tools/hydro_reference.py): LUNA HLLC-MC 65536 cells on 8192, git
b7b0d35 (2026-10-06); self-convergence 1.07e-3 (checked by the fixture).
Those bounds pin LUNA's convergence towards its own limit."""

import functools

import numpy as np
import pytest

from luna.reference_data import reference_path
from tests._athenak import load as load_athenak
from luna.ic_generation.shu_osher import LEFT
from tests._helpers import accuracy_cfg, check, check_tube_budget, evolve

REF = reference_path("luna/shu_osher/shu_osher_mc.npz")  # tools/hydro_reference.py
T, LX = 1.8, 10.0
NS = (256, 512, 1024, 2048)
# Regression only: {(solver, limiter): {N: (L1, L2) of rho}} ~1.3x measured vs LUNA's reference (CPU, FP64).
# The wave train behind the Mach 3 shock is damped out at low N; MC resolves it between 256 and 512 cells, minmod
# between 512 and 1024 (Titarev-Toro: 2048-4096 and beyond 4096). No order bound: not asymptotic.
BOUNDS = {
    ("hllc", 0): {256: (0.91, 0.68), 512: (0.62, 0.5), 1024: (0.28, 0.25), 2048: (0.116, 0.124)},
    # minmod measured (0.702, 0.519), (0.477, 0.385), (0.219, 0.195), (0.0892, 0.0954)
    ("hllc", 1): {256: (0.79, 0.6), 512: (0.32, 0.25), 1024: (0.117, 0.116), 2048: (0.048, 0.066)},
    # MC measured (0.607, 0.461), (0.247, 0.193), (0.0898, 0.0895), (0.0367, 0.0510)
    ("hll", 0): {256: (0.99, 0.68), 512: (0.65, 0.5), 1024: (0.3, 0.26), 2048: (0.12, 0.13)},
    # HLL minmod measured (0.761, 0.522), (0.503, 0.386), (0.229, 0.197), (0.0925, 0.0963)
}
IDS = {("hllc", 0): "minmod", ("hllc", 1): "mc", ("hll", 0): "hll-minmod"}
REST = (1.0, 0.0, 1.0)   # ahead of the shock: at rest, p = 1, so its flux (0, 1, 0) does not see the wave
REF_SELF_L1 = 1.5e-3   # self-convergence L1(rho), 2^15 vs 2^16 cells; measured 1.07e-3



@pytest.fixture(scope="module")
def ref():
    """The stored reference density on 8192 cells, after checking what it is and how good."""
    with np.load(REF) as z:
        assert str(z["problem"]) == "Shu-Osher" and float(z["gamma"]) == 1.4 and float(z["t"]) == T
        assert str(z["riemann_solver"]) == "hllc" and str(z["precision"]) == "fp64"
        check("Shu-Osher reference self-convergence L1(rho)", float(z["self_l1_rho"]), REF_SELF_L1)
        return np.asarray(z["rho"])


@functools.lru_cache(maxsize=None)
def run(n, limiter, t=T, solver="hllc", order=1):
    """(U0, U) on n cells at t (cached: the LUNA-reference and the AthenaK checks share the runs)."""
    from luna.ic_generation.ic import initial_conditions
    cfg = accuracy_cfg(test_problem="Shu-Osher", SPATIAL_DIMENSION=1, Nx=n, Lx=LX, BOUNDARY=1,
                       gamma=1.4, LIMITER=limiter, riemann_solver=solver, ORDER=order)
    U0 = np.asarray(initial_conditions(cfg))
    return U0, np.asarray(evolve(cfg, U0, t)[0])


def errors(n, limiter, ref, t=T, solver="hllc", order=1):
    """(L1, L2) of rho on n cells against the reference averaged onto them; the exact budget checked
    (supersonic inflow, the shock inside the box)."""
    U0, U = run(n, limiter, t, solver, order)
    e, h = U[0] - ref.reshape(n, -1).mean(axis=1), LX / n
    check_tube_budget(f"Shu-Osher {solver} lim{limiter} ORDER={order} N={n}", U0, U, LEFT, REST, t, h, 1.4)
    return float(h * np.abs(e).sum()), float(np.sqrt(h * (e ** 2).sum()))


@pytest.mark.parametrize("solver,limiter", list(BOUNDS), ids=[IDS[k] for k in BOUNDS])
def test_shu_osher_ladder(solver, limiter, ref):
    """Regression only (LUNA's own reference): L1 and L2 of rho at every N, and the budget."""
    for n in NS:
        l1, l2 = errors(n, limiter, ref, solver=solver)
        check(f"Shu-Osher {solver} lim{limiter} N={n} L1 rho", l1, BOUNDS[solver, limiter][n][0])
        check(f"Shu-Osher {solver} lim{limiter} N={n} L2 rho", l2, BOUNDS[solver, limiter][n][1])


def test_bound_catches_a_short_run(ref, ak):
    """Teeth: MC at 1024 stopped at 0.98 t misses its L1 bounds against both references (measured 1.41, 1.41)."""
    l1, l1_ak = errors(1024, 1, ref, t=0.98 * T)[0], errors(1024, 1, ak["rho"], t=0.98 * T)[0]
    print(f"MEASURED Shu-Osher MC N=1024 0.98 t L1 rho vs LUNA ref {l1:.6e}, vs AthenaK {l1_ak:.6e}")
    assert l1 > BOUNDS["hllc", 1][1024][0], f"L1 rho {l1:.4e} of a short run passes the bound"
    assert l1_ak > AK_BOUNDS["hllc", 1][1024][0], f"L1 rho vs AthenaK {l1_ak:.4e} of a short run passes the bound"


# ---------------------------------------------------------------- external gate: AthenaK PPMX 16384
# {(solver, limiter): {N: (L1, L2) of rho}} against AthenaK PPMX 16384, ~1.3x measured (CPU-8, FP64, job 31699149).
# For scale, AthenaK's own L1 against it at 256..2048: PLM 0.725, 0.482, 0.198, 0.0791; PPMX 0.470, 0.217, 0.104,
# 0.0484; EPS_REF 9.0e-3 (8192 vs 16384; vs LUNA's 6.3e-3), 26 % of MC's 2048 value.
AK_BOUNDS = {
    ("hllc", 0): {256: (0.92, 0.68), 512: (0.62, 0.51), 1024: (0.29, 0.26), 2048: (0.12, 0.13)},
    # minmod measured (0.702, 0.520), (0.477, 0.386), (0.218, 0.195), (0.0875, 0.0937)
    ("hllc", 1): {256: (0.79, 0.61), 512: (0.33, 0.26), 1024: (0.12, 0.12), 2048: (0.045, 0.062)},
    # MC measured (0.607, 0.462), (0.247, 0.193), (0.0878, 0.0870), (0.0339, 0.0470)
    ("hllc", 2): {256: (0.83, 0.63), 512: (0.42, 0.34), 1024: (0.14, 0.14), 2048: (0.057, 0.075)},
    # van Leer measured (0.638, 0.481), (0.317, 0.255), (0.107, 0.104), (0.0436, 0.0570)
    ("hll", 0): {256: (0.99, 0.68), 512: (0.66, 0.51), 1024: (0.3, 0.26), 2048: (0.12, 0.13)},
    # HLL minmod measured (0.761, 0.523), (0.503, 0.387), (0.228, 0.196), (0.0908, 0.0946)
}
AK_IDS = {**IDS, ("hllc", 2): "vanleer"}
EPS_REF_BOUND = 0.012                       # the reference's own error; measured 9.01e-3


@pytest.fixture(scope="module")
def ak():
    """AthenaK PPMX 16384 rho, its own error EPS_REF, and AthenaK PLM's L1 against it at the test's N."""
    kw = dict(reconstruct="ppmx", integrator="rk4", rsolver="hllc", cfl_number=0.4, nx1=16384, tlim=T, gamma=1.4,
              dfloor="1e-08", pfloor="1e-08")
    z, _ = load_athenak("shu_osher/so_ppmx_rk4_16384.npz", "so_ppmx_rk4_N16384", **kw)
    plm, _ = load_athenak("shu_osher/so_plm_rk2_2048.npz", "so_plm_rk2_N2048", **(kw | dict(
        reconstruct="plm", integrator="rk2", nx1=2048)))
    assert int(z["ref_N"]) == 16384 and float(z["ref_nonfinite"]) == 0.0
    print(f"MEASURED Shu-Osher AthenaK 16384 vs {int(z['ref_self_N'])} L1 rho = {float(z['ref_L1_self']):.6e}, "
          f"vs LUNA MC 65536 on 8192 = {float(z['ref_L1_vs_luna_at8192']):.6e}")
    eps = max(float(z["ref_L1_self"]), float(z["ref_L1_vs_luna_at8192"]))
    check("Shu-Osher AthenaK reference own error EPS_REF (L1 rho)", eps, EPS_REF_BOUND)
    return dict(rho=np.asarray(z["ref_rho"], np.float64), eps=eps, plm={n: float(plm[f"N{n}_L1"]) for n in NS})


@pytest.mark.parametrize("solver,limiter", list(AK_BOUNDS), ids=[AK_IDS[k] for k in AK_BOUNDS])
def test_shu_osher_athenak(solver, limiter, ak):
    """External gate: L1, L2 of rho at every N against AthenaK PPMX 16384 averaged onto LUNA's cells."""
    for n in NS:
        l1, l2 = errors(n, limiter, ak["rho"], solver=solver)
        check(f"Shu-Osher {solver} lim{limiter} N={n} L1 rho vs AthenaK", l1, AK_BOUNDS[solver, limiter][n][0])
        check(f"Shu-Osher {solver} lim{limiter} N={n} L2 rho vs AthenaK", l2, AK_BOUNDS[solver, limiter][n][1])


def test_vanleer_matches_athenak_plm(ak):
    """Matched method: LUNA van Leer at N is within AthenaK PLM's L1 at N + 2 EPS_REF of the AthenaK reference
    (measured 0.638, 0.317, 0.107, 0.0436 against bounds 0.743, 0.500, 0.216, 0.0971)."""
    for n in NS:
        l1 = errors(n, 2, ak["rho"])[0]
        check(f"Shu-Osher vanleer N={n} L1 rho vs AthenaK (AthenaK PLM {ak['plm'][n]:.4e} + 2 EPS_REF)", l1,
              ak["plm"][n] + 2 * ak["eps"])


def test_matched_bound_catches_first_order(ak):
    """Teeth: LUNA first order (ORDER=0, Godunov) at 1024 misses the matched-method bound (measured 0.641 > 0.216)."""
    l1 = errors(1024, 2, ak["rho"], order=0)[0]
    print(f"MEASURED Shu-Osher first order N=1024 L1 rho vs AthenaK = {l1:.6e}")
    assert l1 > ak["plm"][1024] + 2 * ak["eps"], l1
