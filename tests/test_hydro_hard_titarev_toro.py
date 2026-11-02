"""Titarev-Toro: rho against AthenaK PPMX 65536 cells (the external gate), LUNA's van Leer scheme against AthenaK's
van Leer PLM at the same N, and LUNA's own fine run (regression only).

External gate. Reference athenak/titarev_toro/tt_ppmx_rk4_65536.npz (luna-reference-solutions): AthenaK 886dd2a hydro
kernels + luna_ic pgen (source c52f88f0), binary build_gpu_lunaic sha256 e40b2abd, PPMX + RK4 + HLLC, CFL 0.4,
dfloor = pfloor = 1e-8, FP64, A100, SLURM 31699055 (32768), 31699056 (65536), from the test's own IC
(tools/athenak_refs.py prep/reduce tt; the fixture asserts the scheme from the stored athinput). Its own error
EPS_REF is the larger of its distance to the AthenaK run at half its N and to LUNA MC 65536 (on 8192 cells). Bounds
below are ~1.3x LUNA's measured L1/L2 against it, with EPS_REF and AthenaK's own numbers at the same N alongside.
Matched method. AthenaK PLM is van Leer's harmonic limiter on primitives + RK2 + HLLC (athenak src/reconstruct/
plm.hpp:28-31); LUNA LIMITER=2 is the same limiter in MUSCL-Hancock. "LUNA van Leer is no less accurate than
AthenaK PLM at the same N" gives, by the triangle inequality, L1(LUNA, ref) <= L1(AthenaK PLM, ref) + 2 EPS_REF;
AthenaK PLM's runs at the test's N are athenak/titarev_toro/tt_plm_rk2_8192.npz. No number in that bound is LUNA's.
Regression only: luna/titarev_toro/titarev_toro_mc.npz (tools/hydro_reference.py): LUNA HLLC-MC 65536 cells on 8192, git
8a78a8f (2026-10-06); self-convergence 2.18e-3 (checked by the fixture).
Those bounds pin LUNA's convergence towards its own limit."""

import functools

import numpy as np
import pytest

from luna.reference_data import reference_path
from tests._athenak import load as load_athenak
from tests._helpers import accuracy_cfg, check, evolve

REF = reference_path("luna/titarev_toro/titarev_toro_mc.npz")  # tools/hydro_reference.py
T, LX = 5.0, 10.0
NS = (1024, 2048, 4096, 8192)
# Regression only: {(solver, limiter): {N: (L1, L2) of rho}} ~1.3x measured vs LUNA's reference. Below the threshold the
# waves are damped (L1 ~0.46): MC resolves them between 2048 and 4096, minmod beyond. No order bound.
BOUNDS = {
    ("hllc", 0): {1024: (0.63, 0.30), 2048: (0.59, 0.29), 4096: (0.46, 0.24), 8192: (0.22, 0.12)},
    # minmod measured (0.484, 0.229), (0.457, 0.223), (0.357, 0.184), (0.167, 0.0942)
    ("hllc", 1): {1024: (0.61, 0.29), 2048: (0.47, 0.24), 4096: (0.142, 0.076), 8192: (0.041, 0.022)},
    # MC measured (0.466, 0.225), (0.363, 0.186), (0.110, 0.0585), (0.0312, 0.0168)
    ("hll", 0): {1024: (0.79, 0.33), 2048: (0.81, 0.34), 4096: (0.79, 0.33), 8192: (0.59, 0.26)},
    # HLL minmod measured (0.609, 0.251), (0.623, 0.258), (0.611, 0.256), (0.455, 0.201)
}
# HLL has no contact wave, so it damps the entropy waves: still unresolved at 8192 (HLLC minmod 0.167)
HLL_UNRESOLVED_L1 = 0.35
IDS = {("hllc", 0): "minmod", ("hllc", 1): "mc", ("hll", 0): "hll-minmod"}
# No flux budget: behind the Mach 1.3 shock u - c = -0.77, so interaction waves reach the inflow.
REF_SELF_L1 = 3e-3     # self-convergence L1(rho), 2^15 vs 2^16 cells; measured 2.18e-3



@pytest.fixture(scope="module")
def ref():
    """The stored reference density on 8192 cells, after checking what it is and how good."""
    with np.load(REF) as z:
        assert str(z["problem"]) == "Titarev-Toro" and float(z["gamma"]) == 1.4 and float(z["t"]) == T
        assert str(z["riemann_solver"]) == "hllc" and str(z["precision"]) == "fp64"
        check("Titarev-Toro reference self-convergence L1(rho)", float(z["self_l1_rho"]), REF_SELF_L1)
        return np.asarray(z["rho"])


@functools.lru_cache(maxsize=None)
def run(n, limiter, t=T, solver="hllc", order=1):
    """Final state on n cells at t (cached: the LUNA-reference and the AthenaK checks share the runs)."""
    from luna.ic_generation.ic import initial_conditions
    cfg = accuracy_cfg(test_problem="Titarev-Toro", SPATIAL_DIMENSION=1, Nx=n, Lx=LX, BOUNDARY=1,
                       gamma=1.4, LIMITER=limiter, riemann_solver=solver, ORDER=order)
    return np.asarray(evolve(cfg, np.asarray(initial_conditions(cfg)), t)[0])


def errors(n, limiter, ref, t=T, solver="hllc", order=1):
    """(L1, L2) of rho on n cells against the reference averaged onto them."""
    U = run(n, limiter, t, solver, order)
    e, h = U[0] - ref.reshape(n, -1).mean(axis=1), LX / n
    return float(h * np.abs(e).sum()), float(np.sqrt(h * (e ** 2).sum()))


@pytest.mark.parametrize("solver,limiter", list(BOUNDS), ids=[IDS[k] for k in BOUNDS])
def test_titarev_toro_ladder(solver, limiter, ref):
    """Regression only (LUNA's own reference): L1 and L2 of rho at every N; HLL stays unresolved at the finest N."""
    for n in NS:
        l1, l2 = errors(n, limiter, ref, solver=solver)
        check(f"Titarev-Toro {solver} lim{limiter} N={n} L1 rho", l1, BOUNDS[solver, limiter][n][0])
        check(f"Titarev-Toro {solver} lim{limiter} N={n} L2 rho", l2, BOUNDS[solver, limiter][n][1])
    if solver == "hll":
        check(f"Titarev-Toro hll N={NS[-1]} L1 rho (unresolved)", l1, HLL_UNRESOLVED_L1, lower=True)


def test_bound_catches_a_short_run(ref, ak):
    """Teeth: MC at 4096 stopped at 0.98 t misses its L1 bounds against both references (measured 0.676, 0.675)."""
    l1, l1_ak = errors(4096, 1, ref, t=0.98 * T)[0], errors(4096, 1, ak["rho"], t=0.98 * T)[0]
    print(f"MEASURED Titarev-Toro MC N=4096 0.98 t L1 rho vs LUNA ref {l1:.6e}, vs AthenaK {l1_ak:.6e}")
    assert l1 > BOUNDS["hllc", 1][4096][0], f"L1 rho {l1:.4e} of a short run passes the bound"
    assert l1_ak > AK_BOUNDS["hllc", 1][4096][0], f"L1 rho vs AthenaK {l1_ak:.4e} of a short run passes the bound"


# ---------------------------------------------------------------- external gate: AthenaK PPMX 65536
# {(solver, limiter): {N: (L1, L2) of rho}} against AthenaK PPMX 65536, ~1.3x measured (CPU-8, FP64, job 31699150).
# For scale, AthenaK's own L1 against it at 1024..8192: PLM 0.482, 0.436, 0.244, 0.0748; PPMX 0.425, 0.0782, 0.0445,
# 0.0171; EPS_REF 2.17e-3 (32768 vs 65536; vs LUNA's 2.09e-3).
AK_BOUNDS = {
    ("hllc", 0): {1024: (0.63, 0.3), 2048: (0.6, 0.3), 4096: (0.47, 0.24), 8192: (0.22, 0.13)},
    # minmod measured (0.484, 0.229), (0.458, 0.223), (0.357, 0.184), (0.168, 0.0942)
    ("hllc", 1): {1024: (0.61, 0.3), 2048: (0.48, 0.25), 4096: (0.15, 0.076), 8192: (0.042, 0.022)},
    # MC measured (0.467, 0.225), (0.364, 0.186), (0.110, 0.0585), (0.0317, 0.0166)
    ("hllc", 2): {1024: (0.62, 0.3), 2048: (0.54, 0.27), 4096: (0.25, 0.14), 8192: (0.067, 0.037)},
    # van Leer measured (0.474, 0.227), (0.411, 0.206), (0.187, 0.100), (0.0513, 0.0279)
    ("hll", 0): {1024: (0.8, 0.33), 2048: (0.81, 0.34), 4096: (0.8, 0.34), 8192: (0.6, 0.27)},
    # HLL minmod measured (0.609, 0.252), (0.623, 0.258), (0.611, 0.256), (0.455, 0.201)
}
AK_IDS = {**IDS, ("hllc", 2): "vanleer"}
EPS_REF_BOUND = 2.9e-3                      # the reference's own error; measured 2.17e-3


@pytest.fixture(scope="module")
def ak():
    """AthenaK PPMX 65536 rho, its own error EPS_REF, and AthenaK PLM's L1 against it at the test's N."""
    kw = dict(reconstruct="ppmx", integrator="rk4", rsolver="hllc", cfl_number=0.4, nx1=65536, tlim=T, gamma=1.4,
              dfloor="1e-08", pfloor="1e-08")
    z, _ = load_athenak("titarev_toro/tt_ppmx_rk4_65536.npz", "tt_ppmx_rk4_N65536", **kw)
    plm, _ = load_athenak("titarev_toro/tt_plm_rk2_8192.npz", "tt_plm_rk2_N8192", **(kw | dict(
        reconstruct="plm", integrator="rk2", nx1=8192)))
    assert int(z["ref_N"]) == 65536 and float(z["ref_nonfinite"]) == 0.0
    print(f"MEASURED Titarev-Toro AthenaK 65536 vs {int(z['ref_self_N'])} L1 rho = {float(z['ref_L1_self']):.6e}, "
          f"vs LUNA MC 65536 on 8192 = {float(z['ref_L1_vs_luna_at8192']):.6e}")
    eps = max(float(z["ref_L1_self"]), float(z["ref_L1_vs_luna_at8192"]))
    check("Titarev-Toro AthenaK reference own error EPS_REF (L1 rho)", eps, EPS_REF_BOUND)
    return dict(rho=np.asarray(z["ref_rho"], np.float64), eps=eps, plm={n: float(plm[f"N{n}_L1"]) for n in NS})


@pytest.mark.parametrize("solver,limiter", list(AK_BOUNDS), ids=[AK_IDS[k] for k in AK_BOUNDS])
def test_titarev_toro_athenak(solver, limiter, ak):
    """External gate: L1, L2 of rho at every N against AthenaK PPMX 65536 averaged onto LUNA's cells."""
    for n in NS:
        l1, l2 = errors(n, limiter, ak["rho"], solver=solver)
        check(f"Titarev-Toro {solver} lim{limiter} N={n} L1 rho vs AthenaK", l1, AK_BOUNDS[solver, limiter][n][0])
        check(f"Titarev-Toro {solver} lim{limiter} N={n} L2 rho vs AthenaK", l2, AK_BOUNDS[solver, limiter][n][1])


def test_vanleer_matches_athenak_plm(ak):
    """Matched method: LUNA van Leer at N is within AthenaK PLM's L1 at N + 2 EPS_REF of the AthenaK reference
    (measured 0.474, 0.410, 0.187, 0.0513 against bounds 0.487, 0.441, 0.249, 0.0791)."""
    for n in NS:
        l1 = errors(n, 2, ak["rho"])[0]
        check(f"Titarev-Toro vanleer N={n} L1 rho vs AthenaK (AthenaK PLM {ak['plm'][n]:.4e} + 2 EPS_REF)", l1,
              ak["plm"][n] + 2 * ak["eps"])


def test_matched_bound_catches_first_order(ak):
    """Teeth: LUNA first order (ORDER=0, Godunov) at 4096 misses the matched-method bound (measured 0.490 > 0.249)."""
    l1 = errors(4096, 2, ak["rho"], order=0)[0]
    print(f"MEASURED Titarev-Toro first order N=4096 L1 rho vs AthenaK = {l1:.6e}")
    assert l1 > ak["plm"][4096] + 2 * ak["eps"], l1
