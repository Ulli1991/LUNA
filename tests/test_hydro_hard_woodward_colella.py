"""Woodward-Colella (reflecting walls, T = 0.038): rho against AthenaK PPMX 16384 cells (the external gate), LUNA's
van Leer scheme against AthenaK's van Leer PLM at the same N, and LUNA's own fine run (regression only).

External gate. Reference athenak/woodward_colella/a9_ppmx_rk4_16384.npz (luna-reference-solutions): AthenaK 886dd2a
hydro kernels + luna_ic pgen (source c52f88f0), binary build_gpu_lunaic sha256 e40b2abd, PPMX + RK4 + HLLC, CFL 0.4,
dfloor = pfloor = 1e-8, FP64, A100, SLURM 31107269, from LUNA's own IC (tools/athenak_refs.py prep/reduce a9; the
fixture asserts the scheme from the stored athinput). Its own error EPS_REF is the larger of its distance to AthenaK
PPM4 16384 and to LUNA MC 65536 (on 8192 cells); bounds below are ~1.3x LUNA's measured L1/L2 against it, with EPS_REF
and AthenaK's own numbers at the same N alongside.
Matched method. AthenaK PLM is van Leer's harmonic limiter on primitives + RK2 + HLLC (athenak src/reconstruct/
plm.hpp:28-31); LUNA LIMITER=2 is the same limiter in MUSCL-Hancock. The claim "LUNA van Leer is no less accurate
than AthenaK PLM at the same N" gives, by the triangle inequality, L1(LUNA, ref) <= L1(AK PLM, ref) + 2 EPS_REF, with
L1(AK PLM N, ref) <= L1(AK PLM N, AK PLM 16384) + L1(AK PLM 16384, ref) (the per-N runs are stored only against
AthenaK PLM 16384). No number in that bound comes from LUNA.
Regression only: luna/woodward_colella/woodward_colella_mc.npz (tools/hydro_reference.py): LUNA HLLC-MC 65536 cells
on 8192, git b7b0d35; self-convergence 1.53e-3. Those bounds pin LUNA's convergence towards its own limit."""

import numpy as np
import pytest

import functools

from luna.reference_data import reference_path
from tests._athenak import load as load_athenak
from tests._helpers import accuracy_cfg, check, check_conserved, evolve

REF = reference_path("luna/woodward_colella/woodward_colella_mc.npz")  # tools/hydro_reference.py
T = 0.038
NS = (256, 512, 1024, 2048, 4096)
# Regression only: {(solver, limiter): {N: (L1, L2) of rho}} ~1.3x measured vs LUNA's reference (CPU, FP64, BOUNDARY=2)
BOUNDS = {
    ("hllc", 0): {256: (0.23, 0.57), 512: (0.15, 0.41), 1024: (0.097, 0.29), 2048: (0.059, 0.22), 4096: (0.037, 0.17)},
    # minmod measured (0.178, 0.439), (0.117, 0.319), (0.0747, 0.225), (0.0456, 0.170), (0.0283, 0.129)
    ("hllc", 1): {256: (0.18, 0.48), 512: (0.10, 0.31), 1024: (0.057, 0.22), 2048: (0.033, 0.17), 4096: (0.019, 0.12)},
    # MC measured (0.137, 0.370), (0.0781, 0.239), (0.0439, 0.169), (0.0256, 0.127), (0.0145, 0.0919)
    ("hll", 0): {256: (0.25, 0.61), 512: (0.17, 0.45), 1024: (0.1, 0.31), 2048: (0.064, 0.23), 4096: (0.039, 0.17)},
    # HLL minmod measured (0.194, 0.473), (0.127, 0.347), (0.0808, 0.239), (0.0491, 0.177), (0.0303, 0.134)
}
# fitted over NS, measured -0.1 .. +0.3: minmod L1 0.667, L2 0.443; MC L1 0.809, L2 0.492; HLL minmod
# L1 0.673, L2 0.460
ORDER = {("hllc", 0): ((0.57, 0.97), (0.34, 0.74)), ("hllc", 1): ((0.71, 1.11), (0.39, 0.79)),
         ("hll", 0): ((0.57, 0.97), (0.36, 0.76))}
IDS = {("hllc", 0): "minmod", ("hllc", 1): "mc", ("hll", 0): "hll-minmod"}
REF_SELF_L1 = 2e-3     # self-convergence L1(rho), 2^15 vs 2^16 cells; measured 1.53e-3



@pytest.fixture(scope="module")
def ref():
    """The stored reference density on 8192 cells, after checking what it is and how good."""
    with np.load(REF) as z:
        assert str(z["problem"]) == "Woodward-Colella" and float(z["gamma"]) == 1.4 and float(z["t"]) == T
        assert str(z["riemann_solver"]) == "hllc" and str(z["precision"]) == "fp64"
        assert int(z["boundary"]) == 2 and int(z["limiter"]) == 1
        check("Woodward-Colella reference self-convergence L1(rho)", float(z["self_l1_rho"]), REF_SELF_L1)
        return np.asarray(z["rho"])


@functools.lru_cache(maxsize=None)
def run(n, limiter, t=T, solver="hllc", order=1):
    """(U0, U) on n cells at t (cached: the LUNA-reference and the AthenaK checks share the runs)."""
    from luna.ic_generation.ic import initial_conditions
    cfg = accuracy_cfg(test_problem="Woodward-Colella", SPATIAL_DIMENSION=1, Nx=n, Lx=1.0, BOUNDARY=2,
                       gamma=1.4, LIMITER=limiter, riemann_solver=solver, ORDER=order)
    U0 = np.asarray(initial_conditions(cfg))
    return U0, evolve(cfg, U0, t)[0]


def errors(U, ref):
    """(L1, L2) of rho against the reference averaged onto U's cells."""
    n = U.shape[1]
    e = U[0] - ref.reshape(n, -1).mean(axis=1)
    return float(np.abs(e).sum() / n), float(np.sqrt((e ** 2).sum() / n))


@pytest.mark.parametrize("solver,limiter", list(BOUNDS), ids=[IDS[k] for k in BOUNDS])
def test_woodward_colella_ladder(solver, limiter, ref):
    """Regression only (LUNA's own reference): L1, L2 of rho at every N, orders; mass and energy to round-off."""
    errs = []
    for n in NS:
        U0, U = run(n, limiter, solver=solver)
        check_conserved(f"WC {solver} lim{limiter} N={n}", U0, U, momentum=False)
        l1, l2 = errors(U, ref)
        errs.append((l1, l2))
        check(f"WC {solver} lim{limiter} N={n} L1 rho", l1, BOUNDS[solver, limiter][n][0])
        check(f"WC {solver} lim{limiter} N={n} L2 rho", l2, BOUNDS[solver, limiter][n][1])
    for k, norm in enumerate(("L1", "L2")):
        order = -np.polyfit(np.log(NS), np.log([e[k] for e in errs]), 1)[0]
        check(f"WC {solver} lim{limiter} {norm} order", order, ORDER[solver, limiter][k][0], lower=True)
        check(f"WC {solver} lim{limiter} {norm} order", order, ORDER[solver, limiter][k][1])


def test_bound_catches_a_short_run(ref, ak):
    """Teeth: MC at 2048 stopped at 0.98 t misses its L1 bounds against both references (measured 0.135, 0.135)."""
    _, U = run(2048, 1, t=0.98 * T)
    l1, l1_ak = errors(U, ref)[0], errors(U, ak["rho"])[0]
    print(f"MEASURED WC MC N=2048 0.98 t L1 rho vs LUNA ref {l1:.6e}, vs AthenaK {l1_ak:.6e}")
    assert l1 > BOUNDS["hllc", 1][2048][0], f"L1 rho {l1:.4e} of a short run passes the bound"
    assert l1_ak > AK_BOUNDS["hllc", 1][2048][0], f"L1 rho vs AthenaK {l1_ak:.4e} of a short run passes the bound"


# ---------------------------------------------------------------- external gate: AthenaK PPMX 16384
# {(solver, limiter): {N: (L1, L2) of rho}} against AthenaK PPMX 16384, ~1.3x measured (CPU-8, FP64, job 31699086).
# For scale, AthenaK's own L1 against it: PPMX 0.110, 0.0572, 0.0277, 0.0157; PLM (upper bound, see the docstring)
# 0.163, 0.0997, 0.0573, 0.0324 at 256..2048; EPS_REF 2.78e-3.
AK_BOUNDS = {
    ("hllc", 0): {256: (0.23, 0.57), 512: (0.16, 0.42), 1024: (0.096, 0.29), 2048: (0.058, 0.22), 4096: (0.035, 0.16)},
    # minmod measured (0.177, 0.437), (0.116, 0.317), (0.0732, 0.221), (0.0440, 0.163), (0.0265, 0.119)
    ("hllc", 1): {256: (0.18, 0.48), 512: (0.11, 0.31), 1024: (0.056, 0.22), 2048: (0.032, 0.16), 4096: (0.017, 0.11)},
    # MC measured (0.136, 0.368), (0.0770, 0.236), (0.0424, 0.163), (0.0240, 0.118), (0.0129, 0.0787)
    ("hllc", 2): {256: (0.19, 0.5), 512: (0.12, 0.34), 1024: (0.064, 0.23), 2048: (0.035, 0.17), 4096: (0.02, 0.12)},
    # van Leer measured (0.145, 0.384), (0.0861, 0.256), (0.0488, 0.175), (0.0268, 0.125), (0.0151, 0.0850)
    ("hll", 0): {256: (0.26, 0.62), 512: (0.17, 0.45), 1024: (0.11, 0.31), 2048: (0.062, 0.23), 4096: (0.038, 0.17)},
    # HLL minmod measured (0.193, 0.471), (0.126, 0.345), (0.0792, 0.235), (0.0475, 0.170), (0.0285, 0.124)
}
AK_IDS = {**IDS, ("hllc", 2): "vanleer"}
AK_PLM_NS = (256, 512, 1024, 2048)          # the AthenaK PLM + RK2 runs of a9_plm_rk2_16384.npz
EPS_REF_BOUND = 3.6e-3                      # the reference's own error; measured 2.78e-3 (vs PPM4; vs LUNA 2.53e-3)


@pytest.fixture(scope="module")
def ak():
    """AthenaK PPMX 16384 rho, its own error EPS_REF, and AthenaK PLM's distance to it per N (upper bound)."""
    kw = dict(reconstruct="ppmx", integrator="rk4", rsolver="hllc", cfl_number=0.4, nx1=16384, tlim=T, gamma=1.4)
    z, _ = load_athenak("woodward_colella/a9_ppmx_rk4_16384.npz", "a9_ppmx_rk4_N16384_1d", **kw)
    p4, _ = load_athenak("woodward_colella/a9_ppm4_rk4_16384.npz", "a9_ppm4_rk4_N16384_1d", **(kw | dict(
        reconstruct="ppm4")))
    plm, _ = load_athenak("woodward_colella/a9_plm_rk2_16384.npz", "a9_plm_rk2_N16384_1d", **(kw | dict(
        reconstruct="plm", integrator="rk2")))
    assert float(z["ref_nonfinite"]) == 0.0 and int(z["ref_N"]) == 16384
    rho = np.asarray(z["ref_rho"], np.float64)
    with np.load(REF) as zl:
        luna = np.asarray(zl["rho"], np.float64)
    e_ppm4 = float(np.abs(rho - p4["ref_rho"]).mean())
    e_luna = float(np.abs(rho.reshape(luna.size, -1).mean(1) - luna).mean())
    print(f"MEASURED WC AthenaK PPMX 16384 vs PPM4 16384 L1 rho = {e_ppm4:.6e}, vs LUNA MC 65536 on 8192 = "
          f"{e_luna:.6e}")
    eps = max(e_ppm4, e_luna)
    check("WC AthenaK reference own error EPS_REF (L1 rho)", eps, EPS_REF_BOUND)
    plm_gap = float(np.abs(np.asarray(plm["ref_rho"], np.float64) - rho).mean())
    plm_n = {n: float(plm[f"N{n}_r_L1_rho_1d_akplm"]) + plm_gap for n in AK_PLM_NS}
    print(f"MEASURED WC AthenaK PLM 16384 vs PPMX 16384 L1 rho = {plm_gap:.6e}; AthenaK PLM per N (bound) "
          + ", ".join(f"{n}: {v:.6e}" for n, v in plm_n.items()))
    return dict(rho=rho, eps=eps, plm=plm_n)


@pytest.mark.parametrize("solver,limiter", list(AK_BOUNDS), ids=[AK_IDS[k] for k in AK_BOUNDS])
def test_woodward_colella_athenak(solver, limiter, ak):
    """External gate: L1, L2 of rho at every N against AthenaK PPMX 16384 averaged onto LUNA's cells."""
    for n in NS:
        l1, l2 = errors(run(n, limiter, solver=solver)[1], ak["rho"])
        check(f"WC {solver} lim{limiter} N={n} L1 rho vs AthenaK", l1, AK_BOUNDS[solver, limiter][n][0])
        check(f"WC {solver} lim{limiter} N={n} L2 rho vs AthenaK", l2, AK_BOUNDS[solver, limiter][n][1])


def test_vanleer_matches_athenak_plm(ak):
    """Matched method: LUNA van Leer at N is within AthenaK PLM's L1 at N + 2 EPS_REF of the AthenaK reference
    (measured 0.145, 0.0861, 0.0488, 0.0268 against bounds 0.169, 0.105, 0.0629, 0.0380)."""
    for n in AK_PLM_NS:
        l1 = errors(run(n, 2)[1], ak["rho"])[0]
        check(f"WC vanleer N={n} L1 rho vs AthenaK (AthenaK PLM {ak['plm'][n]:.4e} + 2 EPS_REF)", l1,
              ak["plm"][n] + 2 * ak["eps"])


def test_matched_bound_catches_first_order(ak):
    """Teeth: LUNA first order (ORDER=0, Godunov) at 1024 misses the matched-method bound (measured 0.192 > 0.0629)."""
    l1 = errors(run(1024, 2, order=0)[1], ak["rho"])[0]
    print(f"MEASURED WC first order N=1024 L1 rho vs AthenaK = {l1:.6e}")
    assert l1 > ak["plm"][1024] + 2 * ak["eps"], l1
