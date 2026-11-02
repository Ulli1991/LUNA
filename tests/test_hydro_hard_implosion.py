"""Hui / Liska-Wendroff implosion (LW 2003 sec. 4.7, A10) on the quadrant (0, 0.3)^2 with reflecting walls.

The flow is mirror-symmetric about the diagonal, and LUNA's unsplit step keeps x <-> y exactly on the GPU (FP64,
FP32, recon32, the Pallas kernel, sharded, the production loop): a 1e-14 seed grows to 1e-3 by t = 2.5, so only a
bitwise bound means anything. Plus mass and energy, the diagonal jet at LW's 400^2, and L1/L2(rho) at LW's early time:
- External gate: against AthenaK PPMX 6400^2 (athenak/lw_implosion/a10_ppmx_rk4_6400.npz, luna-reference-solutions):
  AthenaK 886dd2a hydro kernels + luna_ic pgen (source c52f88f0), binary build_gpu_lunaic_nofma sha256 2e963374
  (-fmad=false, exactly mirror symmetric), PPMX + RK4 + HLLC, CFL 0.4, dfloor = pfloor = 1e-8, FP64, A100, from the
  test's exact cell-average IC, block-averaged onto 800^2 (tools/athenak_refs.py prep/reduce a10; the fixture asserts
  the scheme from the stored athinput). Its own error EPS_REF = max(6400 vs 3200, distance to LUNA's 6400^2 MC).
  Bounds ~1.3x LUNA's measured L1/L2 against it; AthenaK's own numbers at the same N alongside.
- Matched method: AthenaK PLM is van Leer's harmonic limiter + RK2 + HLLC (athenak src/reconstruct/plm.hpp:28-31),
  LUNA hllc-vl the same limiter in MUSCL-Hancock. "LUNA van Leer is no less accurate than AthenaK PLM at the same N"
  gives L1(LUNA, ref) <= L1(AthenaK PLM, ref) + 2 EPS_REF (triangle inequality); no number in it comes from LUNA.
- Regression only: against LUNA's own 6400^2 MC reference (tools/implosion_reference.py), and the late-time L1 against
  LUNA's 2048^2 (no AthenaK late reference: the late flow is not converged).
Bounds, jobs and the CPU FMA finding: docs/TEST_PLAN.md (A10)."""
import functools

import jax
import numpy as np
import pytest

from luna.reference_data import reference_path
from tests._athenak import load as load_athenak
from tests._helpers import accuracy_cfg, check, evolve, evolve_staged, gpu_only

G, L = 1.4, 0.3
T_EARLY, T_LATE = 0.045, 2.5
SCHEMES = {"hllc-minmod": ("hllc", 0), "hllc-mc": ("hllc", 1), "hllc-vl": ("hllc", 2),
           "hll-minmod": ("hll", 0), "hll-mc": ("hll", 1), "hll-vl": ("hll", 2)}
REF = reference_path("luna/lw_implosion/lw_implosion_mc.npz")
CPU = jax.default_backend() == "cpu"
# XLA CPU contracts a*b + c*d into fma(a, b, c*d): the x and y terms of a cell are rounded differently, so the
# mirror cell differs in the last bit (exactly 0 with --xla_cpu_max_isa=AVX, which has no FMA; TEST_PLAN A10).
CPU_FMA = pytest.mark.xfail(CPU, strict=True, reason="XLA CPU FMA contraction breaks x <-> y bitwise symmetry "
                            "(~1e-15 FP64, ~1e-6 FP32); exact on the GPU and on CPU without FMA")
SYM_ROUNDOFF = {"fp64": 1e-14, "fp32": 5e-6}   # CPU N=64 to t=0.5, measured max 4.0e-15 / 2.15e-6 (FP32)
CONSERVE = 1e-15                                # |dM|/M, |dE|/E; measured max 3.3e-16 (GPU, CPU)


def cfg_for(n, scheme, nz=None, **kw):
    """The implosion's config on n x n cells (a z-uniform n x n x nz slab if nz)."""
    solver, limiter = SCHEMES[scheme]
    grid = dict(SPATIAL_DIMENSION=3, Nz=nz, Lz=L * nz / n) if nz else dict(SPATIAL_DIMENSION=2)
    return accuracy_cfg(test_problem="LW-Implosion", Nx=n, Ny=n, Lx=L, Ly=L, BOUNDARY=2, gamma=G,
                        riemann_solver=solver, LIMITER=limiter, **grid, **kw)


def initial(n, nz=None):
    """The exact cell-average IC (4, n, n), or the z-uniform slab (5, n, n, nz)."""
    from luna.ic_generation.ic import initial_conditions
    U = np.array(initial_conditions(accuracy_cfg(test_problem="LW-Implosion", SPATIAL_DIMENSION=2, Nx=n,
                                                   Ny=n, Lx=L, Ly=L, gamma=G)))
    if not nz:
        return U
    U = np.repeat(U[..., None], nz, -1)
    return np.concatenate([U[:3], np.zeros_like(U[:1]), U[3:]])


def asym(U):
    """max |q - q^T| over rho, m_x <-> m_y, (m_z), E: x <-> y mirror error, 0 for an exactly symmetric state."""
    U = np.asarray(U)
    T = lambda a: np.swapaxes(a, 0, 1)                    # noqa: E731
    pairs = [(U[0], U[0]), (U[1], U[2]), (U[-1], U[-1])] + ([(U[3], U[3])] if U.shape[0] == 5 else [])
    return max(float(np.abs(a - T(b)).max()) for a, b in pairs)


def check_closed(name, U0, U):
    """|dM|/M and |dE|/E (sums in float64) within CONSERVE (FP64) or the FP32 round-off of evolve."""
    U0, Uf = np.asarray(U0, np.float64), np.asarray(U, np.float64)
    tol = CONSERVE if np.asarray(U).dtype == np.float64 else 1e-6
    check(f"{name} |dM|/M", abs(Uf[0].sum() / U0[0].sum() - 1.0), tol)
    check(f"{name} |dE|/E", abs(Uf[-1].sum() / U0[-1].sum() - 1.0), tol)


@functools.lru_cache(maxsize=None)
def run(n, scheme, times, nz=None, mesh_shape=None, **kw):
    """States at `times` from the IC (cached per precision through the cfg)."""
    return evolve(cfg_for(n, scheme, nz, **kw), initial(n, nz), times, mesh_shape)[0]


def block(a, m):
    """Block average of an (n, n) field onto (m, m)."""
    n = a.shape[0]
    return a.reshape(m, n // m, m, n // m).mean(axis=(1, 3))


@functools.lru_cache(maxsize=None)
def reference():
    """{'early': rho (800, 800) at T_EARLY, 'late': rho (256, 256) at T_LATE} after checking what they are."""
    from tools.implosion_reference import untri
    with np.load(REF) as z:
        assert str(z["problem"]) == "LW-Implosion" and float(z["gamma"]) == G and float(z["box"]) == L
        assert int(z["boundary"]) == 2 and str(z["riemann_solver"]) == "hllc" and int(z["limiter"]) == 1
        assert str(z["precision"]) == "fp64" and float(z["early_t"]) == T_EARLY and float(z["late_t"]) == T_LATE
        check("implosion reference early self-convergence L1(rho)", float(z["early_self_l1_rho"]), REF_SELF_L1[0])
        print(f"MEASURED implosion reference late self-difference L1(rho) = {float(z['late_self_l1_rho']):.6e} "
              "(recorded, not bounded)")
        return {k: untri(np.asarray(z[f"{k}_rho"], np.float64)) for k in ("early", "late")}


def early_errors(n, scheme, ref=None, **kw):
    """(L1, L2) of rho at T_EARLY against `ref` (800^2; default LUNA's reference) averaged onto n x n (n | 800)."""
    ref = reference()["early"] if ref is None else ref
    e = run(n, scheme, (T_EARLY,), **kw)[0][0] - block(ref, n)
    return float(np.abs(e).mean()), float(np.sqrt((e * e).mean()))


def late_l1(n, scheme):
    """L1(rho) at T_LATE against the reference, both averaged onto min(n, 256)^2 cells."""
    m = min(n, 256)
    return float(np.abs(block(run(n, scheme, TIMES_SYM)[-1][0], m) - block(reference()["late"], m)).mean())


# Measured diagonal profiles at t = 2.5 (128..1024^2): the background between the core and the reflected-shock
# dip (rho ~0.80 at s ~0.28) sits at rho >= 0.80, jet gas at <= 0.76; 0.78 splits them, the cap stops at the dip.
JET_RHO, JET_CAP = 0.78, 0.25


def s_jet(rho):
    """Largest distance s < JET_CAP from the origin along the diagonal with rho < JET_RHO (0 if none)."""
    n = rho.shape[0]
    s = np.sqrt(2.0) * (np.arange(n) + 0.5) * L / n
    m = (s < JET_CAP) & (np.diag(rho) < JET_RHO)
    return float(s[m].max()) if m.any() else 0.0


# ---------------------------------------------------------------- fast tier: N = 64, CPU or GPU
NS_FAST, TIMES_FAST = 64, (T_EARLY, 0.5)


def test_initial_state_is_the_cell_average():
    """Exact cell averages: symmetric, N^2/8 cells inside, the half cells, restriction to N/2, luna_ic round trip."""
    from tools import luna_ic
    U, U2 = initial(64), initial(128)
    check("implosion IC asymmetry", asym(U), 0.0)
    f = (1.0 - U[0]) / (1.0 - 0.125)
    check("implosion IC |inside area - N^2/8|", abs(f.sum() - 64 * 64 / 8), 0.0)
    i = np.arange(64)
    half = (i[:, None] + i[None, :]) == 31
    check("implosion IC half cells |rho - 0.5625|", float(np.abs(U[0][half] - 0.5625).max()), 0.0)
    check("implosion IC |block(2N) - N|", float(np.abs(U2.reshape(4, 64, 2, 64, 2).mean((2, 4)) - U).max()), 1e-15)
    check("implosion IC luna_ic round trip", float(np.abs(luna_ic.from_luna5(luna_ic.to_luna5(U), 2) - U).max()), 0.0)


@CPU_FMA
@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_symmetry_is_exact(scheme):
    """max |q - q^T| == 0 at t = 0.045 and 0.5, N = 64."""
    for t, U in zip(TIMES_FAST, run(NS_FAST, scheme, TIMES_FAST)):
        check(f"implosion {scheme} N={NS_FAST} t={t} asymmetry", asym(U), 0.0)


@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_symmetry_to_roundoff_and_conservation(scheme):
    """Asymmetry at round-off on any backend; mass and energy to CONSERVE between the walls."""
    U0 = initial(NS_FAST)
    for t, U in zip(TIMES_FAST, run(NS_FAST, scheme, TIMES_FAST)):
        check(f"implosion {scheme} N={NS_FAST} t={t} asymmetry", asym(U), SYM_ROUNDOFF["fp64"])
        check_closed(f"implosion {scheme} N={NS_FAST} t={t}", U0, U)


@CPU_FMA
@pytest.mark.precision("fp32")
@pytest.mark.parametrize("scheme", ("hllc-minmod", "hllc-mc"))
def test_symmetry_is_exact_fp32(scheme):
    """FP32 state: exact on the GPU."""
    for t, U in zip(TIMES_FAST, run(NS_FAST, scheme, TIMES_FAST, enable_x64=False)):
        check(f"implosion fp32 {scheme} N={NS_FAST} t={t} asymmetry", asym(U), 0.0)


@pytest.mark.precision("fp32")
@pytest.mark.parametrize("scheme", ("hllc-minmod", "hllc-mc"))
def test_symmetry_to_roundoff_fp32(scheme):
    """FP32 state: asymmetry at FP32 round-off on any backend."""
    for t, U in zip(TIMES_FAST, run(NS_FAST, scheme, TIMES_FAST, enable_x64=False)):
        check(f"implosion fp32 {scheme} N={NS_FAST} t={t} asymmetry", asym(U), SYM_ROUNDOFF["fp32"])


@CPU_FMA
@pytest.mark.precision("mixed")
@pytest.mark.parametrize("scheme", ("hllc-minmod", "hllc-mc"))
def test_symmetry_is_exact_mixed(scheme):
    """recon32 (3-D only): the z-uniform 64 x 64 x 4 slab stays z-uniform and x <-> y symmetric."""
    for t, U in zip(TIMES_FAST, run(NS_FAST, scheme, TIMES_FAST, nz=4, hydro_recon32=True)):
        check(f"implosion mixed slab {scheme} t={t} asymmetry", asym(U), 0.0)
        check(f"implosion mixed slab {scheme} t={t} z spread", float(np.abs(U - U[..., :1]).max()), 0.0)


def test_production_loop_is_the_test_loop():
    """evolve_staged (assemble, timestep, step) equals evolve bitwise, so its symmetry is the step's."""
    for t, a, b in zip(TIMES_FAST, evolve_staged(cfg_for(NS_FAST, "hllc-mc"), initial(NS_FAST), TIMES_FAST),
                       run(NS_FAST, "hllc-mc", TIMES_FAST)):
        check(f"implosion staged - evolve t={t}", float(np.abs(a - b).max()), 0.0)


@pytest.mark.parametrize("mesh", ((2, 1), (1, 2)), ids=("x-split", "y-split"))
def test_sharded_equals_one_device(mesh):
    """A seam along x or y only (asymmetric decompositions) changes nothing, bitwise; GPU to t = 2.5 at 400^2."""
    if jax.device_count() < 2:
        pytest.skip("needs >= 2 devices for a real shard seam")
    n, times = (NS_FAST, TIMES_FAST) if CPU else (400, (T_EARLY, 0.5, T_LATE))
    for sc in ("hllc-mc", "hll-minmod"):
        for t, a, b in zip(times, run(n, sc, times, mesh_shape=mesh), run(n, sc, times)):
            check(f"implosion {sc} N={n} mesh {mesh} - one device t={t}", float(np.abs(a - b).max()), 0.0)


def test_symmetry_check_sees_shifted_coordinates():
    """Teeth: the centre rule x + y < 0.15 with y_j = j dy + dy/2 against x_i = (i + 1/2) dx flips 2 half cells."""
    n = NS_FAST
    dx = L / n
    x, y = (np.arange(n) + 0.5) * dx, np.arange(n) * dx + 0.5 * dx
    inside = (x[:, None] + y[None, :]) < 0.5 * L
    rho, p = np.where(inside, 0.125, 1.0), np.where(inside, 0.14, 1.0)
    U0 = np.stack([rho, 0 * rho, 0 * rho, p / (G - 1.0)])
    U = evolve(cfg_for(n, "hllc-mc"), U0, T_EARLY)[0]
    assert asym(U) > SYM_ROUNDOFF["fp64"], f"asymmetry {asym(U):.3e} of a mis-built IC passes"


def test_symmetry_check_sees_a_seed_above_cpu_roundoff():
    """Teeth on any backend: E of one off-diagonal cell scaled by 1 + 1e-12 exceeds the round-off bound at 0.045."""
    U0 = initial(NS_FAST)
    U0[-1, NS_FAST // 2, NS_FAST // 8] *= 1.0 + 1e-12
    a = asym(evolve(cfg_for(NS_FAST, "hllc-mc"), U0, T_EARLY)[0])
    print(f"MEASURED implosion 1e-12 seed N={NS_FAST} asymmetry = {a:.6e}")
    assert a > SYM_ROUNDOFF["fp64"]


# ---------------------------------------------------------------- GPU tier
NS_SYM, TIMES_SYM = 400, (T_EARLY, 0.5, T_LATE)          # LW's grid and times


@gpu_only
@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_symmetry_is_exact_to_late_time(scheme):
    """Bitwise x <-> y at LW's 400^2 to t = 2.5, mass and energy to CONSERVE."""
    U0 = initial(NS_SYM)
    for t, U in zip(TIMES_SYM, run(NS_SYM, scheme, TIMES_SYM)):
        check(f"implosion {scheme} N={NS_SYM} t={t} asymmetry", asym(U), 0.0)
        check_closed(f"implosion {scheme} N={NS_SYM} t={t}", U0, U)


@gpu_only
@pytest.mark.precision("fp32")
@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_symmetry_is_exact_to_late_time_fp32(scheme):
    """FP32 state, 400^2 to t = 2.5."""
    for t, U in zip(TIMES_SYM, run(NS_SYM, scheme, TIMES_SYM, enable_x64=False)):
        check(f"implosion fp32 {scheme} N={NS_SYM} t={t} asymmetry", asym(U), 0.0)


MODES = [pytest.param(m, marks=pytest.mark.precision(m)) for m in ("fp64", "fp32", "mixed")]


@gpu_only
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("scheme", ("hllc-minmod", "hll-minmod"))
def test_pallas_kernel_is_symmetric(mode, scheme, monkeypatch):
    """The Pallas kernel (3-D, minmod) on a 128 x 128 x 4 slab to t = 2.5: exact x <-> y, z-uniform, as XLA."""
    from luna.hydro import muscl_hancock_pallas as mhp
    calls = []
    divergence = mhp.divergence
    monkeypatch.setattr(mhp, "divergence", lambda *a, **k: calls.append(1) or divergence(*a, **k))
    kw = dict(enable_x64=mode != "fp32", hydro_recon32=mode == "mixed")
    pal = run(128, scheme, TIMES_SYM, nz=4, hydro_kernel="pallas", **kw)
    assert calls, "hydro_kernel='pallas' did not reach the Pallas kernel: XLA equals it bitwise, so check the path"
    monkeypatch.setattr(mhp, "divergence", divergence)
    xla = run(128, scheme, TIMES_SYM, nz=4, hydro_kernel="xla", **kw)
    for t, a, b in zip(TIMES_SYM, pal, xla):
        check(f"implosion pallas {mode} {scheme} t={t} asymmetry", asym(a), 0.0)
        check(f"implosion pallas {mode} {scheme} t={t} z spread", float(np.abs(a - a[..., :1]).max()), 0.0)
        print(f"MEASURED implosion pallas - xla {mode} {scheme} t={t} = {float(np.abs(a - b).max()):.6e} (printed)")


@gpu_only
@pytest.mark.parametrize("mode", MODES[:2])
@pytest.mark.parametrize("scheme", ("hllc-mc", "hll-vl", "hllc-minmod"))
def test_pallas_2d_kernel_is_symmetric(mode, scheme, monkeypatch):
    """The 2-D Pallas kernel (any limiter) at LW's 400^2 to t = 2.5: exact x <-> y, as XLA."""
    from luna.hydro import muscl_hancock_pallas as mhp
    calls = []
    divergence = mhp.divergence
    monkeypatch.setattr(mhp, "divergence", lambda *a, **k: calls.append(1) or divergence(*a, **k))
    kw = dict(enable_x64=mode != "fp32")
    pal = run(NS_SYM, scheme, TIMES_SYM, hydro_kernel="pallas", **kw)
    assert calls, "hydro_kernel='pallas' did not reach the Pallas kernel"
    monkeypatch.setattr(mhp, "divergence", divergence)
    xla = run(NS_SYM, scheme, TIMES_SYM, **kw)
    for t, a, b in zip(TIMES_SYM, pal, xla):
        check(f"implosion pallas 2-D {mode} {scheme} t={t} asymmetry", asym(a), 0.0)
        print(f"MEASURED implosion pallas 2-D - xla {mode} {scheme} t={t} = "
              f"{float(np.abs(a - b).max()):.6e} (printed)")


@gpu_only
def test_symmetry_check_sees_a_1e14_seed():
    """Teeth: E of one off-diagonal cell scaled by 1 + 1e-14 is visible at t = 0.045 (measured 1.5e-12)."""
    U0 = initial(NS_SYM)
    U0[-1, NS_SYM // 2, NS_SYM // 8] *= 1.0 + 1e-14
    a = asym(evolve(cfg_for(NS_SYM, "hllc-mc"), U0, T_EARLY)[0])
    print(f"MEASURED implosion 1e-14 seed N={NS_SYM} t={T_EARLY} asymmetry = {a:.6e}")
    assert a > 0.0


NS_EARLY = (100, 200, 400, 800)
# Regression only: {scheme: {N: (L1, L2)}} of rho at T_EARLY against LUNA's 6400^2 reference, ~1.3x measured
# (A100 FP64, job 31110359);
# orders measured -0.1 .. +0.3 (shock and contact: L1 2/3..1, L2 1/3..1/2)
EARLY = {
    "hllc-minmod": {100: (0.0051, 0.017), 200: (0.003, 0.012), 400: (0.0018, 0.0098), 800: (0.00098, 0.0075)},
    # measured (0.00391, 0.0124), (0.00225, 0.00942), (0.00132, 0.00753), (0.000751, 0.00573); orders L1 0.79, L2 0.37
    "hllc-mc": {100: (0.0035, 0.012), 200: (0.0019, 0.0091), 400: (0.0011, 0.0071), 800: (0.00058, 0.0053)},
    # measured (0.00264, 0.00952), (0.00146, 0.00695), (0.00082, 0.00543), (0.000439, 0.00404); orders L1 0.86, L2 0.41
    "hllc-vl": {100: (0.0038, 0.014), 200: (0.0021, 0.0098), 400: (0.0012, 0.0077), 800: (0.00064, 0.0058)},
    # measured (0.00287, 0.0101), (0.00157, 0.00748), (0.000893, 0.0059), (0.000485, 0.00439); orders L1 0.85, L2 0.39
    "hll-minmod": {100: (0.0068, 0.02), 200: (0.0039, 0.016), 400: (0.0024, 0.012), 800: (0.0014, 0.0098)},
    # measured (0.00516, 0.0153), (0.00299, 0.0116), (0.0018, 0.00941), (0.00107, 0.00749); orders L1 0.75, L2 0.34
    "hll-mc": {100: (0.0047, 0.017), 200: (0.0026, 0.012), 400: (0.0015, 0.0094), 800: (0.00081, 0.0071)},
    # measured (0.00361, 0.0125), (0.002, 0.00921), (0.00114, 0.00719), (0.000621, 0.00545); orders L1 0.84, L2 0.39
    "hll-vl": {100: (0.0051, 0.017), 200: (0.0028, 0.013), 400: (0.0016, 0.0099), 800: (0.0009, 0.0076)},
    # measured (0.00388, 0.013), (0.00211, 0.00958), (0.00122, 0.00757), (0.000686, 0.00584); orders L1 0.83, L2 0.38
}
EARLY_ORDER = {
    "hllc-minmod": ((0.69, 1.09), (0.27, 0.67)),
    "hllc-mc": ((0.76, 1.16), (0.31, 0.71)),
    "hllc-vl": ((0.75, 1.15), (0.29, 0.69)),
    "hll-minmod": ((0.65, 1.05), (0.24, 0.64)),
    "hll-mc": ((0.74, 1.14), (0.29, 0.69)),
    "hll-vl": ((0.73, 1.13), (0.28, 0.68)),
}
REF_SELF_L1 = (1e-4,)                 # early: 6400 vs 3200 on 800^2, measured 8.0e-5 (job 31110659)


@gpu_only
@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_early_ladder(scheme):
    """Regression only: L1, L2 of rho at LW's t = 0.045 on 100..800^2 against LUNA's reference, and their orders."""
    errs = [early_errors(n, scheme) for n in NS_EARLY]
    for n, (l1, l2) in zip(NS_EARLY, errs):
        check(f"implosion {scheme} N={n} t={T_EARLY} L1 rho", l1, EARLY[scheme][n][0])
        check(f"implosion {scheme} N={n} t={T_EARLY} L2 rho", l2, EARLY[scheme][n][1])
    for k, norm in enumerate(("L1", "L2")):
        order = -np.polyfit(np.log(NS_EARLY), np.log([e[k] for e in errs]), 1)[0]
        check(f"implosion {scheme} {norm} order", order, EARLY_ORDER[scheme][k][0], lower=True)
        check(f"implosion {scheme} {norm} order", order, EARLY_ORDER[scheme][k][1])


@gpu_only
def test_early_bound_catches_a_short_run():
    """Teeth: HLLC-MC at 400^2 stopped at 0.95 t misses its L1 bounds against LUNA's and AthenaK's reference
    (measured 4.49e-3, 4.49e-3)."""
    U = run(400, "hllc-mc", (0.95 * T_EARLY,))[0][0]
    l1 = float(np.abs(U - block(reference()["early"], 400)).mean())
    l1_ak = float(np.abs(U - block(athenak()["rho"], 400)).mean())
    print(f"MEASURED implosion short run L1 rho = {l1:.6e}, vs AthenaK {l1_ak:.6e}")
    assert l1 > EARLY["hllc-mc"][400][0]
    assert l1_ak > AK_EARLY["hllc-mc"][400][0]


# ---------------------------------------------------------------- external gate: AthenaK PPMX 6400^2 at T_EARLY
# {scheme: {N: (L1, L2)}} of rho at T_EARLY against AthenaK PPMX 6400^2 on N^2, ~1.3x measured (A100 FP64, job
# 31699123). For scale, AthenaK's own L1 against it at 100..800: PLM 3.11e-3, 1.73e-3, 9.87e-4, 5.45e-4; PPMX 2.27e-3,
# 1.27e-3, 6.75e-4, 3.57e-4; EPS_REF 8.68e-5.
AK_EARLY = {
    "hllc-minmod": {100: (0.0051, 0.017), 200: (0.003, 0.013), 400: (0.0018, 0.0097), 800: (0.00098, 0.0074)},
    # measured (0.00392, 0.0124), (0.00225, 0.00938), (0.00132, 0.00741), (0.000753, 0.00568)
    "hllc-mc": {100: (0.0035, 0.013), 200: (0.0019, 0.009), 400: (0.0011, 0.0069), 800: (0.00058, 0.0052)},
    # measured (0.00264, 0.00950), (0.00146, 0.00689), (0.000816, 0.00529), (0.000444, 0.00400)
    "hllc-vl": {100: (0.0038, 0.014), 200: (0.0021, 0.0097), 400: (0.0012, 0.0075), 800: (0.00064, 0.0057)},
    # measured (0.00288, 0.0101), (0.00157, 0.00742), (0.000889, 0.00577), (0.000490, 0.00435)
    "hll-minmod": {100: (0.0068, 0.02), 200: (0.0039, 0.016), 400: (0.0024, 0.013), 800: (0.0014, 0.0097)},
    # measured (0.00516, 0.0154), (0.00299, 0.0116), (0.00180, 0.00929), (0.00107, 0.00740)
    "hll-mc": {100: (0.0048, 0.017), 200: (0.0026, 0.012), 400: (0.0015, 0.0092), 800: (0.00082, 0.007)},
    # measured (0.00362, 0.0125), (0.00200, 0.00916), (0.00113, 0.00705), (0.000624, 0.00537)
    "hll-vl": {100: (0.0051, 0.017), 200: (0.0028, 0.013), 400: (0.0016, 0.0097), 800: (0.0009, 0.0075)},
    # measured (0.00389, 0.0130), (0.00211, 0.00954), (0.00122, 0.00743), (0.000689, 0.00576)
}
EPS_REF_BOUND = 1.1e-4      # max(AthenaK 6400 vs 3200 8.68e-5, vs LUNA 6400 5.92e-5) on 800^2, stored by the reduction


@functools.lru_cache(maxsize=None)
def athenak():
    """AthenaK PPMX 6400^2 rho on 800^2 at T_EARLY, its own error EPS_REF, AthenaK PLM's L1 against it per N."""
    from tools.implosion_reference import untri
    kw = dict(reconstruct="ppmx", integrator="rk4", rsolver="hllc", cfl_number=0.4, nx1=6400, nx2=6400,
              tlim=T_EARLY, gamma=G, dfloor="1e-08", pfloor="1e-08")
    z, _ = load_athenak("lw_implosion/a10_ppmx_rk4_6400.npz", "a10_ppmx_rk4_N6400_ne", **kw)
    plm, _ = load_athenak("lw_implosion/a10_plm_rk2_1024.npz", "a10_plm_rk2_N800_ne", **(kw | dict(
        reconstruct="plm", integrator="rk2", nx1=800, nx2=800)))
    assert float(z["ref_N"]) == 6400 and float(z["ref_nofma"]) == 1.0
    eps = max(float(z["ref_L1_self"]), float(z["ref_L1_vs_luna"]))
    check("implosion AthenaK reference own error EPS_REF (L1 rho on 800^2)", eps, EPS_REF_BOUND)
    return dict(rho=untri(np.asarray(z["ref_rho"], np.float64)), eps=eps,
                plm={n: float(plm[f"N{n}_ne_L1_akppmx"]) for n in NS_EARLY})


@gpu_only
@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_early_ladder_athenak(scheme):
    """External gate: L1, L2 of rho at t = 0.045 on 100..800^2 against AthenaK PPMX 6400^2."""
    ak = athenak()
    for n in NS_EARLY:
        l1, l2 = early_errors(n, scheme, ak["rho"])
        check(f"implosion {scheme} N={n} t={T_EARLY} L1 rho vs AthenaK", l1, AK_EARLY[scheme][n][0])
        check(f"implosion {scheme} N={n} t={T_EARLY} L2 rho vs AthenaK", l2, AK_EARLY[scheme][n][1])


@gpu_only
def test_vanleer_matches_athenak_plm():
    """Matched method: LUNA hllc-vl at N is within AthenaK PLM's L1 at N + 2 EPS_REF of the AthenaK reference
    (measured 2.88e-3, 1.57e-3, 8.89e-4, 4.90e-4 against bounds 3.28e-3, 1.90e-3, 1.16e-3, 7.19e-4)."""
    ak = athenak()
    for n in NS_EARLY:
        check(f"implosion hllc-vl N={n} L1 rho vs AthenaK (AthenaK PLM {ak['plm'][n]:.4e} + 2 EPS_REF)",
              early_errors(n, "hllc-vl", ak["rho"])[0], ak["plm"][n] + 2 * ak["eps"])


@gpu_only
def test_matched_bound_catches_first_order():
    """Teeth: LUNA first order (ORDER=0) at 400^2 misses the matched-method bound (measured 4.72e-3 > 1.16e-3)."""
    ak = athenak()
    l1 = early_errors(400, "hllc-vl", ak["rho"], ORDER=0)[0]
    print(f"MEASURED implosion first order N=400 L1 rho vs AthenaK = {l1:.6e}")
    assert l1 > ak["plm"][400] + 2 * ak["eps"], l1


NS_LATE = (128, 256, 512, 1024)
# Regression only: L1(rho) at t = 2.5 against LUNA's own 2048^2 late reference (tools/implosion_reference.py) on
# min(N, 256)^2. The late flow is not converged (symmetric-noise growth, jets): measured flat in N, 0.011-0.018 for
# every scheme, not monotone, so these pin LUNA and say nothing about convergence. ~1.3x measured, A100 FP64,
# job 31663850: hllc-minmod 0.01341 0.01209 0.01421 0.01534; hllc-mc 0.01158 0.01797 0.01587 0.01232; hllc-vl
# 0.01180 0.01551 0.01500 0.01202; hll-minmod 0.01384 0.01179 0.01094 0.01527; hll-mc 0.01106 0.01132 0.01555
# 0.01394; hll-vl 0.01159 0.01131 0.01667 0.01470. HLLC-MC 128^2 stopped at 0.95 t reads 0.106 (teeth).
LATE = {"hllc-minmod": (0.0174, 0.0157, 0.0185, 0.0199), "hllc-mc": (0.0151, 0.0234, 0.0206, 0.0160),
        "hllc-vl": (0.0153, 0.0202, 0.0195, 0.0156), "hll-minmod": (0.0180, 0.0153, 0.0142, 0.0199),
        "hll-mc": (0.0144, 0.0147, 0.0202, 0.0181), "hll-vl": (0.0151, 0.0147, 0.0217, 0.0191)}


@gpu_only
@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_late_l1(scheme):
    """Regression only: L1(rho) at t = 2.5 against the 2048^2 LUNA reference on min(N, 256)^2, per N."""
    for n, b in zip(NS_LATE, LATE[scheme]):
        check(f"implosion {scheme} N={n} t={T_LATE} L1 rho on {min(n, 256)}^2", late_l1(n, scheme), b)


@gpu_only
def test_late_bound_catches_a_short_run():
    """Teeth: HLLC-MC at 128^2 stopped at 0.95 t = 2.375 misses its late L1 bound."""
    rho = run(128, "hllc-mc", (0.95 * T_LATE,))[0][0]
    l1 = float(np.abs(block(rho, 128) - block(reference()["late"], 128)).mean())
    print(f"MEASURED implosion late short run (0.95 t) N=128 L1 rho = {l1:.6e}")
    assert l1 > LATE["hllc-mc"][0], l1


# s_jet at 400^2 does not depend on the time step (CFL 0.7-0.8, other output stops, symmetric noise <= 1e-8 leave it
# unchanged; critic_a10, jobs 31168340-1) but it moves with t as structures pass along the diagonal (HLLC-MC 0.198,
# 0.124, 0.218, 0.229, 0.149 at t = 2.40..2.60; at 2.6 every scheme reads ~0.149). The median over JET_TIMES
# separates the schemes: jet hllc-mc 0.198, hllc-vl 0.133, hll-mc 0.121; none hllc-minmod 0.077, hll-minmod 0.084,
# hll-vl 0.088 (job 31169063). Bounds: {scheme: (lower, upper)} of the median, near measurement; values printed.
JET_TIMES = (2.40, 2.45, 2.50, 2.55, 2.60)
JET = {"hllc-mc": (0.18, None), "hllc-vl": (0.12, None), "hll-mc": (0.11, None),
       "hllc-minmod": (None, 0.10), "hll-minmod": (None, 0.10), "hll-vl": (None, 0.10)}


@gpu_only
@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_jet(scheme):
    """Median of s_jet over t = 2.40..2.60 on LW's 400^2: above 0.10 for the schemes with a jet, below for the rest."""
    vals = [s_jet(U[0]) for U in run(NS_SYM, scheme, JET_TIMES)]
    print(f"MEASURED implosion {scheme} N={NS_SYM} s_jet at t = {JET_TIMES}: "
          + ", ".join(f"{v:.4f}" for v in vals) + " (printed; t = 2.5 is a regression value, not a bound)")
    med = float(np.median(vals))
    lo, hi = JET[scheme]
    if lo is not None:
        check(f"implosion {scheme} N={NS_SYM} median s_jet", med, lo, lower=True)
    if hi is not None:
        check(f"implosion {scheme} N={NS_SYM} median s_jet", med, hi)
