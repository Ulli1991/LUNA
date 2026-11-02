"""Maclaurin spheroid potential against the analytic solution (Ricker 2008, ApJS 176, 293, sec. 3.1).

A homogeneous oblate spheroid, a1 = a2 = 0.25, rho = 1, centred in a unit box (Ricker's setup), density from
the Maclaurin-Spheroid-3D IC with its ambient removed. Isolated solver (POISSON_BOUNDARY=3) with its own
multipole (l <= 2) face values, and with Ricker's analytic face values; the AMR hierarchy solver
(amr/poisson.py, static level 1 over the central half). Errors are relative: L1 = sum|dphi| / sum|phi|,
Linf = max|dphi| / max|phi|. Ricker plots only an L2-type norm (his Fig. 4, no numbers): O(dx^2) uniform,
dx^1.0-1.2 on his surface-tracking multi-level meshes. Measured here: L1 order 2.0 for both face treatments
and on the one-level AMR; the multipole faces carry ~4.6x the analytic-face error at every N.
The AMR runner (make_amr_self_gravity, amr/poisson.py:291) takes its Poisson BC from cfg.BOUNDARY, so
isolated AMR gravity is reached here only through solve_phi_hierarchy(boundary=3).
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from luna.gravity.particle_mesh import ParticleMeshGravity
from luna.gravity.poisson_bc import make_linear_extrap_dirichlet_halo_3d, make_odd_dirichlet_halo_3d
from luna.gravity.poisson_multigrid import _max_levels, vcycle
from luna.ic_generation import maclaurin3d
from luna.ic_generation.maclaurin3d import maclaurin_3d
from tests._helpers import _with_problem, check, gpu_only

A1, RHO, G, L = 0.25, 1.0, 1.0, 1.0
EXTRA = 8           # V-cycles after FMG: the algebraic error far below the discretisation error


def phi_spheroid(x, y, z, a1, e, G=1.0, rho=1.0):
    """Potential of a homogeneous oblate spheroid (lap phi = 4 pi G rho), centre at the origin.

    Chandrasekhar 1969 ch. 3 eqs. (39)-(40) (Ricker 2008 eqs. 19-24): phi = -pi G rho a1^2 a3 int_lam^inf
    (1 - R^2/(a1^2+u) - z^2/(a3^2+u)) du / Delta, Delta = (a1^2+u) sqrt(a3^2+u), lam the positive root of
    R^2/(a1^2+lam) + z^2/(a3^2+lam) = 1 outside, 0 inside. With t = sqrt(a3^2+u), c = a1 e the integrals are
    closed (the arctan form of Ricker eqs. 25-26, h = c / sqrt(a3^2+lam)); at lam = 0 they are I a1^2, A1, A3
    of Chandrasekhar eq. (36), arctan(c/a3) = arcsin(e).
    """
    a3 = a1 * np.sqrt(1.0 - e * e)
    c = a1 * e
    R2, z2 = x * x + y * y, z * z
    b = a1 * a1 + a3 * a3 - R2 - z2
    q = a1 * a1 * a3 * a3 - R2 * a3 * a3 - z2 * a1 * a1
    lam = np.maximum(0.5 * (-b + np.sqrt(b * b - 4.0 * q)), 0.0)
    t0 = np.sqrt(a3 * a3 + lam)
    at = np.arctan(c / t0)
    j0 = 2.0 * at / c
    j1 = at / c ** 3 - t0 / (c * c * (t0 * t0 + c * c))
    j3 = 2.0 / (c * c) * (1.0 / t0 - at / c)
    return -np.pi * G * rho * a1 * a1 * a3 * (j0 - R2 * j1 - z2 * j3)


def _cells(n):
    x = (np.arange(n) + 0.5) * (L / n) - 0.5 * L
    return np.meshgrid(x, x, x, indexing="ij")


def _cfg(n, e, extra=EXTRA, g=G):
    return _with_problem(maclaurin3d.Params, e=e, a=A1, rho=RHO, test_problem="Maclaurin-Spheroid-3D",
                         SPATIAL_DIMENSION=3, Nx=n, Ny=n, Nz=n, Lx=L, Ly=L, Lz=L, BOUNDARY=4,
                         POISSON_BOUNDARY=3, GRAVITATIONAL_G=g, POISSON_FMG_EXTRA_VCYCLES=extra)


def _rho(cfg):
    """The IC's density without its ambient: rho_in * smoothstep (3-cell blended surface)."""
    amb = 1e-3 * RHO
    return (np.asarray(maclaurin_3d(cfg)[0]) - amb) * RHO / (RHO - amb)


def _errors(phi, ref):
    d = np.abs(np.asarray(phi) - ref)
    return float(d.sum() / np.abs(ref).sum()), float(d.max() / np.abs(ref).max())


@functools.partial(jax.jit, static_argnames=("n_levels", "n_cycles"))
def _analytic_bc_cycles(phi, rhs, dx, faces, n_levels, n_cycles):
    """V-cycles with the analytic potential as the face-centred Dirichlet value (Ricker's convergence study)."""
    top = make_linear_extrap_dirichlet_halo_3d(faces)
    inner = make_odd_dirichlet_halo_3d()
    return jax.lax.fori_loop(0, n_cycles, lambda _, p: vcycle(p, rhs, dx, top, n_levels - 1, 2, 2, 50,
                                                              halo_inner=inner), phi)


def _face_values(n, e):
    """(n+2)^3 with the analytic phi at the six face centres (layout of multipole_dirichlet_face_values)."""
    h = L / n
    x = (np.arange(n) + 0.5) * h - 0.5 * L
    out = np.zeros((n + 2,) * 3)
    A, B = np.meshgrid(x, x, indexing="ij")
    for side, f in ((0, -0.5 * L), (-1, 0.5 * L)):
        out[side, 1:-1, 1:-1] = phi_spheroid(f, A, B, A1, e)
        out[1:-1, side, 1:-1] = phi_spheroid(A, f, B, A1, e)
        out[1:-1, 1:-1, side] = phi_spheroid(A, B, f, A1, e)
    return out


@functools.lru_cache(maxsize=None)
def uniform(n, e, extra=EXTRA, cycles=12, g=G, shift=0):
    """(L1, Linf) of the isolated solve, multipole (production) then analytic faces, against the analytic phi."""
    # g != G (the solver's G) and shift (density rolled by that many cells along x): the teeth test's broken inputs
    cfg = _cfg(n, e, extra, g)
    rho = jnp.asarray(np.roll(_rho(cfg), shift, axis=0))
    ref = phi_spheroid(*_cells(n), A1, e)
    phi = jax.jit(ParticleMeshGravity(cfg).gas_potential)(rho)
    mp = _errors(phi, ref)
    phi_a = _analytic_bc_cycles(phi, 4.0 * np.pi * g * rho, L / n, jnp.asarray(_face_values(n, e)),
                                _max_levels(n), cycles)
    an = _errors(phi_a, ref)
    print(f"MEASURED maclaurin e={e} N={n} g={g} shift={shift}: multipole L1 {mp[0]:.4e} Linf {mp[1]:.4e}; "
          f"analytic-BC L1 {an[0]:.4e} Linf {an[1]:.4e}")
    return mp, an


@functools.lru_cache(maxsize=None)
def amr(n0, e, bs=8, extra=EXTRA, cycles=12):
    """(L1, Linf) on the composite of a two-level hierarchy, level 1 static over the central half."""
    from luna.amr import driver
    from luna.amr.config import AMRConfig
    from luna.amr.poisson import solve_phi_hierarchy
    from luna.amr.state import partition_to_blocks

    acfg = AMRConfig(ndim=3, box=(L,) * 3, base_grid=(n0,) * 3, block_shape=(bs,) * 3, max_levels=1,
                     boundary=1)
    rho_f = _rho(_cfg(2 * n0, e))
    rho_c = rho_f.reshape(n0, 2, n0, 2, n0, 2).mean(axis=(1, 3, 5))
    U0 = jnp.zeros((5, n0, n0, n0)).at[0].set(jnp.asarray(rho_c)).at[4].set(1.0)
    h = driver.build_hierarchy(U0, acfg, thresholds=[0.0], criterion="static", periodic=False)
    lv0, lv1 = h
    na0, na1 = int(lv0.n_active), int(lv1.n_active)
    c1 = np.asarray(lv1.coords[:na1])
    io = np.arange(bs)
    gi = [c1[:, d, None] * bs + io[None, :] for d in range(3)]            # (na1, bs) fine cell index
    blk1 = rho_f[gi[0][:, :, None, None], gi[1][:, None, :, None], gi[2][:, None, None, :]]
    src1 = np.full(lv1.data[:, 0:1].shape, np.nan)
    src1[:na1, 0] = blk1
    src0 = np.full(lv0.data[:, 0:1].shape, np.nan)
    src0[:na0] = np.asarray(partition_to_blocks(jnp.asarray(rho_c)[None], acfg))
    solve = jax.jit(lambda lv, rb: solve_phi_hierarchy(lv, acfg, fourpiG=4.0 * np.pi * G, boundary=3,
                                                       n_vcycles=cycles, fmg_kwargs=dict(n_extra_vcycles=extra),
                                                       rho_blocks=rb))
    phis = solve(h, [jnp.asarray(src0), jnp.asarray(src1)])
    # level 1 against the analytic phi at the fine centres; level 0 where it is not covered
    hf = L / (2 * n0)
    xf = [(g + 0.5) * hf - 0.5 * L for g in gi]
    ref1 = phi_spheroid(xf[0][:, :, None, None], xf[1][:, None, :, None], xf[2][:, None, None, :], A1, e)
    d1 = np.abs(np.asarray(phis[1][:na1, 0]) - ref1)
    covered = np.zeros(na0, bool)
    covered[np.asarray(lv1.parent_slots[:na1])] = True
    c0 = np.asarray(lv0.coords[:na0])
    gc = [c0[:, d, None] * bs + io[None, :] for d in range(3)]
    xc = [(g + 0.5) * (L / n0) - 0.5 * L for g in gc]
    ref0 = phi_spheroid(xc[0][:, :, None, None], xc[1][:, None, :, None], xc[2][:, None, None, :], A1, e)
    d0 = np.abs(np.asarray(phis[0][:na0, 0]) - ref0)[~covered]
    ref0 = ref0[~covered]
    l1 = (d1.sum() * hf ** 3 + d0.sum() * (L / n0) ** 3) / (np.abs(ref1).sum() * hf ** 3
                                                           + np.abs(ref0).sum() * (L / n0) ** 3)
    linf = max(d1.max(), d0.max()) / max(np.abs(ref1).max(), np.abs(ref0).max())
    print(f"MEASURED maclaurin AMR e={e} N0={n0} (level 1 {na1} blocks, eff {2 * n0}): L1 {l1:.4e} "
          f"Linf {linf:.4e}; level-1 L1 {d1.sum() / np.abs(ref1).sum():.4e}")
    return float(l1), float(linf)


def _order(a, b):
    return float(np.log2(a / b))


# -- the analytic potential --------------------------------------------------------------------------

def test_analytic_potential_limits():
    """Sphere limit (e -> 0), interior closed form (Chandrasekhar eq. 36), far field -GM/r, exterior harmonic."""
    r = np.linspace(0.01, 1.0, 200)
    sph = np.where(r <= A1, -2 * np.pi * G * RHO * (A1 ** 2 - r ** 2 / 3), -4 / 3 * np.pi * G * RHO * A1 ** 3 / r)
    for x, y, z in ((r, 0 * r, 0 * r), (0 * r, 0 * r, r), (r / np.sqrt(3),) * 3):
        check("sphere limit e=1e-3 |dphi|/|phi|", np.max(np.abs(phi_spheroid(x, y, z, A1, 1e-3) / sph - 1)), 1e-5)
    e = 0.5
    s, asn = np.sqrt(1 - e * e), np.arcsin(e)
    a1c = s * asn / e ** 3 - (1 - e * e) / e ** 2
    a3c = 2 / e ** 2 - 2 * s * asn / e ** 3
    x, y, z = 0.1, 0.05, 0.08
    closed = -np.pi * G * RHO * (2 * s * asn / e * A1 ** 2 - a1c * (x * x + y * y) - a3c * z * z)
    check("interior vs Chandrasekhar eq. 36", abs(phi_spheroid(x, y, z, A1, e) / closed - 1), 1e-13)
    M = 4 / 3 * np.pi * RHO * A1 ** 3 * s
    check("far field r=100 |phi r/GM + 1|", abs(phi_spheroid(60.0, 0.0, 80.0, A1, e) * 100.0 / (G * M) + 1), 1e-6)
    hh, p = 1e-3, np.array([0.3, 0.1, 0.2])
    lap = sum(phi_spheroid(*(p + hh * np.eye(3)[k]), A1, e) + phi_spheroid(*(p - hh * np.eye(3)[k]), A1, e)
              for k in range(3)) - 6 * phi_spheroid(*p, A1, e)
    check("exterior |lap phi| / (4 pi G rho)", abs(lap / hh ** 2) / (4 * np.pi * G * RHO), 1e-5)
    p = np.array([0.1, 0.05, 0.08])
    lap = sum(phi_spheroid(*(p + hh * np.eye(3)[k]), A1, e) + phi_spheroid(*(p - hh * np.eye(3)[k]), A1, e)
              for k in range(3)) - 6 * phi_spheroid(*p, A1, e)
    check("interior |lap phi / (4 pi G rho) - 1|", abs(lap / hh ** 2 / (4 * np.pi * G * RHO) - 1), 1e-5)


# -- uniform isolated solver -------------------------------------------------------------------------

# measured (L1, Linf), FP64, A100 (CPU equal at 32 and 64 to the printed digits); bounds are 1.15x
UNIFORM = {  # (e, N): (multipole faces, analytic faces)
    (0.5, 32): ((7.7735e-02, 4.8179e-02), (1.6629e-02, 2.4621e-02)),
    (0.5, 64): ((1.9483e-02, 1.3057e-02), (4.2050e-03, 7.1745e-03)),
    (0.5, 128): ((4.8563e-03, 3.4277e-03), (1.0528e-03, 1.9700e-03)),
    (0.5, 256): ((1.2012e-03, 8.7754e-04), (2.6338e-04, 5.2170e-04)),
    (0.9, 32): ((7.8534e-02, 5.3983e-02), (1.8120e-02, 3.3960e-02)),
    (0.9, 64): ((1.9648e-02, 1.5317e-02), (4.5980e-03, 1.0408e-02)),
    (0.9, 128): ((4.7845e-03, 4.1516e-03), (1.1526e-03, 2.9990e-03)),
    (0.9, 256): ((1.0738e-03, 1.0377e-03), (2.8803e-04, 8.2333e-04)),
}
AMR = {(32, 8): (1.9112e-02, 1.2801e-02), (64, 8): (4.7686e-03, 3.3792e-03), (128, 8): (1.2033e-03, 9.1581e-04)}
MARGIN = 1.15
FACES = ("multipole", "analytic-BC")


def _check_uniform(n, e):
    for face, got, want in zip(FACES, uniform(n, e), UNIFORM[e, n]):
        check(f"maclaurin e={e} N={n} {face} L1", got[0], MARGIN * want[0])
        check(f"maclaurin e={e} N={n} {face} Linf", got[1], MARGIN * want[1])


# small cases: 2 extra FMG V-cycles, 4 solve cycles (measured within 0.1 % of the converged 8 / 12 below, 3x cheaper)
CPU = dict(extra=2, cycles=4)
UNIFORM_CPU = ((7.7728e-02, 4.8172e-02), (1.6645e-02, 2.4633e-02))       # e=0.5, N=32
AMR_CPU = (7.6038e-02, 4.7566e-02)                                       # e=0.5, N0=16, 4^3 blocks


def test_isolated_uniform_cpu():
    """32^3: L1 and Linf, both face treatments (the order is on the GPU ladder)."""
    for face, got, want in zip(FACES, uniform(32, 0.5, **CPU), UNIFORM_CPU):
        check(f"maclaurin e=0.5 N=32 {face} L1", got[0], MARGIN * want[0])
        check(f"maclaurin e=0.5 N=32 {face} Linf", got[1], MARGIN * want[1])


# Teeth, CPU-8 FP64, job 31655039 (L1, Linf; bounds 1.15x UNIFORM_CPU): G 5 % high: multipole 0.132, 0.0812;
# analytic faces 0.0312, 0.0425, all four fail. Density one cell off: multipole 0.0786 (passes its L1 bound 0.0894:
# the multipole faces' own error hides it), 0.118; analytic faces 0.0245, 0.0855: the other three fail.
TEETH = {"G+5%": (dict(g=1.05), ((0, 0), (0, 1), (1, 0), (1, 1))),
         "rho-one-cell-off": (dict(shift=1), ((0, 1), (1, 0), (1, 1)))}


@pytest.mark.parametrize("case", list(TEETH))
def test_bound_catches_a_broken_solve(case):
    """Teeth: the solver's G 5 % high fails all four 32^3 bounds; the density one cell off fails three of them."""
    broken, fails = TEETH[case]
    got = uniform(32, 0.5, **CPU, **broken)
    for k, j in fails:
        check(f"maclaurin e=0.5 N=32 {FACES[k]} {('L1', 'Linf')[j]} with {case}", got[k][j],
              MARGIN * UNIFORM_CPU[k][j], lower=True)


@gpu_only
def test_amr_small():
    """Base 16, 4^3 blocks, level 1 over the central half (effective 32): composite errors (12 s CPU, GPU tier)."""
    l1, linf = amr(16, 0.5, bs=4, **CPU)
    check("maclaurin AMR e=0.5 N0=16 L1", l1, MARGIN * AMR_CPU[0])
    check("maclaurin AMR e=0.5 N0=16 Linf", linf, MARGIN * AMR_CPU[1])


@gpu_only
def test_isolated_uniform_ladder_gpu():
    """32 ... 256, e = 0.5 and 0.9: L1, Linf and the fitted orders (L1 measured 1.99-2.06, Linf 1.79-1.94)."""
    ns = (32, 64, 128, 256)
    for e in (0.5, 0.9):
        for n in ns:
            _check_uniform(n, e)
        for k, face in enumerate(FACES):
            l1 = [uniform(n, e)[k][0] for n in ns]
            li = [uniform(n, e)[k][1] for n in ns]
            print(f"MEASURED maclaurin e={e} {face} L1 orders {np.round(-np.diff(np.log2(l1)), 3)} "
                  f"Linf orders {np.round(-np.diff(np.log2(li)), 3)}")
            f1 = float(-np.polyfit(np.log2(ns), np.log2(l1), 1)[0])
            fi = float(-np.polyfit(np.log2(ns), np.log2(li), 1)[0])
            check(f"maclaurin e={e} {face} L1 fitted order", f1, 1.9, lower=True)
            check(f"maclaurin e={e} {face} L1 fitted order", f1, 2.2)
            check(f"maclaurin e={e} {face} Linf fitted order", fi, 1.65, lower=True)


@gpu_only
def test_amr_ladder_gpu():
    """Base 32, 64, 128 with one static level (effective 64 ... 256): L1 order measured 2.00, 1.99."""
    ns = (32, 64, 128)
    res = [amr(n, 0.5) for n in ns]
    for n, (l1, linf) in zip(ns, res):
        check(f"maclaurin AMR e=0.5 N0={n} L1", l1, MARGIN * AMR[n, 8][0])
        check(f"maclaurin AMR e=0.5 N0={n} Linf", linf, MARGIN * AMR[n, 8][1])
    l1 = [r[0] for r in res]
    li = [r[1] for r in res]
    print(f"MEASURED maclaurin AMR L1 orders {np.round(-np.diff(np.log2(l1)), 3)} "
          f"Linf orders {np.round(-np.diff(np.log2(li)), 3)}")
    for k in range(2):
        check(f"maclaurin AMR L1 order {ns[k]}->{ns[k + 1]}", _order(l1[k], l1[k + 1]), 1.9, lower=True)
        check(f"maclaurin AMR Linf order {ns[k]}->{ns[k + 1]}", _order(li[k], li[k + 1]), 1.75, lower=True)
