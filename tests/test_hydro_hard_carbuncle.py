"""Carbuncle family: Quirk's odd-even duct, a strong shock reflecting off a wall, 2D Noh; HLLC and HLL, minmod.

Each is a 2D problem on a thin 3D grid (nz = 2): 3D is the production path."""
import numpy as np
import pytest

from tests._helpers import accuracy_cfg, check, evolve, gpu_only



def cons3(rho, vx, vy, p, g):
    """3D conserved state from 2D primitives (vz = 0)."""
    return np.stack([rho, rho * vx, rho * vy, np.zeros_like(rho),
                     p / (g - 1.0) + 0.5 * rho * (vx * vx + vy * vy)])


def run(U0, shape, box, times, g, solver, order=1):
    """States at `times` on the thin-z grid `shape` (cubic cells of side box[0]/shape[0]), zero-gradient faces."""
    cfg = accuracy_cfg(test_problem="Sod-Shock-3D-AMR", SPATIAL_DIMENSION=3, Nx=shape[0], Ny=shape[1],
                       Nz=shape[2], Lx=box[0], Ly=box[1], Lz=box[2], BOUNDARY=1, gamma=g,
                       riemann_solver=solver, ORDER=order)
    return evolve(cfg, U0, times)[0]


def row_spread(U):
    """(max over x, z of (max_y rho - min_y rho) / mean_y rho, max |v_y|)."""
    rho = U[0]
    spread = (rho.max(1) - rho.min(1)) / rho.mean(1)
    return float(spread.max()), float(np.abs(U[2] / U[0]).max())


# --- Quirk (1994) odd-even duct: a Mach 6 shock runs along x, the centreline row carries a 1e-6 bump ---
QG, QM, QSHAPE = 1.4, 6.0, (800, 20, 2)
QT = np.linspace(0.0, 0.125, 6)[1:]           # shock from x = 0.05 to 0.80


def quirk_ic(shape=QSHAPE, eps=1e-6):
    """Mach-6 shock at x = 0.05 into rho = 1.4, p = 1 (c = 1); rho of row ny/2 times (1 + eps)."""
    nx, ny, nz = shape
    x = (np.arange(nx) + 0.5) / nx
    r2 = (QG + 1) * QM ** 2 / ((QG - 1) * QM ** 2 + 2)
    p2 = 1.0 + 2 * QG / (QG + 1) * (QM ** 2 - 1)
    post = (x < 0.05)[:, None, None] * np.ones(shape)
    rho = np.where(post, 1.4 * r2, 1.4)
    rho[:, ny // 2, :] *= 1.0 + eps
    return cons3(rho, post * QM * (1.0 - 1.0 / r2), 0.0 * rho, np.where(post, p2, 1.0), QG)


def quirk(solver, order=1, shape=QSHAPE):
    """[(row spread, max |v_y|)] at the QT times of the odd-even duct."""
    box = (1.0, shape[1] / shape[0], shape[2] / shape[0])
    return [row_spread(U) for U in run(quirk_ic(shape), shape, box, QT, QG, solver, order)]


# --- strong shock off a wall (Stone et al. 2008, App. C): two Mach-M streams collide at x = 1/2 ---
WG, WSHAPE = 5.0 / 3.0, (200, 50, 2)


def wall_ic(mach, shape=WSHAPE, noise=1e-4, seed=1):
    """rho = 1 (1 + noise U(-1, 1)), c = 1, vx = +-mach toward x = 1/2; its mirror plane is the wall."""
    nx = shape[0]
    x = (np.arange(nx) + 0.5) / nx
    rho = 1.0 + noise * np.random.default_rng(seed).uniform(-1.0, 1.0, shape)
    vx = np.broadcast_to(np.where(x < 0.5, mach, -mach)[:, None, None], shape)
    return cons3(rho, vx, 0.0 * rho, np.full(shape, 1.0 / WG), WG)


def wall(solver, mach, order=1, shape=WSHAPE):
    """[(row spread, max |v_y|)] behind the reflected shocks, at 4 times until they reach x = 0.1 and 0.9."""
    times = np.linspace(0.0, 1.2 / mach, 5)[1:]
    box = (1.0, shape[1] / shape[0], shape[2] / shape[0])
    return [row_spread(U) for U in run(wall_ic(mach, shape), shape, box, times, WG, solver, order)]


# --- 2D Noh (1987): cold gas falls radially in at speed 1; exact shock at r = t/3, rho = 16 behind ---
NG, NT = 5.0 / 3.0, 0.6


def noh_ic(n):
    """rho = 1, p = 1e-6, v = -r_hat on [-1, 1]^2 x thin z."""
    x = (np.arange(n) + 0.5) * 2.0 / n - 1.0
    X, Y = np.meshgrid(x, x, indexing="ij")
    r = np.hypot(X, Y)
    shape = (n, n, 2)
    X, Y, r = (np.broadcast_to(a[..., None], shape) for a in (X, Y, r))
    return cons3(np.ones(shape), -X / r, -Y / r, np.full(shape, 1e-6), NG), X, Y, r


def noh(solver, n=200, order=1):
    """Noh metrics at t = 0.6: L1 rho (r < 0.4), plateau ring spread, shock radius spread, max |v_phi|, rho(0)."""
    U0, X, Y, r = noh_ic(n)
    U = run(U0, (n, n, 2), (2.0, 2.0, 4.0 / n), [NT], NG, solver, order)[0]
    rho = U[0]
    exact = np.where(r < NT / 3.0, 16.0, 1.0 + NT / r)
    inner = r < 0.4
    l1 = float(np.mean(np.abs(rho - exact)[inner]))
    ring = (r > 0.08) & (r < 0.16)
    ring_spread = float(rho[ring].std() / rho[ring].mean())
    th = np.arctan2(Y, X)[..., 0]
    bins = np.linspace(-np.pi, np.pi, 33)
    rs = [r[..., 0][(th >= a) & (th < b) & (rho[..., 0] > 10.0) & (r[..., 0] < 0.4)].max()
          for a, b in zip(bins[:-1], bins[1:])]
    shock_spread = float((max(rs) - min(rs)) / np.mean(rs))
    vphi = float(np.abs((X * U[2] - Y * U[1]) / (r * rho))[inner].max())
    centre = float(rho[r < 0.03].mean())
    return dict(l1=l1, ring=ring_spread, shock=shock_spread, vphi=vphi, centre=centre,
                r_shock=float(np.mean(rs)))


FINDING = "open bug: HLLC carbuncles at grid-aligned strong shocks (O(1) row spread); HLL stays planar"
HLLC_XFAIL = pytest.param("hllc", marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=FINDING))

PLANAR = {"quirk": (5.5e-7, 5.3e-7),     # (row spread, |v_y|) ~1.3x HLL on CPU-8: 4.38e-7, 4.08e-7; HLLC 0.72, 0.62
          "wall5": (2.3e-3, 3.4e-4),     # measured 1.73e-3, 2.64e-4 (seed noise 1e-4); HLLC 0.86, 0.51
          "wall20": (1.7e-3, 1.6e-3)}    # measured 1.36e-3, 1.24e-3; HLLC 1.05, 2.49
CASES = {"quirk": quirk, "wall5": lambda s: wall(s, 5.0), "wall20": lambda s: wall(s, 20.0)}


@pytest.mark.parametrize("solver", ["hll", HLLC_XFAIL])
@pytest.mark.parametrize("case", list(CASES))
def test_shock_stays_planar(case, solver, cases=CASES, bounds=PLANAR):
    """A grid-aligned strong shock with a 1e-6 / 1e-4 seed stays 1D: row spread and |v_y| at every output time."""
    series = cases[case](solver)
    print(f"MEASURED {case} {solver} (spread, |v_y|) vs t: {series}")
    check(f"{case} {solver} max row spread", max(a for a, _ in series), bounds[case][0])
    check(f"{case} {solver} max |v_y|", max(b for _, b in series), bounds[case][1])


# Noh at n = 200, t = 0.6, ~1.3x measured (CPU-8): HLL l1 0.257, ring 0.00378, shock 0.0272, vphi 0.0409,
# 16 - rho(0) 1.07, |r_s - 0.2| 0.00136; HLLC 0.287, 0.0272, 0.0194, 0.0678, 1.68, 0.00310
NOH_BOUND = {"hll": dict(l1=0.33, ring=0.0049, shock=0.035, vphi=0.054, deficit=1.39, r_err=0.0018),
             "hllc": dict(l1=0.37, ring=0.035, shock=0.025, vphi=0.091, deficit=2.2, r_err=0.0040)}


def noh_checks(m, bnd, tag):
    """Check every Noh metric against its bound."""
    m = dict(m, deficit=16.0 - m["centre"], r_err=abs(m["r_shock"] - NT / 3.0))
    for k, b in bnd.items():
        check(f"noh {tag} {k}", m[k], b)


@pytest.mark.parametrize("solver", ["hll", "hllc"])
def test_noh(solver):
    """2D Noh against the exact solution: density L1, symmetry of plateau and shock, v_phi, wall heating."""
    noh_checks(noh(solver), NOH_BOUND[solver], solver)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=FINDING)
def test_noh_hllc_as_isotropic_as_hll():
    """The HLLC Noh plateau is as isotropic as HLL's (it is ~7x less: the carbuncle)."""
    check("noh hllc ring spread vs the HLL bound", noh("hllc")["ring"], NOH_BOUND["hll"]["ring"])


# GPU tier, one MI300A: twice the resolution of every problem.
PLANAR_GPU = {"quirk": (6.8e-7, 3.9e-7),     # measured 5.38e-7, 2.98e-7; HLLC 0.83, 0.64
              "wall20": (5.3e-3, 1.7e-3)}    # measured 4.16e-3, 1.28e-3; HLLC 1.25, 2.15
CASES_GPU = {"quirk": lambda s: quirk(s, 1, (1600, 40, 2)),
             "wall20": lambda s: wall(s, 20.0, 1, (400, 100, 2))}
# n = 400: HLL l1 0.131, ring 0.00173, shock 0.00678, vphi 0.0392, deficit 0.753, r_err 0.00137;
# HLLC 0.162, 0.0192, 0.0129, 0.0806, 0.857, 0.00190
NOH_BOUND_GPU = {"hll": dict(l1=0.17, ring=0.0023, shock=0.0089, vphi=0.051, deficit=0.97, r_err=0.0018),
                 "hllc": dict(l1=0.21, ring=0.025, shock=0.017, vphi=0.105, deficit=1.12, r_err=0.0025)}


@gpu_only
@pytest.mark.parametrize("solver", ["hll", HLLC_XFAIL])
@pytest.mark.parametrize("case", list(CASES_GPU))
def test_shock_stays_planar_gpu(case, solver):
    """The duct at 1600 x 40 and the wall at 400 x 100, on a GPU."""
    test_shock_stays_planar(case, solver, CASES_GPU, PLANAR_GPU)


@gpu_only
@pytest.mark.parametrize("solver", ["hll", "hllc"])
def test_noh_gpu(solver):
    """2D Noh at 400^2, on a GPU."""
    noh_checks(noh(solver, 400), NOH_BOUND_GPU[solver], solver)
