"""Guderley (1942) converging shock on the Cartesian grid against the self-similar exact solution: the fitted
similarity exponent, R_s(t), L1 behind the shock, isotropy, the diagonal mirror and the closed-box budget.

Exact: `src/luna/hydro/exact_guderley.py` (Lazarus 1981 eqs. 2.5-2.9 as in ExactPack's ramsey.py; alpha by
shooting, against Ramsey, Kamm & Bolstad 2012, LA-UR-09-05853, Table 1). Lazarus time: R_s = (-t)^alpha, rho0 =
1, cold gas at rest ahead (p0 = P0, not 0). The surrogate: the exact state at t = -1 (R_s = 1) as cell averages
(centre values, shock cells sub-sampled) in the quadrant (2D, cylindrical j = 2) or octant (3D, spherical j = 3)
[0, L]^d, reflecting faces (BOUNDARY=2: one code for every face, `src/luna/template_config.py:26`), run toward
collapse (t = 0) and stopped before it (R_s = R_FIT[-1]), so the origin cell (A12) never holds the shock.
The far walls stop the inflow: their rarefaction is bounded by the C- characteristic from r = L (dr/dt = u - c
in the exact flow; every signal has dr/dt >= u - c, so the region r < r_c(t) is exact): r_c = 1.20 at t = -0.05
(cyl., L = 2), and the L1 band [0.75, 1.5] R_s keeps >= 0.6 inside r_c at every time (checked: BAND_GAP).
Closed box: mass and energy to round-off in `evolve`.
Measured: R_s(t) from the angle-averaged rho in 0.5 h radial bins (where it rises through (rho0 + rho_s) / 2), in
cells against the exact (the estimator on the exact cell averages, `floor`); alpha fitted to R_s(t) over R_FIT
(collapse time fixed at 0, and free); per-wedge R_s spread (NA wedges); L1 of rho, u_r, p in the band, each over
its post-shock value; the x <-> y mirror (momenta swapped); positivity minima.
Bounds, jobs, hardware: docs/TEST_PLAN.md (Guderley)."""
import functools

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

from luna.hydro import exact_guderley as eg
from tests._helpers import accuracy_cfg, check, evolve, gpu_only

G, RHO0, P0, L = 1.4, 1.0, 1e-6, 2.0
R_L1 = (0.8, 0.5, 0.3, 0.2)                         # R_s at the L1 times
R_FIT = tuple(np.geomspace(0.8, 0.15, 12))          # R_s at the fit times (exact), the last is the end
BAND = (0.75, 1.5)                                  # L1 band in r / R_s
NA = 8                                              # wedges over the quadrant
LIMITERS = {0: "minmod", 1: "mc", 2: "vanleer"}


def lam(j):
    """lambda = 1 / alpha for geometry j at G."""
    return eg.exponent(j, G)


def t_of(R, j):
    """Lazarus time at which the exact shock is at R."""
    return -np.asarray(R) ** lam(j)


@functools.lru_cache(maxsize=None)
def _front(j):
    """The C- characteristic dr/dt = u - c from r = L at t = -1 through the exact flow (dense in t)."""
    def rhs(t, y):
        rho, u, p = eg.solution(np.array([y[0]]), t, j, G, RHO0, P0)
        return [u[0] - np.sqrt(G * p[0] / rho[0])]
    return solve_ivp(rhs, (-1.0, float(t_of(R_FIT[-1], j))), [L], rtol=1e-9, atol=1e-12, dense_output=True).sol


def wall_front(t, j):
    """Innermost radius the far walls' signal can have reached at t (every signal has dr/dt >= u - c)."""
    return float(_front(j)(t)[0])


def exact_cells(X, h, t, j):
    """Exact conserved cell averages (rho, m..., E) at centres X (d, M) of cubes h: centre values; cells the shock
    crosses averaged over 8^d (2D) / 4^d (3D) sub-points."""
    d = X.shape[0]

    def cons(Y):
        r = np.sqrt((Y * Y).sum(0))
        rho, u, p = eg.solution(r, t, j, G, RHO0, P0)
        ur = np.where(r > 0, u / np.maximum(r, 1e-300), 0.0)
        m = [rho * ur * Y[k] for k in range(d)]
        return np.stack([rho] + m + [p / (G - 1.0) + 0.5 * rho * u * u])
    U = cons(X)
    r = np.sqrt((X * X).sum(0))
    cut = np.abs(r - eg.shock_radius(t, j, G)) < 0.5 * np.sqrt(d) * h * 1.01
    if cut.any():
        s = 8 if d == 2 else 4
        off = ((np.arange(s) + 0.5) / s - 0.5) * h
        grids = np.meshgrid(*([off] * d), indexing="ij")
        acc = 0.0
        Xc = X[:, cut]
        for o in zip(*[g.ravel() for g in grids]):
            acc = acc + cons(Xc + np.array(o)[:, None])
        U[:, cut] = acc / s ** d
    return U


def centres(n, d):
    """Cell centres (d, n^d) of [0, L]^d, C order."""
    x = (np.arange(n) + 0.5) * (L / n)
    return np.stack([a.ravel() for a in np.meshgrid(*([x] * d), indexing="ij")])


def initial(n, d):
    """U at t = -1 on n^d cells."""
    j = d
    U = exact_cells(centres(n, d), L / n, -1.0, j)
    return U.reshape((d + 2,) + (n,) * d)


def prims(U, X):
    """(rho, u_r, p) of flattened U at centres X."""
    d = X.shape[0]
    rho = U[0]
    v = U[1:1 + d] / rho
    r = np.sqrt((X * X).sum(0))
    ur = (v * X).sum(0) / np.maximum(r, 1e-300)
    return rho, ur, (G - 1.0) * (U[-1] - 0.5 * rho * (v * v).sum(0))


def _rise(prof, rc, level):
    """Innermost rc where prof rises through level, linear between bins."""
    i = int(np.argmax(prof > level))
    return rc[i - 1] + (level - prof[i - 1]) / (prof[i] - prof[i - 1]) * (rc[i] - rc[i - 1])


def shock_radii(rho, r, wedge, h, rmax):
    """(mean-profile R_s, per-wedge R_s) from rho averaged in 0.5 h radial bins."""
    level = 0.5 * (RHO0 + RHO0 * (G + 1) / (G - 1))
    m = r < rmax
    nb = int(rmax / (0.5 * h)) + 1
    b = (r[m] / (0.5 * h)).astype(int)
    rc = (np.arange(nb) + 0.5) * 0.5 * h

    def one(sel):
        cnt = np.bincount(b[sel], minlength=nb)
        tot = np.bincount(b[sel], rho[m][sel], minlength=nb)
        ok = cnt > 0
        return _rise(tot[ok] / cnt[ok], rc[ok], level)
    w = wedge[m]
    return one(np.ones(b.size, bool)), np.array([one(w == k) for k in range(NA)])


def wedge_of(X):
    """Wedge index: NA azimuth bins of [0, pi/2] (2D); in 3D NA azimuth bins x 1 (the polar angle ignored)."""
    th = np.arctan2(X[1], X[0])
    return np.minimum((th / (0.5 * np.pi) * NA).astype(int), NA - 1)


def analyse(U, n, t, j, floor=True):
    """Metrics of U (C order, (d+2, n...)) at Lazarus time t; see the module docstring."""
    d = U.ndim - 1
    h = L / n
    X = centres(n, d)
    Uf = U.reshape(d + 2, -1)
    r = np.sqrt((X * X).sum(0))
    Rs = float(eg.shock_radius(t, j, G))
    wedge = wedge_of(X)
    R, Rw = shock_radii(Uf[0], r, wedge, h, 2.0 * Rs)
    out = dict(t=t, Rs=Rs, R=R, dR=(R - Rs) / h, wspread=(Rw.max() - Rw.min()) / h, wstd=Rw.std() / h)
    if floor:
        Ue = exact_cells(X, h, t, j)
        Re, Rwe = shock_radii(Ue[0], r, wedge, h, 2.0 * Rs)
        out.update(dR_floor=(Re - Rs) / h, wspread_floor=(Rwe.max() - Rwe.min()) / h)
    band = (r > BAND[0] * Rs) & (r < BAND[1] * Rs)
    num = prims(Uf[:, band], X[:, band])
    ex = prims(exact_cells(X[:, band], h, t, j), X[:, band])
    Rdot = Rs / (lam(j) * -t)
    scale = ((G + 1) / (G - 1) * RHO0, 2 / (G + 1) * Rdot, 2 / (G + 1) * RHO0 * Rdot ** 2)
    out["L1"] = [float(np.mean(np.abs(a - b))) / s for a, b, s in zip(num, ex, scale)]
    rho, _, p = prims(Uf, X)
    out.update(rho_min=float(rho.min()), p_min=float(p.min()), band_gap=wall_front(t, j) - float(r[band].max()))
    T = np.swapaxes(U, 1, 2)[[0, 2, 1] + list(range(3, d + 2))]
    out["mirror"] = float(np.abs(U - T).max() / np.abs(U).max())
    return out


def fit_alpha(ts, Rs):
    """(alpha with collapse at t = 0, alpha and t_c with t_c free) from R_s(t)."""
    a0 = np.polyfit(np.log(-np.asarray(ts)), np.log(Rs), 1)[0]
    (A, a, tc), _ = curve_fit(lambda t, A, a, tc: A * np.clip(tc - t, 1e-12, None) ** a, np.asarray(ts), np.asarray(Rs),
                              p0=(1.0, a0, 0.0), maxfev=20000)
    return float(a0), float(a), float(tc)


def cfg_for(n, d, limiter=0, gamma=G):
    """Quadrant / octant [0, L]^d, reflecting, HLLC, PLM with `limiter`, the test's precision."""
    grid = dict(Nx=n, Ny=n, Nz=n if d == 3 else 1, Lx=L, Ly=L, Lz=L if d == 3 else 1.0)
    return accuracy_cfg(test_problem="Sedov-Blast-3D" if d == 3 else "Gresho-Vortex-2D", SPATIAL_DIMENSION=d,
                        BOUNDARY=2, gamma=gamma, LIMITER=limiter, **grid)


@functools.lru_cache(maxsize=None)
def run(n, d=2, limiter=0, gamma=G):
    """Metrics at the R_FIT and R_L1 times (cached); evolve checks mass and energy to round-off."""
    j = d
    ts = sorted(set(float(t) for t in t_of(R_FIT + R_L1, j)))
    it = iter(ts)
    res, steps = evolve(cfg_for(n, d, limiter, gamma), initial(n, d), [t + 1.0 for t in ts],
                        observe=lambda U: analyse(U, n, next(it), j))
    return {round(r["Rs"], 12): r for r in res}, steps


def ladder(n, d, limiter):
    """Per-run summary: fitted alpha (t_c = 0, free), t_c, max |dR| (cells), L1 at R_L1, wedge spread, mirror."""
    res, steps = run(n, d, limiter)
    fit = [res[round(float(R), 12)] for R in R_FIT]
    a0, a, tc = fit_alpha([r["t"] for r in fit], [r["R"] for r in fit])
    return dict(alpha0=a0, alpha=a, tc=tc, steps=steps, dR=max(abs(r["dR"]) for r in fit),
                floor=max(abs(r["dR_floor"]) for r in fit), wspread=max(r["wspread"] for r in fit),
                mirror=max(r["mirror"] for r in fit), rho_min=min(r["rho_min"] for r in fit),
                p_min=min(r["p_min"] for r in fit), band_gap=min(r["band_gap"] for r in res.values()),
                L1={R: res[round(R, 12)]["L1"] for R in R_L1},
                dR_t=[r["dR"] for r in fit], floor_t=[r["dR_floor"] for r in fit])


NS_CPU = (64, 128)                                  # cells per side of [0, 2]^2: R_s(T0) = 32, 64 cells (256 on the GPU)
NS_GPU = (256, 512, 1024, 2048)
NS_3D = (64, 128, 256)
BAND_GAP = 0.1      # band cells stay this far inside the wall front (r_c - max r); measured min 0.606 (2D), 0.647
# Per (d, limiter), FP64, default dt (CFL 0.8), ~1.3x measured: L1 {N: {R_s: (rho, u_r, p)}} over the post-shock
# values; alpha_err {N: |alpha_fit - alpha|} (t_c = 0), alpha_free the same with t_c free; orders of L1 fitted
# over each ladder (order_cpu: NS_CPU, order_gpu: NS_GPU / NS_3D), measured -0.1 .. +0.3. Jobs: 2D minmod 31383768
# (64-1024), 31383739 (2048), MC 31383740, van Leer 31383743, 3D 31383765 (A100, FP64); CPU = GPU to the printed
# digits at 64-256 (job 31383366, CPU-8).
BOUNDS = {
    (2, 0): dict(
        L1={64: {0.8: (0.031, 0.026, 0.025), 0.5: (0.047, 0.045, 0.042), 0.3: (0.075, 0.067, 0.068), 0.2: (0.12, 0.11, 0.098)},
            128: {0.8: (0.019, 0.015, 0.015), 0.5: (0.028, 0.023, 0.023), 0.3: (0.051, 0.037, 0.04), 0.2: (0.078, 0.049, 0.054)},
            256: {0.8: (0.0099, 0.0078, 0.0076), 0.5: (0.018, 0.012, 0.013), 0.3: (0.033, 0.019, 0.021), 0.2: (0.044, 0.026, 0.029)},
            512: {0.8: (0.0053, 0.004, 0.0041), 0.5: (0.012, 0.0069, 0.0068), 0.3: (0.019, 0.011, 0.011), 0.2: (0.022, 0.016, 0.016)},
            1024: {0.8: (0.0031, 0.0023, 0.0023), 0.5: (0.0061, 0.0037, 0.0035), 0.3: (0.011, 0.0061, 0.0055), 0.2: (0.011, 0.0083, 0.0076)},
            2048: {0.8: (0.0017, 0.0013, 0.0013), 0.5: (0.0033, 0.002, 0.0019), 0.3: (0.0054, 0.0033, 0.0029), 0.2: (0.0061, 0.0046, 0.0042)}},
        # measured L1 rho at 64..2048: R_s 0.8: 0.0234 0.014 0.00755 0.00407 0.00232 0.00127; 0.2: 0.0915 0.0595 0.0338 0.0165 0.00829 0.00462
        order_cpu={0.8: ((0.64, 1.04), (0.70, 1.10), (0.66, 1.06)), 0.2: ((0.52, 0.92), (1.05, 1.45), (0.77, 1.17))},
        # measured (64, 128) 0.742 0.803 0.768 | 0.620 1.158 0.873; over 64..256: 0.817 0.869 0.842 | 0.718 1.035 0.892
        order_gpu={0.8: ((0.75, 1.15), (0.78, 1.18), (0.78, 1.18)), 0.5: ((0.73, 1.13), (0.76, 1.16), (0.82, 1.22)),
                   0.3: ((0.77, 1.17), (0.74, 1.14), (0.84, 1.24)), 0.2: ((0.86, 1.26), (0.74, 1.14), (0.83, 1.23))},
        # measured 0.854 0.875 0.879 | 0.826 0.862 0.924 | 0.868 0.838 0.943 | 0.961 0.835 0.934
        alpha_err={64: 0.05, 128: 0.014, 256: 0.0031, 512: 0.0016, 1024: 0.00047, 2048: 0.00035},
        # measured +0.03792 +0.01067 +0.002322 +0.001229 +0.0003596 +0.0002692
        alpha_free={64: 0.017, 128: 0.0034, 256: 0.0021, 512: 0.0012, 1024: 0.0016, 2048: 0.00052}),
        # measured -0.01243 -0.002605 +0.001558 +0.000871 +0.001216 +0.000399
    (2, 1): dict(
        L1={256: {0.8: (0.0061, 0.0048, 0.0051), 0.5: (0.011, 0.0081, 0.0083), 0.3: (0.02, 0.014, 0.015), 0.2: (0.028, 0.02, 0.024)},
            512: {0.8: (0.0036, 0.0025, 0.0027), 0.5: (0.0081, 0.0048, 0.0046), 0.3: (0.016, 0.0076, 0.0075), 0.2: (0.014, 0.012, 0.013)},
            1024: {0.8: (0.0024, 0.0015, 0.0016), 0.5: (0.0049, 0.0025, 0.0024), 0.3: (0.0092, 0.0041, 0.0038), 0.2: (0.0078, 0.0057, 0.0062)},
            2048: {0.8: (0.0013, 0.00084, 0.00084), 0.5: (0.0026, 0.0015, 0.0013), 0.3: (0.0046, 0.0025, 0.0022), 0.2: (0.0041, 0.0032, 0.0032)}},
        # measured L1 rho at 256..2048: R_s 0.8: 0.00464 0.00276 0.00179 0.000984; 0.2: 0.0208 0.0107 0.00593 0.00312
        order_gpu={0.8: ((0.63, 1.03), (0.72, 1.12), (0.75, 1.15)), 0.5: ((0.60, 1.00), (0.75, 1.15), (0.81, 1.21)),
                   0.3: ((0.61, 1.01), (0.73, 1.13), (0.82, 1.22)), 0.2: ((0.80, 1.20), (0.78, 1.18), (0.86, 1.26))},
        # measured 0.734 0.821 0.859 | 0.700 0.854 0.911 | 0.713 0.833 0.926 | 0.907 0.882 0.969
        alpha_err={256: 0.0025, 512: 0.00084, 1024: 4e-05, 2048: 0.0001},
        # measured +0.001885 +0.0006448 -2.925e-05 +7.623e-05
        alpha_free={256: 0.0018, 512: 0.00018, 1024: 0.0014, 2048: 0.00052}),
        # measured -0.001326 -0.000133 +0.001006 +0.000394
    (2, 2): dict(
        L1={256: {0.8: (0.0076, 0.0056, 0.0057), 0.5: (0.015, 0.0091, 0.0095), 0.3: (0.028, 0.015, 0.016), 0.2: (0.032, 0.021, 0.024)},
            512: {0.8: (0.0043, 0.003, 0.0031), 0.5: (0.0094, 0.0054, 0.0052), 0.3: (0.017, 0.0083, 0.0082), 0.2: (0.016, 0.013, 0.013)},
            1024: {0.8: (0.0026, 0.0018, 0.0018), 0.5: (0.0051, 0.0028, 0.0026), 0.3: (0.0088, 0.0048, 0.0042), 0.2: (0.0084, 0.0068, 0.0062)},
            2048: {0.8: (0.0014, 0.00095, 0.00094), 0.5: (0.0027, 0.0016, 0.0014), 0.3: (0.0045, 0.0027, 0.0023), 0.2: (0.0045, 0.0035, 0.0033)}},
        # measured L1 rho at 256..2048: R_s 0.8: 0.0058 0.00328 0.00192 0.00104; 0.2: 0.0245 0.0123 0.00642 0.00345
        order_gpu={0.8: ((0.72, 1.12), (0.74, 1.14), (0.76, 1.16)), 0.5: ((0.73, 1.13), (0.76, 1.16), (0.83, 1.23)),
                   0.3: ((0.78, 1.18), (0.72, 1.12), (0.83, 1.23)), 0.2: ((0.84, 1.24), (0.75, 1.15), (0.85, 1.25))},
        # measured 0.822 0.842 0.864 | 0.830 0.861 0.932 | 0.887 0.820 0.933 | 0.943 0.855 0.956
        alpha_err={256: 0.0026, 512: 0.0012, 1024: 0.00024, 2048: 0.00018},
        # measured +0.001925 +0.0008888 +0.0001796 +0.0001367
        alpha_free={256: 0.0003, 512: 0.00042, 1024: 0.0014, 2048: 0.00051}),
        # measured -0.000229 +0.000321 +0.001013 +0.000392
    (3, 0): dict(
        L1={64: {0.8: (0.026, 0.022, 0.021), 0.5: (0.045, 0.037, 0.052), 0.3: (0.11, 0.054, 0.086), 0.2: (0.19, 0.088, 0.13)},
            128: {0.8: (0.015, 0.012, 0.013), 0.5: (0.035, 0.019, 0.029), 0.3: (0.075, 0.029, 0.049), 0.2: (0.14, 0.047, 0.081)},
            256: {0.8: (0.0085, 0.0061, 0.007), 0.5: (0.024, 0.0096, 0.016), 0.3: (0.048, 0.016, 0.024), 0.2: (0.059, 0.026, 0.041)}},
        # measured L1 rho at 64..256: R_s 0.8: 0.0196 0.0115 0.00652; 0.2: 0.145 0.100 0.0446
        order_gpu={0.8: ((0.69, 1.09), (0.80, 1.20), (0.66, 1.06)), 0.5: ((0.36, 0.76), (0.86, 1.26), (0.78, 1.18)),
                   0.3: ((0.48, 0.88), (0.79, 1.19), (0.83, 1.23)), 0.2: ((0.75, 1.15), (0.80, 1.20), (0.72, 1.12))},
        # measured 0.794 0.900 0.764 | 0.462 0.958 0.884 | 0.583 0.885 0.934 | 0.848 0.895 0.822
        alpha_err={64: 0.0014, 128: 0.012, 256: 0.0083},
        # measured -0.001073 -0.009093 -0.006314 (not monotone: the shock goes from 0.36 cells behind to 0.27 ahead)
        alpha_free={64: 0.026, 128: 0.012, 256: 0.0055}),
        # measured +0.01925 +0.00887 +0.0042
}
SHOCK_LAG = {(2, 0): 0.65, (2, 1): 0.40, (2, 2): 0.47, (3, 0): 0.47}   # max |R_s - R_exact| / h, fit times, every N;
# measured 0.496 0.302 0.360 0.357 (the estimator on the exact cell averages: <= 0.095 (2D), 0.147 (3D))
WEDGE_SPREAD = {(2, 0): 1.04, (2, 1): 1.03, (2, 2): 1.07, (3, 0): 0.75}  # max - min of the NA wedge R_s / h;
# measured 0.799 0.788 0.819 0.576 (exact cell averages: <= 0.50)
MIRROR = {(2, 0): 1.4e-11, (2, 1): 4.6e-12, (2, 2): 3.6e-12, (3, 0): 2.3e-14}  # max |U - U^T| / max |U|;
# measured 1.07e-11 3.51e-12 2.72e-12 1.75e-14 (round-off, growing with N toward collapse; <= 5e-15 at N <= 256)


def check_run(tag, s, B, n, d, lim):
    """The per-N checks of one ladder entry."""
    for R, b in B["L1"][n].items():
        for k, name in enumerate(("rho", "u_r", "p")):
            check(f"{tag} N={n} R_s={R} L1 {name}", s["L1"][R][k], b[k])
    check(f"{tag} N={n} |alpha_fit - alpha|", abs(s["alpha0"] - 1.0 / lam(d)), B["alpha_err"][n])
    check(f"{tag} N={n} |alpha_fit(t_c free) - alpha|", abs(s["alpha"] - 1.0 / lam(d)), B["alpha_free"][n])
    print(f"MEASURED {tag} N={n} alpha_fit = {s['alpha0']:.9f} (t_c = 0), {s['alpha']:.9f} (t_c free: {s['tc']:+.3e}); "
          f"exact {1.0 / lam(d):.9f}; steps {s['steps']}")
    check(f"{tag} N={n} max |R_s - R_exact| / h", s["dR"], SHOCK_LAG[d, lim])
    check(f"{tag} N={n} wedge spread of R_s / h", s["wspread"], WEDGE_SPREAD[d, lim])
    check(f"{tag} N={n} mirror x<->y", s["mirror"], MIRROR[d, lim])
    check(f"{tag} N={n} wall front - band max r", s["band_gap"], BAND_GAP, lower=True)
    check(f"{tag} N={n} min p / P0", s["p_min"] / P0, 1.0 - 1e-9, lower=True)
    check(f"{tag} N={n} min rho", s["rho_min"], 0.1, lower=True)


def check_orders(tag, out, ns, orders):
    """L1 orders fitted over ns at each R_s against [lo, hi]."""
    for R, lohi in orders.items():
        for k, name in enumerate(("rho", "u_r", "p")):
            o = -np.polyfit(np.log(ns), np.log([out[n]["L1"][R][k] for n in ns]), 1)[0]
            check(f"{tag} R_s={R} L1 {name} order", o, lohi[k][0], lower=True)
            check(f"{tag} R_s={R} L1 {name} order", o, lohi[k][1])


def test_exact_solution():
    """alpha against RKB Table 1 (9 digits); the strong-shock jump; R from the adiabatic integral against its ODE;
    the radial Euler equations by central differences; ExactPack's state() at three points."""
    for (j, g), a in [(k, v) for k, v in eg.ALPHA_RKB.items() if k[1] in (1.4, 5.0 / 3.0)]:
        check(f"Guderley j={j} gamma={g:.4f} |alpha - RKB Table 1|", abs(1.0 / eg.exponent(j, g) - a), 6e-10)
        sim = eg.similarity(j, g)
        x = -np.geomspace(1.0, 1e-5, 300)
        check(f"Guderley j={j} gamma={g:.4f} max |R_integral / R_ode - 1|",
              float(np.abs(sim.vcr(x)[2] / sim.r_ode(x) - 1).max()), 1.2e-8)
    for j in (2, 3):
        t = -0.3
        Rs = float(eg.shock_radius(t, j, G))
        Rdot = Rs / (lam(j) * -t)
        rho, u, p = eg.solution(np.array([Rs * (1 + 1e-12)]), t, j, G)
        jump = max(abs(rho[0] / ((G + 1) / (G - 1)) - 1), abs(u[0] / (-2 / (G + 1) * Rdot) - 1),
                   abs(p[0] / (2 / (G + 1) * Rdot ** 2) - 1))
        check(f"Guderley j={j} strong-shock jump max rel error", jump, 1e-10)
        worst = 0.0
        for r0, t0 in [(1.2, -1.0), (2.5, -1.0), (1.05, -0.5), (0.3, -0.05), (1.0, -0.2)]:
            hh = 1e-5 * r0
            P = {k: [v[0] for v in eg.solution(np.array([r0 + a * hh]), t0 + b * hh, j, G)]
                 for k, (a, b) in dict(c=(0, 0), rp=(1, 0), rm=(-1, 0), tp=(0, 1), tm=(0, -1)).items()}
            rho, u, p = P["c"]
            dr = [(P["rp"][i] - P["rm"][i]) / (2 * hh) for i in range(3)]
            dt = [(P["tp"][i] - P["tm"][i]) / (2 * hh) for i in range(3)]
            mass = (dt[0] + u * dr[0] + rho * (dr[1] + (j - 1) * u / r0)) / (abs(rho * dr[1]) + abs(u * dr[0]))
            mom = (dt[1] + u * dr[1] + dr[2] / rho) / (abs(u * dr[1]) + abs(dr[2] / rho))
            ent = (dt[2] / p - G * dt[0] / rho + u * (dr[2] / p - G * dr[0] / rho)) * abs(r0 / u)
            worst = max(worst, abs(mass), abs(mom), abs(ent))
        check(f"Guderley j={j} Euler residual (central differences, h = 1e-5 r)", worst, 1e-7)
    # ExactPack ramsey.state() (lanl/ExactPack, run with this lambda; its eexp is minutes): (j, r, t): (rho, u, p)
    ep = {(2, 1.2, -1.0): (7.7278300211, -0.6352153067, 0.6281888655),
          (2, 0.5, -0.3): (8.5845864710, -0.7337602127, 0.9572967617),
          (3, 2.0, -1.0): (14.9342552457, -0.3572849870, 0.4710778514)}
    for (j, r0, t0), want in ep.items():
        got = [v[0] for v in eg.solution(np.array([r0]), t0, j, G)]
        check(f"Guderley j={j} r={r0} t={t0} vs ExactPack max rel", max(abs(a / b - 1) for a, b in zip(got, want)), 1.1e-8)


def test_guderley_cylindrical_cpu():
    """2D quadrant, minmod, N = 64, 128: L1 behind the shock, fitted alpha, shock lag, wedge spread, mirror, wall
    front clearance, positivity at every N; the L1 orders; mass and energy to round-off (in evolve)."""
    B, tag, out = BOUNDS[2, 0], "Guderley cyl minmod", {}
    for n in NS_CPU:
        out[n] = ladder(n, 2, 0)
        check_run(tag, out[n], B, n, 2, 0)
    check_orders(tag, out, NS_CPU, B["order_cpu"])


@gpu_only
@pytest.mark.parametrize("limiter", (0, 1, 2), ids=[LIMITERS[k] for k in (0, 1, 2)])
def test_guderley_cylindrical_gpu(limiter):
    """2D quadrant, N = 256..2048 (R_s(T0) to 1024 cells), minmod, MC, van Leer: as the CPU test."""
    B, tag, out = BOUNDS[2, limiter], f"Guderley cyl {LIMITERS[limiter]}", {}
    for n in NS_GPU:
        out[n] = ladder(n, 2, limiter)
        check_run(tag, out[n], B, n, 2, limiter)
    check_orders(tag, out, NS_GPU, B["order_gpu"])


@gpu_only
def test_guderley_spherical_gpu():
    """3D octant, minmod, N = 64..256: as the 2D test (wedges in azimuth only)."""
    B, tag, out = BOUNDS[3, 0], "Guderley sph minmod", {}
    for n in NS_3D:
        out[n] = ladder(n, 3, 0)
        check_run(tag, out[n], B, n, 3, 0)
    check_orders(tag, out, NS_3D, B["order_gpu"])


def test_bound_catches_wrong_gamma():
    """Teeth: the same IC run with gamma = 5/3 misses the N = 64 L1 and alpha bounds."""
    res, _ = run(64, 2, 0, 5.0 / 3.0)
    fit = [res[round(float(R), 12)] for R in R_FIT]
    a0 = fit_alpha([r["t"] for r in fit], [r["R"] for r in fit])[0]
    l1 = res[round(0.5, 12)]["L1"][0]
    print(f"MEASURED teeth gamma=5/3: alpha_fit {a0:.6f}, L1 rho at R_s=0.5 {l1:.4e}")
    assert abs(a0 - 1.0 / lam(2)) > BOUNDS[2, 0]["alpha_err"][64], f"alpha {a0} of a gamma=5/3 run passes"
    assert l1 > BOUNDS[2, 0]["L1"][64][0.5][0], f"L1 rho {l1:.4e} of a gamma=5/3 run passes"
