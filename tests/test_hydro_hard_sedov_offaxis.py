"""Off-axis Sedov-Taylor (2D, cylindrical): a blast centred off every grid symmetry point against the exact
solution, for isotropy and grid orientation.

The blast sits at XC, a fixed point of the unit box that is no cell centre, cell corner or symmetry point of
the grid at any N (its sub-cell offset changes with N); the on-centre run puts it on the box centre, a cell
corner, where the grid is mirror- and x <-> y symmetric. The energy E is the exact cell average of a Gaussian
of fixed width SIGMA (separable erf differences, renormalised so the cells hold E to round-off): the same
continuum IC at every N. Periodic box, the shock stays clear of it: mass, energy and momentum to round-off.

Exact: Kamm 2000 (LA-UR-00-6055) closed form, cylindrical j = 2 (Kamm & Timmes 2007, LA-UR-07-2849),
`src/luna/hydro/exact_sedov.py`. Measured, about the true centre: the angle-mean shock radius against
R_s(T) in cells; its scatter over NA wedges (rms of R(theta) / R_s) and that of the post-shock peak; the
m = 1 (a drift of the blast centre) and m = 4 (grid orientation) Fourier amplitudes of R(theta) in cells, the
m = 4 of the peak; L1(rho) / rho2 in the band BAND R_s. The estimator's own lattice floor is measured on an
exactly isotropic field (`synthetic`). 3D: 48 equal-area wedges, minmod, on the GPU.
Bounds, jobs, hardware: docs/TEST_PLAN.md (off-axis Sedov)."""
import functools
import math

import numpy as np
import pytest

from luna.hydro import exact_sedov as se
from tests._helpers import accuracy_cfg, check, evolve, gpu_only

G, RHO0, P_BG, E0, T = 1.4, 1.0, 1e-5, 1.0, 0.09     # R_s(T) = 0.301
SIGMA = 0.004                                       # Gaussian width of the deposit
XC_OFF = (0.4637, 0.5219, 0.5147)                   # off every grid point at every N (2D: x, y)
XC_ON = (0.5, 0.5, 0.5)                                # the box centre: a cell corner, grid-symmetric
NA = 32                                             # angle wedges
BAND = (0.4, 1.2)                                   # L1 band in r / R_s
LIMITERS = {0: "minmod", 1: "mc", 2: "vanleer"}


def initial(n, xc, ndim=2):
    """(rho, m..., E) on n^ndim cells of the unit box: ambient at rest plus E0 as exact Gaussian cell averages."""
    edges = np.arange(n + 1) / n
    w = []
    for c in xc:
        phi = np.array([0.5 * math.erf((e - c) / (math.sqrt(2.0) * SIGMA)) for e in edges])
        d = np.diff(phi)
        w.append(d / d.sum())
    e = E0 * functools.reduce(np.multiply.outer, w) * n ** ndim
    rho = np.full(e.shape, RHO0)
    return np.stack([rho] + [np.zeros_like(rho)] * ndim + [P_BG / (G - 1.0) + e])


def cfg_for(n, limiter=0, ndim=2):
    """The run's config: periodic unit box, HLLC, PLM with `limiter`, FP64 (the test's precision)."""
    grid = dict(Nx=n, Ny=n, Nz=n if ndim == 3 else 1)
    return accuracy_cfg(test_problem="Sedov-Blast-3D" if ndim == 3 else "Gresho-Vortex-2D", SPATIAL_DIMENSION=ndim,
                        BOUNDARY=0, gamma=G, LIMITER=limiter, t_end=T, **grid)


def coords(n, xc, ndim=2):
    """(r, theta) of the cell centres about xc; theta the azimuth (2D) or the equal-area wedge label (3D)."""
    x = (np.arange(n) + 0.5) / n
    if ndim == 2:
        X, Y = np.meshgrid(x - xc[0], x - xc[1], indexing="ij")
        return np.hypot(X, Y), np.arctan2(Y, X)
    X, Y, Z = np.meshgrid(x - xc[0], x - xc[1], x - xc[2], indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z)
    return r, (np.arctan2(Y, X), Z / np.maximum(r, 1e-300))


def wedges(theta, ndim=2):
    """Wedge index of every cell: NA azimuth wedges (2D); 8 azimuth x 6 cos(polar) bins, equal area (3D)."""
    if ndim == 2:
        return np.minimum(((theta + np.pi) / (2 * np.pi) * NA).astype(int), NA - 1), NA
    phi, mu = theta
    a = np.minimum(((phi + np.pi) / (2 * np.pi) * 8).astype(int), 7)
    b = np.minimum(((mu + 1.0) / 2.0 * 6).astype(int), 5)
    return a * 6 + b, 48


def _crossing(prof, x, level):
    """Outermost x beyond the peak where prof falls through level, linear between bins."""
    i0 = int(np.argmax(prof))
    j = i0 + int(np.argmax(prof[i0:] < level))
    return x[j - 1] + (level - prof[j - 1]) / (prof[j] - prof[j - 1]) * (x[j] - x[j - 1])


def shock_and_peak(r, rho, k, nk, dx, rmax):
    """Per wedge, from the wedge-averaged rho in 0.5 dx radial bins: the outermost radius where it falls through
    (peak + rho0) / 2, and the peak."""
    nr = int(rmax / (0.5 * dx)) + 1
    m = r < rmax
    idx = k[m] * nr + (r[m] / (0.5 * dx)).astype(int)
    cnt = np.bincount(idx, minlength=nk * nr).reshape(nk, nr)
    tot = np.bincount(idx, rho[m], minlength=nk * nr).reshape(nk, nr)
    rc = (np.arange(nr) + 0.5) * 0.5 * dx
    R, P = np.empty(nk), np.empty(nk)
    for w in range(nk):
        ok = cnt[w] > 0
        prof, x = tot[w][ok] / cnt[w][ok], rc[ok]
        P[w] = prof.max()
        R[w] = _crossing(prof, x, 0.5 * (P[w] + RHO0))
    return R, P


def fourier(a, m):
    """Amplitude of the cos/sin(m theta) mode of a over the NA wedge centres."""
    ang = (np.arange(NA) + 0.5) / NA * 2 * np.pi - np.pi
    return 2.0 * abs(np.sum(a * np.exp(1j * m * ang))) / NA


def analyse(U, n, xc, ndim=2):
    """Metrics of the end state about xc (see the module docstring)."""
    dx = 1.0 / n
    rho_ex_fn = functools.partial(se.solution, t=T, E0=E0, rho0=RHO0, gamma=G, j=ndim)
    Rs = rho_ex_fn(np.zeros(1))[3]
    r, theta = coords(n, xc, ndim)
    k, nk = wedges(theta, ndim)
    R, P = shock_and_peak(r, U[0], k, nk, dx, 1.5 * Rs)
    out = dict(dR=(R.mean() - Rs) / dx, sR=R.std() / Rs, sP=P.std() / P.mean(), peak=P.mean())
    if ndim == 2:
        out.update(m1=fourier(R, 1) / dx, m4=fourier(R, 4) / dx, m4P=fourier(P, 4) / P.mean())
    band = (r > BAND[0] * Rs) & (r < BAND[1] * Rs)
    rho_ex = rho_ex_fn(r[band])[0]
    rho2 = RHO0 * (G + 1) / (G - 1)
    out["L1"] = float(np.mean(np.abs(U[0][band] - rho_ex))) / rho2
    out["Rs_cells"] = Rs / dx
    return {key: float(v) for key, v in out.items()}


def synthetic(n, xc, ndim=2, width=1.5):
    """An exactly isotropic rho: the exact profile smoothed in r by a Gaussian of `width` cells, at the cell
    centres about xc; the estimator's lattice artefact is what analyse() finds in it."""
    Rs = se.solution(np.zeros(1), T, E0, RHO0, G, ndim)[3]
    dx = 1.0 / n
    rr = np.arange(0.0, 2.0 * Rs, 0.02 * dx)
    prof = se.solution(rr, T, E0, RHO0, G, ndim)[0]
    s = width / 0.02
    ker = np.exp(-0.5 * (np.arange(-int(5 * s), int(5 * s) + 1) / s) ** 2)
    pad = np.concatenate([prof[::-1], prof, np.full(ker.size, RHO0)])
    sm = np.convolve(pad, ker / ker.sum(), mode="same")[prof.size:2 * prof.size]
    r, _ = coords(n, xc, ndim)
    return np.interp(r, rr, sm, right=RHO0)[None]


def end_state(n, limiter=0, on_centre=False, ndim=2, mesh_shape=None):
    """(U at T, steps, xc); evolve checks mass, energy, momentum to round-off (periodic)."""
    xc = (XC_ON if on_centre else XC_OFF)[:ndim]
    U, steps = evolve(cfg_for(n, limiter, ndim), initial(n, xc, ndim), T, mesh_shape)
    return U, steps, xc


@functools.lru_cache(maxsize=None)
def run(n, limiter=0, on_centre=False, ndim=2):
    """The metrics at T (cached)."""
    U, steps, xc = end_state(n, limiter, on_centre, ndim)
    m = analyse(U, n, xc, ndim)
    print(f"MEASURED sedov-offaxis {ndim}D N={n} {LIMITERS[limiter]} {'on' if on_centre else 'off'}-centre "
          f"steps={steps} " + " ".join(f"{a}={b:.6g}" for a, b in m.items()))
    return m


# ---------------------------------------------------------------- measured, A100 FP64 (CPU: N = 64, 2D)
# Jobs 31377452 (N = 64; CPU-8 31379583 equal to 6 digits), 31377027, 31377015-17 (2D), 31377019 (3D);
# docs/TEST_PLAN.md (off-axis Sedov).
# (dR, sR, sP, m1, m4, m4P, L1) per (limiter, on_centre) and N. Bounds: |dR - measured| <= DR_TOL cells, the
# rest <= MARGIN x measured; m1 off-centre <= measured + M1_TOL (the estimator floor), on-centre <= M1_SYM.
KEYS = ("dR", "sR", "sP", "m1", "m4", "m4P", "L1")
DR_TOL, MARGIN, M1_TOL, M1_SYM = 0.05, 1.1, 0.003, 1e-8
CPU_N = 64
MEAS_CPU = {
    (0, False): (0.902, 0.0176, 0.0621, 0.115, 0.449, 0.0857, 0.0571),
    (0, True): (0.920, 0.0236, 0.0630, 0.0, 0.643, 0.0872, 0.0552),
}
GPU_NS = (256, 512, 1024, 2048)
MEAS_GPU = {
    (0, False): {256: (0.983, 7.03e-4, 0.0228, 0.0139, 0.0373, 0.0318, 0.0229),
                 512: (1.001, 4.09e-4, 0.0177, 0.00232, 0.0819, 0.0251, 0.0137),
                 1024: (0.998, 4.04e-4, 0.0116, 0.00547, 0.174, 0.0164, 0.00766),
                 2048: (0.951, 2.31e-4, 0.00997, 0.00667, 0.196, 0.0137, 0.00407)},
    (0, True): {256: (0.999, 6.65e-4, 0.0236, 0.0, 0.0318, 0.0333, 0.0228),
                512: (1.009, 1.77e-4, 0.0176, 0.0, 0.0330, 0.0248, 0.0136),
                1024: (1.017, 4.34e-4, 0.0118, 0.0, 0.186, 0.0167, 0.00760),
                2048: (0.939, 2.09e-4, 0.00794, 0.0, 0.180, 0.0112, 0.00401)},
    (1, False): {256: (0.770, 8.86e-4, 0.0212, 0.0325, 0.0193, 0.0294, 0.0194),
                 512: (0.748, 2.97e-4, 0.0142, 0.0101, 0.0586, 0.0200, 0.0110),
                 1024: (0.689, 2.07e-4, 0.0110, 0.00504, 0.0764, 0.0152, 0.00582),
                 2048: (0.593, 8.39e-5, 0.00864, 0.00293, 0.0112, 0.0117, 0.00286)},
    (1, True): {256: (0.796, 8.18e-4, 0.0205, 0.0, 0.00268, 0.0284, 0.0195),
                512: (0.752, 2.58e-4, 0.0135, 0.0, 0.0506, 0.0190, 0.0111),
                1024: (0.688, 1.45e-4, 0.00939, 0.0, 0.0551, 0.0133, 0.00587),
                2048: (0.591, 7.19e-5, 0.00672, 0.0, 0.00817, 0.00939, 0.00283)},
    (2, False): {256: (0.824, 8.60e-4, 0.0214, 0.0266, 0.0421, 0.0298, 0.0203),
                 512: (0.811, 4.09e-4, 0.0153, 0.00794, 0.0861, 0.0215, 0.0118),
                 1024: (0.773, 3.46e-4, 0.0108, 0.00219, 0.148, 0.0151, 0.00631),
                 2048: (0.686, 9.58e-5, 0.00853, 0.00493, 0.0711, 0.0115, 0.00318)},
    (2, True): {256: (0.847, 7.43e-4, 0.0210, 0.0, 0.0221, 0.0293, 0.0203),
                512: (0.814, 3.14e-4, 0.0145, 0.0, 0.0669, 0.0205, 0.0118),
                1024: (0.778, 3.23e-4, 0.00956, 0.0, 0.141, 0.0135, 0.00632),
                2048: (0.694, 1.13e-4, 0.00781, 0.0, 0.0916, 0.0108, 0.00318)},
}
# L1 order over GPU_NS, measured off / on: minmod 0.831 / 0.837, MC 0.921 / 0.926, van Leer 0.893 / 0.892
ORDER_LO = {0: 0.78, 1: 0.87, 2: 0.84}
L1_COST = 1.03          # L1 off-centre / on-centre at every GPU N; measured max 1.016 (minmod 2048)
# 3D minmod (dR, sR, sP, L1): 128^3 off / on, 256^3 off (on, recorded: 1.071, 8.01e-4, 0.0154, 0.0217)
MEAS_3D = {(128, False): (1.001, 1.24e-3, 0.0204, 0.0327), (128, True): (1.003, 1.12e-3, 0.0194, 0.0326),
           (256, False): (1.073, 8.93e-4, 0.0176, 0.0216)}
# Estimator floor: analyse() of synthetic() at 2048^2, (sR, m1, m4) off / on centre, measured
FLOOR = {False: (1.52e-5, 0.00640, 0.00264), True: (1.74e-5, 0.0, 0.00734)}


def check_metrics(tag, m, meas, on_centre, keys=KEYS):
    """Every metric in `m` against its measured value under the margin rules above."""
    for key, v in zip(keys, meas):
        if key == "dR":
            check(f"{tag} |dR - {v}| (cells)", abs(m[key] - v), DR_TOL)
        elif key == "m1":
            check(f"{tag} m1 (cells)", m[key], M1_SYM if on_centre else v + M1_TOL)
        else:
            check(f"{tag} {key}", m[key], MARGIN * v)


@pytest.mark.parametrize("gamma, alpha", [(1.4, 0.9840), (5.0 / 3.0, 0.5643)])
def test_exact_sedov_cylindrical_alpha(gamma, alpha):
    """The closed form's cylindrical (j = 2) alpha against Kamm & Timmes 2007 (LA-UR-07-2849)."""
    check(f"Sedov j=2 alpha gamma={gamma:.4f} rel err", abs(se.sedov_alpha(gamma, 2) - alpha) / alpha, 2e-4)


@pytest.mark.parametrize("on_centre", [False, True], ids=["off", "on"])
def test_estimator_floor(on_centre):
    """An exactly isotropic field (the smoothed exact profile) at 2048^2 gives a scatter and m = 1, 4 modes far
    below the runs'; analysed about a centre half a cell off, m = 1 reads that half cell (teeth)."""
    n = 2048
    xc = (XC_ON if on_centre else XC_OFF)[:2]
    rho = synthetic(n, xc)
    m = analyse(rho, n, xc)
    tag = f"sedov-offaxis estimator floor N={n} {'on' if on_centre else 'off'}-centre"
    sR, m1, m4 = FLOOR[on_centre]
    check(f"{tag} sR", m["sR"], MARGIN * sR)
    check(f"{tag} m1 (cells)", m["m1"], M1_SYM if on_centre else m1 + M1_TOL)
    check(f"{tag} m4 (cells)", m["m4"], MARGIN * m4)
    shifted = analyse(rho, n, (xc[0] + 0.5 / n, xc[1]))
    check(f"{tag} m1 about xc + dx/2 (cells)", shifted["m1"], 0.45, lower=True)
    check(f"{tag} m1 about xc + dx/2 (cells)", shifted["m1"], 0.55)


@pytest.mark.parametrize("on_centre", [False, True], ids=["off", "on"])
def test_sedov_offaxis_cpu(on_centre):
    """N = 64, minmod: shock radius, scatter over angle, L1; mass, energy, momentum to round-off (in evolve);
    on the box centre the m = 1 mode is round-off (mirror symmetry)."""
    m = run(CPU_N, 0, on_centre)
    check_metrics(f"sedov-offaxis 2D N={CPU_N} minmod {'on' if on_centre else 'off'}", m,
                  MEAS_CPU[0, on_centre], on_centre)


def test_bound_catches_a_wrong_energy():
    """Teeth: 1.05 E0 deposited (R_s 1.2 %, ~0.24 cells high at N = 64) fails the off-centre CPU dR band."""
    xc = XC_OFF[:2]
    U = initial(CPU_N, xc)
    U[-1] = P_BG / (G - 1.0) + 1.05 * (U[-1] - P_BG / (G - 1.0))
    m = analyse(evolve(cfg_for(CPU_N), U, T)[0], CPU_N, xc)
    print(f"MEASURED sedov-offaxis teeth 1.05 E0 N={CPU_N}: dR={m['dR']:.4g} L1={m['L1']:.4g}")
    check(f"sedov-offaxis teeth 1.05 E0 |dR - {MEAS_CPU[0, False][0]}| (cells)", abs(m["dR"] - MEAS_CPU[0, False][0]),
          DR_TOL, lower=True)


def test_sedov_offaxis_sharded():
    """The off-centre N = 64 run on a (2, 4) mesh is bitwise the one-device run."""
    import jax
    if len(jax.devices()) < 8:
        pytest.skip("needs 8 devices (CPU: --xla_force_host_platform_device_count=8)")
    one = end_state(CPU_N)[0]
    eight = end_state(CPU_N, mesh_shape=(2, 4))[0]
    check("sedov-offaxis N=64 max |U_(2,4) - U_1|", float(np.abs(eight - one).max()), 0.0)


@gpu_only
@pytest.mark.parametrize("limiter", list(LIMITERS), ids=list(LIMITERS.values()))
def test_sedov_offaxis_ladder_gpu(limiter):
    """N = 256..2048, off and on centre: every metric per N, the L1 order, and what the offset costs in L1."""
    name = LIMITERS[limiter]
    out = {(n, on): run(n, limiter, on) for on in (False, True) for n in GPU_NS}
    for on in (False, True):
        tag = f"sedov-offaxis 2D {name} {'on' if on else 'off'}"
        for key in ("L1", "sR", "m4"):
            order = -np.polyfit(np.log(GPU_NS), np.log([out[n, on][key] for n in GPU_NS]), 1)[0]
            print(f"MEASURED {tag} {key} order = {order:.4f}")
        for n in GPU_NS:
            check_metrics(f"{tag} N={n}", out[n, on], MEAS_GPU[limiter, on][n], on)
        order = -np.polyfit(np.log(GPU_NS), np.log([out[n, on]["L1"] for n in GPU_NS]), 1)[0]
        check(f"{tag} L1 order", order, ORDER_LO[limiter], lower=True)
    for n in GPU_NS:
        off, on = out[n, False], out[n, True]
        print(f"MEASURED sedov-offaxis 2D {name} N={n} off/on: sR {off['sR'] / on['sR']:.3f} "
              f"m4 {off['m4'] / on['m4']:.3f} L1 {off['L1'] / on['L1']:.4f}")
        check(f"sedov-offaxis 2D {name} N={n} L1 off / on", off["L1"] / on["L1"], L1_COST)


@gpu_only
def test_sedov_offaxis_3d_gpu():
    """3D, minmod: 128^3 off and on centre, 256^3 off centre (48 equal-area wedges)."""
    for (n, on), meas in MEAS_3D.items():
        check_metrics(f"sedov-offaxis 3D N={n} minmod {'on' if on else 'off'}", run(n, 0, on, 3), meas, on,
                      keys=("dR", "sR", "sP", "L1"))
