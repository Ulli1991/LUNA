"""Boost penalty of the fixed grid: density square, Sod and KH, each boosted by 0, 1, 10, 100 c_s, against exact or rest-frame answers.

A fixed grid is not Galilean invariant; these pin how fast the error grows with the boost (Robertson et al. 2010)."""
import numpy as np
import pytest

from luna.hydro import exact_riemann as er
from tests._helpers import accuracy_cfg, check, evolve, gpu_only



# --- density square: rho = 2 in [0.25, 0.75]^2, 1 outside, p = 1, gamma 5/3, boosted along (1, 1)/sqrt 2 ---
SG, ST = 5.0 / 3.0, 0.2
SC = np.sqrt(SG)                 # c_s of the ambient gas


def square_avg(n, shift):
    """Exact cell averages of the periodic square translated by `shift` = (sx, sy)."""
    e = np.arange(n + 1) / n
    def frac(s):
        """Fraction of each cell inside [0.25, 0.75] + s, periodic."""
        lo = np.mod(0.25 + s, 1.0)
        out = np.zeros(n)
        for a in (lo - 1.0, lo, lo + 1.0):
            out += np.clip(np.minimum(e[1:], a + 0.5) - np.maximum(e[:-1], a), 0.0, None) * n
        return out
    return 1.0 + np.outer(frac(shift[0]), frac(shift[1]))


def square(n, boost, order=1, frame=1.0):
    """L1(rho - exact) at ST of the square boosted by `boost` c_s; `frame` scales the exact shift (teeth: 0)."""
    v = boost * SC / np.sqrt(2.0)
    rho = square_avg(n, (0.0, 0.0))
    U0 = np.stack([rho, rho * v, rho * v, 1.0 / (SG - 1.0) + rho * v * v])
    cfg = accuracy_cfg(test_problem="Square-Advected", SPATIAL_DIMENSION=2, Nx=n, Ny=n, Lx=1.0,
                       Ly=1.0, BOUNDARY=0, gamma=SG, ORDER=order)
    U, _ = evolve(cfg, U0, ST)
    return float(np.mean(np.abs(U[0] - square_avg(n, (frame * v * ST,) * 2))))


# --- Sod in a periodic double tube R | L | R (L on [0.25, 0.75]), boosted along x ---
SOD = er.TORO[1][:2]
DG, DT = 1.4, 0.1
DC = np.sqrt(DG)                 # c_s of the left state


def sod_exact(n, t, shift, sub=64):
    """Cell-averaged exact (rho, u, p) of the double tube, comoving frame, cells shifted by -shift."""
    f = (np.arange(sub) + 0.5) / sub
    x = ((np.arange(n)[:, None] + f[None, :]) / n - shift) % 1.0
    rho, u, p = er.double_tube(*SOD, DG, x, t)
    m, mu = rho.mean(1), (rho * u).mean(1)
    E = (p / (DG - 1.0) + 0.5 * rho * u * u).mean(1)
    return m, mu / m, (DG - 1.0) * (E - 0.5 * mu * mu / m)


def sod(n, boost, order=1, frame=1.0):
    """L1 of (rho, u - v, p) at DT of the double Sod tube boosted by `boost` c_s."""
    v = boost * DC
    rho, u, p = sod_exact(n, 1e-12, 0.0)
    u = u + v
    U0 = np.stack([rho, rho * u, p / (DG - 1.0) + 0.5 * rho * u * u])
    cfg = accuracy_cfg(test_problem="Sod-Shock", SPATIAL_DIMENSION=1, Nx=n, Lx=1.0, BOUNDARY=0,
                       gamma=DG, ORDER=order)
    U, _ = evolve(cfg, U0, DT)
    ex = sod_exact(n, DT, frame * v * DT)
    num = (U[0], U[1] / U[0] - v, (DG - 1.0) * (U[2] - 0.5 * U[1] ** 2 / U[0]))
    return [float(np.mean(np.abs(a - b))) for a, b in zip(num, ex)]


# --- KH: the smooth double shear layer of tests/test_hydro_accuracy_kh.py, boosted along x ---
KT = np.arange(0.0, 4.01, 0.25)
KFIT = (2.0, 4.0)


def kh(nx, boost, order=1, seed=1e-5):
    """|sigma / sigma_theory - 1| of the seeded mode's growth rate, boosted along x by `boost` c_s."""
    from luna.hydro.exact_kh_linear import geometry, linear_theory, profile
    g = geometry()
    ny = 2 * nx
    x = (np.arange(nx) + 0.5) * g["Lx"] / nx
    y = (np.arange(ny) + 0.5) * g["Ly"] / ny
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = profile(Y, g["Ly"], g["d"], g["dU"]) + boost * g["cs"]
    env = sum(np.exp(-((Y - y0) / (4 * g["d"])) ** 2) for y0 in (0.25 * g["Ly"], 0.75 * g["Ly"]))
    v = seed * np.sin(g["k"] * X) * env
    rho, p = np.full_like(X, g["rho"]), np.full_like(X, g["p"])
    U0 = np.stack([rho, rho * u, rho * v, p / (g["gamma"] - 1) + 0.5 * rho * (u * u + v * v)])
    cfg = accuracy_cfg(test_problem="Kelvin-Helmholtz", SPATIAL_DIMENSION=2, Nx=nx, Ny=ny,
                       Lx=g["Lx"], Ly=g["Ly"], BOUNDARY=0, gamma=g["gamma"], ORDER=order)
    states, _ = evolve(cfg, U0, KT)
    amp = np.array([np.sqrt(np.sum(np.abs(np.fft.rfft(U[2] / U[0], axis=0)[1]) ** 2)) for U in states])
    fit = (KT >= KFIT[0]) & (KT <= KFIT[1])
    sigma = np.polyfit(KT[fit], np.log(amp[fit]), 1)[0]
    theory = linear_theory(g["k"], g["Ly"], g["d"], g["dU"], g["cs"], g["gamma"], 256)
    return abs(sigma / theory - 1.0)


def penalty(ns, errs, target):
    """N at which the log-log interpolated error ladder `errs`(ns) reaches `target` (extrapolated past the end)."""
    lx, ly = np.log(ns), np.log(errs)
    k = int(np.clip(np.searchsorted(-ly, -np.log(target)), 1, len(ns) - 1))
    s = (ly[k] - ly[k - 1]) / (lx[k] - lx[k - 1])
    return float(np.exp(lx[k - 1] + (np.log(target) - ly[k - 1]) / s))


# Bounds ~1.3x the values measured on CPU-8 (minmod HLLC, default dt), which follow as comments.
SQ_NS = (64, 128, 256)
SQ_BOUND = {1: (0.043, 0.029, 0.021), 10: (0.105, 0.066, 0.043), 100: (0.21, 0.15, 0.096)}
# measured b=1 0.0328, 0.0221, 0.0155; b=10 0.0807, 0.0506, 0.0326; b=100 0.164, 0.114, 0.0732; b=0 exactly 0
SQ_PENALTY = {10: 5.2, 100: 18.5}   # N / 64 to recover the b=1 error at N = 64; measured 3.96, 14.2 (extrapolated)
SOD_NS = (100, 200, 400, 800, 1600)
SOD_BOUND = {
    0: (0.012, 0.0073, 0.0043, 0.0024, 0.0013),
    # measured 0.00937, 0.00559, 0.00333, 0.00183, 0.00100
    1: (0.012, 0.0069, 0.0037, 0.0022, 0.0013),
    # measured 0.00928, 0.00528, 0.00283, 0.00169, 0.00100
    10: (0.034, 0.018, 0.01, 0.0055, 0.0031),
    # measured 0.0264, 0.0140, 0.00766, 0.00425, 0.00240
    100: (0.059, 0.042, 0.025, 0.013, 0.0075),
    # measured 0.0457, 0.0327, 0.0194, 0.0104, 0.00573
}
SOD_PENALTY = {1: 1.3, 10: 4.1, 100: 11.7}   # N / 100 to recover the rest-frame L1 rho at N = 100; measured 0.987, 3.17, 9.00
KH_BOUND = {0: 0.0384, 1: 0.042, 10: 0.126}  # nx = 64; measured 0.0296, 0.0323, 0.0968


def test_square_boost():
    """Square: exact at rest, L1 bound per boost and N, and the N needed at 10 and 100 c_s for the 1 c_s error."""
    check("square b=0 N=64 L1", square(64, 0), 1e-13)
    err = {b: [square(n, b) for n in SQ_NS] for b in SQ_BOUND}
    for b, es in err.items():
        for n, e, bnd in zip(SQ_NS, es, SQ_BOUND[b]):
            check(f"square b={b} N={n} L1 rho", e, bnd)
    for b, bnd in SQ_PENALTY.items():
        check(f"square b={b} resolution penalty vs b=1 at N=64", penalty(SQ_NS, err[b], err[1][0]) / 64, bnd)


def test_sod_boost():
    """Boosted double Sod tube: L1 rho bound per boost and N, and the resolution penalty vs the rest frame."""
    err = {b: [sod(n, b) for n in SOD_NS] for b in SOD_BOUND}
    for b, es in err.items():
        print(f"MEASURED sod b={b} L1 (rho, u, p) {es}")
        for n, e, bnd in zip(SOD_NS, es, SOD_BOUND[b]):
            check(f"sod b={b} N={n} L1 rho", e[0], bnd)
    rest = [e[0] for e in err[0]]
    for b, bnd in SOD_PENALTY.items():
        check(f"sod b={b} resolution penalty at N=100", penalty(SOD_NS, [e[0] for e in err[b]], rest[0]) / 100, bnd)


@pytest.mark.parametrize("boost", list(KH_BOUND))
def test_kh_boost(boost):
    """KH growth-rate error at 64 x 128 with the layer boosted along x by 0, 1, 10 c_s."""
    check(f"kh b={boost} nx=64 |sigma/theory - 1|", kh(64, boost), KH_BOUND[boost])


# GPU tier, one MI300A (ROCm); 64-256 share the CPU bounds.
SQ_NS_GPU = SQ_NS + (512, 1024)
SQ_BOUND_GPU = {b: SQ_BOUND[b] + v for b, v in
                {1: (0.0125, 0.0092), 10: (0.0283, 0.0176), 100: (0.061, 0.039)}.items()}
# measured at 512, 1024: b=1 0.00956, 0.00707; b=10 0.0217, 0.0135; b=100 0.0467, 0.0295
SQ_PENALTY_GPU = {10: 5.2, 100: 17.7}   # interpolated, no extrapolation; measured 3.96, 13.6


@gpu_only
def test_square_boost_gpu():
    """The square ladder up to 1024^2: the 100 c_s penalty interpolated instead of extrapolated."""
    err = {b: [square(n, b) for n in SQ_NS_GPU] for b in SQ_BOUND_GPU}
    for b, es in err.items():
        for n, e, bnd in zip(SQ_NS_GPU, es, SQ_BOUND_GPU[b]):
            check(f"square b={b} N={n} L1 rho", e, bnd)
    for b, bnd in SQ_PENALTY_GPU.items():
        check(f"square b={b} resolution penalty vs b=1 at N=64", penalty(SQ_NS_GPU, err[b], err[1][0]) / 64, bnd)


SOD_NS_GPU = (800, 1600, 3200, 6400)
SOD_BOUND_GPU = {0: (0.00078, 0.00045), 1: (0.00075, 0.00044), 10: (0.0018, 0.00104), 100: (0.0041, 0.0023)}
# measured at 3200, 6400: b=0 0.000598, 0.000346; b=1 0.000576, 0.000339; b=10 0.00138, 0.000798; b=100 0.00317, 0.00178
SOD_PENALTY_GPU = {1: 1.17, 10: 3.66, 100: 10.1}   # N / 800 for the rest-frame error at 800; measured 0.901, 2.81, 7.75
KH_NS_GPU = (64, 128, 256)
KH_BOUND_GPU = {0: (0.0384, 0.0125, 0.0082), 10: (0.126, 0.028, 0.0117)}
# measured b=0 0.0296, 0.00957, 0.00625; b=10 0.0968, 0.0217, 0.00898
KH_PENALTY_GPU = 2.25     # N / 64 at 10 c_s for the rest-frame error at 64; measured 1.73
KH_100_BOUND = 0.185      # 128 x 256 at 100 c_s; measured 0.142 (rest frame 0.00957)


@gpu_only
def test_sod_boost_gpu():
    """The Sod ladder 800-6400 cells: the resolution penalty at N = 800, interpolated for every boost."""
    lad = {b: [sod(n, b)[0] for n in SOD_NS_GPU] for b in SOD_BOUND_GPU}
    for b, es in lad.items():
        for n, e, bnd in zip(SOD_NS_GPU[2:], es[2:], SOD_BOUND_GPU[b]):
            check(f"sod b={b} N={n} L1 rho", e, bnd)
    for b, bnd in SOD_PENALTY_GPU.items():
        check(f"sod b={b} resolution penalty at N=800", penalty(SOD_NS_GPU, lad[b], lad[0][0]) / 800, bnd)


@gpu_only
def test_kh_boost_gpu():
    """KH at 64-256 for 0 and 10 c_s with the 10 c_s resolution penalty, and 128 x 256 at 100 c_s."""
    err = {b: [kh(n, b) for n in KH_NS_GPU] for b in KH_BOUND_GPU}
    for b, es in err.items():
        for n, e, bnd in zip(KH_NS_GPU, es, KH_BOUND_GPU[b]):
            check(f"kh b={b} nx={n} |sigma/theory - 1|", e, bnd)
    check("kh b=10 resolution penalty at nx=64", penalty(KH_NS_GPU, err[10], err[0][0]) / 64, KH_PENALTY_GPU)
    check("kh b=100 nx=128 |sigma/theory - 1|", kh(128, 100), KH_100_BOUND)
