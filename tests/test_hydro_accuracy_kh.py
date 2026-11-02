"""Kelvin-Helmholtz linear growth: the exact compressible eigenmode of the layer (luna.hydro.exact_kh_linear.linear_mode),
seeded small, against its exact growth: v_y norms at T and the growth rate, with orders."""
import functools

import numpy as np
import pytest

from tests._helpers import accuracy_cfg, check, evolve, gpu_only

TIMES = np.arange(0.0, 4.01, 0.25)
AMP = 1e-6                          # max |v_y| of the seed; 1e-7 and +-AMP agree to 4 digits
NS, NS_GPU = (32, 64), (64, 128, 256)
NAMES = ("L1", "L2", "Linf", "sigma")
# {N: bound}, 1.3x measured, orders -0.1 .. +0.3: (v_y - exact) / (AMP e^{sigma T}) and |sigma_fit / sigma - 1|
BOUNDS = dict(L1={32: 0.23, 64: 0.057}, L1_order=(1.91, 2.31),
    L2={32: 0.3, 64: 0.076}, L2_order=(1.86, 2.26),
    Linf={32: 0.75, 64: 0.25}, Linf_order=(1.52, 1.92),
    sigma={32: 0.22, 64: 0.04}, sigma_order=(2.35, 2.75))
# L1 0.174, 0.0433; 2.01 | L2 0.227, 0.0582; 1.96 | Linf 0.572, 0.186; 1.62 | sigma 0.164, 0.03; 2.45
BOUNDS_GPU = dict(L1={64: 0.057, 128: 0.013, 256: 0.0031}, L1_order=(2.01, 2.41),
    L2={64: 0.076, 128: 0.017, 256: 0.0041}, L2_order=(2.01, 2.41),
    Linf={64: 0.25, 128: 0.058, 256: 0.016}, Linf_order=(1.90, 2.30),
    sigma={64: 0.04, 128: 0.0081, 256: 0.002}, sigma_order=(2.08, 2.48))
# L1 0.0433, 0.00958, 0.00232; 2.11 | L2 0.0582, 0.0128, 0.00314; 2.11 | Linf 0.186, 0.0445, 0.0117; 2.00 | sigma 0.03, 0.00618, 0.00146; 2.18


@functools.lru_cache(maxsize=None)
def mode():
    from luna.hydro.exact_kh_linear import geometry, linear_mode
    g = geometry()
    return g, *linear_mode(g)


def seeded(nx, amp):
    """Conserved U on nx x 2nx: the layer plus amp times the eigenmode, both as the test grid sees them."""
    from luna.hydro.exact_kh_linear import cell_average_y, profile
    g, s, q, _ = mode()
    ny = 2 * nx
    x = (np.arange(nx) + 0.5) * g["Lx"] / nx
    y = (np.arange(ny) + 0.5) * g["Ly"] / ny
    ex = np.exp(1j * g["k"] * x) * np.sinc(g["k"] * g["Lx"] / nx / (2 * np.pi))   # cell average of e^{ikx}
    up, vp, pp = (amp * np.real(ex[:, None] * cell_average_y(q[i], g["Ly"], ny)[None, :]) for i in range(3))
    u = profile(y, g["Ly"], g["d"], g["dU"])[None, :] + up
    rho, p = g["rho"] + pp / g["cs"] ** 2, g["p"] + pp                          # isentropic: rho' = p' / cs^2
    return np.stack([rho, rho * u, rho * vp, p / (g["gamma"] - 1) + 0.5 * rho * (u * u + vp * vp)])


def exact_v(nx, t):
    """The cell-averaged v_y of the unit mode at t."""
    from luna.hydro.exact_kh_linear import cell_average_y
    g, s, q, _ = mode()
    x = (np.arange(nx) + 0.5) * g["Lx"] / nx
    ex = np.exp(1j * g["k"] * x) * np.sinc(g["k"] * g["Lx"] / nx / (2 * np.pi))
    return np.real(ex[:, None] * cell_average_y(q[1], g["Ly"], 2 * nx)[None, :] * np.exp(s * t))


@functools.lru_cache(maxsize=None)
def errors(nx, amp=AMP, frac=1.0, order=1):
    """(L1, L2, Linf) of (v_y - exact) / (amp e^{sigma T}) at T = 4 for a run stopped at frac * T, and
    |sigma_fit / sigma - 1| from ln |v_y mode| over the run."""
    g, s, _, _ = mode()
    times = TIMES * frac
    cfg = accuracy_cfg(test_problem="Kelvin-Helmholtz", SPATIAL_DIMENSION=2, Nx=nx, Ny=2 * nx,
                       Lx=g["Lx"], Ly=g["Ly"], BOUNDARY=0, gamma=g["gamma"], ORDER=order)
    states, _ = evolve(cfg, seeded(nx, amp), times)
    v = [U[2] / U[0] for U in states]
    a = np.array([np.sqrt(np.sum(np.abs(np.fft.rfft(w, axis=0)[1]) ** 2)) for w in v])
    sigma = np.polyfit(times, np.log(a), 1)[0]
    T = TIMES[-1]
    e = (v[-1] - amp * exact_v(nx, T)) / (amp * np.exp(s.real * T))
    return float(np.abs(e).mean()), float(np.sqrt((e * e).mean())), float(np.abs(e).max()), abs(sigma / s.real - 1)


def gate(ns, b):
    """Every norm at every N under its bound, and its order inside its band."""
    res = [dict(zip(NAMES, errors(n))) for n in ns]
    print(f"MEASURED kh {','.join(map(str, ns))} = {res}")
    for nm in NAMES:
        vals = [r[nm] for r in res]
        for n, v in zip(ns, vals):
            check(f"kh N={n} {nm}", v, b[nm][n])
        order = -np.polyfit(np.log(ns), np.log(vals), 1)[0]
        check(f"kh {nm} order", order, b[f"{nm}_order"][0], lower=True)
        check(f"kh {nm} order", order, b[f"{nm}_order"][1])


def test_kh_linear():
    """v_y norms and the growth rate at 32 and 64, with orders."""
    gate(NS, BOUNDS)


def test_kh_seed_is_linear():
    """The same norms and rate at AMP / 10: the seed is in the linear regime (10 AMP is not: 1.6e-3)."""
    a, b = errors(64), errors(64, amp=0.1 * AMP)
    for nm, x, y in zip(NAMES, a, b):
        check(f"kh N=64 {nm} at AMP / 10 over at AMP - 1", abs(y / x - 1), 1e-3)


@pytest.mark.parametrize("kw", [dict(frac=0.9), dict(order=0)], ids=["short", "first-order"])
def test_bound_catches_a_broken_run(kw):
    """Teeth: stopped at 0.9 T the L1 at 64 misses its bound; first order misses the growth-rate bound."""
    r = dict(zip(NAMES, errors(64, **kw)))
    assert r["L1"] > BOUNDS["L1"][64] if "frac" in kw else r["sigma"] > BOUNDS["sigma"][64]


@gpu_only
def test_kh_linear_gpu():
    """The same at 64 x 128 .. 256 x 512, on a GPU."""
    gate(NS_GPU, BOUNDS_GPU)
