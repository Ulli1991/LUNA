"""Single-mode Rayleigh-Taylor, Liska & Wendroff 2003 sec. 4.6: (0, 1/6) x (0, 1), g = 0.1 in -y, rho 2 over 1,
all walls reflecting. Linear gate: the exact eigenmode of the smoothed, compressible, wall-bounded background
(a 1D ODE, shooting), seeded +AMP and -AMP: half the difference has no drift and no even orders in AMP."""
import functools

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from tests._helpers import accuracy_cfg, check, evolve_staged, gpu_only

GAM, G, LX = 1.4, 0.1, 1.0 / 6.0               # LW give no gamma here; 1.4 and P_I as Springel 2010 sec. 8.8
K = 6.0 * np.pi                                # half a wavelength across the box: u = 0 on the x walls
P_I, DELTA = 2.5, 0.025                        # pressure at the interface; tanh width ours (LW give none)
AMP, T = 1e-7, 4.0                             # max |v_y| of the seed (1e-8 agrees); ~2.6 e-folds


def background(y):
    """rho, d rho / dy, p: rho = 3/2 + tanh((y - 1/2) / DELTA) / 2 in exact hydrostatic balance."""
    s = (y - 0.5) / DELTA
    rho = 1.5 + 0.5 * np.tanh(s)
    p = P_I - G * (1.5 * (y - 0.5) + 0.5 * DELTA * np.log(np.cosh(s)))
    return rho, 0.5 / DELTA / np.cosh(s) ** 2, p


def _rhs(y, q, sig):
    """(W, P)' for u = U sin kx, v = W cos kx, rho', p' = (R, P) cos kx, all ~ exp(sig t)."""
    W, P = q
    rho, drho, p = background(y)
    div = K * K * P / (sig * rho)                          # k U, from sig rho U = k P
    Wp = -(sig * P - W * rho * G + GAM * p * div) / (GAM * p)
    R = -(W * drho + rho * (div + Wp)) / sig
    return [Wp, -sig * rho * W - G * R]


def _shoot(sig, dense=False):
    sol = solve_ivp(_rhs, (0.0, 1.0), [0.0, 1.0], args=(sig,), rtol=1e-12, atol=1e-14, dense_output=dense)
    return sol if dense else sol.y[0, -1]


@functools.lru_cache(maxsize=None)
def sigma():
    """The fundamental (fastest) mode: the largest root of W(1) = 0 (FD eigen-solve agrees to 1e-6)."""
    s = np.linspace(0.05, 1.2, 116)
    f = [_shoot(x) for x in s]
    return max(brentq(_shoot, s[i], s[i + 1], xtol=1e-15) for i in range(len(s) - 1) if f[i] * f[i + 1] < 0)


def initial(nx, ny, s=1.0, q=5):
    """Exact cell averages of the background plus s times the eigenmode with max |v_y| = AMP."""
    hx, hy = LX / nx, 1.0 / ny
    gx, gw = np.polynomial.legendre.leggauss(q)
    yq = (np.arange(ny)[:, None] + 0.5 + 0.5 * gx[None, :]) * hy           # (ny, q)
    avg = lambda f: (0.5 * gw * f).sum(-1)                                   # noqa: E731
    rho0, _, p0 = background(yq)
    xc = (np.arange(nx) + 0.5) * hx
    fx = np.sinc(K * hx / (2 * np.pi))                                       # cell average of cos, sin in x
    cx, sx = (fx * np.cos(K * xc))[:, None, None], (fx * np.sin(K * xc))[:, None, None]
    if s:
        sig = sigma()
        sol = _shoot(sig, dense=True)
        W, P = sol.sol(yq.ravel()).reshape(2, ny, q)
        rho, drho, p = background(yq)
        div = K * K * P / (sig * rho)
        Wp = -(sig * P - W * rho * G + GAM * p * div) / (GAM * p)
        R, U = -(W * drho + rho * (div + Wp)) / sig, K * P / (sig * rho)
        a = s * AMP / np.abs(sol.sol(np.linspace(0, 1, 4001))[0]).max()
        R, U, W, P = a * R, a * U, a * W, a * P
    else:
        R = U = W = P = np.zeros_like(yq)
    # linear in the seed: rho u ~ rho0 U sin, rho v ~ rho0 W cos; second order (even in s) dropped
    rho = avg(rho0)[None] + cx[..., 0] * avg(R)[None]
    mx = sx[..., 0] * avg(rho0 * U)[None]
    my = cx[..., 0] * avg(rho0 * W)[None]
    E = avg(p0)[None] / (GAM - 1.0) + cx[..., 0] * avg(P)[None] / (GAM - 1.0)
    return np.stack([rho, mx, my, E])


SCHEMES = {"hllc-minmod": ("hllc", 0), "hll-minmod": ("hll", 0), "hllc-mc": ("hllc", 1)}
NS, NS_GPU, NAMES = (32, 64, 128), (128, 256, 512), ("L1", "L2", "Linf", "sigma", "drift", "energy")      # Nx; Ny = 4 Nx
# {scheme: bounds}, 1.3x measured, orders -0.1 .. +0.3: field norms, |sigma_meas / sigma - 1|, background drift
# (first order: mirrored, not hydrostatic, wall ghosts), background |d(E + rho g y)| / E. HLL smears the contact.
BOUNDS = {
    "hllc-minmod": dict(L1={32: 0.026, 64: 0.0082, 128: 0.0024}, L1_order=(1.61, 2.01),
        L2={32: 0.061, 64: 0.02, 128: 0.0057}, L2_order=(1.60, 2.00),
        Linf={32: 0.31, 64: 0.11, 128: 0.043}, Linf_order=(1.33, 1.73),
        sigma={32: 0.068, 64: 0.019, 128: 0.0051}, sigma_order=(1.77, 2.17),
        drift={32: 5.5e-05, 64: 2.9e-05, 128: 1.4e-05}, drift_order=(0.89, 1.29),
        energy={32: 4.6e-07, 64: 1.2e-07, 128: 2.9e-08}, energy_order=(1.90, 2.30)),
    # L1 0.0195, 0.00628, 0.00182; 1.71 | L2 0.0463, 0.0149, 0.00437; 1.70 | Linf 0.236, 0.0822, 0.0327; 1.43
    # sigma 0.0521, 0.0144, 0.00392; 1.87 | drift 4.18e-05, 2.18e-05, 1.05e-05; 0.99 | energy 3.51e-07, 8.76e-08, 2.19e-08; 2.00
    "hll-minmod": dict(L1={32: 0.084, 64: 0.038, 128: 0.015}, L1_order=(1.15, 1.55),
        L2={32: 0.24, 64: 0.13, 128: 0.055}, L2_order=(0.97, 1.37),
        Linf={32: 1.5, 64: 0.94, 128: 0.5}, Linf_order=(0.66, 1.06),
        sigma={32: 0.46, 64: 0.18, 128: 0.06}, sigma_order=(1.36, 1.76),
        drift={32: 0.29, 64: 0.19, 128: 0.086}, drift_order=(0.77, 1.17),
        energy={32: 4.1e-05, 64: 1.5e-05, 128: 4.8e-06}, energy_order=(1.45, 1.85)),
    # L1 0.0639, 0.0286, 0.0113; 1.25 | L2 0.184, 0.0974, 0.0417; 1.07 | Linf 1.09, 0.718, 0.38; 0.76
    # sigma 0.349, 0.132, 0.0461; 1.46 | drift 0.217, 0.139, 0.0655; 0.87 | energy 3.13e-05, 1.13e-05, 3.64e-06; 1.55
    "hllc-mc": dict(L1={32: 0.0057, 64: 0.00073, 128: 0.00012}, L1_order=(2.73, 3.13),
        L2={32: 0.018, 64: 0.0024, 128: 0.00048}, L2_order=(2.49, 2.89),
        Linf={32: 0.12, 64: 0.027, 128: 0.014}, Linf_order=(1.46, 1.86),
        sigma={32: 0.017, 64: 0.002, 128: 0.00024}, sigma_order=(2.97, 3.37),
        drift={32: 5.4e-05, 64: 2.9e-05, 128: 1.4e-05}, drift_order=(0.88, 1.28),
        energy={32: 4e-07, 64: 9.9e-08, 128: 2.5e-08}, energy_order=(1.90, 2.30)),
    # L1 0.00437, 0.000554, 8.69e-05; 2.83 | L2 0.0132, 0.00177, 0.000362; 2.59 | Linf 0.0902, 0.0202, 0.0104; 1.56
    # sigma 0.0127, 0.00148, 0.000181; 3.07 | drift 4.09e-05, 2.16e-05, 1.05e-05; 0.98 | energy 3.03e-07, 7.58e-08, 1.89e-08; 2.00
}
BOUNDS_GPU = {
    "hllc-minmod": dict(L1={128: 0.0024, 256: 0.00064, 512: 0.00017}, L1_order=(1.82, 2.22),
        L2={128: 0.0057, 256: 0.0016, 512: 0.00042}, L2_order=(1.79, 2.19),
        Linf={128: 0.043, 256: 0.016, 512: 0.0053}, Linf_order=(1.40, 1.80),
        sigma={128: 0.0051, 256: 0.0014, 512: 0.00034}, sigma_order=(1.87, 2.27),
        drift={128: 1.4e-05, 256: 7.1e-06, 512: 3.6e-06}, drift_order=(0.88, 1.28),
        energy={128: 2.9e-08, 256: 7.2e-09, 512: 1.8e-09}, energy_order=(1.90, 2.30)),
    # L1 0.00182, 0.00049, 0.000127; 1.92 | L2 0.00437, 0.00122, 0.000319; 1.89 | Linf 0.0327, 0.0121, 0.00406; 1.50
    # sigma 0.00392, 0.00106, 0.000257; 1.97 | drift 1.05e-05, 5.43e-06, 2.71e-06; 0.98 | energy 2.19e-08, 5.46e-09, 1.37e-09; 2.00
    "hll-minmod": dict(L1={128: 0.015, 256: 0.0047, 512: 0.0012}, L1_order=(1.72, 2.12),
        L2={128: 0.055, 256: 0.018, 512: 0.005}, L2_order=(1.62, 2.02),
        Linf={128: 0.5, 256: 0.21, 512: 0.078}, Linf_order=(1.24, 1.64),
        sigma={128: 0.06, 256: 0.018, 512: 0.0041}, sigma_order=(1.84, 2.24),
        drift={128: 0.086, 256: 0.031, 512: 0.0095}, drift_order=(1.49, 1.89),
        energy={128: 4.8e-06, 256: 1.4e-06, 512: 3.3e-07}, energy_order=(1.82, 2.22)),
    # L1 0.0113, 0.00356, 0.00091; 1.82 | L2 0.0417, 0.0137, 0.00383; 1.72 | Linf 0.38, 0.156, 0.0596; 1.34
    # sigma 0.0461, 0.0133, 0.00312; 1.94 | drift 0.0655, 0.0237, 0.00725; 1.59 | energy 3.64e-06, 1.02e-06, 2.53e-07; 1.92
    "hllc-mc": dict(L1={128: 0.00012, 256: 2.6e-05, 512: 6.3e-06}, L1_order=(1.98, 2.38),
        L2={128: 0.00048, 256: 0.00013, 512: 3.3e-05}, L2_order=(1.83, 2.23),
        Linf={128: 0.014, 256: 0.0063, 512: 0.0025}, Linf_order=(1.13, 1.53),
        sigma={128: 0.00024, 256: 4.5e-05, 512: 9.8e-06}, sigma_order=(2.20, 2.60),
        drift={128: 1.4e-05, 256: 7.1e-06, 512: 3.6e-06}, drift_order=(0.88, 1.28),
        energy={128: 2.5e-08, 256: 6.2e-09, 512: 1.6e-09}, energy_order=(1.90, 2.30)),
    # L1 8.69e-05, 1.98e-05, 4.84e-06; 2.08 | L2 0.000362, 9.81e-05, 2.48e-05; 1.93 | Linf 0.0104, 0.00481, 0.0019; 1.23
    # sigma 0.000181, 3.4e-05, 7.47e-06; 2.30 | drift 1.05e-05, 5.41e-06, 2.71e-06; 0.98 | energy 1.89e-08, 4.73e-09, 1.18e-09; 2.00
}


def cfg_for(nx, scheme, sign=1.0):
    solver, limiter = SCHEMES[scheme]
    return accuracy_cfg(test_problem="Rayleigh-Taylor-2D", SPATIAL_DIMENSION=2, Nx=nx, Ny=4 * nx, Lx=LX, Ly=1.0,
                        BOUNDARY=2, gamma=GAM, riemann_solver=solver, LIMITER=limiter,
                        external_potential="uniform", external_axis=1, external_g=-sign * G)


@functools.lru_cache(maxsize=None)
def run(nx, scheme, s, t=T, sign=1.0):
    """The background seeded with s times the eigenmode, at t."""
    return evolve_staged(cfg_for(nx, scheme, sign), initial(nx, 4 * nx, s), t)


@functools.lru_cache(maxsize=None)
def errors(nx, scheme, t=T, sign=1.0):
    """(L1, L2, Linf) of (dm_y - exact) / (AMP e^{sig T}), the measured growth rate, the background drift
    max |rho - rho(0)| and its |d(E + rho g y)| / E; d = (U+ - U-) / 2, the background B = (U+ + U-) / 2."""
    Up, Um, B0 = run(nx, scheme, 1.0, t, sign), run(nx, scheme, -1.0, t, sign), initial(nx, 4 * nx, 0.0)
    B = 0.5 * (Up + Um)                                                         # + O(AMP^2)
    g = np.exp(sigma() * T)
    d, d0 = 0.5 * (Up[2] - Um[2]), initial(nx, 4 * nx)[2] - B0[2]
    e = (d - g * d0) / (AMP * g)
    ratio = (d * d0).sum() / (d0 * d0).sum()
    y = (np.arange(4 * nx) + 0.5) / (4 * nx)
    etot = lambda V: float((V[3] + sign * G * V[0] * y).sum())                  # noqa: E731
    return (float(np.abs(e).mean()), float(np.sqrt((e * e).mean())), float(np.abs(e).max()),
            float(np.log(abs(ratio)) / t), float(np.abs(B[0] - B0[0]).max()), abs(etot(B) / etot(B0) - 1.0))


def series(nx, scheme):
    r = errors(nx, scheme)
    return dict(zip(NAMES, (r[0], r[1], r[2], abs(r[3] / sigma() - 1.0), r[4], r[5])))


def gate(scheme, ns, b):
    """Every norm at every N under its bound, and its order inside its band."""
    res = [series(n, scheme) for n in ns]
    print(f"MEASURED rt {scheme} {','.join(map(str, ns))} = {res}")
    for nm in NAMES:
        vals = [r[nm] for r in res]
        for n, v in zip(ns, vals):
            check(f"rt {scheme} N={n} {nm}", v, b[nm][n])
        order = -np.polyfit(np.log(ns), np.log(vals), 1)[0]
        check(f"rt {scheme} {nm} order", order, b[f"{nm}_order"][0], lower=True)
        check(f"rt {scheme} {nm} order", order, b[f"{nm}_order"][1])


def test_sigma_against_incompressible_limit():
    """Sanity: a smoothed interface grows slower than the sharp incompressible sqrt(A g k) = 0.793."""
    assert 0.6 < sigma() < np.sqrt(G * K / 3.0)


@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_rayleigh_taylor_linear(scheme):
    """Field norms, growth rate, background drift and energy at every N, with orders; mass in evolve_staged."""
    gate(scheme, NS, BOUNDS[scheme])


@pytest.mark.parametrize("kw", [dict(t=0.9 * T), dict(sign=-1.0)], ids=["short", "flipped-g"])
def test_bound_catches_a_broken_run(kw):
    """Teeth: stopped at 0.9 T, or g pointing up (stable), the L1 at 64 misses its bound (0.032, 0.14)."""
    assert errors(64, "hllc-minmod", **kw)[0] > BOUNDS["hllc-minmod"]["L1"][64]


@gpu_only
@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_rayleigh_taylor_linear_gpu(scheme):
    """128, 256, 512 on a GPU: the CPU test's norms, growth rate, drift and energy, with orders."""
    gate(scheme, NS_GPU, BOUNDS_GPU[scheme])
