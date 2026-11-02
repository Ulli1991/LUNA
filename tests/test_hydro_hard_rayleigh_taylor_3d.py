"""Single-mode Rayleigh-Taylor in 3D, linear: the A13 stratification (rho 2 over 1, tanh width 0.025, g = 0.1,
gamma 1.4, p = 2.5 at the interface) with gravity along x, all walls reflecting, height 2/3. The mode is
v_x = W(x) cos(ky y) cos(kz z) with |k| = 6 pi in the minimal wall box Ly = pi / ky, Lz = pi / kz: the
compressible eigenmode depends on ky^2 + kz^2 only (the A13 shooting ODE with K^2 -> |k|^2, solved for this
height). Modes (a, b), (ky, kz) = |k| (a, b) / 5: axis (5, 0) and oblique (3, 4) at the same h, so their rates
must agree (isotropy); (4, 3) is (3, 4) with y and z swapped (mirror symmetry). Seeded +AMP and -AMP as A13.
Bounds, jobs, hardware: docs/TEST_PLAN.md (Rayleigh-Taylor 3D)."""
import functools

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from tests._helpers import accuracy_cfg, check, evolve_staged, gpu_only

GAM, G, P_I, DELTA = 1.4, 0.1, 2.5, 0.025      # as A13 (tests/test_hydro_hard_rayleigh_taylor.py)
HEIGHT, XI = 2.0 / 3.0, 1.0 / 3.0              # box along g and the interface: walls e^{-|k| XI} = 0.002 away
KABS = 6.0 * np.pi                             # |k|, as A13's K
AMP, T = 1e-7, 4.0                             # max |v_x| of the seed; ~2.6 e-folds
MODES = {"axis": (5, 0), "oblique": (3, 4), "swapped": (4, 3)}
NZ_AXIS = 4                                    # cells along z for the axis mode (uniform in z)


def background(x, xi=XI):
    """rho, d rho / dx, p: rho = 3/2 + tanh((x - xi) / DELTA) / 2 in exact hydrostatic balance."""
    s = (x - xi) / DELTA
    rho = 1.5 + 0.5 * np.tanh(s)
    p = P_I - G * (1.5 * (x - xi) + 0.5 * DELTA * np.log(np.cosh(s)))
    return rho, 0.5 / DELTA / np.cosh(s) ** 2, p


def _mode(x, W, P, sig, k2, xi):
    """(W', R, U) where k U = div_h, from sig rho u_h = k_h P and continuity, energy."""
    rho, drho, p = background(x, xi)
    div = k2 * P / (sig * rho)
    Wp = -(sig * P - W * rho * G + GAM * p * div) / (GAM * p)
    return Wp, -(W * drho + rho * (div + Wp)) / sig, rho


def _rhs(x, q, sig, k2, xi):
    W, P = q
    Wp, R, rho = _mode(x, W, P, sig, k2, xi)
    return [Wp, -sig * rho * W - G * R]


def _shoot(sig, k2, height, xi, dense=False):
    sol = solve_ivp(_rhs, (0.0, height), [0.0, 1.0], args=(sig, k2, xi), method="DOP853", rtol=1e-12, atol=1e-14,
                    dense_output=dense)
    return sol if dense else sol.y[0, -1]


@functools.lru_cache(maxsize=None)
def sigma(k2=KABS ** 2, height=HEIGHT, xi=XI):
    """The fundamental mode: the largest root of W(height) = 0, scanned down from 1.2 in steps of 0.01 as A13."""
    hi, f_hi = 1.2, _shoot(1.2, k2, height, xi)
    while hi > 0.05:
        lo = hi - 0.01
        f_lo = _shoot(lo, k2, height, xi)
        if f_lo * f_hi < 0:
            return brentq(_shoot, lo, hi, args=(k2, height, xi), xtol=1e-15)
        hi, f_hi = lo, f_lo
    raise ValueError("no growing mode above 0.05")


def grid(j, mode):
    """(Nx, Ny, Nz, Ly, Lz, ky, kz) at h = 1 / (72 j): 48 j cells high, 24 j cells per wavelength."""
    a, b = MODES[mode]
    ky, kz = KABS * a / 5.0, KABS * b / 5.0
    ny = 60 * j // a
    nz = 60 * j // b if b else NZ_AXIS
    h = 1.0 / (72 * j)
    return 48 * j, ny, nz, ny * h, nz * h, ky, kz


@functools.lru_cache(maxsize=None)
def initial(j, mode, s=1.0, q=5):
    """Exact cell averages of the background plus s times the eigenmode with max |v_x| = AMP; (5, Nx, Ny, Nz)."""
    nx, ny, nz, _, _, ky, kz = grid(j, mode)
    h = 1.0 / (72 * j)
    gx, gw = np.polynomial.legendre.leggauss(q)
    xq = (np.arange(nx)[:, None] + 0.5 + 0.5 * gx[None, :]) * h                 # (nx, q)
    avg = lambda f: (0.5 * gw * f).sum(-1)                                       # noqa: E731

    def trig(k, n):
        """Cell averages of cos(k y), sin(k y) on n cells of width h."""
        c = (np.arange(n) + 0.5) * h
        f = np.sinc(k * h / (2 * np.pi))
        return f * np.cos(k * c), f * np.sin(k * c)

    cy, sy = trig(ky, ny)
    cz, sz = trig(kz, nz)
    rho0, _, p0 = background(xq)
    if s:
        sig, k2 = sigma(), ky * ky + kz * kz
        sol = _shoot(sig, k2, HEIGHT, XI, dense=True)
        W, P = sol.sol(xq.ravel()).reshape(2, nx, q)
        _, R, rho = _mode(xq, W, P, sig, k2, XI)
        V = P / (sig * rho)                                                      # u_y = ky V, u_z = kz V
        a = s * AMP / np.abs(sol.sol(np.linspace(0, HEIGHT, 4001))[0]).max()
        R, V, W, P = a * R, a * V, a * W, a * P
    else:
        R = V = W = P = np.zeros_like(xq)
    o = lambda f, g, k: avg(f)[:, None, None] * g[None, :, None] * k[None, None, :]  # noqa: E731
    rho = avg(rho0)[:, None, None] + o(R, cy, cz)
    mx = o(rho0 * W, cy, cz)
    my = o(rho0 * V * ky, sy, cz)
    mz = o(rho0 * V * kz, cy, sz)
    E = avg(p0)[:, None, None] / (GAM - 1.0) + o(P, cy, cz) / (GAM - 1.0)
    return np.stack([np.broadcast_to(f, (nx, ny, nz)) for f in (rho, mx, my, mz, E)])


SCHEMES = {"hllc-minmod": ("hllc", 0), "hllc-mc": ("hllc", 1)}
NAMES = ("L1", "L2", "Linf", "sigma", "drift", "energy")


def cfg_for(j, mode, scheme, sign=1.0):
    solver, limiter = SCHEMES[scheme]
    nx, ny, nz, ly, lz, _, _ = grid(j, mode)
    return accuracy_cfg(test_problem="Rayleigh-Taylor-3D", SPATIAL_DIMENSION=3, Nx=nx, Ny=ny, Nz=nz,
                        Lx=HEIGHT, Ly=ly, Lz=lz, BOUNDARY=2, gamma=GAM, riemann_solver=solver, LIMITER=limiter,
                        external_potential="uniform", external_axis=0, external_g=-sign * G)


@functools.lru_cache(maxsize=None)
def run(j, mode, scheme, s, t, sign=1.0):
    """The background seeded with s times the eigenmode, at t / 2 and t."""
    return tuple(evolve_staged(cfg_for(j, mode, scheme, sign), initial(j, mode, s), [0.5 * t, t]))


@functools.lru_cache(maxsize=None)
def errors(j, mode, scheme, t, sign=1.0, early=False):
    """({L1, L2, Linf of (dm_x - exact) / (AMP e^{sig t}), |sigma_meas / sigma - 1|, the background drift
    max |rho - rho(0)|, |d(E + rho g x)| / E}, sigma_meas); d = (U+ - U-) / 2, the background B = (U+ + U-) / 2.
    `early`: the states at t / 2 against the exact mode at t (a run stopped short)."""
    Up, Um = (run(j, mode, scheme, s, t, sign)[0 if early else 1] for s in (1.0, -1.0))
    B0 = initial(j, mode, 0.0)
    B = 0.5 * (Up + Um)                                                         # + O(AMP^2)
    g = np.exp(sigma() * t)
    d, d0 = 0.5 * (Up[1] - Um[1]), initial(j, mode)[1] - B0[1]
    e = (d - g * d0) / (AMP * g)
    rate = float(np.log(abs((d * d0).sum() / (d0 * d0).sum())) / t)
    x = ((np.arange(B0.shape[1]) + 0.5) / (72 * j))[:, None, None]
    etot = lambda V: float((V[4] + sign * G * V[0] * x).sum())                  # noqa: E731
    vals = (float(np.abs(e).mean()), float(np.sqrt((e * e).mean())), float(np.abs(e).max()),
            abs(rate / sigma() - 1.0), float(np.abs(B[0] - B0[0]).max()), abs(etot(B) / etot(B0) - 1.0))
    return dict(zip(NAMES, vals)), rate


def gate(scheme, js, t, b):
    """Both modes' norms at every N under their bounds, and |sigma_oblique / sigma_axis - 1| (isotropy);
    with two or more N, every order inside its band."""
    res = {m: [errors(j, m, scheme, t)[0] for j in js] for m in ("axis", "oblique")}
    res["iso"] = [{"iso": abs(errors(j, "oblique", scheme, t)[1] / errors(j, "axis", scheme, t)[1] - 1.0)} for j in js]
    print(f"MEASURED rt3d {scheme} j={js} = {res}")
    for key, rows in res.items():
        for nm in rows[0]:
            vals = [r[nm] for r in rows]
            bb = b[key] if key == "iso" else b[key][nm]
            for j, v in zip(js, vals):
                check(f"rt3d {scheme} {key} j={j} {nm}", v, bb[j])
            if len(js) > 1:
                order = -np.polyfit(np.log(js), np.log(vals), 1)[0]
                lo, hi = b[f"{key}_order"] if key == "iso" else b[key][f"{nm}_order"]
                check(f"rt3d {scheme} {key} {nm} order", order, lo, lower=True)
                check(f"rt3d {scheme} {key} {nm} order", order, hi)


def _b(js, **kw):
    """{name: {j: bound}, name_order: (lo, hi)} from name=(bounds..., (lo, hi))."""
    return {k: (v if k.endswith("_order") else dict(zip(js, v))) for k, v in kw.items()}


JS, JS_GPU, T_CPU = (1,), (2, 4, 8), 1.0       # j: 24 j cells per wavelength; the CPU run stops at T / 4
# 1.3x measured, orders -0.1 .. +0.3. CPU: job 31383738 (CPU-8, FP64); GPU: jobs 31384675, 31384676 (A100, FP64).
# Field norms, |sigma_meas / sigma - 1|, background drift max |rho - rho0| (first order: mirrored, not
# hydrostatic, wall ghosts), |d(E + rho g x)| / E; iso = |sigma_oblique / sigma_axis - 1|.
BOUNDS = {"hllc-minmod": dict(
    axis=_b(JS, L1=(0.06,), L2=(0.101,), Linf=(0.41,), sigma=(0.32,), drift=(3e-05,), energy=(5.8e-07,)),
    # L1 0.0459 | L2 0.0770 | Linf 0.309 | sigma 0.239 | drift 2.25e-05 | energy 4.40e-07
    oblique=_b(JS, L1=(0.03,), L2=(0.057,), Linf=(0.33,), sigma=(0.25,), drift=(3e-05,), energy=(5.8e-07,)),
    # L1 0.0229 | L2 0.0435 | Linf 0.248 | sigma 0.186 | drift 2.25e-05 | energy 4.40e-07
    iso={1: 0.091}, iso_order=None)}                                         # 0.0695
BOUNDS_GPU = {
    "hllc-minmod": dict(
        axis=_b(JS_GPU, L1=(0.042, 0.014, 0.004), L1_order=(1.58, 1.98),
                L2=(0.085, 0.031, 0.0094), L2_order=(1.49, 1.89),
                Linf=(0.38, 0.17, 0.07), Linf_order=(1.11, 1.51),
                sigma=(0.075, 0.021, 0.0054), sigma_order=(1.80, 2.20),
                drift=(5.1e-05, 2.6e-05, 1.4e-05), drift_order=(0.88, 1.28),
                energy=(5.7e-07, 1.5e-07, 3.6e-08), energy_order=(1.90, 2.30)),
        # L1 0.0316, 0.0105, 0.00307; 1.68 | L2 0.0654, 0.0231, 0.00717; 1.59 | Linf 0.288, 0.128, 0.0535; 1.21
        # sigma 0.0571, 0.0156, 0.00408; 1.90 | drift 3.92e-05, 1.95e-05, 1.01e-05; 0.98 | energy 4.37e-07, 1.09e-07, 2.73e-08; 2.00
        oblique=_b(JS_GPU, L1=(0.021, 0.0067, 0.002), L1_order=(1.60, 2.00),
                   L2=(0.046, 0.015, 0.0042), L2_order=(1.61, 2.01),
                   Linf=(0.26, 0.088, 0.036), Linf_order=(1.32, 1.72),
                   sigma=(0.058, 0.017, 0.0044), sigma_order=(1.76, 2.16),
                   drift=(5.1e-05, 2.6e-05, 1.4e-05), drift_order=(0.88, 1.28),
                   energy=(5.7e-07, 1.5e-07, 3.6e-08), energy_order=(1.90, 2.30)),
        # L1 0.0156, 0.00515, 0.00148; 1.70 | L2 0.0347, 0.0111, 0.00322; 1.71 | Linf 0.192, 0.0675, 0.0270; 1.42
        # sigma 0.0441, 0.0125, 0.00336; 1.86 | drift, energy as axis
        iso={2: 0.018, 4: 0.0041, 8: 0.00095}, iso_order=(2.03, 2.43)),       # 0.0138, 0.00313, 0.000727; 2.13
    "hllc-mc": dict(
        axis=_b(JS_GPU, L1=(0.0077, 0.0011, 0.00021), L1_order=(2.50, 2.90),
                L2=(0.018, 0.0034, 0.00097), L2_order=(2.00, 2.40),
                Linf=(0.093, 0.045, 0.022), Linf_order=(0.95, 1.35),
                sigma=(0.013, 0.0014, 0.0002), sigma_order=(2.94, 3.34),
                drift=(5.3e-05, 2.6e-05, 1.4e-05), drift_order=(0.89, 1.29),
                energy=(5e-07, 1.3e-07, 3.1e-08), energy_order=(1.90, 2.30)),
        # L1 0.00587, 0.000788, 0.00016; 2.60 | L2 0.0138, 0.00261, 0.000746; 2.10 | Linf 0.0714, 0.0340, 0.0167; 1.05
        # sigma 0.00988, 0.00103, 0.000147; 3.04 | drift 4.02e-05, 1.97e-05, 1.02e-05; 0.99 | energy 3.78e-07, 9.43e-08, 2.36e-08; 2.00
        oblique=_b(JS_GPU, L1=(0.004, 0.00049, 0.00011), L1_order=(2.51, 2.91),
                   L2=(0.011, 0.0014, 0.00034), L2_order=(2.36, 2.76),
                   Linf=(0.086, 0.02, 0.011), Linf_order=(1.45, 1.85),
                   sigma=(0.012, 0.0012, 0.00018), sigma_order=(2.89, 3.29),
                   drift=(5.3e-05, 2.6e-05, 1.4e-05), drift_order=(0.89, 1.29),
                   energy=(5e-07, 1.3e-07, 3.1e-08), energy_order=(1.90, 2.30)),
        # L1 0.00304, 0.00037, 8.17e-05; 2.61 | L2 0.00788, 0.00105, 0.000259; 2.46 | Linf 0.0661, 0.0149, 0.00775; 1.55
        # sigma 0.00861, 0.000899, 0.000136; 2.99 | drift, energy as axis
        iso={2: 0.0017, 4: 0.00018, 8: 1.5e-05}, iso_order=(3.33, 3.73)),     # 0.00128, 0.000136, 1.10e-05; 3.43
}


def test_sigma_below_the_sharp_limit():
    """Sanity: the smoothed, compressible, wall-bounded rate is below the sharp incompressible sqrt(A g k) = 0.793."""
    print(f"MEASURED rt3d sigma = {sigma():.9f}")
    assert 0.6 < sigma() < np.sqrt(G * KABS / 3.0)


def test_rayleigh_taylor_3d_linear():
    """Axis and oblique modes, 24 cells per wavelength, to T / 4: norms, rate, isotropy, drift, energy;
    mass in evolve_staged."""
    gate("hllc-minmod", JS, T_CPU, BOUNDS["hllc-minmod"])


def test_bound_catches_a_broken_run():
    """Teeth: stopped at t / 2, the oblique L1 misses its bound."""
    v = errors(1, "oblique", "hllc-minmod", T_CPU, early=True)[0]["L1"]
    print(f"MEASURED rt3d stopped-early oblique L1 = {v:.6e}")
    assert v > BOUNDS["hllc-minmod"]["oblique"]["L1"][1]


@gpu_only
def test_bound_catches_a_flipped_g():
    """Teeth: g pointing up (stable), the axis mode's L1 at 48 cells per wavelength misses its bound."""
    v = errors(2, "axis", "hllc-minmod", T, sign=-1.0)[0]["L1"]
    print(f"MEASURED rt3d flipped-g axis L1 = {v:.6e}")
    assert v > BOUNDS_GPU["hllc-minmod"]["axis"]["L1"][2]


@gpu_only
def test_sigma_is_the_2d_rate():
    """The ODE at height 1, interface 1/2 is A13's: the same sigma."""
    from tests import test_hydro_hard_rayleigh_taylor as rt2
    check("rt3d sigma(H=1) / sigma_A13 - 1", abs(sigma(KABS ** 2, 1.0, 0.5) / rt2.sigma() - 1.0), 8.3e-14)          # 6.4e-14


@gpu_only
def test_mode_mirror_symmetry():
    """(4, 3) is (3, 4) with y <-> z (and m_y <-> m_z) swapped, to round-off, at 24 cells per wavelength, T."""
    a = run(1, "oblique", "hllc-minmod", 1.0, T)[1]
    b = run(1, "swapped", "hllc-minmod", 1.0, T)[1][[0, 1, 3, 2, 4]].transpose(0, 1, 3, 2)
    check("rt3d y<->z swap max|dm_x| / AMP", float(np.abs(a[1] - b[1]).max()) / AMP, 3.4e-08)                # 2.6e-08, round-off
    check("rt3d y<->z swap max|dU| / max|U|", float(np.abs(a - b).max() / np.abs(a).max()), 8.3e-15)  # 6.3e-15


@gpu_only
@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_rayleigh_taylor_3d_linear_gpu(scheme):
    """48, 96, 192 cells per wavelength to T on a GPU: both modes' norms, rate, drift, energy and the isotropy, with orders."""
    gate(scheme, JS_GPU, T, BOUNDS_GPU[scheme])
