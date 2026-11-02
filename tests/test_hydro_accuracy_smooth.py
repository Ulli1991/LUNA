"""Smooth and advection problems with known answers: sound waves, Gresho vortex, square and sine advection."""
import numpy as np
import pytest

from luna.ic_generation import gresho_vortex
from tests._helpers import accuracy_cfg, check, check_conserved, evolve, gpu_only

GAMMA = 5.0 / 3.0
SOLVERS = ("hllc", "hll")
PROBLEMS = ("sound", "sound_adv", "gresho", "square", "sine")
# {(problem, solver): bounds} ~1.3x measured; orders measured -0.1 .. +0.3 (CPU, FP64, minmod, default dt).
# Linf only for smooth problems (not the square); minmod clips extrema, so its order is ~1.2-1.6.
BOUNDS = {
    ("sound", "hllc"): dict(L1={32: 0.02, 64: 0.0056, 128: 0.0015, 256: 0.00041}, L2={32: 0.025, 64: 0.0082, 128: 0.0026, 256: 0.00084},
        L1_order=(1.76, 2.16), L2_order=(1.53, 1.93),
        Linf={32: 0.057, 64: 0.024, 128: 0.01, 256: 0.0042}, Linf_order=(1.16, 1.56)),
    # measured L1 0.0151, 0.00429, 0.00117, 0.000315; L2 0.0193, 0.0063, 0.00202, 0.000649; orders 1.861, 1.633; Linf 0.044, 0.0188, 0.00783, 0.00322, order 1.258
    ("sound", "hll"): dict(L1={32: 0.02, 64: 0.0056, 128: 0.0015, 256: 0.00041}, L2={32: 0.025, 64: 0.0082, 128: 0.0026, 256: 0.00084},
        L1_order=(1.76, 2.16), L2_order=(1.53, 1.93),
        Linf={32: 0.057, 64: 0.024, 128: 0.01, 256: 0.0042}, Linf_order=(1.16, 1.56)),
    # measured L1 0.0151, 0.00429, 0.00117, 0.000315; L2 0.0193, 0.0063, 0.00202, 0.000649; orders 1.861, 1.633; Linf 0.044, 0.0188, 0.00783, 0.00322, order 1.258
    ("sound_adv", "hllc"): dict(L1={32: 0.021, 64: 0.0059, 128: 0.0017, 256: 0.00043}, L2={32: 0.026, 64: 0.0082, 128: 0.0025, 256: 0.00068},
        L1_order=(1.77, 2.17), L2_order=(1.65, 2.05),
        Linf={32: 0.053, 64: 0.021, 128: 0.0073, 256: 0.0021}, Linf_order=(1.45, 1.85)),
    # measured L1 0.0165, 0.00452, 0.00127, 0.000331; L2 0.02, 0.00633, 0.00189, 0.000522; orders 1.874, 1.752; Linf 0.0404, 0.0165, 0.0056, 0.0016, order 1.554
    ("sound_adv", "hll"): dict(L1={32: 0.022, 64: 0.0061, 128: 0.0017, 256: 0.00045}, L2={32: 0.027, 64: 0.0081, 128: 0.0024, 256: 0.00067},
        L1_order=(1.76, 2.16), L2_order=(1.66, 2.06),
        Linf={32: 0.052, 64: 0.021, 128: 0.0072, 256: 0.0021}, Linf_order=(1.46, 1.86)),
    # measured L1 0.0167, 0.00467, 0.00133, 0.000345; L2 0.0204, 0.00623, 0.00187, 0.000518; orders 1.861, 1.763; Linf 0.0402, 0.0163, 0.00555, 0.00158, order 1.556
    ("gresho", "hllc"): dict(L1={32: 0.18, 64: 0.07, 128: 0.03}, L2={32: 0.22, 64: 0.089, 128: 0.041},
        L1_order=(1.2, 1.6), L2_order=(1.09, 1.49),
        Linf={32: 0.58, 64: 0.32, 128: 0.18}, Linf_order=(0.75, 1.15),
        Lz_core={32: 0.23, 64: 0.098, 128: 0.037}, Lz_core_order=(1.23, 1.63)),
    # measured L1 0.138, 0.0538, 0.0229; L2 0.166, 0.0687, 0.0318; orders 1.296, 1.195; |dL_z|/L_z 0.0332, 0.000844, 0.000725 (ratio to the core loss <= 0.18); core loss 0.18, 0.0751, 0.0284, order 1.335; Linf 0.443, 0.246, 0.137, order 0.845
    ("gresho", "hll"): dict(L1={32: 0.27, 64: 0.16, 128: 0.069}, L2={32: 0.4, 64: 0.23, 128: 0.1},
        L1_order=(0.88, 1.28), L2_order=(0.86, 1.26),
        Linf={32: 1.0, 64: 0.68, 128: 0.38}, Linf_order=(0.62, 1.02),
        Lz_core={32: 0.77, 64: 0.29, 128: 0.093}, Lz_core_order=(1.43, 1.83)),
    # measured L1 0.205, 0.125, 0.0529; L2 0.304, 0.177, 0.0801; orders 0.977, 0.962; |dL_z|/L_z 0.369, 0.000994, 0.0227 (ratio <= 0.62); core loss 0.596, 0.224, 0.0717, order 1.527; Linf 0.792, 0.526, 0.292, order 0.720
    ("square", "hllc"): dict(L1={64: 0.098, 128: 0.064}, L2={64: 0.19, 128: 0.15},
        L1_order=(0.53, 0.93), L2_order=(0.23, 0.63)),
    # measured L1 0.0757, 0.0489; L2 0.147, 0.117; orders 0.629, 0.326
    ("square", "hll"): dict(L1={64: 0.1, 128: 0.066}, L2={64: 0.19, 128: 0.15},
        L1_order=(0.53, 0.93), L2_order=(0.23, 0.63)),
    # measured L1 0.0786, 0.0507; L2 0.149, 0.119; orders 0.631, 0.328
    ("sine", "hllc"): dict(L1={32: 0.031, 64: 0.0096, 128: 0.003}, L2={32: 0.045, 64: 0.015, 128: 0.0048},
        L1_order=(1.58, 1.98), L2_order=(1.52, 1.92),
        Linf={32: 0.15, 64: 0.066, 128: 0.028}, Linf_order=(1.1, 1.5)),
    # measured L1 0.024, 0.00735, 0.00233; L2 0.035, 0.0116, 0.00368; orders 1.683, 1.623; Linf 0.112, 0.0506, 0.0214, order 1.195
    ("sine", "hll"): dict(L1={32: 0.039, 64: 0.013, 128: 0.0041}, L2={32: 0.058, 64: 0.021, 128: 0.0067},
        L1_order=(1.53, 1.93), L2_order=(1.46, 1.86),
        Linf={32: 0.2, 64: 0.095, 128: 0.042}, Linf_order=(1.01, 1.41)),
    # measured L1 0.03, 0.0101, 0.00315; L2 0.0449, 0.016, 0.00514; orders 1.626, 1.564; Linf 0.151, 0.0733, 0.0325, order 1.110
}
LZ_REDISTRIBUTED = 0.8   # |dL_z| of the box / the core's loss: L_z moves out, it is not lost


def _ic(cfg):
    """The registered IC of cfg.test_problem as numpy."""
    from luna.ic_generation.ic import initial_conditions
    return np.asarray(initial_conditions(cfg))


def sound(n, solver="hllc", amp=1e-6):
    """A right-going sound wave (c = 1) after one period: (U0, U, (rho - exact) per amplitude)."""
    cfg = accuracy_cfg(test_problem="Soundwave", SPATIAL_DIMENSION=1, Nx=n, Lx=1.0,
                       BOUNDARY=0, gamma=GAMMA, riemann_solver=solver)
    # The registered IC has amp 1e-4, whose steepening (~amp^2) floors the error near N = 1024.
    s = amp * np.sin(2 * np.pi * (np.arange(n) + 0.5) / n)
    rho, u, p = 1.0 + s, s, 0.6 + s
    U0 = np.stack([rho, rho * u, p / (GAMMA - 1.0) + 0.5 * rho * u * u])
    U, _ = evolve(cfg, U0, 1.0)
    return U0, U, (U[0] - U0[0]) / amp


def sound_advected(n, solver="hllc", t=1.0):
    """The advected sound wave (all three waves): (U0, U, (rho - exact linear solution) per amplitude)."""
    cfg = accuracy_cfg(test_problem="Soundwave-Advected", SPATIAL_DIMENSION=1, Nx=n, Lx=1.0,
                       BOUNDARY=0, gamma=GAMMA, riemann_solver=solver)
    U0 = _ic(cfg)
    U, _ = evolve(cfg, U0, t)
    eps, k, c = 1e-6, 2 * np.pi, np.sqrt(GAMMA)
    x = (np.arange(n) + 0.5) / n
    s = lambda v: eps * np.sin(k * (x - v * t))           # dp = drho = du = eps sin(kx) at t=0
    rho = 1.0 + (1.0 - 1.0 / c ** 2) * s(1.0) + 0.5 * (1.0 + c) / c ** 2 * s(1.0 + c) \
        + 0.5 * (1.0 - c) / c ** 2 * s(1.0 - c)
    return U0, U, (U[0] - rho) / eps


def gresho(n, solver="hllc", t=1.0, mmax=0.1):
    """The steady Gresho vortex at t (peak speed 1): (U0, U, |v - v_exact|)."""
    cfg = accuracy_cfg(test_problem="Gresho-Vortex-2D", SPATIAL_DIMENSION=2, Nx=n, Ny=n,
                       Lx=1.0, Ly=1.0, BOUNDARY=0, gamma=GAMMA, problem=gresho_vortex.Params(Mmax=mmax),
                       riemann_solver=solver)
    U0 = _ic(cfg)
    U, _ = evolve(cfg, U0, t)
    return U0, U, np.hypot(*(U[1:3] / U[0] - U0[1:3] / U0[0]))


def advect(problem, n, solver="hllc"):
    """Diagonal advection at u = v = 1 over one period (square, or one sine wavelength): (U0, U, rho - rho_0)."""
    from luna.ic_generation import soundwave_advected2d
    kw = (dict(test_problem="Soundwave-Advected-2D", problem=soundwave_advected2d.Params(waves=1, amp=0.5))
          if problem == "sine" else dict(test_problem="Square-Advected"))
    cfg = accuracy_cfg(SPATIAL_DIMENSION=2, Nx=n, Ny=n, Lx=1.0, Ly=1.0, BOUNDARY=0, gamma=GAMMA,
                       riemann_solver=solver, **kw)
    U0 = _ic(cfg)
    U, _ = evolve(cfg, U0, 1.0)
    return U0, U, U[0] - U0[0]


RUNS = {"sound": sound, "sound_adv": sound_advected, "gresho": gresho,
        "square": lambda n, solver="hllc": advect("square", n, solver),
        "sine": lambda n, solver="hllc": advect("sine", n, solver)}


def angular_momentum(U0, U, r_core=0.4):
    """(|dL_z| / L_z(0) of the box, -dL_z / L_z(0) inside r_core) about the box centre (2D, unit box).

    Cartesian FV conserves L_z only to truncation: numerical viscosity moves it out of the vortex
    (the core loss, which converges) and the box total is the small, sign-changing remainder."""
    n = U0.shape[1]
    x, y = np.meshgrid((np.arange(n) + 0.5) / n - 0.5, (np.arange(n) + 0.5) / n - 0.5, indexing="ij")
    lz = lambda V: x * V[2] - y * V[1]                    # noqa: E731
    L0, dL = lz(U0).sum(), lz(U) - lz(U0)
    return float(abs(dL.sum()) / L0), float(-dL[np.hypot(x, y) < r_core].sum() / L0)


def _ladder(name, solver, bounds):
    """L1, L2 at each N, their orders, conservation to round-off (periodic); Gresho: the L_z drift."""
    ns = sorted(bounds["L1"])
    errs, core = [], []
    for n in ns:
        U0, U, e = RUNS[name](n, solver)
        check_conserved(f"{name} {solver} N={n}", U0, U)
        l1, l2, linf = float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e * e))), float(np.abs(e).max())
        errs.append((l1, l2, linf))
        check(f"{name} {solver} N={n} L1", l1, bounds["L1"][n])
        if "L2" in bounds:
            check(f"{name} {solver} N={n} L2", l2, bounds["L2"][n])
        if "Linf" in bounds:
            check(f"{name} {solver} N={n} Linf", linf, bounds["Linf"][n])
        if "Lz_core" in bounds:
            total, lost = angular_momentum(U0, U)
            core.append(lost)
            check(f"{name} {solver} N={n} core L_z loss", lost, bounds["Lz_core"][n])
            check(f"{name} {solver} N={n} |dL_z| / core loss", total / lost, LZ_REDISTRIBUTED)
    for k, norm in ((k, nm) for k, nm in enumerate(("L1", "L2", "Linf")) if nm in bounds):
        order = -np.polyfit(np.log(ns), np.log([e[k] for e in errs]), 1)[0]
        check(f"{name} {solver} {norm} order", order, bounds[f"{norm}_order"][0], lower=True)
        check(f"{name} {solver} {norm} order", order, bounds[f"{norm}_order"][1])
    if core:
        order = -np.polyfit(np.log(ns), np.log(core), 1)[0]
        check(f"{name} {solver} core L_z loss order", order, bounds["Lz_core_order"][0], lower=True)
        check(f"{name} {solver} core L_z loss order", order, bounds["Lz_core_order"][1])


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("name", PROBLEMS)
def test_smooth_ladder(name, solver):
    """L1 and L2 at every resolution, the orders, conservation (and Gresho's angular momentum)."""
    _ladder(name, solver, BOUNDS[name, solver])


GPU_BOUNDS = {  # one MI300A, HLLC; L1 only until re-measured on viper (L2, L_z, HLL)
    "sound": dict(L1={256: 0.00041, 512: 0.00011, 1024: 2.8e-5}, L1_order=(1.83, 2.23)),
    # measured 0.000315, 8.36e-5, 2.18e-5; order 1.926
    "gresho": dict(L1={128: 0.03, 256: 0.013, 512: 0.005}, L1_order=(1.2, 1.6)),
    # measured 0.0229, 0.00955, 0.00379; order 1.297
    "square": dict(L1={256: 0.041, 512: 0.026}, L1_order=(0.55, 0.95)),
    # measured 0.0314, 0.0200; order 0.648
    "sine": dict(L1={256: 0.0009, 512: 0.00026}, L1_order=(1.73, 2.13)),
    # measured 0.000686, 0.000193; order 1.831
}


@gpu_only
@pytest.mark.parametrize("name", list(GPU_BOUNDS))
def test_smooth_ladder_gpu(name):
    """The same at production-like resolutions, on a GPU."""
    _ladder(name, "hllc", GPU_BOUNDS[name])
