"""Single-mode Richtmyer-Meshkov in a closed tube (A14), heavy -> light and light -> heavy: the interface ripple
psi(t) against the exact compressible linear theory (tools/rm_linear.py: Cobos-Campos & Wouchuk 2016, 2017),
theta-group states (Thornber et al. 2017: gamma 5/3, Mach 1.8439, densities 3 and 1), seeded +psi0 and -psi0."""
import functools

import numpy as np
import pytest

from luna.reference_data import reference_path
from tests._helpers import accuracy_cfg, check, evolve, gpu_only

G, MACH, P0, LAM = 5.0 / 3.0, 1.8439, 1.0, 1.0
K, KPSI0, TAU = 2.0 * np.pi / LAM, 0.01, 20.0   # one mode; k psi0 = 0.01 (linear); window up to k c_af t = TAU
TAU0, DT_OUT = 2.0, 0.02                        # before TAU0 the reflected wave still sits at the interface
CASES = {"heavy-light": (3.0, 1.0), "light-heavy": (1.0, 3.0)}      # rho on the shock side, rho beyond
# driver, gap shock-interface, section beyond: wall-reflected waves reach the interface only after the window
GEOM = {"heavy-light": (4.0, 0.5, 6.0), "light-heavy": (9.0, 0.5, 8.0)}
SCHEMES = {"hllc-minmod": ("hllc", 0), "hllc-mc": ("hllc", 1)}
NS, NS_GPU = (32, 64), (128, 256, 512)          # cells per wavelength; the box is half a wavelength wide
NAMES = ("L1", "L2", "Linf", "rate")
AK_REF = reference_path("athenak/richtmyer_meshkov/rm_ppmx_rk4_512.npz")
BOUNDS = {
    ('heavy-light', 'hllc-mc'): dict(L1={32: 0.5, 64: 0.3},
        L2={32: 0.58, 64: 0.35},
        Linf={32: 1.1, 64: 0.68},
        rate={32: 0.33, 64: 0.21},
        L1_order=(0.65, 1.05),
        L2_order=(0.63, 1.03),
        Linf_order=(0.52, 0.92),
        rate_order=(0.58, 0.98)),
    # measured L1 0.383, 0.228; 0.75 | L2 0.44, 0.265; 0.73 | Linf 0.8, 0.519; 0.62 | rate 0.248, 0.155; 0.68
    ('heavy-light', 'hllc-minmod'): dict(L1={32: 0.66, 64: 0.45},
        L2={32: 0.76, 64: 0.52},
        Linf={32: 1.5, 64: 1.1},
        rate={32: 0.44, 64: 0.3},
        L1_order=(0.47, 0.87),
        L2_order=(0.47, 0.87),
        Linf_order=(0.40, 0.80),
        rate_order=(0.47, 0.87)),
    # measured L1 0.503, 0.34; 0.57 | L2 0.583, 0.394; 0.57 | Linf 1.09, 0.772; 0.50 | rate 0.335, 0.225; 0.57
    ('light-heavy', 'hllc-mc'): dict(L1={32: 0.82, 64: 0.55},
        L2={32: 0.94, 64: 0.63},
        Linf={32: 1.8, 64: 1.2},
        rate={32: 0.32, 64: 0.22},
        L1_order=(0.49, 0.89),
        L2_order=(0.48, 0.88),
        Linf_order=(0.56, 0.96),
        rate_order=(0.47, 0.87)),
    # measured L1 0.627, 0.417; 0.59 | L2 0.723, 0.482; 0.58 | Linf 1.36, 0.856; 0.66 | rate 0.244, 0.165; 0.57
    ('light-heavy', 'hllc-minmod'): dict(L1={32: 1.2, 64: 0.81},
        L2={32: 1.4, 64: 0.94},
        Linf={32: 2.6, 64: 1.7},
        rate={32: 0.46, 64: 0.33},
        L1_order=(0.39, 0.79),
        L2_order=(0.40, 0.80),
        Linf_order=(0.49, 0.89),
        rate_order=(0.39, 0.79)),
    # measured L1 0.872, 0.619; 0.49 | L2 1.02, 0.722; 0.50 | Linf 1.95, 1.3; 0.59 | rate 0.354, 0.252; 0.49
}
BOUNDS_GPU = {
    ('heavy-light', 'hllc-mc'): dict(L1={128: 0.19, 256: 0.13, 512: 0.08},
        L2={128: 0.23, 256: 0.15, 512: 0.091},
        Linf={128: 0.41, 256: 0.26, 512: 0.17},
        rate={128: 0.13, 256: 0.083, 512: 0.052},
        L1_order=(0.53, 0.93),
        L2_order=(0.54, 0.94),
        Linf_order=(0.57, 0.97),
        rate_order=(0.57, 0.97)),
    # measured L1 0.146, 0.0959, 0.0611; 0.63 | L2 0.169, 0.111, 0.0699; 0.64 | Linf 0.312, 0.198, 0.124; 0.67 | rate 0.0988, 0.0637, 0.0393; 0.67
    ('heavy-light', 'hllc-minmod'): dict(L1={128: 0.29, 256: 0.19, 512: 0.13},
        L2={128: 0.34, 256: 0.22, 512: 0.15},
        Linf={128: 0.63, 256: 0.4, 512: 0.26},
        rate={128: 0.2, 256: 0.13, 512: 0.084},
        L1_order=(0.52, 0.92),
        L2_order=(0.52, 0.92),
        Linf_order=(0.55, 0.95),
        rate_order=(0.51, 0.91)),
    # measured L1 0.222, 0.143, 0.0939; 0.62 | L2 0.257, 0.166, 0.109; 0.62 | Linf 0.478, 0.301, 0.195; 0.65 | rate 0.149, 0.0974, 0.064; 0.61
    ('light-heavy', 'hllc-mc'): dict(L1={128: 0.34, 256: 0.21, 512: 0.13},
        L2={128: 0.39, 256: 0.24, 512: 0.14},
        Linf={128: 0.69, 256: 0.42, 512: 0.25},
        rate={128: 0.14, 256: 0.08, 512: 0.048},
        L1_order=(0.63, 1.03),
        L2_order=(0.63, 1.03),
        Linf_order=(0.63, 1.03),
        rate_order=(0.64, 1.04)),
    # measured L1 0.257, 0.157, 0.093; 0.73 | L2 0.297, 0.18, 0.107; 0.73 | Linf 0.527, 0.321, 0.191; 0.73 | rate 0.102, 0.0609, 0.0367; 0.74
    ('light-heavy', 'hllc-minmod'): dict(L1={128: 0.55, 256: 0.36, 512: 0.23},
        L2={128: 0.64, 256: 0.42, 512: 0.27},
        Linf={128: 1.2, 256: 0.75, 512: 0.49},
        rate={128: 0.23, 256: 0.15, 512: 0.095},
        L1_order=(0.52, 0.92),
        L2_order=(0.52, 0.92),
        Linf_order=(0.51, 0.91),
        rate_order=(0.51, 0.91)),
    # measured L1 0.417, 0.272, 0.177; 0.62 | L2 0.485, 0.317, 0.206; 0.62 | Linf 0.874, 0.575, 0.373; 0.61 | rate 0.17, 0.112, 0.0726; 0.61
}


@functools.lru_cache(maxsize=None)
def setup(case):
    """Theory (rm_setup) and tube: shock at xs, interface at xi, walls at 0 and L, the window's end t_end."""
    from tools.rm_linear import rm_setup
    ri, rt = CASES[case]
    s = rm_setup(G, G, ri, rt, P0, MACH, K, 1.0)
    drv, gap, right = GEOM[case]
    t_hit = gap / s["D_i"]
    return dict(s=s, ri=ri, rt=rt, xs=drv, xi=drv + gap, L=drv + gap + right, t_hit=t_hit,
                t_end=t_hit + TAU / (K * s["c_trans_f"]))


def initial(case, n, sign, nsub=128):
    """Cell averages (rho, mx, my, E), (4, n L, n/2): driver | gas at rest | gas at rest, the interface at
    xi + sign psi0 cos(k y) (subsampled in y next to it), on y in [0, lam/2] between walls."""
    d = setup(case)
    s, psi0 = d["s"], KPSI0 / K
    nx, ny = int(round(d["L"] * n)), n // 2
    h, hy = d["L"] / nx, 0.5 * LAM / ny
    a = np.arange(nx) * h
    f_i = np.clip((d["xi"] - a) / h, 0, 1)[:, None] * np.ones((1, ny))
    near = np.where(np.abs(a + 0.5 * h - d["xi"]) < psi0 + 2 * h)[0]
    ys = (np.arange(ny)[:, None] + (np.arange(nsub)[None, :] + 0.5) / nsub) * hy
    f_i[near] = np.clip((d["xi"] + sign * psi0 * np.cos(K * ys)[None] - a[near, None, None]) / h, 0, 1).mean(-1)
    f_s = np.clip((d["xs"] - a) / h, 0, 1)[:, None] * np.ones((1, ny))
    r1, u1, p1 = s["rho_inc1"], s["U1"], s["p1"]
    rho = f_s * r1 + (f_i - f_s) * d["ri"] + (1 - f_i) * d["rt"]
    E = f_s * (p1 / (G - 1) + 0.5 * r1 * u1 * u1) + (1 - f_s) * P0 / (G - 1)
    return np.stack([rho, f_s * r1 * u1, np.zeros_like(rho), E])


def amplitude(case, t, rho):
    """cos(k y) amplitude of the per-column crossing of the mean post-shock density near the expected interface."""
    d = setup(case)
    s = d["s"]
    mid = 0.5 * (s["rho_inc_f"] + float(s["rho_trans_f"]))
    nx, ny = rho.shape
    h = d["L"] / nx
    x = (np.arange(nx) + 0.5) * h
    xe = d["xi"] + s["U_interface"] * (t - d["t_hit"])
    w = np.where(np.abs(x - xe) < 6 * h + 8 * KPSI0 / K)[0]          # the ripple reaches 5.9 psi0 by TAU
    xc = np.empty(ny)
    for j in range(ny):
        r = rho[w, j]
        c = np.where(np.diff(np.sign(r - mid)) != 0)[0]
        if len(c) == 0:
            return np.nan
        xs = x[w][c] + (mid - r[c]) * h / (r[c + 1] - r[c])
        xc[j] = xs[np.argmin(np.abs(xs - xe))]          # a wiggle can add a crossing: take the interface's
    cy = np.cos(K * (np.arange(ny) + 0.5) * 0.5 * LAM / ny)
    return float((xc * cy).sum() / (cy * cy).sum())


@functools.lru_cache(maxsize=None)
def history(case, n, scheme, frac=1.0, order=1):
    """(tau, psi/psi0), the odd part (psi(+psi0) - psi(-psi0)) / 2; a run stopped at frac reports its outputs
    at the full window's times (a broken clock)."""
    d = setup(case)
    solver, limiter = SCHEMES[scheme]
    cfg = accuracy_cfg(test_problem="Kelvin-Helmholtz", SPATIAL_DIMENSION=2, Nx=int(round(d["L"] * n)), Ny=n // 2,
                       Lx=d["L"], Ly=0.5 * LAM, BOUNDARY=2, gamma=G, riemann_solver=solver, LIMITER=limiter,
                       ORDER=order)
    times = np.arange(1, int(d["t_end"] / DT_OUT) + 1) * DT_OUT
    a = []
    for sg in (1.0, -1.0):
        rho, _ = evolve(cfg, initial(case, n, sg), frac * times, observe=lambda U: U[0])
        a.append([amplitude(case, t, r) for t, r in zip(frac * times, rho)])
    return K * d["s"]["c_trans_f"] * (times - d["t_hit"]), (np.array(a[0]) - np.array(a[1])) / (2 * KPSI0 / K)


def errors(case, n, scheme, **kw):
    """(L1, L2, Linf) of psi/psi0 - exact over TAU0 <= tau <= TAU, and the relative error of the fitted slope."""
    from tools.rm_linear import psi_history
    tau, psi = history(case, n, scheme, **kw)
    m = (tau >= TAU0) & (tau <= TAU)
    assert np.isfinite(psi[m]).all(), f"{case} N={n} {scheme}: interface lost inside the window"
    exact = psi_history(setup(case)["s"], tau[m])[0]
    e = psi[m] - exact
    rate = abs(np.polyfit(tau[m], psi[m], 1)[0] / np.polyfit(tau[m], exact, 1)[0] - 1)
    return float(np.abs(e).mean()), float(np.sqrt((e * e).mean())), float(np.abs(e).max()), float(rate)


def gate(case, scheme, ns, b):
    """Every norm at every N under its bound, and its order inside its band."""
    res = [dict(zip(NAMES, errors(case, n, scheme))) for n in ns]
    print(f"MEASURED rm {case} {scheme} {','.join(map(str, ns))} = {res}")
    for nm in NAMES:
        vals = [r[nm] for r in res]
        for n, v in zip(ns, vals):
            check(f"rm {case} {scheme} N={n} {nm}", v, b[case, scheme][nm][n])
        order = -np.polyfit(np.log(ns), np.log(vals), 1)[0]
        check(f"rm {case} {scheme} {nm} order", order, b[case, scheme][f"{nm}_order"][0], lower=True)
        check(f"rm {case} {scheme} {nm} order", order, b[case, scheme][f"{nm}_order"][1])


KEYS = [(c, s) for c in CASES for s in SCHEMES]


def test_theory_reproduces_the_worked_examples():
    """rm_linear's asymptotic velocity matches CW17 App. B (HF100/10, -0.284574934) and CW16 App. D (case 5,
    0.145287), in units k psi0 D_i."""
    from tools.rm_linear import rm_linear
    hf, sf6 = rm_linear(1.45, 1.8, 0.0706, 15.3), rm_linear(1.0935, 1.4, 5.25, 2.86)
    assert abs(hf["ui"] + 0.284574934) < 1e-9 and abs(sf6["ui"] - 0.145287) < 1e-6, (hf["ui"], sf6["ui"])


@pytest.mark.parametrize("case", list(CASES))
def test_history_matches_the_bromwich_inversion(case):
    """The exact curve psi(tau) (Bessel series) against the independent numerical Bromwich inversion over the
    window (measured 1.3e-15 / 3.6e-15; scipy 1.16.3's itj0y0 made it 5e-10 / 1e10, docs/history/CODE_NOTES.md)."""
    from tools.rm_linear import _laplace_psi, psi_history
    s = setup(case)["s"]
    tau = np.array([TAU0, 5.0, 10.0, 15.0, TAU])
    pb, pl = psi_history(s, tau)[0], _laplace_psi(s["raw"], tau)[0]
    check(f"rm {case} Bessel vs Bromwich", float(np.abs(pb - pl).max() / np.abs(pl).max()), 1e-12)


@pytest.mark.parametrize("case,scheme", KEYS, ids=[f"{c}-{s}" for c, s in KEYS])
def test_richtmyer_meshkov(case, scheme):
    """psi(t) against the exact linear theory, and the growth-rate error, at 32 and 64 cells per wavelength."""
    gate(case, scheme, NS, BOUNDS)


@pytest.mark.parametrize("kw", [dict(frac=0.8), dict(order=0)], ids=["short", "first-order"])
def test_bound_catches_a_broken_run(kw):
    """Teeth: stopped at 0.8 of the window, or first order, light -> heavy minmod at 64 misses its L1 bound."""
    assert errors("light-heavy", 64, "hllc-minmod", **kw)[0] > BOUNDS["light-heavy", "hllc-minmod"]["L1"][64]


@gpu_only
@pytest.mark.parametrize("case,scheme", KEYS, ids=[f"{c}-{s}" for c, s in KEYS])
def test_richtmyer_meshkov_gpu(case, scheme):
    """The same at 128, 256, 512 on a GPU; AthenaK PPMX + RK4 at 512 against the same exact curve for context."""
    gate(case, scheme, NS_GPU, BOUNDS_GPU)
    with np.load(AK_REF) as z:
        print(f"MEASURED rm {case} AthenaK 512 L1 = {float(z[f'{case}_L1']):.4e} (LUNA {scheme} 512 above)")
