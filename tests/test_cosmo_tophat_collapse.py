"""Pressureless top-hat collapse in Einstein-de Sitter (A16, part B): a uniform cold sphere (comoving radius 0.15,
z = 50, turnaround at a = 0.5) against the exact closed-FRW interior, R = A (1 - cos th), t = B (th - sin th),
itself checked against a direct integration of R'' = -G M / R^2 (tools/tophat.py). The box outside is lowered so
the mean is exactly 1: by Gauss the interior then feels exactly the sphere's overdensity despite the mean-subtracted
Poisson solve; periodic images enter at l = 4 only (R0 = 0.1 and 0.15 agree). Gated: the mean rho~ of the inner
half (the 8 centre cells keep a ~1 % cold-flow spike where the flow converges on a point), the interior v~ slope,
and the Layzer-Irvine budget d(K~ + U~ + W~) = W~ dln a (super-comoving, gamma 5/3; W~ on the 7-point stencil)."""
import numpy as np

from tests._helpers import accuracy_cfg, check, evolve_staged, gpu_only

A_I, R0, A_TA = 1.0 / 51.0, 0.15, 0.5
P0 = 1e-10                     # p~ and its floor: cold flow needs the floor; 1e-12 .. 1e-9 agree to 1e-5
A_SNAP = (0.1, 0.25, 0.5, 0.65)
N_LI = 160                     # log-spaced snapshots for the budget (Simpson; the quadrature error is 2e-4 of it)


def tophat():
    """The exact solution with turnaround at A_TA."""
    from scipy.optimize import brentq
    from luna.cosmology.exact_tophat import TopHat
    return TopHat(brentq(lambda d: TopHat(d, A_I).a_turnaround() - A_TA, 1e-3, 0.2), A_I)


def test_exact_solution_two_routes():
    """The parametric closed-FRW solution against the shell ODE, and 9 pi^2 / 16 at turnaround."""
    th = tophat()
    a = np.array([0.05, 0.25, A_TA, 0.65, 0.75])
    check("tophat parametric vs ODE", float(np.abs(th.one_plus_delta(a) / th.integrate(a) - 1).max()), 1e-9)
    check("tophat 1 + delta at turnaround - 9 pi^2/16", abs(th.one_plus_delta([A_TA])[0] - 9 * np.pi ** 2 / 16), 1e-9)


def ic(N, th):
    """rho~ = 1 + delta_i inside R0, lowered outside to mean 1; v~ the growing mode scaled to the exact interior."""
    x = (np.arange(N) + 0.5) / N - 0.5
    X = np.stack(np.meshgrid(x, x, x, indexing="ij"))
    inside = np.sqrt((X ** 2).sum(0)) < R0
    delta_i = th.one_plus_delta([A_I])[0] - 1.0
    rho = np.where(inside, 1.0 + delta_i, 1.0 - delta_i * inside.mean() / (1.0 - inside.mean()))
    k = 2 * np.pi * np.fft.fftfreq(N, 1.0 / N)
    K = np.stack(np.meshgrid(k, k, k, indexing="ij"))
    k2 = (K ** 2).sum(0)
    k2[0, 0, 0] = 1.0
    dk = np.fft.fftn(rho - 1.0)
    s = 3.0 * A_I ** 2 * th.vpec_over_r([A_I])[0] / delta_i      # grad lap^-1 delta = delta x / 3 inside
    v = np.stack([np.real(np.fft.ifftn(-1j * K[i] * dk / k2)) for i in range(3)]) * s
    U = np.zeros((5, N, N, N))
    U[0], U[1:4] = rho, rho * v
    U[-1] = 1.5 * P0 + 0.5 * rho * (v ** 2).sum(0)
    return U, X


def cfg_for(N, **kw):
    from luna.ic_generation import zeldovich
    return accuracy_cfg(**{**dict(test_problem="Zeldovich-Pancake", SPATIAL_DIMENSION=3, Nx=N, Ny=N, Nz=N, Lx=1.0,
                                  Ly=1.0, Lz=1.0, BOUNDARY=0, gamma=5.0 / 3.0, CosmologicalIntegration=True,
                                  cosmo_Omega_m=1.0, cosmo_Omega_L=0.0, cosmo_H0=1.0, a_init=A_I,
                                  a_end=max(A_SNAP), cosmo_p_floor=P0, cosmo_T_floor=0.0,
                                  problem=zeldovich.Params()), **kw})


def run(N, a_max=max(A_SNAP), signed=False, **kw):
    """({a: (|core rho~ / exact - 1|, |v~ slope / exact - 1|)}, {a: |Layzer-Irvine residual| / |W~(a)|})."""
    from luna.setup.clock import make_clock
    th = tophat()
    U0, X = ic(N, th)
    cfg = cfg_for(N, a_end=a_max, **kw)
    clock = make_clock(cfg)
    r = np.sqrt((X ** 2).sum(0))
    k = 2 * np.pi * np.fft.fftfreq(N, 1.0 / N)
    lam = sum(np.meshgrid(*[-4 * N * N * np.sin(k / (2 * N)) ** 2] * 3, indexing="ij"))
    lam[0, 0, 0] = 1.0
    snaps = [a for a in A_SNAP if a <= a_max]
    a_li = np.geomspace(A_I, a_max, N_LI + 1)
    a_all = np.union1d(a_li[1:], snaps)
    taus = [clock.tau_of_a(a) for a in a_all]
    a_of = dict(zip(taus, a_all))

    def energies(U, a):
        """(E~, W~) per unit volume, W~ = 1/2 <delta phi~> on the 7-point stencil, lap phi~ = 3/2 a delta."""
        d = U[0] - 1.0
        pk = 1.5 * a * np.fft.fftn(d) / lam
        pk[0, 0, 0] = 0.0
        return float(U[-1].mean()), 0.5 * float((d * np.real(np.fft.ifftn(pk))).mean())

    def observe(U, tau):
        a = a_of[tau]
        g = None
        if signed or any(abs(a - s_) < 1e-12 for s_ in snaps):   # signed: every output, errors with their sign
            ex = th.one_plus_delta([a])[0]
            core = r < 0.5 * R0 * (ex / th.one_plus_delta([A_I])[0]) ** (-1.0 / 3.0)
            vx = U[1][core] / U[0][core]
            slope = (vx * X[0][core]).sum() / (X[0][core] ** 2).sum()
            g = (float(U[0][core].mean() / ex - 1), float(slope / (a * a * th.vpec_over_r([a])[0]) - 1))
            g = g if signed else (abs(g[0]), abs(g[1]))
        dp = max(abs(float(U[d].sum() - U0[d].sum())) for d in (1, 2, 3)) / float(np.abs(U[1:4]).sum())
        return a, energies(U, a), g, dp

    obs = evolve_staged(cfg, U0, taus, observe=observe)
    gates = {round(a, 6): g for a, _, g, _ in obs if g is not None}
    dp = max(o[3] for o in obs)
    keep = [(a, *e) for a, e, _, _ in obs if np.min(np.abs(a_li - a)) < 1e-14]
    a, E, W = map(np.array, zip((A_I, *energies(U0, A_I)), *keep))
    lna, tot = np.log(a), E + W
    li = {}
    for s_ in snaps:
        j = int(np.argmin(abs(a - s_)))
        j -= j % 2
        x, w = lna[:j + 1], W[:j + 1]
        simpson = np.sum((x[2::2] - x[:-2:2]) / 6 * (w[:-2:2] + 4 * w[1:-1:2] + w[2::2]))
        li[s_] = float(abs(tot[j] - tot[0] - simpson) / abs(W[j]))
    return gates, li, dp


# Measured on one MI300A (minmod, dloga_max 0.01, 2026-10-08); bounds 1.3 x measured, orders measured -0.1 / +0.3.
# The 64^3 core error at a = 0.65 (0.38) is unresolved (3.6 cells across) and not gated on the CPU.
CORE = {64: {0.1: 1.1e-3, 0.25: 5.8e-3, 0.5: 0.11},
        128: {0.1: 2.5e-4, 0.25: 8.0e-4, 0.5: 3.8e-3, 0.65: 5.1e-2},
        256: {0.1: 4.6e-5, 0.25: 1.4e-4, 0.5: 5.8e-4, 0.65: 1.4e-3}}
SLOPE = {64: {0.1: 5.1e-3, 0.25: 9.3e-3, 0.5: 9.0e-2},
         128: {0.1: 1.3e-3, 0.25: 1.9e-3, 0.5: 4.3e-3, 0.65: 3.3e-2},
         256: {0.1: 2.9e-4, 0.25: 3.9e-4, 0.5: 8.5e-4, 0.65: 1.9e-3}}
LI = {64: 2.8e-3, 128: 6.2e-3, 256: 3.7e-3}     # max over the gated a: the residual changes sign, so no order
ORDER = {"core": {0.25: (2.42, 2.82), 0.5: (2.61, 3.01)}, "slope": {0.25: (2.13, 2.53), 0.5: (2.25, 2.65)}}


# max over the run of max_d |sum (rho v~)_d| / sum |rho v~|: periodic, the central-difference force on the 7-point
# potential is antisymmetric (sum rho grad phi = 0) and the sphere is centred, so 0 up to the solve's tolerance.
P_TOL = 1e-13   # measured 4.2e-17 (64^3, A100)


def gate(N, gates, li, dp):
    """Check every gated snapshot of one run, and the momentum."""
    check(f"tophat N={N} max |sum rho v~| / sum |rho v~|", dp, P_TOL)
    for a, b in CORE[N].items():
        check(f"tophat N={N} a={a} |core rho~ / exact - 1|", gates[a][0], b)
        check(f"tophat N={N} a={a} |v~ slope / exact - 1|", gates[a][1], SLOPE[N][a])
    check(f"tophat N={N} max Layzer-Irvine |residual| / |W~|", max(li[a] for a in CORE[N]), LI[N])


def test_tophat_collapse():
    """64^3: the core density and v~ slope to turnaround, and the Layzer-Irvine budget."""
    gate(64, *run(64))


def test_bound_catches_one_percent_gravity(monkeypatch):
    """Teeth: gravity 1 % too strong (a mis-set Omega_m or H0) misses every kind of 64^3 bound."""
    from luna.setup.clock import CosmoClock
    orig = CosmoClock.kick_scale
    monkeypatch.setattr(CosmoClock, "kick_scale", lambda self, a: orig(self, a) * 1.01)
    gates, li, _ = run(64)
    assert gates[0.25][0] > CORE[64][0.25] and gates[0.1][1] > SLOPE[64][0.1]
    assert max(li[a] for a in CORE[64]) > LI[64]


@gpu_only
def test_tophat_collapse_gpu():
    """128^3 and 256^3 through 1 + delta = 20, and the convergence order of the core density and v~ slope."""
    res = {N: run(N) for N in (128, 256)}
    for N, (gates, li, dp) in res.items():
        gate(N, gates, li, dp)
    for key, i in (("core", 0), ("slope", 1)):
        for a, (lo, hi) in ORDER[key].items():
            q = float(np.log2(res[128][0][a][i] / res[256][0][a][i]))
            check(f"tophat {key} order 128->256 a={a}", q, lo, lower=True)
            check(f"tophat {key} order 128->256 a={a}", q, hi)
