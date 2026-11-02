"""The analytic references of the Jeans, linear-wave and Zel'dovich tests, against independent derivations.

- Jeans (`tests/test_gravity_jeans_wave.py`): omega^2 = c^2 k^2 - 4 pi G rho0 recomputed from the G the test
  passes to LUNA; the eigenvalues of the linearised Euler + Poisson operator for a plane wave (numpy eig); the
  test's initial mode propagated by expm of that operator reproduces its exact answer (1 after a period, e after
  an e-fold).
- Linear waves (`tests/test_hydro_accuracy_linear_waves.py`, 1D and oblique 2D): the test's perturbation is a
  right eigenvector of the Euler flux Jacobian n . dF/dU at the background (complex-step Jacobian, independent of
  `src/luna/hydro/jacobian.py`) with the eigenvalue its period assumes.
- Zel'dovich (`luna.ic_generation.zeldovich.exact`): the Lagrangian q recovered from exact()'s velocity maps
  forward onto x; rho = 1 / (1 + D psi'(q)) and the mass coordinate int_0^x rho = q; the growth D(a) and the
  caustic a_c from the linear growth ODE in super-comoving time (Martel & Shapiro 1998).

Bounds are FP64 round-off (u = 2^-53) with the stated amplification, or the stated integrator tolerance.
"""
import numpy as np
import pytest
import scipy.linalg
from scipy.integrate import solve_ivp

import tests.test_gravity_jeans_wave as jw
import tests.test_hydro_accuracy_linear_waves as lw
from tests._helpers import check

U = np.finfo(float).eps / 2


# ------------------------------------------------------------------------------------------------------------ Jeans
def jeans_operator(k, rho0, p0, gam, G):
    """d/dt of the amplitudes (rho, u_par, p) of Im(z exp(i k.x)): continuity, momentum with the Poisson force
    -grad phi, phi_hat = -4 pi G rho_hat / k^2, adiabatic pressure."""
    return np.array([[0.0, -1j * k * rho0, 0.0],
                     [4j * np.pi * G / k, 0.0, -1j * k / rho0],
                     [0.0, -1j * k * gam * p0, 0.0]])


def jeans_mode(nj, n=8):
    """The complex amplitude (rho, u_par, p) of jw.initial(n, nj), fitted to sin and cos of k.x over the cells."""
    U0 = jw.initial(n, nj, amp=1.0)                       # linear in amp; amp = 1 keeps the fit at round-off
    shape = U0.shape[1:]
    h = [b / s for b, s in zip(jw.BOX, shape)]
    X = np.meshgrid(*[(np.arange(s) + 0.5) * hh for s, hh in zip(shape, h)], indexing="ij")
    th = sum(k * x for k, x in zip(jw.KVEC, X)).ravel()
    mom = U0[1:4]
    fields = [U0[0] - jw.RHO0, np.tensordot(jw.KVEC / jw.K, mom, 1) / jw.RHO0,
              (jw.GAM - 1.0) * (U0[4] - 0.5 * (mom * mom).sum(0) / jw.RHO0) - jw.P0]
    B = np.stack([np.sin(th), np.cos(th)], 1)
    coef = [np.linalg.lstsq(B, f.ravel(), rcond=None)[0] for f in fields]
    return np.array([c[0] + 1j * c[1] for c in coef])


def jeans_check(nj, g_scale=1.0):
    """(|omega^2_closed - omega2|/(c k)^2, max eig miss / |M|, |expm(M t) z - growth z| / |z|) for the test's G
    times g_scale."""
    G = jw.cfg_for(32, nj, "hllc-minmod", g_scale=g_scale).GRAVITATIONAL_G
    c2 = jw.GAM * jw.P0 / jw.RHO0
    w2 = c2 * jw.K ** 2 - 4.0 * np.pi * G * jw.RHO0
    w2_test = jw.omega2(nj)
    M = jeans_operator(jw.K, jw.RHO0, jw.P0, jw.GAM, G)
    ev = np.linalg.eigvals(M)
    root = np.sqrt(complex(-w2_test))
    eig_miss = float(max(np.abs(ev - w).min() for w in (-root, 0.0, root)) / np.linalg.norm(M, 2))
    z = jeans_mode(nj)
    t, growth = (2 * np.pi / np.sqrt(w2_test), 1.0) if w2_test > 0 else (1 / np.sqrt(-w2_test), np.e)
    E = scipy.linalg.expm(M * t)
    prop = float(np.abs(E @ z - growth * z).max() / np.abs(z).max())
    Mt = M * t                                             # real 6 x 6 form for expm_cond
    R = np.block([[Mt.real, -Mt.imag], [Mt.imag, Mt.real]])
    budget = float(scipy.linalg.expm_cond(R)) + np.linalg.norm(E, 2) + growth
    return abs(w2 - w2_test) / (c2 * jw.K ** 2), eig_miss, prop, budget


@pytest.mark.parametrize("nj", (0.5, 1.5))
def test_jeans_dispersion_and_mode(nj):
    """omega^2 closed form from the cfg's G equals the test's omega2 (8 u); the operator's eigenvalues are
    0, +-sqrt(-omega2) (16 u |M|, LAPACK backward error); expm(M t) of the test's initial mode gives its exact
    answer 1 | e. Its bound 4 u (kappa_exp(M t) + |expm(M t)| + growth): scaling-and-squaring has backward error
    <= u (Al-Mohy & Higham 2009, SIAM J. Matrix Anal. Appl. 31, 970), so the expm forward error is <= kappa_exp u
    (Higham 2008, Functions of Matrices, sec. 3.1; kappa_exp by scipy.linalg.expm_cond); the round-off of the
    fitted z passes through expm(M t) - growth I; 4 for the 3-term products."""
    dw, de, dp, budget = jeans_check(nj)
    check(f"Jeans nj={nj} |omega^2 closed - omega2| / (c k)^2", dw, 8 * U)
    check(f"Jeans nj={nj} max |eig(M) - (0, +-sqrt(-omega2))| / |M|", de, 16 * U)
    check(f"Jeans nj={nj} |expm(M t) z - (1|e) z| / |z| (kappa_exp + |expm| + growth = {budget:.3g})", dp,
          4 * budget * U)


def test_jeans_teeth_wrong_g():
    """Teeth: with 2 G in the cfg the closed form, the eigenvalues and the propagated mode all miss."""
    for nj in (0.5, 1.5):
        dw, de, dp, budget = jeans_check(nj, g_scale=2.0)
        print(f"MEASURED teeth Jeans nj={nj} 2G: {dw:.3e}, {de:.3e}, {dp:.3e}")
        assert dw > 8 * U and de > 16 * U and dp > 4 * budget * U


# ----------------------------------------------------------------------------------------------------- linear waves
def flux_n(Uc, n, g):
    """Euler flux along n of conserved (rho, m_1..m_d, E); complex-safe (no conjugation)."""
    rho, m, E = Uc[0], Uc[1:-1], Uc[-1]
    v = m / rho
    un = (v * n).sum()
    p = (g - 1.0) * (E - 0.5 * rho * (v * v).sum())
    return np.concatenate([[rho * un], rho * v * un + p * n, [(E + p) * un]])


def flux_jacobian(Uc, n, g, h=1e-30):
    """n . dF/dU by complex step: exact to round-off, no differencing error."""
    cols = []
    for i in range(len(Uc)):
        e = np.zeros(len(Uc), complex)
        e[i] = 1j * h
        cols.append(flux_n(Uc.astype(complex) + e, n, g).imag / h)
    return np.array(cols).T


LW_CASES = [(d, m) for d in (1, 2) for m in lw.MODES[d]]


def wave_check(sdim, mode, g=lw.G, speed=None):
    """(|A r - lam r| / (|A| |r|), min |eig(A) - lam| / |A|, sound: |r_hat - e_hat|) for the test's vector r."""
    _, _, nhat, lam_len = lw.geometry(sdim, 8)
    # initial() is linear in eps (E carries no eps^2 term): eps = +-1 isolates r at round-off
    (Up, period), (Um, _), (U0, _) = (lw.initial(sdim, 8, mode, eps=e) for e in (1.0, -1.0, 0.0))
    d = ((Up - Um) / 2.0).reshape(Up.shape[0], -1)
    r = d[:, int(np.argmax(np.abs(d).sum(0)))]
    Ub = U0.reshape(U0.shape[0], -1)[:, 0]
    A = flux_jacobian(Ub, nhat, g)
    if speed is None:
        speed = lam_len / period                                # the speed the test's period assumes
        speed = -speed if mode == "sound_l" else speed
    an = np.linalg.norm(A, 2)
    res = float(np.linalg.norm(A @ r - speed * r) / (an * np.linalg.norm(r)))
    w, V = np.linalg.eig(A)
    i = int(np.argmin(np.abs(w - speed)))
    vec = np.inf
    if mode.startswith("sound"):
        e, rr = V[:, i].real, r / np.linalg.norm(r)
        e = e / np.linalg.norm(e) * np.sign(e @ rr)
        vec = float(np.abs(rr - e).max())
    return res, float(abs(w[i] - speed) / an), vec


@pytest.mark.parametrize("sdim,mode", LW_CASES, ids=[f"{d}d-{m}" for d, m in LW_CASES])
def test_linear_wave_eigenvectors(sdim, mode):
    """The test's perturbation r (normalisation: drho = 1 for sound and entropy, |dv| = 1 for shear) satisfies
    A r = lam r with the speed its period assumes (c = 1 for sound, -1 for sound_l, |v0| = 0.5 otherwise),
    lam is an eigenvalue of numpy eig(A), and for sound r / |r| equals numpy's unit eigenvector. Bounds: 32 u
    (complex-step A exact to round-off; 4-5 term products), LAPACK backward error 16 u |A| (eigenvector:
    eigenvalue gap >= 0.5 |A|, so 64 u)."""
    res, ev, vec = wave_check(sdim, mode)
    check(f"wave {sdim}D {mode} |A r - lam r| / (|A| |r|)", res, 32 * U)
    check(f"wave {sdim}D {mode} min |eig(A) - lam| / |A|", ev, 16 * U)
    if mode.startswith("sound"):
        check(f"wave {sdim}D {mode} |r_hat - eig vector|", vec, 64 * U)


def test_linear_wave_teeth():
    """Teeth: sound against the gamma = 7/5 Jacobian, entropy and shear at the sound speed 1 instead of |v0|."""
    for sdim, mode in LW_CASES:
        res = wave_check(sdim, mode, g=1.4)[0] if mode.startswith("sound") else wave_check(sdim, mode, speed=1.0)[0]
        print(f"MEASURED teeth wave {sdim}D {mode}: {res:.3e}")
        assert res > 32 * U


# ------------------------------------------------------------------------------------------------------- Zel'dovich
def zel():
    """(exact, EdS cosmology, a_caustic default) of luna.ic_generation.zeldovich."""
    from luna.cosmology import Cosmology
    from luna.ic_generation import zeldovich
    return zeldovich.exact, Cosmology(Omega_m=1.0, Omega_L=0.0), zeldovich.Params().a_caustic


FRACS = (0.2, 0.5, 0.9, 0.99, 0.999)                  # a / a_c


def zel_roundtrip(frac, a_c_wrong=None):
    """(max |q - A sin(k q)/k - x|, max |rho (1 - A cos k q) - 1|) with q from exact()'s v, EdS: dD/dtau = a^1.5."""
    exact, eds, a_c = zel()
    a, k = frac * a_c, 2.0 * np.pi
    q0 = np.linspace(0.002, 0.19, 60)                      # k q < 1.2: arcsin on its principal branch
    x = q0 - frac * np.sin(k * q0) / k
    rho, v = (np.asarray(f) for f in exact(a, x, 1.0, a_c, eds, 1.0))
    a_c_use = a_c if a_c_wrong is None else a_c_wrong
    A = a / a_c_use
    q = np.arcsin(-v * k * a_c_use / a ** 1.5) / k
    return float(np.abs(q - A * np.sin(k * q) / k - x).max()), float(np.abs(rho * (1 - A * np.cos(k * q)) - 1).max())


@pytest.mark.parametrize("frac", FRACS)
def test_zeldovich_roundtrip(frac):
    """q recovered from exact()'s velocity maps forward onto x, and rho (1 - A cos kq) = 1, at a / a_c = frac up to
    0.999. Bound 16 u / (1 - A): the map's derivative 1 - A cos kq >= 1 - A amplifies round-off in q."""
    dx, drho = zel_roundtrip(frac)
    bound = 16 * U / (1 - frac)
    check(f"Zel'dovich a/a_c={frac} max |forward map(q) - x|", dx, bound)
    check(f"Zel'dovich a/a_c={frac} max |rho (1 - A cos kq) - 1|", drho, bound)


def mass_misses(rho_a_scale=1.0):
    """{a/a_c: max |int_0^x rho - q|}, rho from exact() at rho_a_scale times a (1: the same a as q)."""
    exact, eds, a_c = zel()
    k, xg, wg = 2.0 * np.pi, *np.polynomial.legendre.leggauss(200)
    out = {}
    for frac in (0.2, 0.5, 0.9):
        a = frac * a_c
        q0 = np.array([0.05, 0.1, 0.19])
        X = q0 - frac * np.sin(k * q0) / k
        nodes = (X[:, None] * (xg[None, :] + 1) / 2).ravel()
        rho = np.asarray(exact(a * rho_a_scale, nodes, 1.0, a_c, eds, 1.0)[0]).reshape(len(X), -1)
        mass = (rho * wg[None, :]).sum(1) * X / 2
        v = np.asarray(exact(a, X, 1.0, a_c, eds, 1.0)[1])
        q = np.arcsin(-v * k * a_c / a ** 1.5) / k
        out[frac] = float(np.abs(mass - q).max())
    return out


def test_zeldovich_mass_coordinate():
    """int_0^x rho dx (200-node Gauss-Legendre) equals the Lagrangian q of x, a / a_c = 0.2, 0.5, 0.9 (bound
    1e-12: rho analytic, nearest singularity at distance >= (1 - A) / k from [0, x])."""
    for frac, m in mass_misses().items():
        check(f"Zel'dovich a/a_c={frac} max |int_0^x rho - q|", m, 1e-12)


def test_zeldovich_mass_coordinate_teeth():
    """Teeth: rho taken 1 % later in a than the velocity misses the mass coordinate at every a."""
    m = mass_misses(rho_a_scale=1.01)
    print(f"MEASURED teeth Zel'dovich mass coordinate, rho at 1.01 a: {m}")
    assert min(m.values()) > 1e-12


def growth(Om, OL, a_i, a_targets):
    """D(a) / D(a_i) at a_targets from D'' = 1.5 Om a D in tau, da/dtau = a^3 E(a) (H0 = 1), growing mode at
    a = 1e-4 (EdS limit there); solve_ivp DOP853, rtol 1e-13."""
    def f(t, y):
        a, D, Dp = y
        return [a ** 3 * np.sqrt(Om / a ** 3 + OL), Dp, 1.5 * Om * a * D]
    a0 = 1e-4
    y0 = [a0, a0, a0 ** 1.5 * np.sqrt(Om + OL * a0 ** 3)]
    out = {}
    for at in sorted(set([a_i, *a_targets])):
        ev = lambda t, y, at=at: y[0] - at                # noqa: E731
        ev.terminal = True
        s = solve_ivp(f, (0.0, 1e9), y0, method="DOP853", rtol=1e-13, atol=1e-30, events=ev)
        out[at] = s.y_events[0][0][1]
    return {at: out[at] / out[a_i] for at in a_targets}


def test_zeldovich_growth_and_caustic():
    """exact()'s amplitude A(a) = 1 - 1 / rho(x = 0) equals a_i/a_c D(a)/D(a_i) from the growth ODE (EdS, a_i =
    0.3), and the ODE's caustic (A = 1) is a_c = 1.5 (zeldovich.py Params): bound 1e-10 (rtol 1e-13 over ~10^4
    e-folds of tau steps)."""
    exact, eds, a_c = zel()
    a_i, fr = 0.3, np.array([0.4, 0.6, 0.9, 1.2, 1.45])
    G = growth(1.0, 0.0, a_i, list(fr))
    A_exact = np.array([1 - 1 / float(np.asarray(exact(a, np.zeros(1), 1.0, a_c, eds, 1.0)[0])[0]) for a in fr])
    A_ode = a_i / a_c * np.array([G[a] for a in fr])
    check("Zel'dovich max |A_exact - A_ode|", float(np.abs(A_exact - A_ode).max()), 1e-10)
    # caustic: A_ode linear in a in EdS; root of a_i/a_c D(a)/D(a_i) = 1 by the secant through 1.2, 1.45
    s = (A_ode[-1] - A_ode[-2]) / (fr[-1] - fr[-2])
    check("Zel'dovich |a_caustic(ODE) - 1.5|", abs(fr[-1] + (1 - A_ode[-1]) / s - a_c), 1e-9)


def test_zeldovich_teeth():
    """Teeth: LCDM (0.3, 0.7) growth misses a/a_c by far more than 1e-10; exact() refuses LCDM; a wrong a_c in
    the inversion misses the round-trip."""
    from luna.cosmology import Cosmology
    exact = zel()[0]
    G = growth(0.3, 0.7, 0.3, [0.9, 1.45])
    miss = max(abs(0.3 * G[a] - a) for a in (0.9, 1.45))
    print(f"MEASURED teeth Zel'dovich LCDM growth miss {miss:.3e}")
    assert miss > 1e-10
    with pytest.raises(ValueError, match="EdS"):
        exact(0.5, np.zeros(1), 1.0, 1.5, Cosmology(Omega_m=0.3, Omega_L=0.7), 1.0)
    dx = zel_roundtrip(0.5, a_c_wrong=1.5 * 0.99)[0]
    print(f"MEASURED teeth Zel'dovich wrong a_c round-trip {dx:.3e}")
    assert dx > 16 * U / 0.5
