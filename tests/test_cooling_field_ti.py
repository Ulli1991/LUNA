"""Field (1965) thermal instability: growth and decay rates of linear modes about a Koyama-Inutsuka equilibrium
against the exact dispersion relation.

Net loss per unit mass L(rho, T) = (n_e n_H Lambda(T) - n_H Gamma) / rho, LUNA's rate (`src/luna/sources/cooling.py`,
`_thermo`), with constant heating Gamma: at an equilibrium T L_T = alpha Gamma n_H / rho and rho L_rho = Gamma n_H / rho,
alpha = dln Lambda / dln T of the production table interpolant (finite difference of `_cooling_lambda` inside one
log-log bin, where it is an exact power law). Field's cubic (Field 1965 ApJ 142, 531, Sect. II; quoted with his
k_T, k_rho, k_K as eq. 3 of Klimchuk 2026, arXiv:2602.10902; Field's own equation number not read), no conduction, in
s = n tau, q = k c tau, tau = u / (n_H Gamma) the heating time, c the adiabatic sound speed:
    s^3 + alpha s^2 + q^2 s + q^2 (alpha - 1) / gamma = 0,
derived again here from the linearised Euler + energy equations (p1 = -n^2 rho1 / k^2). Limits: q -> inf, the
condensation root -> (1 - alpha) / gamma (isobaric, Field's eq. 31 per Klimchuk 2026; unstable iff (dL/dT)_p < 0, i.e.
alpha < 1); q -> 0, a root -> -alpha (isochoric, c k_T tau; KI's Lambda rises with T, so isochorically stable) and the
condensation root -> q sqrt((1 - alpha) / (gamma alpha)). Checked in `test_dispersion_limits`.

Runs: the exact real eigenmode (rho, v, p) of the root, cell-averaged, amplitude 1e-4, through the production staged
loop (cooling stage before and after the hydro, `src/luna/setup/evolve.py:38`) on a periodic 3D grid with one cell
across (plane wave along x; at an angle in x-y on the GPU). Measured: the fitted d ln|rho_k| / dt over two e-folds
against the root. Cooling is operator-split with forward-Euler sub-steps of dt_frac t_cool, t_cool = T / |dT/dt|, which
is infinite at equilibrium: each half-step is ONE forward-Euler step of h = dt / 2, so the rate carries an O(dt / tau)
splitting error, shown by a CFL study at small q. Bounds, jobs, hardware: docs/TEST_PLAN.md (Field 1965)."""
import numpy as np
import pytest

from tests._helpers import accuracy_cfg, check, evolve_staged, gpu_only

GAMMA, MU, X_H, NE = 5.0 / 3.0, 0.6, 0.76, 1.0       # KI-faithful n_e = n_H; X_H = units.HYDROGEN_MASSFRAC
UNITS = dict(UnitLength_in_cm=3.085678e18, UnitMass_in_g=1.989e33, UnitVelocity_in_cm_per_s=1.0e5)   # pc, Msun, km/s
EPS = 1.0e-4                                          # max of |rho1/rho|, |p1/p|, |T1/T| at t = 0
FOLDS = 2.0                                           # fit window in e-folds
NT = 8                                                # fit samples after t = 0
T_UNSTABLE, T_STABLE = 1000.0, 80.0                   # K: alpha ~ 0.59 (isobarically unstable), ~ 1.65 (stable)


def roots(alpha, q):
    """The three roots s of Field's cubic s^3 + alpha s^2 + q^2 s + q^2 (alpha - 1) / gamma."""
    return np.roots([1.0, alpha, q * q, q * q * (alpha - 1.0) / GAMMA])


def mode_rate(alpha, q, mode):
    """s of the condensation mode (largest real root) or of the isochoric mode (real root nearest -alpha)."""
    r = roots(alpha, q)
    real = r[np.abs(r.imag) < 1e-9 * np.abs(r).max()].real
    if mode == "condensation":
        return real.max()
    s = real[np.argmin(np.abs(real + alpha))]
    assert abs(s + alpha) < 0.2 * alpha, (alpha, q, r)   # else the isochoric root is in a complex pair
    return s


def equilibrium(T_target):
    """(T0, alpha, n_H0 [cm^-3], tau [s]) at the midpoint of the production KI table bin holding T_target."""
    import jax.numpy as jnp

    from luna.sources.cooling import _cooling_lambda, koyama_inutsuka_2002
    from luna.units import BOLTZMANN_CGS
    tab = koyama_inutsuka_2002()
    i = int(np.searchsorted(tab.log_T, np.log10(T_target))) - 1
    T0 = 10.0 ** (0.5 * (tab.log_T[i] + tab.log_T[i + 1]))
    lt, ll = jnp.asarray(tab.log_T), jnp.asarray(tab.log_Lambda)
    lam = lambda T: float(_cooling_lambda(jnp.float64(T), lt, ll))   # noqa: E731
    f = 1.0 + 1e-3                                    # stays in the bin: half-width 0.0128 dex
    alpha = np.log(lam(T0 * f) / lam(T0 / f)) / (2.0 * np.log(f))
    n_H = tab.heating / (NE * lam(T0))
    tau = BOLTZMANN_CGS * T0 / ((GAMMA - 1.0) * MU * X_H * tab.heating)
    return T0, alpha, n_H, tau


def setup(T_target, q, n, mode="condensation", angle=(1, 0), limiter=0, cfl=0.8):
    """(cfg, U0, kvec, rho0, n_exact [1/code time], tau [code], dt/tau nominal) for one eigenmode run."""
    import jax

    from luna.units import BOLTZMANN_CGS, PROTON_MASS_CGS, UnitSystem
    units = UnitSystem(**UNITS)
    T0, alpha, n_H, tau_s = equilibrium(T_target)
    rho_cgs = n_H * PROTON_MASS_CGS / X_H
    rho0 = rho_cgs / units.UnitDensity_in_cgs
    p0 = rho_cgs * BOLTZMANN_CGS * T0 / (MU * PROTON_MASS_CGS) / units.UnitPressure_in_cgs
    c = np.sqrt(GAMMA * p0 / rho0)
    tau = tau_s / units.UnitTime_in_s
    s = mode_rate(alpha, q, mode)
    k, nrate = q / (c * tau), s / tau
    mx, my = angle
    L = 2.0 * np.pi * np.hypot(mx, my) / k
    ny = n if my else 1
    kvec = np.array([mx, my]) * 2.0 * np.pi / L
    ndev = len(jax.devices())
    cfg = accuracy_cfg(test_problem="Sedov-Blast-3D", SPATIAL_DIMENSION=3, BOUNDARY=0, gamma=GAMMA, LIMITER=limiter,
                       CFL=cfl, Nx=n, Ny=ny, Nz=1, Lx=L, Ly=L, Lz=L, MESH_SHAPE=(ndev, 1, 1), units=units,
                       cooling_enabled=True, cooling_table="koyama_inutsuka", cooling_T_floor=1.0,
                       cooling_X_H=X_H, cooling_ne_per_nH=NE, cooling_mu=MU)
    # cell averages of the plane wave: sinc factors per axis
    dx, dy = L / n, L / ny
    x = (np.arange(n) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    th = (kvec[0] * x[:, None] + kvec[1] * y[None, :])[:, :, None]
    w = np.sinc(kvec[0] * dx / (2 * np.pi)) * np.sinc(kvec[1] * dy / (2 * np.pi))
    eps = EPS / max(1.0, GAMMA * s * s / (q * q), abs(1.0 + GAMMA * s * s / (q * q)))
    rho = rho0 * (1.0 + eps * w * np.cos(th))
    vmag = -(nrate * eps / k) * w * np.sin(th)
    p = p0 - nrate ** 2 * rho0 * eps / k ** 2 * w * np.cos(th)
    mom = [rho * vmag * kvec[0] / k, rho * vmag * kvec[1] / k, np.zeros_like(rho)]
    E = p / (GAMMA - 1.0) + 0.5 * rho * vmag ** 2
    U0 = np.stack([rho, *mom, E])
    dt_tau = cfl / (c * (1.0 / dx + 1.0 / dy + 1.0 / L)) / tau
    return cfg, U0, (kvec, x, y), rho0, nrate, tau, dt_tau


def staged_jit(cfg, U0, times, teeth=False):
    """(states at `times`, steps): production's step order for a cooling-only run (cooling(dt/2), hydro(dt),
    cooling(dt/2), `src/luna/setup/evolve.py:36`) with the assembled hydro and the stage's apply_cooling, in one
    jitted loop (evolve_staged dispatches the cooling scan eagerly: ~0.4 s a step); checked against evolve_staged."""
    import jax
    import jax.numpy as jnp

    from luna.setup.assemble import assemble
    from luna.setup.ics import Initial
    from luna.setup.mesh import build_mesh, decomposition
    from luna.setup.run_kind import run_kind
    from luna.sources.cooling import apply_cooling
    kind = run_kind(cfg)
    layout = decomposition(cfg, kind)
    mesh, _ = build_mesh(cfg, layout, kind)
    phys = assemble(cfg, mesh, kind, layout, Initial(cfg=cfg, U=np.asarray(U0)), is_main=False)
    assert [s.name for s in phys.stages] == ["cooling"] and not phys.projections, phys.stages
    hyd, solver = phys.hydro, phys.cooling.solver

    @jax.jit
    def advance(U, t, n, t1):
        def body(c):
            U, t, n = c
            dt = jnp.minimum(hyd.compute_dt(U), t1 - t)
            U = U if teeth else apply_cooling(U, 0.5 * dt, cfg, solver)     # teeth: drop the first half
            U = apply_cooling(hyd.step(U, dt), 0.5 * dt, cfg, solver)
            return U, jnp.where(dt > 0, t + dt, jnp.inf), n + 1
        return jax.lax.while_loop(lambda c: c[1] < t1 * (1 - 1e-13), body, (U, t, n))

    U = jax.device_put(jnp.asarray(U0, jnp.float64), hyd.sharding)
    t, n, out = jnp.float64(0.0), jnp.int32(0), []
    for t1 in times:
        U, t, n = advance(U, t, n, jnp.float64(t1))
        Uh = np.asarray(U)
        assert np.isfinite(float(t)) and np.isfinite(Uh).all(), (float(t), t1)
        check(f"N={cfg.Nx} t={t1:.4g} |dM|/M", abs(Uh[0].sum() / U0[0].sum() - 1.0), 1e-13)
        out.append(Uh)
    return out, int(n)


def measure(T_target, q, n, mode="condensation", teeth=False, **kw):
    """(s_exact, s_measured, max fit residual of ln|rho_k|, dt / tau) of one eigenmode run."""
    cfg, U0, (kvec, x, y), rho0, nrate, tau, dt_tau = setup(T_target, q, n, mode, **kw)
    ph = np.exp(-1j * (kvec[0] * x[:, None] + kvec[1] * y[None, :]))[:, :, None]

    def pres(U):
        return (GAMMA - 1.0) * (U[4] - 0.5 * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / U[0])

    p0 = pres(U0).mean()

    def amp(U):                                       # isochoric mode: rho1 / rho ~ 1e-3 p1 / p, so p is measured
        f = pres(U) / p0 - 1.0 if mode == "isochoric" else U[0] / rho0 - 1.0
        return abs(np.sum(f * ph)) * 2.0 / U[0].size

    ts = np.arange(NT + 1) * FOLDS / abs(nrate) / NT
    Us, steps = staged_jit(cfg, U0, ts[1:], teeth)
    a = np.log(np.array([amp(U0)] + [amp(U) for U in Us]))
    fit = np.polyfit(ts, a, 1)
    res = np.abs(a - np.polyval(fit, ts)).max()
    s_ex, s_m = nrate * tau, fit[0] * tau
    print(f"FIELD T0~{T_target:g}K {mode} q={q:g} N={n} angle={kw.get('angle', (1, 0))} lim={kw.get('limiter', 0)} "
          f"cfl={kw.get('cfl', 0.8)} teeth={teeth} dt/tau={dt_tau:.3e} s_exact={s_ex:.6f} s_meas={s_m:.6f} "
          f"rel={s_m / s_ex - 1.0:+.3e} fitres={res:.1e} steps={steps}")
    return s_ex, s_m, res, dt_tau


@gpu_only
def test_jit_loop_matches_staged():
    """The jitted loop reproduces the production staged loop (evolve_staged) over the same times."""
    cfg, U0, *_ , nrate, tau, _ = setup(T_UNSTABLE, 10.0, 16)
    ts = [0.1 * tau]
    a = staged_jit(cfg, U0, ts)[0]
    b = evolve_staged(cfg, U0, ts)
    for x, y in zip(a, b):
        r, e = np.abs(y[0]).max(), np.abs(y[-1]).max()
        scale = np.array([r, *[np.sqrt(r * e)] * 3, e])        # momenta against sqrt(rho E), as check_conserved
        check("max_var |jit - staged| / scale_var", (np.abs(x - y).max(axis=(1, 2, 3)) / scale).max(), 1e-13)


@pytest.mark.parametrize("ndim", [1, 2])
def test_cooling_in_1d_2d(ndim):
    """Cooling a 1D/2D state equals, bitwise, cooling the same state in the 3D layout with zero extra momenta."""
    from dataclasses import replace

    import jax.numpy as jnp

    from luna.sources.cooling import ExplicitCoolingSolver, apply_cooling, koyama_inutsuka_2002
    from luna.template_config import Config
    from luna.units import UnitSystem
    units = UnitSystem(**UNITS)
    solver = ExplicitCoolingSolver(table=koyama_inutsuka_2002(), T_floor=1.0)
    rng = np.random.default_rng(1965)
    shape = (4,) * ndim
    rho = rng.uniform(0.05, 0.5, shape)
    mom = rng.uniform(-0.1, 0.1, (ndim,) + shape)
    U = np.concatenate([rho[None], mom, (rng.uniform(0.5, 2.0, shape) * rho + 0.5 * (mom ** 2).sum(0) / rho)[None]])
    U3 = np.concatenate([U[:-1], np.zeros((3 - ndim,) + shape), U[-1:]])
    cool = lambda U_, d: np.asarray(apply_cooling(   # noqa: E731
        jnp.asarray(U_), 1e-3, replace(Config(), units=units, gamma=GAMMA, SPATIAL_DIMENSION=d), solver))
    V, V3 = cool(U, ndim), cool(U3, 3)
    assert not np.array_equal(V[-1], U[-1])                       # it cools
    np.testing.assert_array_equal(V[:-1], U[:-1])                 # rho and momenta untouched
    np.testing.assert_array_equal(V[-1], V3[-1])                  # the 3D result, bitwise
    print(f"{ndim}D: max |dE/E| {np.abs(V[-1] / U[-1] - 1).max():.3e}, equal to the 3D embedding bitwise")


def test_cubic_vs_linearised():
    """Field's cubic (alpha, tau) equals the eigenvalues of the dimensional linearised 1D equations built from LUNA's
    own dT/dt(rho, T) (central differences about the equilibrium), q = 0.3 .. 100, unstable and stable."""
    from dataclasses import replace

    import jax.numpy as jnp

    from luna.sources.cooling import ExplicitCoolingSolver, _thermo, koyama_inutsuka_2002
    from luna.template_config import Config
    from luna.units import BOLTZMANN_CGS, PROTON_MASS_CGS, UnitSystem
    units = UnitSystem(**UNITS)
    cfg = replace(Config(), units=units, gamma=GAMMA, SPATIAL_DIMENSION=1)
    solver = ExplicitCoolingSolver(table=koyama_inutsuka_2002(), mu=MU, T_floor=1.0, X_H=X_H, ne_per_nH=NE)
    R = BOLTZMANN_CGS / (MU * PROTON_MASS_CGS)

    def F(rho_cgs, T):                                # LUNA's dT/dt [K/s] at (rho, T), cooling + heating
        U = jnp.asarray([[rho_cgs / units.UnitDensity_in_cgs], [0.0], [1.0]])
        return float(_thermo(U, 0.0, cfg, solver)[1](jnp.float64(T))[0])

    for T_target in (T_UNSTABLE, T_STABLE):
        T0, alpha, n_H, tau = equilibrium(T_target)
        rho0 = n_H * PROTON_MASS_CGS / X_H
        h = 1e-6
        F_T = (F(rho0, T0 * (1 + h)) - F(rho0, T0 * (1 - h))) / (2 * h * T0)
        F_r = (F(rho0 * (1 + h), T0) - F(rho0 * (1 - h), T0)) / (2 * h * rho0)
        c = np.sqrt(GAMMA * R * T0)
        for q in (0.3, 1.0, 10.0, 100.0):
            k = q / (c * tau)
            # (rho1, v1, T1) e^{ikx + nt}: continuity, momentum with p = rho R T, DT/Dt = F - (gamma-1) T div v
            M = np.array([[0.0, -1j * k * rho0, 0.0],
                          [-1j * k * R * T0 / rho0, 0.0, -1j * k * R],
                          [F_r, -1j * k * (GAMMA - 1.0) * T0, F_T]])
            ev = np.linalg.eigvals(M) * tau
            r = roots(alpha, q)
            err = max(np.abs(ev - x).min() for x in r) / np.abs(r).max()
            check(f"T0={T0:.0f} q={q:g}: max |eig - root| / max|root|", err, 6e-9)


def test_dispersion_limits():
    """The cubic's roots reach the isobaric, isochoric and small-q condensation limits; the table alpha is KI's."""
    T0, alpha, n_H, tau = equilibrium(T_UNSTABLE)
    a_ki = 0.5 + 92.0 / T0                               # KI low-T term dominates at 1e3 K (Lya term < 1e-18)
    print(f"T0={T0:.2f} K alpha_table={alpha:.5f} alpha_KI={a_ki:.5f} n_H={n_H:.4f} tau={tau:.4e} s")
    check("|alpha_table / alpha_KI - 1| at T0", abs(alpha / a_ki - 1.0), 1e-4)
    for a in (alpha, equilibrium(T_STABLE)[1], 0.3):
        check(f"alpha={a:.3f} q=1e4: |s / ((1-a)/g) - 1|", abs(mode_rate(a, 1e4, "condensation")
                                                           / ((1.0 - a) / GAMMA) - 1.0), 1e-7)
        check(f"alpha={a:.3f} q=1e-5: |s_iso / -a - 1|", abs(mode_rate(a, 1e-5, "isochoric") / -a - 1.0), 1e-7)
    small = mode_rate(alpha, 1e-6, "condensation") / 1e-6
    check("q=1e-6: |s / (q sqrt((1-a)/(g a))) - 1|", abs(small / np.sqrt((1 - alpha) / (GAMMA * alpha)) - 1.0), 1e-5)
    for q in (0.1, 1.0, 10.0):                           # every root satisfies the cubic
        r = roots(alpha, q)
        check(f"q={q}: max |cubic(root)|", np.abs(np.polyval([1.0, alpha, q * q, q * q * (alpha - 1) / GAMMA],
                                                                r)).max(), 1e-12)


def test_condensation_rate_cpu():
    """Unstable KI equilibrium, q = 10, N = 16, minmod: the growth rate against the cubic."""
    s_ex, s_m, res, _ = measure(T_UNSTABLE, 10.0, 16)
    check("q=10 N=16 |s/s_exact - 1|", abs(s_m / s_ex - 1.0), 4.3e-3)



def order(xs, errs):
    """Fitted order p of errs ~ xs^-p (xs: N, or 1 / CFL)."""
    return -np.polyfit(np.log(xs), np.log(errs), 1)[0]


def ladder(label, xs, runs, bounds, p_min):
    """Bound every |s / s_exact - 1| of runs() over xs and the fitted order."""
    errs = []
    for x, b in zip(xs, bounds):
        s_ex, s_m, _, _ = runs(x)
        errs.append(abs(s_m / s_ex - 1.0))
        check(f"{label} x={x} |s/s_exact - 1|", errs[-1], b)
    if p_min is not None:                             # None: the error changes sign, no order to fit
        check(f"{label} order", order(xs, errs), p_min, lower=True)
    return errs


NS = (16, 32, 64, 128, 256)


@pytest.mark.parametrize("q,bounds,p_min", [
    (0.3, (3.6e-2, 2.5e-2, 1.45e-2, 7.4e-3, 3.8e-3), 0.65),
    (1.0, (2.8e-2, 1.9e-2, 1.0e-2, 5.3e-3, 2.7e-3), 0.75),
    (10.0, (4.3e-3, 2.6e-3, 1.4e-3, 7.3e-4, 3.7e-4), 0.75),
    (100.0, (4.3e-4, 2.6e-4, 1.45e-4, 7.4e-5, 3.8e-5), 0.75)])
@gpu_only
def test_convergence_gpu(q, bounds, p_min):
    """Unstable equilibrium, minmod, CFL 0.8: the growth-rate error over N at q = k c tau."""
    ladder(f"unstable q={q:g}", NS, lambda n: measure(T_UNSTABLE, q, n), bounds, p_min)


@gpu_only
def test_stable_decay_gpu():
    """Stable cold equilibrium (alpha ~ 1.66), q = 10: the decay rate over N."""
    ladder("stable q=10", NS[:4], lambda n: measure(T_STABLE, 10.0, n), (1.35e-2, 7.8e-3, 4.1e-3, 2.1e-3), 0.75)


@gpu_only
def test_dt_study_gpu():
    """N fixed, CFL halved: the rate error is the O(dt / tau) cooling split (q = 0.3, 10; isochoric q = 0.03)."""
    cfls = (0.8, 0.4, 0.2, 0.1, 0.05)
    inv = [1.0 / c for c in cfls]
    ladder("dt q=0.3 N=64", inv, lambda x: measure(T_UNSTABLE, 0.3, 64, cfl=1.0 / x),
           (1.42e-2, 6.8e-3, 3.15e-3, 1.3e-3, 3.8e-4), 1.15)
    ladder("dt q=10 N=64", inv[:4], lambda x: measure(T_UNSTABLE, 10.0, 64, cfl=1.0 / x),
           (1.41e-3, 6.2e-4, 2.2e-4, 1.7e-5), 1.95)
    # dt below the sample spacing (0.42 tau) at every CFL, else the output times cap it
    small = (0.1, 0.05, 0.025, 0.0125)
    ladder("dt isochoric q=0.03 N=64", [1.0 / c for c in small],
           lambda x: measure(T_UNSTABLE, 0.03, 64, mode="isochoric", cfl=1.0 / x),
           (5.4e-2, 2.9e-2, 1.5e-2, 7.6e-3), 0.85)


# signed s / s_exact - 1 at N = 16 .. 128, A100 FP64, job 31663849: negative and falling, then it changes sign between
# 32 and 64 (what sets the positive part at fixed dt is not isolated here), so no order is fitted. Each N is held to
# a two-sided band of 0.3 |measured| about its measured value (the 1.3x rule, signed), the N = 64 crossing included
SPATIAL_MEASURED = (-2.604e-3, -4.411e-4, 1.298e-5, 1.148e-4)


@gpu_only
def test_spatial_ladder_gpu():
    """Regression only: dt fixed (CFL proportional to N), q = 10, the signed rate error per N in a +-30 % band."""
    got = []
    for n in NS[:4]:
        s_ex, s_m, _, _ = measure(T_UNSTABLE, 10.0, n, cfl=0.025 * n / 16)
        got.append(s_m / s_ex - 1.0)
        print(f"MEASURED spatial q=10 dt fixed N={n} s/s_exact - 1 = {got[-1]:+.6e}")
    for n, g, want in zip(NS[:4], got, SPATIAL_MEASURED):
        check(f"spatial q=10 dt fixed N={n} s/s_exact - 1", g, want - 0.3 * abs(want), lower=True)
        check(f"spatial q=10 dt fixed N={n} s/s_exact - 1", g, want + 0.3 * abs(want))


@gpu_only
def test_angle_gpu():
    """The plane wave at angle atan(2) in x-y, q = 10, against the same root."""
    ladder("angle (1,2) q=10", NS[1:4], lambda n: measure(T_UNSTABLE, 10.0, n, angle=(1, 2)),
           (2.1e-3, 1.35e-3, 7.6e-4), 0.6)


@pytest.mark.parametrize("limiter,bounds", [(1, (5.0e-3, 2.8e-3, 1.45e-3)), (2, (4.9e-3, 2.8e-3, 1.45e-3))])
@gpu_only
def test_limiters_gpu(limiter, bounds):
    """MC and van Leer at q = 10."""
    ladder(f"limiter {limiter} q=10", NS[:3], lambda n: measure(T_UNSTABLE, 10.0, n, limiter=limiter), bounds, 0.75)


@gpu_only
def test_teeth_gpu():
    """A run that drops the first cooling half-step (half the cooling) must fail the q = 10 N = 64 bound."""
    s_ex, s_m, _, _ = measure(T_UNSTABLE, 10.0, 64, teeth=True)
    check("teeth q=10 N=64 |s/s_exact - 1|", abs(s_m / s_ex - 1.0), 1.4e-3, lower=True)
