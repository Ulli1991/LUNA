"""Regression only: an n = 1 polytrope (gamma = 2) under LUNA's multigrid self-gravity (isolated BCs), started at rest.

It does not stay at rest: the scheme is not well balanced. At 32^3 (CPU-8, FP64, RAMSES dt rule) over 10 sound
crossings: with gravity_energy kick the central density drifts 23 % (spurious Mach 0.05); with flux (what 'auto'
resolves to now) 5 % (spurious Mach 0.11). Bounds ~1.3x LUNA's own values per mode (no target but zero): they hold
the defect at its size, they do not validate. On 8aba9c1 the unpinned test ran flux against the kick bounds and failed
(Mach 0.112 > 0.065, job 31654893); each mode is now pinned. The kick 10-crossing drift bound followed the code:
0.228 (917ae22, measured 0.175) -> 0.264 (9280e3a, 0.203, dt rule) -> 0.228 (bdc075b, 0.203 kept) -> 0.30 (b84f577,
0.233, RAMSES dt rule): loosened as the drift grew with dt-rule changes (traded for a lower spurious Mach), each time
after a re-measurement. Fixing the drift is a solver change."""
import numpy as np
import pytest

from tests._helpers import RunAborted, _build_cfg_for_test, _where, check, gpu_only

R, RHO_C, G, GAMMA = 0.35, 1.0, 1.0, 2.0
K = 2.0 * G * R * R / np.pi            # n = 1: R = pi r_n, r_n^2 = K / (2 pi G)
T_SC = R / np.sqrt(GAMMA * K * RHO_C)  # central sound-crossing time of R, sqrt(pi)/2
AMB = 1e-3                             # ambient density / rho_c, at the central sound speed (a cold one drains to vacuum)



def radius(n):
    """Cell-centre distance from the box centre on an n^3 unit box."""
    x = (np.arange(n) + 0.5) / n - 0.5
    return np.sqrt(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2)


def polytrope_U(n, k_fac=1.0, cold=False):
    """Conserved U of the n = 1 polytrope rho_c sinc(r/R), P = k_fac K rho^2 inside R; `cold`: P = K rho^2 outside too."""
    r = radius(n)
    inside = r < R
    rho = np.where(inside, RHO_C * np.sinc(r / R), AMB * RHO_C)
    P = np.where(inside, k_fac * K * rho ** 2, K * rho ** 2 if cold else AMB * K * RHO_C ** 2)
    z = np.zeros_like(rho)
    return np.stack([rho, z, z, z, P / (GAMMA - 1.0)])


def diag(U, U0):
    """(max Mach, mass-weighted rms Mach) inside 0.8 R, central-density drift, profile L1 / rho_c inside R, min P."""
    r = radius(U.shape[1])
    rho, v = U[0], U[1:4] / U[0]
    P = (GAMMA - 1.0) * (U[4] - 0.5 * rho * (v ** 2).sum(0))
    mach = np.sqrt((v ** 2).sum(0)) / np.sqrt(GAMMA * P / rho)
    core, star = r < 0.8 * R, r < R
    c = r <= r.min() * (1 + 1e-9)
    return dict(mach_max=float(mach[core].max()),
                mach_rms=float(np.sqrt((rho * mach ** 2)[core].sum() / rho[core].sum())),
                drho_c=float(rho[c].mean() / U0[0][c].mean() - 1.0),
                l1=float(np.abs(rho - U0[0])[star].mean() / RHO_C),
                dmass=float(rho.sum() / U0[0].sum() - 1.0),
                p_min=float(P.min()) if np.isfinite(U).all() else -np.inf)


def run(n, times, k_fac=1.0, cold=False, **kw):
    """Diagnostics of the polytrope on n^3 at each of `times` (in T_SC), production step via amuse_luna.stepper."""
    import jax.numpy as jnp
    from amuse_luna.stepper import build
    # the registered IC is a placeholder: U0 replaces it
    opts = dict(test_problem="Soundwave-3D", SPATIAL_DIMENSION=3, Nx=n, Ny=n, Nz=n,
                Lx=1.0, Ly=1.0, Lz=1.0, BOUNDARY=4, POISSON_BOUNDARY=3, SELF_GRAVITY=True,
                GRAVITATIONAL_G=G, gamma=GAMMA, ORDER=1, LIMITER=0, riemann_solver="hllc",
                enable_x64=True, write_snapshots=False, boundary_mass_budget=True,
                t_end=float(max(times)) * T_SC)
    cfg = _build_cfg_for_test(**{**opts, **kw})
    U0 = polytrope_U(n, k_fac, cold)
    st = build(cfg, U0=jnp.asarray(U0))
    out = []
    for t in times:
        st.evolve_to(t * T_SC)
        reached = st.t >= t * T_SC * (1 - 1e-12) and bool(np.isfinite(np.asarray(st.U)).all())
        print(f"RUN [{_where()}] polytrope N={n} t={st.t / T_SC:.6f} t_end={t} T_sc steps={st.step} reached={reached}")
        if not reached:
            raise RunAborted(f"polytrope n={n} stopped at t={st.t / T_SC:.6f} T_sc of {t} (step {st.step})")
        out.append(diag(np.asarray(st.U), U0))
        # diode faces: box mass - initial == the accumulated face inflow - outflow (off the step, bitwise)
        acc = np.asarray(st.sources["boundary_mass"])
        out[-1]["budget"] = abs(out[-1]["dmass"] * U0[0].sum() / n ** 3 - (acc[0] - acc[1]).sum()) / (U0[0].sum() / n ** 3)
        print(f"MEASURED wb n={n} t={t} T_sc step={st.step} {out[-1]}", flush=True)
    return out


def summary(out):
    """Asserted: max-over-time max Mach, |rho_c drift| after one crossing and its max, final rms Mach and profile L1."""
    return dict(mach_max=max(d["mach_max"] for d in out),
                mach_rms=out[-1]["mach_rms"],
                drho_c1=abs(out[0]["drho_c"]),
                drho_c=max(abs(d["drho_c"]) for d in out),
                l1=out[-1]["l1"])


TIMES = (1, 2, 5, 10)
BUDGET_TOL = 1e-13   # measured 1.9e-16, 2.6e-16, 2.0e-16 at 32, 64, 128 (A100)
# ~1.3x measured on CPU-8, RAMSES dt rule, CFL 0.8, per gravity_energy (job 31655329, both; flux also 31654893):
# kick 0.0502, 0.0139, 0.0470, 0.233, 0.0745 (the bounds since b84f577); flux, which 'auto' resolves to here since the
# flux mode landed, 0.112, 0.0251, 0.0526, 0.0526, 0.00891: a quarter of the drift, twice the spurious Mach. Under the
# retired per-axis 0.4 rule (kick): 0.0773, 0.0154, 0.0789, 0.203, 0.0694 (docs/TEST_PLAN.md).
BOUNDS = {(32, "kick"): dict(mach_max=0.065, mach_rms=0.018, drho_c1=0.061, drho_c=0.30, l1=0.097),
          (32, "flux"): dict(mach_max=0.146, mach_rms=0.0326, drho_c1=0.068, drho_c=0.068, l1=0.0116)}


def _assert(n, out, bounds):
    """Check every summary quantity of a run against its bound, and the face mass budget to round-off."""
    for k, v in summary(out).items():
        check(f"wb n={n} {k}", v, bounds[k])
    check(f"wb n={n} max |dM - (in - out)| / M0", max(d["budget"] for d in out), BUDGET_TOL)


@pytest.mark.parametrize("energy", ["kick", "flux"])
def test_polytrope_drift_from_rest(energy, n=32):
    """Regression only: 32^3 over 10 crossings, spurious Mach and central drift, each gravity energy mode."""
    _assert(n, run(n, TIMES, gravity_energy=energy), BOUNDS[n, energy])


def test_unbalanced_ic_is_caught():
    """Teeth: K 10% above balance breaks the flux one-crossing drift bound (measured 0.208 against 0.068)."""
    drift = abs(run(32, TIMES[:1], k_fac=1.1, gravity_energy="flux")[0]["drho_c"])
    check("wb n=32 K x1.1 drho_c1", drift, BOUNDS[32, "flux"]["drho_c1"], lower=True)


# kick, measured on MI300A, 2 GPUs, under the retired per-axis 0.4 dt rule, not re-measured since (stale):
# 64^3 0.0505, 0.0104, 0.0127, 0.0917, 0.0328; 128^3 0.0372, 0.0122, 0.00326, 0.0291, 0.0124
GPU_BOUNDS = {64: dict(mach_max=0.062, mach_rms=0.0133, drho_c1=0.016, drho_c=0.111, l1=0.041),
              128: dict(mach_max=0.048, mach_rms=0.0152, drho_c1=0.0041, drho_c=0.038, l1=0.0152)}


@gpu_only
@pytest.mark.parametrize("n", list(GPU_BOUNDS))
def test_polytrope_drift_from_rest_gpu(n):
    """Regression only: the kick hold at 64^3 and 128^3, on a GPU."""
    _assert(n, run(n, TIMES, gravity_energy="kick"), GPU_BOUNDS[n])


COLD = "open bug: a cold ambient drains to vacuum through the diode faces; its pressure goes negative near 2.85 T_sc"


def cold_first_bad(n, t_end=3.0):
    """Step by step to t_end T_sc with a cold ambient: None, or the first step whose state is not finite with rho, P > 0."""
    import jax.numpy as jnp
    from amuse_luna.stepper import build
    opts = dict(test_problem="Soundwave-3D", SPATIAL_DIMENSION=3, Nx=n, Ny=n, Nz=n, Lx=1.0, Ly=1.0, Lz=1.0,
                BOUNDARY=4, POISSON_BOUNDARY=3, SELF_GRAVITY=True, GRAVITATIONAL_G=G, gamma=GAMMA, ORDER=1,
                LIMITER=0, riemann_solver="hllc", enable_x64=True, write_snapshots=False, t_end=t_end * T_SC)
    st = build(_build_cfg_for_test(**opts), U0=jnp.asarray(polytrope_U(n, cold=True)))
    while st.t < t_end * T_SC * (1 - 1e-12):
        dt = min(st.compute_dt(), t_end * T_SC - st.t)
        if not dt > 0:
            return dict(step=st.step, t=st.t / T_SC, why=f"dt={dt}", rho_min=np.nan, p_min=np.nan)
        st.step_once(dt)
        U = np.asarray(st.U)
        rho = U[0]
        P = (GAMMA - 1.0) * (U[4] - 0.5 * (U[1:4] ** 2).sum(0) / rho)
        finite = bool(np.isfinite(U).all())
        if not finite or rho.min() <= 0.0 or P.min() <= 0.0:
            why = "non-finite" if not finite else ("rho <= 0" if rho.min() <= 0.0 else "P <= 0")
            return dict(step=st.step, t=st.t / T_SC, why=why, rho_min=float(rho.min()), p_min=float(P.min()))
    print(f"RUN [{_where()}] cold n={n} t={st.t / T_SC:.6f} T_sc of {t_end} steps={st.step} reached=True")
    return None


@gpu_only
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=COLD)
def test_cold_ambient_stays_positive_gpu(n=96):
    """A cold ambient (P = K rho^2 outside the star too) keeps rho, P > 0 and a finite state for 3 crossing times."""
    bad = cold_first_bad(n)
    print(f"MEASURED wb cold n={n} first bad state: {bad}")
    if bad is not None and not (bad["why"] == "P <= 0" and bad["rho_min"] > 0.0 and 2.0 <= bad["t"] <= 3.0):
        raise RunAborted(f"cold ambient failed, but not as the open bug (P <= 0 first, at 2-3 T_sc): {bad}")
    assert bad is None, f"cold ambient: P <= 0 at step {bad['step']}, t = {bad['t']:.3f} T_sc (min P {bad['p_min']:.3e})"


LANE = "open bug: the registered Lane-Emden-3D IC at n = 1 puts rho = 0 at the surface"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=LANE)
def test_registered_lane_emden_n1_ic():
    """The registered Lane-Emden-3D IC at n = 1: rho > 0, the analytic sinc profile above the ambient, a valid state."""
    from luna.hydro.physics_checks import validate_state
    from luna.ic_generation import lane_emden3d
    from luna.ic_generation.ic import initial_conditions
    n = 32
    cfg = _build_cfg_for_test(test_problem="Lane-Emden-3D", SPATIAL_DIMENSION=3, Nx=n, Ny=n, Nz=n, Lx=1.0,
                              Ly=1.0, Lz=1.0, gamma=GAMMA, GRAVITATIONAL_G=G, enable_x64=True,
                              problem=lane_emden3d.Params(n=1.0, rho_c=RHO_C, R=R, rho_ambient=AMB))
    U = np.asarray(initial_conditions(cfg), dtype=np.float64)
    r = radius(n)
    exact = np.where(r < R, np.maximum(RHO_C * np.sinc(r / R), AMB * RHO_C), AMB * RHO_C)
    err = float(np.abs(U[0] - exact).max())
    print(f"MEASURED lane-emden n=1 max |rho - sinc| = {err:.3e}, min rho = {float(U[0].min()):.3e}")
    assert float(U[0].min()) > 0.0, f"min rho {float(U[0].min()):.3e}"
    if err >= 1e-6:
        raise ValueError(f"rho > 0 but max |rho - sinc| = {err:.3e}")
    validate_state(U, GAMMA)
