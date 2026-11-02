"""Shared builders for the LUNA test suite: AMR configs, blob/uniform ICs,
and the short-run production-path driver used by the regression tests."""

import numpy as np
import jax.numpy as jnp

GAMMA = 5.0 / 3.0


PRECISION = "fp64"      # the running test's mode, set by tests/conftest.py
# Config per mode; production defaults gravity to mixed, the tests set it from the mode.
_MODE_CFG = {"fp64": dict(enable_x64=True, hydro_recon32=False, gravity_proper_mixed=False, gravity_precision="double"),
             "fp32": dict(enable_x64=False, hydro_recon32=False, gravity_proper_mixed=True, gravity_precision="single"),
             "mixed": dict(enable_x64=True, hydro_recon32=True, gravity_proper_mixed=True, gravity_precision="mixed")}
_GRAVITY_MODE = {"fp64": "double", "fp32": None, "mixed": "mixed"}   # effective solve mode; None: not checked


class PrecisionMismatch(ValueError):
    """A run's cfg disagrees with the test's precision mode."""


def check_precision(cfg):
    """Refuse a cfg whose precision (enable_x64, hydro_recon32, the effective gravity solve) disagrees with the
    test's mode or with JAX."""
    import jax
    from luna.setup.cosmo_common import gravity_precision_for
    mode = _MODE_CFG[PRECISION]
    x64, recon32 = mode["enable_x64"], mode["hydro_recon32"]
    got = (bool(cfg.enable_x64), bool(getattr(cfg, "hydro_recon32", False)))
    if got != (x64, recon32) or bool(jax.config.jax_enable_x64) != x64:
        raise PrecisionMismatch(f"test precision {PRECISION!r} needs enable_x64={x64}, hydro_recon32={recon32} "
                                f"(JAX x64 {x64}); got cfg {got}, JAX x64 {bool(jax.config.jax_enable_x64)}")
    want = _GRAVITY_MODE[PRECISION]
    grav = gravity_precision_for(cfg, bool(getattr(cfg, "CosmologicalIntegration", False))).mode
    if cfg.SELF_GRAVITY and want is not None and grav != want:
        raise PrecisionMismatch(f"test precision {PRECISION!r} needs the gravity solve in {want}; cfg gives {grav}")
    if recon32 and cfg.SPATIAL_DIMENSION != 3:
        raise PrecisionMismatch(f"precision 'mixed' (hydro_recon32) is 3-D only; got SPATIAL_DIMENSION={cfg.SPATIAL_DIMENSION}")


def cfg3d(n, bs=16, **kw):
    """3D AMRConfig on the unit box: (n,)*3 base grid, (bs,)*3 blocks."""
    from luna.amr.config import AMRConfig
    return AMRConfig(ndim=3, box=(1.0,) * 3, base_grid=(n,) * 3,
                     block_shape=(bs,) * 3, **kw)


def blob_U(n, amp=0.5, sigma=0.09, p_amp=0.3, vx_amp=0.0, sigma2=None, E0=None):
    """Gaussian-blob conserved IC (5, n, n, n) on the unit box.

    rho = 1 + amp*g, p = 1 + p_amp*g with g = exp(-r^2 / (2 sigma^2))
    (or exp(-r^2 / sigma2) when sigma2 is given), vx = vx_amp*sin(2 pi x).
    E0, if given, overrides the total energy with that uniform value.
    """
    x = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2
    g = np.exp(-r2 / (2 * sigma ** 2 if sigma2 is None else sigma2))
    rho = 1.0 + amp * g
    p = 1.0 + p_amp * g
    u = vx_amp * np.sin(2 * np.pi * X)
    U = np.zeros((5, n, n, n))
    U[0] = rho
    U[1] = rho * u
    U[4] = E0 if E0 is not None else p / (GAMMA - 1.0) + 0.5 * rho * u * u
    return jnp.asarray(U)


def uniform_U(rho):
    """3D conserved state from a density field: uniform pressure, v = 0."""
    E = np.ones_like(rho) / (GAMMA - 1.0)
    z = np.zeros_like(rho)
    return jnp.asarray(np.stack([rho, z, z, z, E], axis=0))


def _build_cfg_for_test(**overrides):
    """Construct a Config with quiet/fast test defaults and the test's precision; allow overrides."""
    from luna.template_config import Config

    quiet = dict(PLOT_AFTER_SIM=False, **_MODE_CFG[PRECISION],
                 enable_diagnostics=True, enable_checkpointing=False)
    return Config(**{**quiet, **overrides})


def _with_problem(params_cls, **kw):
    """A test Config whose `params_cls` fields among `kw` go to cfg.problem."""
    from dataclasses import fields
    names = {f.name for f in fields(params_cls)}
    problem = params_cls(**{k: kw.pop(k) for k in list(kw) if k in names})
    return _build_cfg_for_test(problem=problem, **kw)


def _make_step_fns(cfg):
    """(compute_dt_fn, step_fn) from production's create_solver_distributed on a one-device mesh,
    so limiter, kernel, fused and recon32 resolve exactly as in a run."""
    import jax
    from jax.sharding import Mesh

    from luna.setup.hydro import create_solver_distributed
    from luna.utils.grid_params import GridParams

    sdim = cfg.SPATIAL_DIMENSION
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape((1,) * sdim), ("x", "y", "z")[:sdim])
    compute_dt, step, _ = create_solver_distributed(cfg, GridParams.from_config(cfg), mesh)
    return compute_dt, step


def _where():
    """The running pytest test id, for the RUN lines."""
    import os
    return os.environ.get("PYTEST_CURRENT_TEST", "-").split(" ")[0]


def _run_short(cfg, U=None, conserve=True):
    """Drive a short simulation (no main.py machinery) from U (default: the cfg's IC) to t_end; raises RunAborted short of it.

    Pure hydro: in a closed box (`conserve`) the end state is checked against U as in `evolve`."""
    from luna.ic_generation.ic import initial_conditions

    check_precision(cfg)
    U = initial_conditions(cfg) if U is None else U
    U0 = np.asarray(U)
    compute_dt_fn, step_fn = _make_step_fns(cfg)
    t, n = 0.0, 0
    while t < cfg.t_end * (1 - 1e-12):
        dt = min(float(compute_dt_fn(U)), cfg.t_end - t)
        if not dt > 1e-15 * cfg.t_end:
            raise RunAborted(f"{cfg.test_problem}: dt={dt} at t={t} of {cfg.t_end} after {n} steps")
        U = step_fn(U, dt)
        t, n = t + dt, n + 1
    U = np.asarray(U)
    if not np.isfinite(U).all():
        raise RunAborted(f"{cfg.test_problem}: non-finite state at t_end after {n} steps")
    print(f"RUN [{_where()}] {cfg.test_problem} N={cfg.Nx} t={t:.6e} t_end={cfg.t_end:.6e} steps={n} reached=True")
    if conserve and cfg.BOUNDARY in (0, 2):
        check_conserved(f"{cfg.test_problem} N={cfg.Nx} t={cfg.t_end:.4g}", U0, U, momentum=cfg.BOUNDARY == 0)
    return U


def accuracy_cfg(**kw):
    """Config for an accuracy run: minmod, HLLC, the default dt rule, the test's precision, overridable."""
    base = dict(ORDER=1, LIMITER=0, riemann_solver="hllc", **_MODE_CFG[PRECISION],
                PLOT_AFTER_SIM=False, enable_checkpointing=False)
    return _build_cfg_for_test(**{**base, **kw})


MAX_STEPS = 5_000_000


class RunAborted(RuntimeError):
    """A run that stopped before its target time: step cap, a non-positive or NaN dt, or a non-finite state."""


def evolve(cfg, U0, times, mesh_shape=None, max_steps=MAX_STEPS, conserve=True, observe=None, faces=None):
    """(states at `times`, steps) from U0 at t=0 on `mesh_shape`; raises RunAborted unless every time is reached.
    With `observe`, observe(U) is kept at each time instead of the state.

    The step is pure hydro, so in a closed box (`conserve`) each state is checked against U0: mass and
    energy to round-off, momentum too when periodic (walls push). faces: the first-order face count (audit C8)
    the run must give, (lo, hi), required with cfg.allow_first_order_faces (check_faces)."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh

    from luna.setup.hydro import create_solver_distributed
    from luna.utils.grid_params import GridParams

    check_precision(cfg)
    sdim = cfg.SPATIAL_DIMENSION
    shape = tuple(mesh_shape or (1,) * sdim)
    if len(jax.devices()) < int(np.prod(shape)):
        raise RunAborted(f"mesh {shape} needs {int(np.prod(shape))} devices, {len(jax.devices())} visible")
    devs = np.asarray(jax.devices()[:int(np.prod(shape))]).reshape(shape)
    mesh = Mesh(devs, ("x", "y", "z")[:sdim])
    grid = GridParams.from_config(cfg)
    compute_dt, step, sharding = create_solver_distributed(cfg, grid, mesh, face_report=True)

    # FP32: t is float32, and a plain sum of the steps' dt drifts ~sqrt(steps) ulp from the time the state is at
    # (3e-6 of t over 4.5k steps, A5); Kahan-compensated there, t - c is that time. FP64 is unchanged (c = 0).
    kahan = not cfg.enable_x64

    @jax.jit
    def advance(U, t, c, t1, n, fr):
        def body(s):
            U, t, c, n, _, fr = s
            dt = jnp.minimum(compute_dt(U), (t1 - t) + c if kahan else t1 - t)
            ok = dt > 0                          # False for a NaN or non-positive dt: stop, state untouched
            if kahan:
                y = dt - c
                tn = t + y
                c, t = jnp.where(ok, (tn - t) - y, c), jnp.where(ok, tn, t)
            else:
                t = jnp.where(ok, t + dt, t)
            U1, rep = step(U, dt)
            k = jnp.sum(rep["n"]).astype(jnp.int32)
            first = (fr["n"] == 0) & (k > 0)  # the face report: a total, the first step with one and its cells
            fr = {"n": fr["n"] + jnp.where(ok, k, 0), "step": jnp.where(first, n + 1, fr["step"]),
                  "t": jnp.where(first, t, fr["t"]), "cells": jnp.where(first, rep["cells"], fr["cells"]),
                  "faces": jnp.where(first, rep["faces"], fr["faces"])}
            return jnp.where(ok, U1, U), t, c, n + 1, ok, fr
        go = lambda s: s[4] & ((s[1] - s[2] if kahan else s[1]) < t1 * (1 - 1e-13)) & (s[3] < max_steps)  # noqa: E731
        return jax.lax.while_loop(go, body, (U, t, c, n, jnp.bool_(True), fr))

    U = jax.device_put(jnp.asarray(U0, dtype=jnp.float64 if cfg.enable_x64 else jnp.float32), sharding)
    dt0 = float(compute_dt(U))
    t, c, n, out = jnp.float64(0.0), jnp.float64(0.0), jnp.int32(0), []
    k = 8 * int(np.prod(shape))                  # integrators.FACE_REPORT_CELLS per shard
    fr = {"n": jnp.int32(0), "step": jnp.int32(0), "t": t, "cells": jnp.full((k, sdim), -1, jnp.int32),
          "faces": jnp.zeros((k,), jnp.int32)}
    for t1 in np.atleast_1d(times):
        tj = jnp.float64(t1)                     # FP32 runs: float32(t1), the time the loop can land on
        U, t, c, n, ok, fr = advance(U, t, c, tj, n, fr)
        _face_report(cfg, grid, fr)
        Uh = np.asarray(U)
        reached = bool(ok) and float(t) - float(c) >= float(tj) * (1 - 1e-12) and bool(np.isfinite(Uh).all())
        print(f"RUN [{_where()}] {cfg.test_problem} N={cfg.Nx} t={float(t):.6e} t_end={t1:.6e} steps={int(n)} "
              f"t_end/dt0={t1 / dt0:.4g} reached={reached}")
        if not reached:
            raise RunAborted(f"stopped at t={float(t):.6e} of {t1:.6e} after {int(n)} steps "
                             f"(dt ok={bool(ok)}, finite={bool(np.isfinite(Uh).all())}, cap {max_steps})")
        if conserve and cfg.BOUNDARY in (0, 2):
            check_conserved(f"{cfg.test_problem} N={cfg.Nx} t={t1:.4g}", U0, Uh, momentum=cfg.BOUNDARY == 0)
        out.append(observe(Uh) if observe else Uh)
    check_faces(cfg, int(fr["n"]), faces)
    return (out if np.ndim(times) else out[0]), int(n)


def check_faces(cfg, n, faces):
    """The run's first-order face count (audit C8) within faces = (lo, hi); without allow_first_order_faces it is 0
    (else the run was refused). LUNA_FACES_MEASURE=1 prints it instead of checking (to set the bounds)."""
    import os
    allow = bool(cfg.allow_first_order_faces)
    print(f"FACES [{_where()}] {cfg.test_problem} N={cfg.Nx} LIMITER={cfg.LIMITER} {cfg.riemann_solver}: "
          f"{n} first-order faces (allow={allow})")
    if not allow or os.environ.get("LUNA_FACES_MEASURE") == "1":
        return
    if faces is None:
        raise ValueError("allow_first_order_faces=True: state the expected first-order face count, faces=(lo, hi)")
    lo, hi = faces
    assert lo <= n <= hi, f"first-order faces {n} outside [{lo}, {hi}] (audit C8)"


def _face_report(cfg, grid, fr):
    """Raise RunAborted naming the first step and cells with an unphysical Hancock face (audit C8) unless
    cfg.allow_first_order_faces; then print the count (setup/face_report.py)."""
    from luna.setup.face_report import FaceLog, _cells
    n = int(fr["n"])
    if n == 0:
        return
    where = FaceLog(cfg, grid, False).where(_cells(fr))
    head = f"{n} unphysical Hancock face states (audit C8) from step {int(fr['step'])} t={float(fr['t']):.6e}, {where}"
    if not cfg.allow_first_order_faces:
        raise RunAborted(f"{head}: refused (allow_first_order_faces=False)")
    print(f"FACES [{_where()}] {cfg.test_problem} N={cfg.Nx}: {head}, first order")


def at_n(n, n1, v1, n2, v2):
    """v at a literature resolution n from power-of-2 runs n1 < n < n2 (power law between them)."""
    return v1 * (v2 / v1) ** (np.log(n / n1) / np.log(n2 / n1))


def evolve_staged(cfg, U0, times, conserve=True, observe=None, faces=None):
    """U at each of `times` through the production loop (assemble, timestep, step): sources, stages and
    projections run as in a run. Closed box: mass checked to round-off (energy changes by source work).
    With `observe`, observe(U, t) is kept at each time instead of the state; faces: as in evolve. A refused
    unphysical face (setup/face_report.py) raises RunAborted naming step, t and cell."""
    import jax
    import jax.numpy as jnp

    from luna.setup.assemble import assemble
    from luna.setup.evolve import initial_state, step, timestep
    from luna.setup.face_report import FaceLog
    from luna.setup.ics import Initial
    from luna.setup.mesh import build_mesh, decomposition
    from luna.setup.run_kind import run_kind

    check_precision(cfg)
    kind = run_kind(cfg)
    layout = decomposition(cfg, kind)
    mesh, _ = build_mesh(cfg, layout, kind)
    dtype = jnp.float64 if cfg.enable_x64 else jnp.float32
    physics = assemble(cfg, mesh, kind, layout, Initial(cfg=cfg, U=np.asarray(U0)), is_main=False)
    U = jax.device_put(jnp.asarray(U0, dtype), physics.hydro.sharding)
    state = initial_state(Initial(cfg=cfg, U=U), physics)
    if physics.gravity is not None:                          # the opening solve, as luna.setup.evolve.evolve
        state = physics.gravity.init(state)
    out, log = [], FaceLog(cfg, physics.hydro.grid, is_main=False)
    for t_end in np.atleast_1d(times):
        while state.t < t_end:
            dt, t_new = timestep(state, physics, t_end)
            reports = []
            state = step(state, dt, t_new, physics, reports=reports)
            if reports:
                log.push(reports[0], state.step, float(t_new))
                try:
                    log.flush()
                except RuntimeError as e:
                    raise RunAborted(str(e)) from None
        U = np.asarray(state.U)
        if not np.isfinite(U).all():
            raise RunAborted(f"{cfg.test_problem}: non-finite state at t={state.t}")
        if conserve and cfg.BOUNDARY in (0, 2):
            tol = 1e-13 if U.dtype == np.float64 else 1e-5
            check(f"{cfg.test_problem} N={cfg.Nx} t={t_end:.4g} staged |dM|/M",
                  abs(U[0].sum() / np.asarray(U0)[0].sum() - 1.0), tol)
        out.append(observe(U, t_end) if observe else U)
    check_faces(cfg, log.total, faces)
    return out if np.ndim(times) else out[0]


def check(name, value, bound, lower=False):
    """Print the measured value and assert value <= bound (>= bound if `lower`)."""
    print(f"MEASURED {name} = {value:.6e} ({'>=' if lower else '<='} {bound:.6e})")
    ok = value >= bound if lower else value <= bound
    assert ok, f"{name}: measured {value:.6e}, bound {'>=' if lower else '<='} {bound:.6e}"


def check_conserved(name, U0, U, *, momentum=True, energy=True, tol=None):
    """Bound |dM|/M, if `energy` |dE|/E and, if `momentum`, every |dP_d| / sqrt(M E) (closed or periodic box);
    tol defaults to round-off of U's precision (1e-13 FP64, 1e-5 FP32)."""
    U0, U = np.asarray(U0, np.float64), np.asarray(U)
    tol = tol if tol is not None else (1e-13 if U.dtype == np.float64 else 1e-5)
    M, E = U0[0].sum(), U0[-1].sum()
    check(f"{name} |dM|/M", abs(U[0].sum() / M - 1.0), tol)
    if energy:
        check(f"{name} |dE|/E", abs(U[-1].sum() / E - 1.0), tol)
    for d in range(U0.ndim - 1 if momentum else 0):
        check(f"{name} |dP_{'xyz'[d]}|/sqrt(ME)", abs(U[1 + d].sum() - U0[1 + d].sum()) / np.sqrt(M * E), tol)


def self_gravity_energies(U, h, G):
    """(E_kin, E_th, W) of periodic self-gravitating gas U (5, nx, ny, nz) on cubes of side h: W = 1/2 sum rho phi
    dV, phi the FFT solution of the 7-point Laplacian (the operator the multigrid solves), mean density removed."""
    U = np.asarray(U, np.float64)
    rho, dV = U[0], h ** 3
    ke = 0.5 * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / rho
    lap = sum(((2 * np.cos(2 * np.pi * np.fft.fftfreq(n)) - 2) / h ** 2).reshape([-1 if a == b else 1 for b in range(3)])
              for a, n in enumerate(rho.shape))
    lap[0, 0, 0] = 1.0
    phi = np.real(np.fft.ifftn(4 * np.pi * G * np.fft.fftn(rho - rho.mean()) / lap))
    return float(ke.sum() * dV), float((U[4] - ke).sum() * dV), float(0.5 * (rho * phi).sum() * dV)


def check_tube_budget(name, U0, U, WL, WR, t, h, gamma, tol=1e-12):
    """1D Riemann problem with the waves inside the box: every total changes by t (F(W_L) - F(W_R)), as
    the end faces see constant states; error relative to the totals of rho and E, momentum to sqrt(M E).
    U, U0 are (rho, rho u, E)
    rows (3, N); W = (rho, u, p) or (rho, u, p, v) with a transverse v."""
    def flux(W):
        rho, u, p, v = (*W, 0.0)[:4]
        return np.array([rho * u, rho * u * u + p, u * (p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v) + p)])
    U0, U = np.asarray(U0, np.float64), np.asarray(U)
    got, want = h * (U.sum(axis=1) - U0.sum(axis=1)), t * (flux(WL) - flux(WR))
    M, E = h * np.abs(U0[0]).sum(), h * np.abs(U0[2]).sum()
    scale = (M, np.sqrt(M * E), E)
    for k, comp in enumerate(("mass", "momentum", "energy")):
        check(f"{name} budget {comp}", abs(got[k] - want[k]) / scale[k], tol)


def check_mirror(name, U, axis, tol=0.0):
    """Bound max |U - its mirror about the box centre along `axis`| (normal momentum negated) / max |U|."""
    U = np.asarray(U)
    R = np.flip(U, 1 + axis).copy()
    R[1 + axis] *= -1.0
    check(f"{name} mirror {'xyz'[axis]}", float(np.abs(U - R).max() / np.abs(U).max()), tol)


def gpu_only(fn):
    """Mark a test gpu + slow and skip it on the CPU backend."""
    import jax
    import pytest
    skip = pytest.mark.skipif(jax.default_backend() == "cpu", reason="large resolution: GPU only")
    return pytest.mark.gpu(pytest.mark.slow(skip(fn)))
