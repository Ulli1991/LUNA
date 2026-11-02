"""Warm-started gravity solves along a sequence of sources, each step scored against a converged solve.

The source sequences are the cases of tests/test_gravity_*.py; `run` drives the backend as the gravity stage does.
"""
import functools
import os
from dataclasses import replace

import jax

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from luna.gravity.backend import DELTA, PROPER, LocalGravity, PoissonOpts, ShardedGravity  # noqa: E402
from luna.gravity.poisson_multigrid import solve_poisson_fmg  # noqa: E402


# Bound on the per-step potential and acceleration error, per (case, grid side), at the defaults
# (tol 1e-4, extrapolated, mixed): 1.3 times the largest error of the opening cold FMG and the warm solves.
BOUNDS = {
    ("jeans", 32): 2.3e-3, ("dt_jump", 32): 8.6e-3, ("restart", 32): 1.3e-2,
    ("turbulence_fast", 32): 1.3e-2, ("turbulence_slow", 32): 1.1e-2,
    ("collapse_periodic", 32): 4.2e-3, ("collapse_isolated", 32): 3.5e-3, ("sinks", 32): 4.9e-3,
    ("feedback", 32): 1.3e-2, ("cosmo", 32): 1.4e-2,
    ("jeans", 128): 1.7e-4, ("dt_jump", 128): 8.4e-4, ("restart", 128): 1.4e-3,
    ("turbulence_fast", 128): 1.4e-3, ("turbulence_slow", 128): 1.1e-3,
    ("collapse_periodic", 128): 5.7e-4, ("collapse_isolated", 128): 4.4e-4,
    ("sinks", 128): 6.7e-4, ("feedback", 128): 1.4e-3, ("cosmo", 128): 1.5e-3,
}

# The same with fixed cycles (gravity_warm_tol=0): 1.5 times the largest error of the two-cycle
# FP64 solves (cosmo: one mixed cycle).
FIXED_BOUNDS = {
    ("jeans", 32): 3.0e-3, ("dt_jump", 32): 1.0e-2, ("restart", 32): 1.5e-2,
    ("turbulence_fast", 32): 2.2e-2, ("turbulence_slow", 32): 1.25e-2,
    ("collapse_periodic", 32): 5.0e-3, ("collapse_isolated", 32): 4.1e-3, ("sinks", 32): 6.3e-3,
    ("feedback", 32): 1.9e-2, ("cosmo", 32): 2.5e-2,
    ("jeans", 128): 1.5e-3, ("dt_jump", 128): 9.1e-3, ("restart", 128): 1.6e-2,
    ("turbulence_fast", 128): 2.4e-2, ("turbulence_slow", 128): 7.4e-3,
    ("collapse_periodic", 128): 6.2e-3, ("collapse_isolated", 128): 4.9e-3,
    ("sinks", 128): 3.5e-3, ("feedback", 128): 1.7e-2, ("cosmo", 128): 2.7e-2,
}


# Proposed bounds for the extrapolated, convergence-controlled, mixed-precision solve, on its warm
# steps only: 1.5 times what that solve reaches. LUNA_TEST_GRAVITY_BOUNDS=tight selects them.
TIGHT = {
    ("jeans", 32): 1.6e-3, ("dt_jump", 32): 1.8e-3, ("restart", 32): 2.7e-3,
    ("turbulence_fast", 32): 1.4e-3, ("turbulence_slow", 32): 2.1e-3,
    ("collapse_periodic", 32): 1.1e-3, ("collapse_isolated", 32): 7.2e-4, ("sinks", 32): 1.9e-3,
    ("feedback", 32): 2.9e-3, ("cosmo", 32): 2.1e-3,
    ("jeans", 128): 1.5e-3, ("dt_jump", 128): 2.1e-3, ("restart", 128): 2.4e-3,
    ("turbulence_fast", 128): 1.6e-3, ("turbulence_slow", 128): 2.3e-3,
    ("collapse_periodic", 128): 1.3e-3, ("collapse_isolated", 128): 9.2e-4,
    ("sinks", 128): 7.4e-4, ("feedback", 128): 2.6e-3, ("cosmo", 128): 2.1e-3,
}


def bounds(cfg=None):
    """The bound table for `cfg` (default: the suite's config): fixed or convergence-controlled cycles."""
    cfg = config(32) if cfg is None else cfg
    return FIXED_BOUNDS if cfg.gravity_warm_tol == 0 else BOUNDS


def tight_bounds():
    """Whether the suite checks the proposed per-case bounds (TIGHT) instead of BOUNDS."""
    return os.environ.get("LUNA_TEST_GRAVITY_BOUNDS", "") == "tight"


def assert_tracks(name, N, err, bad, restarts=()):
    """Errors within the case's bound, and the broken solve's beyond it."""
    print(f"MEASURED {name} N={N}: e_g max {err[:, 1].max():.3e} e_psi max {err[:, 0].max():.3e} "
          f"broken e_g max {bad[:, 1].max():.3e}")
    if not tight_bounds():
        bound = bounds()[(name, N)]
        assert err[:, 1].max() <= bound, (name, err[:, 1])
        assert err[:, 0].max() <= bound, (name, err[:, 0])
        assert bad[:, 1].max() > bound, (name, bad[:, 1])
        return
    warm = [n for n in range(len(err)) if n and n not in restarts]
    cold = [n for n in range(len(err)) if n not in warm]
    bound = TIGHT[(name, N)]
    assert err[warm].max() <= bound, (name, err[warm])
    assert err[cold].max() <= BOUNDS[(name, N)], (name, err[cold])
    caught = bad[warm, 1].max() > bound or bad[cold, 1].max() > BOUNDS[(name, N)]
    assert caught, (name, bad[:, 1])


def env_overrides():
    """Config overrides from LUNA_TEST_GRAVITY ("field=value+field=value"): the suite on another solve setting."""
    from luna.setup.cli import overrides
    spec = os.environ.get("LUNA_TEST_GRAVITY", "")
    return overrides(dict(kv.split("=", 1) for kv in spec.split("+") if kv))


def config(N, boundary=0, **over):
    """A cubic self-gravity config of side N (box 1), with the LUNA_TEST_GRAVITY overrides."""
    from luna.template_config import Config
    over = {**env_overrides(), **over}
    cfg = replace(Config(), SPATIAL_DIMENSION=3, Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0,
                  SELF_GRAVITY=True, GRAVITATIONAL_G=1.0, BOUNDARY=0 if boundary == 0 else 1,
                  POISSON_BOUNDARY=boundary)
    return replace(cfg, **over)


def backend(cfg, mesh=None, comoving=False):
    """The backend `build_gravity` would make for `cfg` (sharded when `mesh` is given)."""
    from luna.gravity.particle_mesh import ParticleMeshGravity
    from luna.setup.cosmo_common import gravity_precision_for
    prec = gravity_precision_for(cfg, comoving)
    opts = PoissonOpts.from_config(cfg, cfg.dm_warm_vcycles if comoving else cfg.gravity_warm_vcycles)
    mode = DELTA if comoving else PROPER
    if mesh is None:
        return LocalGravity(ParticleMeshGravity(cfg), cfg.Lx, opts, prec, mode, 1.0)
    N3 = (cfg.Nx, cfg.Ny, cfg.Nz)
    return ShardedGravity(mesh, N3, cfg.Lx / cfg.Nx, cfg.Lx, int(cfg.POISSON_BOUNDARY), opts, prec,
                          mode, 1.0)


def source(rho, boundary):
    """lap psi = rho - <rho> (periodic) or rho, as the stage builds it with 4 pi G = 1."""
    rho = jnp.asarray(rho, jnp.float64)
    return rho - jnp.mean(rho) if boundary == 0 else rho


@functools.lru_cache(maxsize=None)
def _converged_program(N, boundary):
    return jax.jit(lambda s: solve_poisson_fmg(s, 1.0 / N, 1.0, boundary, n_extra_vcycles=30,
                                               n_coarsest=50))


def converged(src, boundary):
    """The discrete solution: FMG plus 30 V(2,2) cycles."""
    phi = _converged_program(src.shape[0], boundary)(jnp.asarray(src, jnp.float64))
    return phi - jnp.mean(phi) if boundary == 0 else phi


def _accel(psi, boundary):
    mode = "wrap" if boundary == 0 else "edge"
    p = jnp.pad(psi, 1, mode=mode)
    N = psi.shape[0]
    return jnp.stack([(p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]),
                      (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]),
                      (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2])]) * (-0.5 * N)


def score(psi, ref, boundary):
    """(potential error, acceleration error): rms of the difference over rms of the reference."""
    psi = jnp.asarray(np.asarray(psi), jnp.float64)
    d = psi - ref
    if boundary in (0, 1, 2):
        d = d - jnp.mean(d)
    r = ref - jnp.mean(ref) if boundary in (0, 1, 2) else ref
    e_psi = float(jnp.sqrt(jnp.mean(d ** 2) / jnp.mean(r ** 2)))
    ga, gr = _accel(psi, boundary), _accel(ref, boundary)
    e_g = float(jnp.sqrt(jnp.sum((ga - gr) ** 2) / jnp.sum(gr ** 2)))
    return e_psi, e_g


def place(be, x):
    """`x` as the backend holds a grid."""
    if not be.sharded:
        return jnp.asarray(x)
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P
    return jax.device_put(jnp.asarray(x), NamedSharding(be.mesh, P("x", "y", "z")))


def run(be, steps, boundary, restart_at=(), broken=None, keep=None, events=(), info=None):
    """Per-step (psi, acceleration) errors along [(rho, dt), ...] through `Gravity._solve`."""
    from luna.gravity.kdk import Gravity
    from luna.setup.state import State
    stage = Gravity(be, None, [], True)
    errs, state = [], State()
    for n, (rho, dt) in enumerate(steps):
        src = np.asarray(source(rho, boundary))
        s = place(be, src)
        be.last_info = None
        if n in events:
            stage.events[0] = True
        if n and (broken == "stale" or broken == ("skip", n)):
            state = state.update(psi_prev=state.psi, dt_prev=float(dt))
        elif state.psi is None or n in restart_at:
            stage = Gravity(be, None, [], True)
            state = State(psi=be.solve(s, None))
        else:
            psi, state = stage._solve(state, s, dt)
            state = state.update(psi=psi)
        errs.append(score(state.psi, converged(src, boundary), boundary))
        if keep is not None:
            keep.append(np.asarray(state.psi))
        if info is not None:
            li = be.last_info
            info.append((int(li[0]), float(li[1])) if li is not None else None)
    return np.array(errs)


# ---- source sequences: lists of (rho, dt) on an N^3 box of side 1 -------------------------------

def _xyz(N):
    x = (np.arange(N) + 0.5) / N
    return np.meshgrid(x, x, x, indexing="ij")


def jeans(N, n=16, dt=0.05, growth=1.0, amp=1e-2):
    """Linear Jeans growth: rho = 1 + amp exp(growth t) (cos 2 pi x + cos 2 pi (y + z) / 2)."""
    X, Y, Z = _xyz(N)
    mode = np.cos(2 * np.pi * X) + 0.5 * np.cos(2 * np.pi * (Y + Z))
    return [(1.0 + amp * np.exp(growth * k * dt) * mode, dt) for k in range(n)]


def _sphere(N, r, centre=(0.5, 0.5, 0.5), width=1.5):
    X, Y, Z = _xyz(N)
    d = np.sqrt((X - centre[0]) ** 2 + (Y - centre[1]) ** 2 + (Z - centre[2]) ** 2)
    return 0.5 * (1.0 - np.tanh((d - r) * N / width))


def collapse(N, n=30, r0=0.3, shrink=0.95, floor=1e-2):
    """Homologous collapse of a uniform sphere: radius r0 s, density s^-3, s *= shrink per step, dt ~ s^1.5."""
    out, s = [], 1.0
    for _ in range(n):
        out.append((floor + _sphere(N, r0 * s) / s ** 3, 0.01 * s ** 1.5))
        s *= shrink
    return out


SINK_EVENTS = (6, 7, 13)


def sinks(N, n=20, events=SINK_EVENTS, r_sink=2.5):
    """A contracting clump whose core mass is removed at `events` (a sink forms, then accretes)."""
    out, removed = [], np.zeros((N, N, N))
    seq = collapse(N, n=n, r0=0.25, shrink=0.97)
    centres = [(0.5, 0.5, 0.5), (0.5 + 3.0 / N, 0.5, 0.5), (0.5, 0.5 - 3.0 / N, 0.5)]
    for k, (rho, dt) in enumerate(seq):
        if k in events:
            c = centres[list(events).index(k) % len(centres)]
            removed = np.maximum(removed, _sphere(N, r_sink / N, c, width=0.5))
        out.append((rho * (1.0 - 0.999 * removed), dt))
    return out


def moving_blob(N, dts, v=(1.0, 0.5, 0.0), amp=10.0, w=0.08):
    """A Gaussian over-density moving at velocity v; `dts` sets the step sizes (dt jumps and drops)."""
    X, Y, Z = _xyz(N)
    out, t = [], 0.0
    for dt in dts:
        c = [(0.3 + vi * t) % 1.0 for vi in v]
        d2 = sum(np.minimum(np.abs(A - ci), 1.0 - np.abs(A - ci)) ** 2 for A, ci in zip((X, Y, Z), c))
        out.append((1.0 + amp * np.exp(-0.5 * d2 / w ** 2), dt))
        t += dt
    return out


def turbulence(N, n=12, dt=0.02, t_corr=1.0, sigma=1.0, k_max=3, seed=1, t0=0.0):
    """Log-normal density of an Ornstein-Uhlenbeck random field in modes |k| <= k_max (driven turbulence)."""
    rng = np.random.default_rng(seed)
    X, Y, Z = _xyz(N)
    ks = [(i, j, k) for i in range(-k_max, k_max + 1) for j in range(-k_max, k_max + 1)
          for k in range(0, k_max + 1) if 0 < i * i + j * j + k * k <= k_max ** 2]
    phases = [2 * np.pi * (i * X + j * Y + k * Z) for i, j, k in ks]
    a = rng.standard_normal((len(ks), 2))
    f = np.exp(-dt / t_corr)
    out = []
    for step in range(int(round(t0 / dt)) + n):
        a = f * a + np.sqrt(1 - f * f) * rng.standard_normal(a.shape)
        if step < int(round(t0 / dt)):
            continue
        s = sum(ai[0] * np.cos(p) + ai[1] * np.sin(p) for ai, p in zip(a, phases))
        s = sigma * s / np.sqrt(len(ks))
        out.append((np.exp(s - 0.5 * sigma ** 2), dt))
    return out


def feedback(N, n=12, at=5, dt=0.02, boost=20.0):
    """Turbulent gas with a mass injection burst at step `at` that then spreads out."""
    base = turbulence(N, n=n, dt=dt, seed=3)
    out = []
    for k, (rho, d) in enumerate(base):
        if k >= at:
            w = (1.5 + 0.7 * (k - at)) / N
            X, Y, Z = _xyz(N)
            g = np.exp(-0.5 * ((X - 0.4) ** 2 + (Y - 0.6) ** 2 + (Z - 0.5) ** 2) / w ** 2)
            rho = rho + boost * g * (1.5 / N / w) ** 3
        out.append((rho, d))
    return out
