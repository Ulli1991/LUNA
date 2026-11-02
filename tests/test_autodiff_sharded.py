"""The hydro gradient across a shard seam.

tests/test_autodiff_identities.py checks the differentiated step on one
device, where the halo exchange is a ppermute to itself. Here the grid is
split, so every halo crosses a device boundary and the adjoint has to carry
the cotangent back across it. Checked on 2 devices (the smallest seam), and
on (2,2,1) and (4,1,1) where the CPU mesh has the devices -- with two
devices on a periodic axis the left and right neighbour are the same device,
so a halo sent the wrong way round is invisible there and only 4 can see it.

  1. <J v, w> == <v, J^T w> for one step and for a checkpointed rollout
  2. total-mass gradient == 1 on density, 0 elsewhere (a dropped seam
     contribution breaks it immediately)
  3. the gradient on N devices == the gradient on 1, to round-off
  4. autodiff == central differences at seam cells, at a generic state (see
     generic_state for why generic)
  5. a negative control: an adjoint that drops the seam must fail 1-3

Run on 8 virtual CPU devices (conftest) and once on 2 GPUs:
    JAX_PLATFORMS=rocm XLA_FLAGS= pytest tests/test_autodiff_sharded.py
"""
import numpy as np
import pytest

import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P   # noqa: E402

from tests.test_autodiff_identities import (                  # noqa: E402
    GAMMA, N, TOL, _dot_product_test, _hydro_step, _random_like, _smooth_state)

pytestmark = pytest.mark.fast

NSTEP = 4
SHARDED = [(2, 1, 1), (2, 2, 1), (4, 1, 1)]


def _mesh(shape):
    n = int(np.prod(shape))
    if jax.device_count() < n:
        pytest.skip(f"mesh {shape} needs {n} devices, have {jax.device_count()}")
    return Mesh(np.array(jax.devices()[:n]).reshape(shape), ("x", "y", "z"))


def _put(U, mesh):
    return jax.device_put(U, NamedSharding(mesh, P(None, "x", "y", "z")))


def _rollout(mesh, n=NSTEP, limiter=0):
    """The production step in a checkpointed fixed-length scan, as
    luna.autodiff.rollout.build_simulate runs it."""
    step = _hydro_step(mesh, limiter=limiter)

    def run(U):
        body = jax.checkpoint(lambda U, _: (step(U), None))
        return jax.lax.scan(body, U, None, length=n)[0]
    return run


def _loss(mesh, limiter=0):
    run = _rollout(mesh, limiter=limiter)
    return lambda U: 0.5 * jnp.sum(run(U) ** 2)


# ---------------------------------------------------------------------------
# 1. the adjoint is the transpose, across the seam
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHARDED)
def test_step_adjoint_is_the_transpose_across_the_seam(shape):
    mesh = _mesh(shape)
    rel = _dot_product_test(_hydro_step(mesh), _put(_smooth_state(), mesh))
    assert rel < TOL, f"mesh {shape}: <Jv,w> vs <v,J^T w> disagree by {rel:.2e}"


@pytest.mark.parametrize("shape", SHARDED)
def test_rollout_adjoint_is_the_transpose_across_the_seam(shape):
    mesh = _mesh(shape)
    rel = _dot_product_test(_rollout(mesh), _put(_smooth_state(), mesh))
    assert rel < TOL, f"mesh {shape}, {NSTEP} steps: transpose mismatch {rel:.2e}"


# ---------------------------------------------------------------------------
# 2. conservation fixes the gradient, across the seam
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHARDED)
def test_total_mass_gradient_is_exact_across_the_seam(shape):
    mesh = _mesh(shape)
    run = _rollout(mesh)
    g = jax.jit(jax.grad(lambda U: jnp.sum(run(U)[0])))(_put(_smooth_state(), mesh))
    err_rho = float(jnp.max(jnp.abs(g[0] - 1.0)))
    err_rest = float(jnp.max(jnp.abs(g[1:])))
    assert err_rho < TOL, f"mesh {shape}: d(mass)/d(rho) deviates from 1 by {err_rho:.2e}"
    assert err_rest < TOL, f"mesh {shape}: d(mass)/d(momentum,energy) = {err_rest:.2e}"


# ---------------------------------------------------------------------------
# 3. N devices == 1 device
# ---------------------------------------------------------------------------

def _derivatives(mesh):
    """(gradient of the loss, J v, J^T w) of the rollout on `mesh`, as numpy."""
    U = _put(_smooth_state(), mesh)
    v = _put(_random_like(U, 3), mesh)
    w = _put(_random_like(U, 4), mesh)
    run = _rollout(mesh)

    @jax.jit
    def f(U, v, w):
        g = jax.grad(lambda U: 0.5 * jnp.sum(run(U) ** 2))(U)
        Jv = jax.jvp(run, (U,), (v,))[1]
        JTw = jax.vjp(run, U)[1](w)[0]
        return g, Jv, JTw
    return [np.asarray(a) for a in f(U, v, w)]


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.mark.parametrize("shape", SHARDED)
def test_sharded_derivatives_equal_the_single_device_ones(shape):
    """Same arithmetic per cell whatever the layout, so round-off only.
    MEASURED on 8 CPU devices and on 2 MI300A: see docs/AUTODIFF.md."""
    got = _derivatives(_mesh(shape))
    ref = _derivatives(_mesh((1, 1, 1)))
    for name, a, b in zip(("grad", "J v", "J^T w"), got, ref):
        assert _rel(a, b) < TOL, f"mesh {shape}: {name} differs from 1 device by {_rel(a, b):.2e}"


# ---------------------------------------------------------------------------
# 4. finite differences at the seam
# ---------------------------------------------------------------------------

def generic_state(seed=0):
    """A smooth periodic state with no symmetry: every primitive varies in
    every direction (four random Fourier waves each), so no limiter argument
    is exactly zero -- unlike _smooth_state, whose v_z == 0 and x-uniform
    v_x put minmod at a = b = 0, where it has no derivative."""
    rng = np.random.default_rng(seed)
    x = (np.arange(N) + 0.5) / N
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    def f():
        out = 0
        for _ in range(4):
            k = rng.integers(-2, 3, 3)
            out = out + rng.normal() * np.sin(2 * np.pi * (k[0] * X + k[1] * Y + k[2] * Z)
                                              + rng.uniform(0, 2 * np.pi))
        return out / np.abs(out).max()
    rho, p = 1 + 0.2 * f(), 1 + 0.2 * f()
    v = [0.1 * f() for _ in range(3)]
    E = p / (GAMMA - 1) + 0.5 * rho * sum(w * w for w in v)
    return jnp.asarray(np.stack([rho, *(rho * w for w in v), E]))


# (variable, x, y, z): both sides of the interior seam at x = N/2, the periodic
# wrap at x = N-1 and x = 0, every variable.
SEAM_CELLS = [(0, N // 2 - 1, 5, 9), (0, N // 2, 5, 9), (0, N - 1, 3, 3), (1, N // 2, 8, 2),
              (4, N // 2 - 1, 12, 7), (2, 0, 4, 11), (3, N // 2, 13, 1)]


@pytest.mark.parametrize("idx", SEAM_CELLS)
def test_gradient_at_the_seam_matches_central_differences(idx):
    """Minmod on (2,1,1), at a generic state (no limiter at a = b = 0), loss
    <w, U(T)> with a fixed random w; best central difference over h = 1e-3
    .. 1e-6 (kinks near some cells need the small h). MEASURED best: 5.6e-10
    to 5.7e-8, identical on 1 device and on 2. Not 1/2<U,U>: its value is
    6.7e3, and the rounding of that, over h, is 3e-6 of the 0.024 gradient
    of a momentum cell -- the floor it measured there, on 1 device and 2."""
    mesh = _mesh((2, 1, 1))
    U = _put(generic_state(), mesh)
    w = _put(_random_like(U, 9), mesh)
    run = _rollout(mesh)
    loss = jax.jit(lambda U, w: jnp.vdot(w, run(U)))
    g = float(jax.jit(jax.grad(loss))(U, w)[idx])
    err = []
    for h in (1e-3, 1e-4, 1e-5, 1e-6):
        fd = (float(loss(U.at[idx].add(h), w)) - float(loss(U.at[idx].add(-h), w))) / (2 * h)
        err.append(abs(g - fd) / max(abs(fd), 1e-300))
    assert min(err) < 1e-6, f"cell {idx}: best central-difference agreement {min(err):.2e} of {err}"


# ---------------------------------------------------------------------------
# 5. negative control: an adjoint that drops the seam
# ---------------------------------------------------------------------------

def _seam_dropping_ppermute(real):
    """ppermute whose reverse rule loses what crosses a device boundary (the
    forward and the one-device self-exchange stay correct)."""
    def ppermute(x, axis_name, perm):
        cross = any(s != d for s, d in perm)

        @jax.custom_vjp
        def f(x):
            return real(x, axis_name, perm)

        def bwd(_, ct):
            back = [(d, s) for s, d in perm]
            return (jnp.zeros_like(ct) if cross else real(ct, axis_name, back),)
        f.defvjp(lambda x: (f(x), None), bwd)
        return f(x)
    return ppermute


def test_a_dropped_seam_in_the_adjoint_is_caught(monkeypatch):
    """The gates above must notice an adjoint that drops the seam: the
    transpose identity (J v from the correct tangent), the mass identity and
    the comparison with one device all fail. A gate nobody saw fail is not
    evidence."""
    mesh = _mesh((2, 1, 1))
    U = _put(_smooth_state(), mesh)
    v, w = _put(_random_like(U, 0), mesh), _put(_random_like(U, 1), mesh)
    Jv = jax.jit(lambda U, v: jax.jvp(_rollout(mesh), (U,), (v,))[1])(U, v)
    g_ref = np.asarray(_derivatives(_mesh((1, 1, 1)))[0])

    monkeypatch.setattr(jax.lax, "ppermute", _seam_dropping_ppermute(jax.lax.ppermute))
    run = _rollout(mesh)
    JTw = jax.jit(lambda U, w: jax.vjp(run, U)[1](w)[0])(U, w)
    a, b = float(jnp.vdot(Jv, w)), float(jnp.vdot(v, JTw))
    rel = abs(a - b) / max(abs(a), abs(b))
    g_mass = jax.jit(jax.grad(lambda U: jnp.sum(run(U)[0])))(U)
    g = np.asarray(jax.jit(jax.grad(lambda U: 0.5 * jnp.sum(run(U) ** 2)))(U))

    assert rel > 1e3 * TOL, f"a dropped seam did not move the transpose identity ({rel:.2e})"
    assert float(jnp.max(jnp.abs(g_mass[0] - 1.0))) > 1e3 * TOL, "mass identity blind to the seam"
    assert _rel(g, g_ref) > 1e3 * TOL, "1-vs-2 device comparison blind to the seam"
