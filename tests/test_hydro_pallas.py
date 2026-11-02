"""The fused MUSCL-Hancock kernel (hydro/muscl_hancock_pallas.py) against the
XLA step it replaces (the wide 2-ghost path of muscl_hancock._muscl_hancock_3d).

It is the same arithmetic in a different compiler, so it agrees to round-off,
not bitwise; the comparison is normalised by the largest |dU|, because a
cell where the fluxes nearly cancel has a tiny dU and any relative error
there is noise. Conservation is exact either way: on a periodic box sum(dU)
vanishes to round-off iff every face flux is used with the same value on both
sides, which the kernel gets by computing each face twice from the same
inputs. On a CPU the kernel runs in Pallas's interpreter, which cannot run
inside a shard_map, so the production step (always a shard_map) is tested on
GPUs; the CPU tests call the kernel directly."""
import numpy as np
import pytest

import jax
import jax.numpy as jnp


from luna.hydro import muscl_hancock_pallas as mhp                   # noqa: E402
from luna.hydro.conversions import cons_to_prim, prim_to_cons       # noqa: E402
from luna.hydro.muscl_hancock import _muscl_hancock_3d               # noqa: E402

GAMMA = 5.0 / 3.0
GPU = jax.default_backend() in ("gpu", "rocm", "cuda")


def _prims(n, seed=0, shock=True):
    """Periodic, fully 3-D, with a density jump of 50 across half the box."""
    rng = np.random.default_rng(seed)
    x = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    k = 2.0 * np.pi
    rho = 1.0 + 0.3 * np.sin(k * X) * np.cos(k * Y) * np.cos(k * Z)
    if shock:
        rho = rho * np.where(X < 0.5, 1.0, 50.0)
    v = [0.4 * np.cos(k * X) * np.sin(k * Y), 0.4 * np.sin(k * Z) * np.cos(k * X),
         0.4 * np.cos(k * Y) * np.sin(k * Z)]
    P = 1.0 + 0.2 * np.cos(k * X) * np.cos(k * Y) * np.sin(k * Z) + 0.05 * rng.random(X.shape)
    return jnp.asarray(np.stack([rho, *v, P]))


def _both(W, dt, recon32=False, solver="hllc"):
    n = W.shape[1]
    inv = (float(n),) * 3
    Wp = jnp.pad(W, [(0, 0)] + [(2, 2)] * 3, mode="wrap")
    xla = jax.jit(lambda Wp: _muscl_hancock_3d(Wp, dt, gamma=GAMMA, limiter=0, riemann_solver=solver,
                                               inv_dx=inv[0], inv_dy=inv[1], inv_dz=inv[2],
                                               boundary=0, wide=True, recon32=recon32))(Wp)
    pal = jax.jit(lambda Wp: mhp.divergence(Wp, dt, gamma=GAMMA, inv=inv, recon32=recon32,
                                            riemann_solver=solver))(Wp)
    return np.asarray(xla), np.asarray(pal)


def _agreement(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(a)))


def _imbalance(dU):
    """|sum dU_k| / sum |dU_k|, worst component: 0 up to round-off iff the
    face fluxes telescope."""
    return max(float(abs(dU[k].sum()) / np.abs(dU[k]).sum()) for k in range(len(dU)))


@pytest.mark.parametrize("shock", [False, True])
def test_the_kernel_is_the_xla_divergence(shock):
    n = 16 if not GPU else 64
    a, b = _both(_prims(n, shock=shock), 0.2 / n / 3.0)
    assert _agreement(a, b) < 1e-12


def test_the_kernel_conserves_on_a_periodic_box():
    n = 16 if not GPU else 64
    a, b = _both(_prims(n), 0.2 / n / 3.0)
    assert _imbalance(a) < 1e-13            # the XLA step: the reference level
    assert _imbalance(b) < 1e-13


# FP32 and recon32 (measured 2026-10-10, A100 64^3 and CPU interpreter 16^3, smooth / shock):
# Pallas against XLA's own recon32 path is bitwise on the GPU and 2e-7 / 5e-8 (FP32),
# 3e-7 / 4e-8 (mixed) in the interpreter, where XLA's CPU backend contracts FMAs per
# expression. recon32 differs from FP64 by 2e-7 .. 3e-6: the teeth that it reconstructs in FP32.
@pytest.mark.precision("fp32")
@pytest.mark.parametrize("shock", [False, True])
def test_the_fp32_kernel_is_the_xla_divergence(shock):
    n = 16 if not GPU else 64
    a, b = _both(_prims(n, shock=shock), 0.2 / n / 3.0)
    assert b.dtype == np.float32
    assert _agreement(a, b) < 1e-6
    assert _imbalance(b) < 5e-8             # FP32 round-off, as XLA's (1.2e-8 .. 1.5e-8)


@pytest.mark.precision("mixed")
@pytest.mark.parametrize("shock", [False, True])
def test_the_recon32_kernel_is_the_xla_recon32_divergence(shock):
    n = 16 if not GPU else 64
    W, dt = _prims(n, shock=shock), 0.2 / n / 3.0
    a, b = _both(W, dt, recon32=True)
    _, c = _both(W, dt)
    assert b.dtype == np.float64
    assert _agreement(a, b) < 1e-6
    assert _agreement(c, b) > 1e-8           # FP32 reconstruction, not FP64
    assert _imbalance(a) < 1e-13             # XLA: FP64 fluxes telescope
    # The kernel computes each face twice. Triton emits the same code for both (2.8e-17 at
    # 64^3); the CPU interpreter may contract an FMA in one and not the other, which the
    # FP32 face state turns into 1e-9 (measured 1.1e-9 smooth, 5e-11 shock at 16^3).
    assert _imbalance(b) < (1e-13 if GPU else 3e-9)


def _stepper(mesh, kernel, n, recon32=False, solver="hllc"):
    from luna.time_integration.integrators import create_muscl_hancock_modular_step
    from luna.utils.halo_exchange_shardmap import make_halo_shardmap
    dx = 1.0 / n
    return create_muscl_hancock_modular_step(
        make_halo_shardmap(mesh, 3), GAMMA, 1, 0, dx, dx, dx, 1 / dx, 1 / dx, 1 / dx,
        0, 3, 0.4, riemann_solver=solver, use_shardmap=True, mesh=mesh, kernel=kernel,
        recon32=recon32)[1]


def _mesh(ndev):
    from jax.sharding import Mesh
    return Mesh(np.array(jax.devices()[:ndev]).reshape(ndev, 1, 1), ("x", "y", "z"))


def _split_vs_fused(W, dt, recon32=False, solver="hllc"):
    inv = tuple(float(n) for n in W.shape[1:])
    Wp = jnp.pad(W, [(0, 0)] + [(2, 2)] * 3, mode="wrap")
    f = {s: jax.jit(lambda Wp, s=s: mhp.divergence(Wp, dt, gamma=GAMMA, inv=inv, recon32=recon32,
                                                   split=s, riemann_solver=solver))(Wp)
         for s in (False, True)}
    return np.asarray(f[False]), np.asarray(f[True])


def _same_arithmetic(a, b, mode):
    """Bitwise under Triton; the CPU interpreter contracts FMAs per expression, so the two-pass
    split rounds a few cells differently there (measured 1.1e-14 abs, 0.4 % of values, FP64)."""
    if GPU:
        np.testing.assert_array_equal(a, b)
    else:
        assert _agreement(a, b) < (1e-12 if mode == "fp64" else 1e-6)


# The split path (predictor pass, faces pass) does the fused kernel's arithmetic.
@pytest.mark.parametrize("mode", [
    "fp64", pytest.param("fp32", marks=pytest.mark.precision("fp32")),
    pytest.param("mixed", marks=pytest.mark.precision("mixed"))])
@pytest.mark.parametrize("shock", [False, True])
def test_the_split_path_is_the_fused_kernel(mode, shock):
    n = 16 if not GPU else 64
    a, b = _split_vs_fused(_prims(n, shock=shock), 0.2 / n / 3.0, recon32=mode == "mixed")
    _same_arithmetic(a, b, mode)
    assert abs(_imbalance(a) - _imbalance(b)) < (1e-15 if mode == "fp64" else 1e-8)


def test_the_split_path_on_a_non_cubic_block():
    """Per-axis predictor tiles, the last one shifted back (a shard of a (2,1,1) mesh is not cubic)."""
    rng = np.random.default_rng(3)
    shape = (8, 24, 40) if not GPU else (32, 48, 96)
    rho = 1.0 + 0.5 * rng.random(shape)
    v = [0.3 * (rng.random(shape) - 0.5) for _ in range(3)]
    P = 1.0 + 0.5 * rng.random(shape)
    a, b = _split_vs_fused(jnp.asarray(np.stack([rho, *v, P])), 1e-3)
    _same_arithmetic(a, b, "fp64")


def test_the_split_path_refuses_scalars():
    W = jnp.pad(_prims(16), [(0, 0)] + [(2, 2)] * 3, mode="wrap")
    with pytest.raises(NotImplementedError, match="passive scalars"):
        mhp.divergence(jnp.concatenate([W, W[:1]]), 1e-3, gamma=GAMMA, inv=(16.0,) * 3,
                       n_scalars=1, split=True)


# HLL in the same kernels: the XLA HLL step (fluxes._hll_core) is the reference.
@pytest.mark.parametrize("mode", [
    "fp64", pytest.param("fp32", marks=pytest.mark.precision("fp32")),
    pytest.param("mixed", marks=pytest.mark.precision("mixed"))])
def test_the_hll_kernel_is_the_xla_hll_divergence(mode):
    n = 16 if not GPU else 64
    W, dt = _prims(n, shock=True), 0.2 / n / 3.0
    a, b = _both(W, dt, recon32=mode == "mixed", solver="hll")
    assert _agreement(a, b) < (1e-12 if mode == "fp64" else 1e-6)
    # FP32 round-off; mixed in the CPU interpreter as in the HLLC recon32 test above
    bound = 5e-8 if mode == "fp32" else (3e-9 if mode == "mixed" and not GPU else 1e-13)
    assert _imbalance(b) < bound
    _, c = _both(W, dt, recon32=mode == "mixed")
    assert _agreement(c, b) > 1e-6                    # it is HLL, not HLLC
    f, s = _split_vs_fused(W, dt, recon32=mode == "mixed", solver="hll")
    _same_arithmetic(f, s, mode)


def test_hll_with_passive_scalars_is_refused():
    W = jnp.pad(_prims(16), [(0, 0)] + [(2, 2)] * 3, mode="wrap")
    with pytest.raises(NotImplementedError, match="HLLC only"):
        mhp.divergence(jnp.concatenate([W, W[:1]]), 1e-3, gamma=GAMMA, inv=(16.0,) * 3,
                       n_scalars=1, riemann_solver="hll")


# (agreement, imbalance) bounds per mode for three production steps.
_STEP_BOUNDS = {"fp64": (1e-11, 1e-12), "fp32": (1e-5, 1e-6), "mixed": (1e-5, 1e-12)}


@pytest.mark.skipif(not GPU, reason="the production step is a shard_map: GPU")
@pytest.mark.parametrize("ndev,mode", [
    (1, "fp64"), (2, "fp64"), (1, "fp64-hll"), (2, "fp64-hll"),
    pytest.param(1, "fp32", marks=pytest.mark.precision("fp32")),
    pytest.param(2, "fp32", marks=pytest.mark.precision("fp32")),
    pytest.param(1, "mixed", marks=pytest.mark.precision("mixed")),
    pytest.param(2, "mixed", marks=pytest.mark.precision("mixed"))])
def test_the_production_step_on_gpus(ndev, mode):
    if jax.device_count() < ndev:
        pytest.skip(f"needs {ndev} GPUs")
    from jax.sharding import NamedSharding, PartitionSpec as P
    n = 64
    mesh = _mesh(ndev)
    U0 = jax.device_put(prim_to_cons(_prims(n), GAMMA), NamedSharding(mesh, P(None, "x", "y", "z")))
    dt = 0.2 / n / 3.0
    out = {}
    for kernel in ("xla", "pallas"):
        step = _stepper(mesh, kernel, n, recon32=mode == "mixed",
                        solver="hll" if mode.endswith("hll") else "hllc")
        U = U0
        for _ in range(3):
            U = step(U, dt)
        out[kernel] = np.asarray(U)
    a, b = out["xla"], out["pallas"]
    dU = (b - np.asarray(U0))
    agree, imb = _STEP_BOUNDS[mode.removesuffix("-hll")]
    assert float(np.max(np.abs(a - b)) / np.max(np.abs(a - np.asarray(U0)))) < agree
    assert _imbalance(dU) < imb            # three steps, the seam included on 2 GPUs
    assert np.all(np.asarray(cons_to_prim(jnp.asarray(b), GAMMA))[4] > 0.0)


@pytest.mark.skipif(not GPU, reason="the Triton lowering is the GPU path")
def test_a_weakly_typed_state_compiles():
    """A state built from Python numbers is weakly typed; Pallas's Triton
    lowering turned its tiles into FP32 beside FP64 ones and refused to
    compile (the Sedov runner on one GPU). The kernel makes its input strong."""
    from jax._src.lax.lax import _convert_element_type
    n = 64
    W = _prims(n)
    Wp = jnp.pad(W, [(0, 0)] + [(2, 2)] * 3, mode="wrap")
    dt = 0.2 / n / 3.0

    weak = _convert_element_type(Wp, jnp.dtype("float64"), weak_type=True)
    assert jax.typeof(weak).weak_type
    a = np.asarray(jax.jit(lambda Wp: mhp.divergence(Wp, dt, gamma=GAMMA, inv=(float(n),) * 3))(weak))
    b = np.asarray(jax.jit(lambda Wp: mhp.divergence(Wp, dt, gamma=GAMMA, inv=(float(n),) * 3))(Wp))
    np.testing.assert_array_equal(a, b)


# ---- every dimension, limiter, solver and precision (2026-10-12) ----------------------------
# The XLA reference is the wide step of muscl_hancock_divergence (1-D/2-D face fluxes, 3-D
# _muscl_hancock_3d); limiters are slopes.slope_limiter and the jacobian rows are jacobian.py's
# in both. recon32 is 3-D only, in XLA and Pallas alike.
LIMITERS = {0: "minmod", 1: "mc", 2: "vanleer"}
_SHAPES = {1: (64,), 2: (16, 24), 3: (8, 12, 16)} if not GPU else \
    {1: (4096,), 2: (128, 192), 3: (32, 48, 64)}


def _prims_nd(shape, seed=0, shock=True):
    """Periodic, every axis varying, with a density jump of 50 across half the box."""
    rng = np.random.default_rng(seed)
    nd = len(shape)
    xs = np.meshgrid(*[(np.arange(n) + 0.5) / n for n in shape], indexing="ij")
    k = 2.0 * np.pi
    rho = 1.0 + 0.3 * np.prod([np.sin(k * x + i) for i, x in enumerate(xs)], axis=0)
    if shock:
        rho = rho * np.where(xs[0] < 0.5, 1.0, 50.0)
    v = [0.4 * np.cos(k * xs[(d + 1) % nd]) * np.sin(k * xs[d]) for d in range(nd)]
    P = 1.0 + 0.2 * np.cos(k * xs[-1]) + 0.05 * rng.random(shape)
    return jnp.asarray(np.stack([rho, *v, P]))


def _inv(shape):
    return tuple(float(n) for n in shape) + (1.0,) * (3 - len(shape))


def _xla_nd(Wp, dt, limiter, solver, recon32=False):
    from luna.hydro.muscl_hancock import muscl_hancock_divergence
    nd = Wp.ndim - 1
    inv = _inv(tuple(n - 4 for n in Wp.shape[1:]))
    return np.asarray(jax.jit(lambda Wp: muscl_hancock_divergence(
        Wp, dt, gamma=GAMMA, limiter=limiter, riemann_solver=solver, spatial_dim=nd,
        inv_dx=inv[0], inv_dy=inv[1], inv_dz=inv[2], boundary=0, order=1, wide=True,
        recon32=recon32))(Wp))


def _pallas_nd(Wp, dt, limiter, solver, recon32=False, split=None, n_scalars=0):
    inv = _inv(tuple(n - 4 for n in Wp.shape[1:]))
    return np.asarray(jax.jit(lambda Wp: mhp.divergence(
        Wp, dt, gamma=GAMMA, inv=inv, recon32=recon32, split=split, riemann_solver=solver,
        limiter=limiter, n_scalars=n_scalars))(Wp))


@pytest.mark.parametrize("nd", [1, 2])
def test_low_dim_passive_scalars(nd):
    """A scalar fraction of 1 advects as the density, bitwise; the gas rows do not see it."""
    shape = _SHAPES[nd]
    W = _prims_nd(shape)
    Wp = jnp.pad(jnp.concatenate([W, jnp.ones_like(W[:1])]), [(0, 0)] + [(2, 2)] * nd,
                 mode="wrap")
    dt = 0.2 / max(shape) / 3.0
    d = _pallas_nd(Wp, dt, 1, "hllc", split=False, n_scalars=1)
    g = _pallas_nd(Wp[:-1], dt, 1, "hllc", split=False)
    np.testing.assert_array_equal(d[-1], d[0])
    _same_arithmetic(d[:-1], g, "fp64")


def test_recon32_outside_3d_is_refused():
    Wp = jnp.pad(_prims_nd(_SHAPES[2]), [(0, 0), (2, 2), (2, 2)], mode="wrap")
    with pytest.raises(NotImplementedError, match="recon32 in 2-D"):
        mhp.divergence(Wp, 1e-3, gamma=GAMMA, inv=(1.0,) * 3, recon32=True)


def test_an_unknown_limiter_is_refused():
    from luna.hydro.slopes import SmoothTangentMinmod
    Wp = jnp.pad(_prims_nd(_SHAPES[1]), [(0, 0), (2, 2)], mode="wrap")
    for lim in (3, SmoothTangentMinmod(1e-3)):
        with pytest.raises(NotImplementedError, match="limiter"):
            mhp.divergence(Wp, 1e-3, gamma=GAMMA, inv=(1.0,) * 3, limiter=lim)


# ---- the production step: which path runs, and what it refuses -------------------------------
def _mesh_nd(shape):
    from jax.sharding import Mesh
    n = int(np.prod(shape))
    return Mesh(np.array(jax.devices()[:n]).reshape(shape), ("x", "y", "z")[:len(shape)])


def _step_nd(mesh, nd, n, kernel, limiter=0, solver="hllc", boundary=0, order=1, fused=False,
             recon32=False, autodiff=False, n_scalars=0):
    from luna.time_integration.integrators import create_muscl_hancock_modular_step
    from luna.utils.halo_exchange_shardmap import make_halo_shardmap
    dx = 1.0 / n
    return create_muscl_hancock_modular_step(
        make_halo_shardmap(mesh, nd), GAMMA, order, limiter, dx, dx, dx, 1 / dx, 1 / dx, 1 / dx,
        boundary, nd, 0.4, riemann_solver=solver, use_shardmap=True, mesh=mesh, kernel=kernel,
        fused=fused, recon32=recon32, autodiff=autodiff, n_scalars=n_scalars)[1]


def _state_nd(mesh, nd, n):
    from jax.sharding import NamedSharding, PartitionSpec as P
    U = prim_to_cons(_prims_nd((n,) * nd), GAMMA)
    return jax.device_put(U, NamedSharding(mesh, P(None, *("x", "y", "z")[:nd])))


def _pallas_calls(step, *args):
    return str(jax.make_jaxpr(step)(*args)).count("pallas_call")


@pytest.mark.parametrize("nd", [1, 2, 3])
@pytest.mark.parametrize("limiter", [0, 1, 2])
def test_pallas_is_the_kernel_that_runs(nd, limiter):
    """hydro_kernel='pallas' traces Pallas kernels into the step, 'xla' none (the A10 check:
    2-D and MC/van Leer ran XLA silently before)."""
    mesh = _mesh_nd((1,) * nd)
    U = _state_nd(mesh, nd, 16)
    assert _pallas_calls(_step_nd(mesh, nd, 16, "pallas", limiter), U, 1e-3) >= 1
    assert _pallas_calls(_step_nd(mesh, nd, 16, "xla", limiter), U, 1e-3) == 0
    auto = _pallas_calls(_step_nd(mesh, nd, 16, "auto", limiter), U, 1e-3)
    assert (auto >= 1) == GPU


# Each, with hydro_kernel='pallas': (step kwargs, the words of the refusal).
_REFUSED = [
    (dict(nd=3, order=0), "ORDER=0"),
    (dict(nd=2, order=0), "ORDER=0"),
    (dict(nd=3, fused=True), "hydro_fused=True"),
    (dict(nd=3, autodiff=True), "autodiff_mode"),
    (dict(nd=1, autodiff=True), "autodiff_mode"),
    (dict(nd=2, recon32=True), "hydro_recon32 in 2-D"),
    (dict(nd=1, recon32=True), "hydro_recon32 in 1-D"),
    (dict(nd=1, boundary=4), "BOUNDARY=4 in 1-D"),
    (dict(nd=2, boundary=4), "BOUNDARY=4 in 2-D"),
    (dict(nd=3, limiter=None), "limiter"),
    (dict(nd=3, solver="hll", n_scalars=1), "passive scalars with HLL"),
]


@pytest.mark.parametrize("kw,words", _REFUSED, ids=[w for _, w in _REFUSED])
def test_pallas_refuses_what_it_cannot_run(kw, words):
    """'pallas' never falls back to XLA; 'auto' takes XLA there quietly."""
    import re
    from luna.hydro.slopes import SmoothTangentMinmod
    kw = dict(kw)
    nd = kw.pop("nd")
    if "limiter" in kw:
        kw["limiter"] = SmoothTangentMinmod(1e-3)
    mesh = _mesh_nd((1,) * nd)
    with pytest.raises(NotImplementedError, match=re.escape(words) + ".*'xla'"):
        _step_nd(mesh, nd, 16, "pallas", **kw)
    if kw.get("n_scalars"):
        return                            # scalars have no XLA step at all
    step = _step_nd(mesh, nd, 16, "auto", **kw)
    assert _pallas_calls(step, _state_nd(mesh, nd, 16), 1e-3) == 0


@pytest.mark.skipif(not GPU, reason="the production step is a shard_map: GPU")
@pytest.mark.parametrize("nd,mesh_shape,limiter,boundary", [
    (2, (2, 1), 1, 0), (2, (1, 2), 2, 1), (2, (2, 1), 0, 2),
    (3, (2, 1, 1), 1, 0), (3, (1, 2, 1), 2, 1), (3, (1, 1, 2), 1, 2),
    (1, (2,), 2, 0)])
def test_sharded_pallas_is_one_device(nd, mesh_shape, limiter, boundary):
    """N shards equal one device, bitwise (tests/test_seam_consistency.py:117), every dimension."""
    n_dev = int(np.prod(mesh_shape))
    if jax.device_count() < n_dev:
        pytest.skip(f"needs {n_dev} GPUs")
    n = {1: 512, 2: 64, 3: 32}[nd]
    out = []
    for shape in ((1,) * nd, mesh_shape):
        mesh = _mesh_nd(shape)
        step = _step_nd(mesh, nd, n, "pallas", limiter, boundary=boundary)
        U = _state_nd(mesh, nd, n)
        for _ in range(3):
            U = step(U, 0.2 / n / 3.0)
        out.append(np.asarray(U))
    np.testing.assert_array_equal(out[0], out[1])


@pytest.mark.skipif(not GPU, reason="the production step is a shard_map: GPU")
@pytest.mark.parametrize("nd", [1, 2])
@pytest.mark.parametrize("limiter", [0, 1, 2])
def test_low_dim_production_step_on_a_gpu(nd, limiter):
    """Three production steps, Pallas against XLA, zero-gradient walls."""
    n = {1: 1024, 2: 128}[nd]
    mesh = _mesh_nd((1,) * nd)
    U0 = _state_nd(mesh, nd, n)
    out = {}
    for kernel in ("xla", "pallas"):
        step = _step_nd(mesh, nd, n, kernel, limiter, boundary=1)
        U = U0
        for _ in range(3):
            U = step(U, 0.2 / n / 3.0)
        out[kernel] = np.asarray(U)
    U0 = np.asarray(U0)
    assert _agreement(out["xla"] - U0, out["pallas"] - U0) < 1e-11
