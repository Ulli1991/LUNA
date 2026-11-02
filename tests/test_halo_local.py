"""The single-device halo (utils/halo_local.py) and the mesh-free hydro step against the shard_map
path on a one-device mesh: bitwise, for every boundary, halo width and dimension."""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jax.sharding import Mesh, PartitionSpec as P

from luna.hydro.conversions import prim_to_cons
from luna.time_integration.integrators import create_muscl_hancock_modular_step
from luna.utils.halo_exchange_shardmap import make_halo_shardmap
from luna.utils.halo_local import pad_halo_local
from luna.utils.shmap import shard_map


GAMMA = 5.0 / 3.0
GPU = jax.default_backend() in ("gpu", "rocm", "cuda")
NAMES = ("x", "y", "z")
SHAPE = {1: (12,), 2: (10, 8), 3: (8, 6, 4)}


def _mesh(sdim):
    return Mesh(np.asarray(jax.devices()[:1]).reshape((1,) * sdim), NAMES[:sdim])


def _mesh_halo(W, boundary, sdim, width, axes=None):
    spec = P(None, *NAMES[:sdim])
    halo = make_halo_shardmap(_mesh(sdim), sdim)
    return shard_map(lambda w: halo(w, boundary, sdim, width=width, axes=axes), mesh=_mesh(sdim),
                     in_specs=(spec,), out_specs=spec)(W)


def _field(sdim, seed=0):
    """Distinct values everywhere, velocities of both signs, so every ghost and sign is visible."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal((2 + sdim,) + SHAPE[sdim]))


CASES = [(s, b, w) for s in (1, 2, 3) for b in (0, 1, 2, 4) for w in (1, 2)]


@pytest.mark.parametrize("sdim,boundary,width", CASES)
def test_local_halo_is_the_mesh_halo(sdim, boundary, width):
    W = _field(sdim)
    a = np.asarray(pad_halo_local(W, boundary, sdim, width=width))
    b = np.asarray(_mesh_halo(W, boundary, sdim, width))
    assert a.shape == b.shape
    assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("sdim,boundary,width",
                         [(s, b, w) for s in (2, 3) for b in (0, 1) for w in (1, 2)])
def test_per_axis_halo_is_the_mesh_halo(sdim, boundary, width):
    W = _field(sdim, seed=1)
    for axes in [(0,), (sdim - 1,), tuple(range(sdim))]:
        a = np.asarray(pad_halo_local(W, boundary, sdim, width=width, axes=axes))
        b = np.asarray(_mesh_halo(W, boundary, sdim, width, axes=axes))
        assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("sdim", [1, 2, 3])
def test_a_wrong_boundary_is_caught(sdim):
    """Teeth: every boundary pair differs somewhere, so a mixed-up mapping cannot pass the comparison."""
    W = _field(sdim)
    W = W.at[1:1 + sdim].set(jnp.abs(W[1:1 + sdim]))      # inflow at the low faces: the diode acts
    for width in (1, 2):
        for b_local in (0, 1, 2, 4):
            for b_mesh in (0, 1, 2, 4):
                if b_local != b_mesh:
                    a = np.asarray(pad_halo_local(W, b_local, sdim, width=width))
                    b = np.asarray(_mesh_halo(W, b_mesh, sdim, width))
                    assert a.tobytes() != b.tobytes(), (b_local, b_mesh, width)


@pytest.mark.parametrize("sdim", [1, 2, 3])
def test_wide_reflective_ghosts_mirror(sdim):
    """Two reflective ghosts are the edge cells mirrored, normal velocity negated, the rest as is."""
    W = _field(sdim)
    Wp = np.asarray(pad_halo_local(W, 2, sdim, width=2))
    inner = (slice(None),) + (slice(2, -2),) * sdim
    for d in range(sdim):
        ax = 1 + d
        def take(a, i):
            sl = list(inner)
            sl[ax] = i
            return a[tuple(sl)]
        sign = np.ones((2 + sdim,) + (1,) * (sdim - 1))
        sign[1 + d] = -1.0
        for g, c in ((0, 3), (1, 2), (-1, -4), (-2, -3)):   # ghost layer, the interior cell it mirrors
            assert np.array_equal(take(Wp, g), sign * take(Wp, c)), (d, g)


def _state(sdim, kind, n=16):
    """Sod tube along x (with a transverse shear) or a Sedov-like hot spot in a moving medium."""
    shape = (n,) * sdim
    xs = np.meshgrid(*[(np.arange(n) + 0.5) / n for _ in range(sdim)], indexing="ij")
    if kind == "sod":
        rho = np.where(xs[0] < 0.5, 1.0, 0.125)
        Pr = np.where(xs[0] < 0.5, 1.0, 0.1)
    else:
        r2 = sum((x - 0.5) ** 2 for x in xs)
        rho = np.ones(shape)
        Pr = 1e-2 + 50.0 * np.exp(-r2 / 0.01)
    vel = [0.3 * np.sin(2 * np.pi * xs[(d + 1) % sdim]) * (1 if d % 2 == 0 else -1) for d in range(sdim)]
    W = np.stack([rho, *vel, Pr])
    return prim_to_cons(jnp.asarray(W), GAMMA)


def _steps(sdim, boundary, order, fused=False, recon32=False, kernel="xla"):
    n = 16
    dx = 1.0 / n
    args = (GAMMA, order, 0, dx, dx, dx, 1 / dx, 1 / dx, 1 / dx, boundary, sdim, 0.4)
    kw = dict(riemann_solver="hllc", fused=fused, recon32=recon32, kernel=kernel)
    local = create_muscl_hancock_modular_step(None, *args, **kw)
    mesh = _mesh(sdim)
    meshed = create_muscl_hancock_modular_step(make_halo_shardmap(mesh, sdim), *args,
                                               use_shardmap=True, mesh=mesh, **kw)
    return local, meshed


def _run(fns, U, n_steps=3):
    dt_fn, step = fns
    for _ in range(n_steps):
        dt = dt_fn(U)
        U = step(U, dt)
    return np.asarray(U), float(dt)


STEP_CASES = ([(s, b, o, False, False) for s in (1, 2, 3) for b in (0, 1, 2, 4) for o in (0, 1)]
              + [(3, b, 1, True, False) for b in (0, 1)]
              + [(3, b, 1, False, True) for b in (0, 1, 2, 4)])


@pytest.mark.parametrize("kind", ["sod", "sedov"])
@pytest.mark.parametrize("sdim,boundary,order,fused,recon32", STEP_CASES)
def test_mesh_free_step_is_the_mesh_step(kind, sdim, boundary, order, fused, recon32):
    U0 = _state(sdim, kind)
    local, meshed = _steps(sdim, boundary, order, fused=fused, recon32=recon32)
    a, dta = _run(local, U0)
    b, dtb = _run(meshed, U0)
    assert dta == dtb
    assert a.tobytes() == b.tobytes()
    assert not np.array_equal(a, np.asarray(U0))


@pytest.mark.skipif(not GPU, reason="the Pallas kernel under shard_map needs a GPU")
@pytest.mark.parametrize("kind", ["sod", "sedov"])
@pytest.mark.parametrize("boundary", [0, 1])
def test_mesh_free_pallas_step_is_the_mesh_step(kind, boundary):
    U0 = _state(3, kind, n=16)
    local, meshed = _steps(3, boundary, 1, kernel="pallas")
    a, _ = _run(local, U0)
    b, _ = _run(meshed, U0)
    assert a.tobytes() == b.tobytes()
