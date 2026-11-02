"""Fast unit tests for LUNA's core numerical kernels.

These test individual pure functions in isolation — no simulations, no
shard_map JIT of full steps — so the whole file runs in well under a second
and forms the `pytest -m "not slow"` smoke tier. Tolerances are set for the
default float32 backend (the suite does not force x64).

Covered:
  - cons<->prim round-trip identity (1D/2D/3D)
  - slope limiters: minmod/moncen/vanleer properties + dispatch
  - HLLC flux consistency: F(W, W) == the analytic Euler flux
  - compute_dt == CFL * dx / (|v| + c_s)
  - shard_map periodic halo wraps correctly (1-device mesh)
"""
import numpy as np
import jax
import jax.numpy as jnp

from luna.hydro.conversions import cons_to_prim, prim_to_cons
from luna.hydro.slopes import minmod, moncen, vanleer, slope_limiter
from luna.hydro.fluxes import hllc_flux_primitive_1d

GAMMA = 5.0 / 3.0


# --------------------------------------------------------------------------
# cons <-> prim
# --------------------------------------------------------------------------
def _random_prim(ndim, n=8, seed=0):
    """Physical primitive state of correct rank: shape (Nvar, *[n]*ndim) with
    Nvar = ndim + 2 ([rho, v_1..v_ndim, P]); rho>0, P>0, |v| order unity. The
    conversion functions infer the spatial dimension from the array rank."""
    rng = np.random.default_rng(seed)
    shape = (n,) * ndim
    rho = rng.uniform(0.3, 3.0, shape).astype(np.float32)
    vels = [rng.uniform(-2.0, 2.0, shape).astype(np.float32) for _ in range(ndim)]
    P = rng.uniform(0.2, 5.0, shape).astype(np.float32)
    return jnp.asarray(np.stack([rho, *vels, P]))


def test_cons_prim_roundtrip():
    """prim -> cons -> prim must return the original primitive state."""
    for ndim in (1, 2, 3):
        W = _random_prim(ndim)
        W_rt = cons_to_prim(prim_to_cons(W, GAMMA), GAMMA)
        assert jnp.allclose(W, W_rt, rtol=1e-5, atol=1e-5), f"round-trip failed in {ndim}D"


def test_prim_cons_energy_definition():
    """Conservative energy = P/(gamma-1) + 1/2 rho v^2 (3D)."""
    W = _random_prim(3)
    U = prim_to_cons(W, GAMMA)
    rho, vx, vy, vz, P = W
    E_expected = P / (GAMMA - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz)
    assert jnp.allclose(U[4], E_expected, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------
# slope limiters
# --------------------------------------------------------------------------
def test_minmod_properties():
    a = jnp.asarray([2.0, -2.0, 2.0, -3.0, 5.0, 0.0])
    b = jnp.asarray([3.0, -3.0, -3.0, 2.0, 5.0, 1.0])
    out = np.asarray(minmod(a, b))
    # opposite signs -> 0
    assert out[2] == 0.0 and out[3] == 0.0
    # same sign -> sign * min(|a|,|b|)
    assert np.isclose(out[0], 2.0) and np.isclose(out[1], -2.0)
    assert np.isclose(out[4], 5.0)
    # bounded by both arguments everywhere
    assert np.all(np.abs(out) <= np.minimum(np.abs(a), np.abs(b)) + 1e-6)


def test_limiters_zero_at_extremum():
    """At a local extremum (opposite-sign slopes) every limiter returns 0,
    which is what guarantees monotonicity (TVD)."""
    dl = jnp.asarray([1.0, -2.0, 3.0])
    dr = jnp.asarray([-1.0, 2.0, -0.5])  # opposite sign to dl
    for lim in (minmod, moncen, vanleer):
        assert jnp.allclose(lim(dl, dr), 0.0), f"{lim.__name__} nonzero at extremum"


def test_vanleer_smooth_value():
    """Van Leer harmonic mean for equal slopes returns the slope itself."""
    s = jnp.asarray([2.0, 0.5, 1.0])
    assert jnp.allclose(vanleer(s, s), s, rtol=1e-6)


def test_slope_limiter_dispatch():
    dl = jnp.asarray([2.0, -1.5, 0.7])
    dr = jnp.asarray([3.0, -2.5, 0.2])
    assert jnp.allclose(slope_limiter(dl, dr, 0), minmod(dl, dr))
    assert jnp.allclose(slope_limiter(dl, dr, 1), moncen(dl, dr))
    assert jnp.allclose(slope_limiter(dl, dr, 2), vanleer(dl, dr))


# --------------------------------------------------------------------------
# HLLC flux consistency
# --------------------------------------------------------------------------
def test_hllc_consistency_no_jump():
    """With no discontinuity (W_L == W_R == W) the HLLC flux must reduce to
    the exact analytic Euler flux F(W)."""
    W = _random_prim(1)               # (3, n): [rho, vx, P]
    F = np.asarray(hllc_flux_primitive_1d(W, W, 0, GAMMA))
    rho, vx, P = np.asarray(W)
    E = P / (GAMMA - 1.0) + 0.5 * rho * vx * vx
    F_exact = np.stack([rho * vx, rho * vx * vx + P, vx * (E + P)])
    assert np.allclose(F, F_exact, rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------
# compute_dt
# --------------------------------------------------------------------------
def test_compute_dt_cfl():
    """compute_dt on a uniform 1D state equals CFL * dx / (|v| + c_s)."""
    from luna.time_integration.integrators import _make_compute_dt_fn

    rho, vx, P = 1.5, 0.7, 2.0
    CFL, dx = 0.4, 1.0 / 64
    W = jnp.asarray(np.stack([np.full(32, rho), np.full(32, vx), np.full(32, P)],
                             dtype=np.float32))
    U = prim_to_cons(W, GAMMA)
    dt = float(_make_compute_dt_fn(GAMMA, CFL, 1, dx, dx, dx)(U))
    cs = np.sqrt(GAMMA * P / rho)
    dt_expected = CFL * dx / (abs(vx) + cs)
    assert np.isclose(dt, dt_expected, rtol=1e-5)


# --------------------------------------------------------------------------
# shard_map periodic halo
# --------------------------------------------------------------------------
def test_halo_periodic_1d_wraps():
    """The shard_map halo on a 1-device mesh adds 1 periodic ghost each side:
    left ghost = last interior cell, right ghost = first interior cell."""
    from jax.sharding import Mesh, PartitionSpec as P
    from luna.utils.shmap import shard_map
    from luna.utils.halo_exchange_shardmap import make_halo_shardmap

    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape((1,)), ("x",))
    halo = make_halo_shardmap(mesh, 1)
    U = jnp.asarray(np.arange(3 * 4, dtype=np.float32).reshape(3, 4))  # (Nvar, Nx)
    padded = jax.jit(shard_map(
        lambda u: halo(u, 0, 1), mesh=mesh,
        in_specs=P(None, "x"), out_specs=P(None, "x")))(U)
    padded = np.asarray(padded)
    assert padded.shape == (3, 6)
    assert np.allclose(padded[:, 1:-1], np.asarray(U))   # interior unchanged
    assert np.allclose(padded[:, 0], np.asarray(U)[:, -1])   # left ghost = last cell
    assert np.allclose(padded[:, -1], np.asarray(U)[:, 0])   # right ghost = first cell
