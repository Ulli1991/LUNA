"""The prototype chemistry coupling: passive scalars in the Pallas hydro, and the network-5 stage.

CPU: the kernel is called directly (the interpreter cannot run in a shard_map); GPU: the production step."""
from dataclasses import replace

import numpy as np
import pytest

import jax
import jax.numpy as jnp


from luna import units as un                                         # noqa: E402
from luna.hydro import muscl_hancock_pallas as mhp                   # noqa: E402

from luna.hydro.conversions import cons_to_prim, prim_to_cons       # noqa: E402
from luna.sources import chemistry as ch                             # noqa: E402
from luna.template_config import Config                              # noqa: E402
from tests._helpers import check                                     # noqa: E402

GAMMA = 5.0 / 3.0
GPU = jax.default_backend() in ("gpu", "rocm", "cuda")
UNITS = un.UnitSystem(UnitLength_in_cm=un.PARSEC_CGS, UnitMass_in_g=un.SOLAR_MASS_CGS,
                      UnitVelocity_in_cm_per_s=1e5)


def _prims(n, seed=0):
    """Periodic, fully 3-D, a density jump of 50 across half the box."""
    rng = np.random.default_rng(seed)
    x = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    k = 2.0 * np.pi
    rho = (1.0 + 0.3 * np.sin(k * X) * np.cos(k * Y) * np.cos(k * Z)) * np.where(X < 0.5, 1.0, 50.0)
    v = [0.4 * np.cos(k * X) * np.sin(k * Y), 0.4 * np.sin(k * Z) * np.cos(k * X),
         0.4 * np.cos(k * Y) * np.sin(k * Z)]
    P = 1.0 + 0.2 * np.cos(k * X) * np.cos(k * Y) * np.sin(k * Z) + 0.05 * rng.random(X.shape)
    return np.stack([rho, *v, P])


def _fractions(n, seed=1):
    """Three scalar fractions: uniform, a sharp step, random cells in [0, 1e-4]."""
    rng = np.random.default_rng(seed)
    x = (np.arange(n) + 0.5) / n
    X, Y, _ = np.meshgrid(x, x, x, indexing="ij")
    step = np.where((Y > 0.25) & (Y < 0.6), 0.4, 1e-3)
    return np.stack([np.full(X.shape, 0.3), step, 1e-4 * rng.random(X.shape)])


def _advect(W, X, dt, n_scalars=3, inv=None):
    """One wrap-padded divergence of the kernel (gas rows, scalar rows); dx = 1/n by default."""
    inv = float(W.shape[1]) if inv is None else float(inv)
    Wp = jnp.pad(jnp.concatenate([W, X])[: 5 + n_scalars], [(0, 0)] + [(2, 2)] * 3, mode="wrap")
    f = jax.jit(lambda Wp: mhp.divergence(Wp, dt, gamma=GAMMA, inv=(inv,) * 3,
                                          n_scalars=n_scalars))
    d = np.asarray(f(Wp))
    return d[:5], d[5:]


def test_scalars_leave_the_gas_rows_bitwise_alone():
    n = 16
    W, X = _prims(n), _fractions(n)
    dt = 0.2 / n / 3.0
    gas0, _ = _advect(W, X, dt, n_scalars=0)
    gas3, _ = _advect(W, X, dt)
    np.testing.assert_array_equal(gas0, gas3)


def test_scalar_fluxes_telescope_and_ride_on_the_mass_flux():
    n = 16
    W, X = _prims(n), _fractions(n)
    dt = 0.2 / n / 3.0
    dU, dS = _advect(W, X, dt)
    rho = W[0]
    S = rho * X
    for k in range(3):
        assert abs(dS[k].sum()) / np.abs(dS[k]).sum() < 1e-13          # face fluxes telescope
    for k in range(5):                                                 # and so do the gas rows (periodic)
        check(f"chemistry advection |sum dU_{k}| / sum |dU_{k}|", abs(dU[k].sum()) / np.abs(dU[k]).sum(), 1e-13)
    rho1 = rho + dt * dU[0]
    x_uniform = (S[0] + dt * dS[0]) / rho1
    assert np.max(np.abs(x_uniform - 0.3)) < 1e-14                     # F_S = F_rho X exactly
    x_step = (S[1] + dt * dS[1]) / rho1
    assert x_step.min() >= 1e-3 * (1 - 1e-12) and x_step.max() <= 0.4 * (1 + 1e-12)


def _ism_box(n, seed=3):
    """Physical units (pc, Msun, km/s): n_H 0.3..300, T 30..1e4 K, Mach ~1 flows, sharp CO edges."""
    from luna.chemistry import tables as tb
    rng = np.random.default_rng(seed)
    x = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    k = 2.0 * np.pi
    lg_n = 0.5 + 2.0 * np.sin(k * X) * np.cos(k * Y) + 0.3 * rng.random(X.shape)
    n_h = 10.0 ** lg_n
    temp = np.clip(1e4 / n_h ** 1.2, 30.0, 1e4)
    rho = n_h * (1.0 + 4.0 * tb.ABHE) * tb.MH / UNITS.UnitDensity_in_cgs
    P = n_h * 1.1 * tb.KBOLTZ * temp / UNITS.UnitPressure_in_cgs
    v = [5.0 * np.sin(k * Y), 5.0 * np.cos(k * Z), 5.0 * np.sin(k * X)]
    W = np.stack([rho, *v, P])
    x_h2 = np.clip(0.5 * (lg_n - 0.5) / 2.5, 1e-6, 0.49)
    x_hp = 1e-2 / n_h
    x_co = np.where(lg_n > 1.5, 1.41e-4, 0.0)
    return W, np.stack([x_h2, x_hp, x_co])


def _elements(rho, x):
    """(H nuclei, C nuclei) totals with HI and C+ derived as the network does."""
    x_hi = np.maximum(1.0 - 2.0 * x[0] - x[1], 0.0)
    x_cp = np.maximum(ch.ABUNDC - x[2], 0.0)
    return np.sum(rho * (x_hi + 2.0 * x[0] + x[1])), np.sum(rho * (x_cp + x[2]))


def test_advection_and_chemistry_conserve_hydrogen_and_carbon():
    from jax.sharding import Mesh
    n = 8
    box = 10.0                                   # pc
    cfg = replace(Config(), units=UNITS, Lx=box, Ly=box, Lz=box, Nx=n, Ny=n, Nz=n,
                  chemistry="net5", gamma=GAMMA)
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1, 1), ("x", "y", "z"))
    consts = ch.ChemConsts(cfg)
    apply = ch.make_apply(consts, mesh)
    W, x = _ism_box(n)
    U = np.asarray(prim_to_cons(jnp.asarray(W), GAMMA))
    S = U[0] * x
    carry = {"S": jnp.asarray(S), "tdust": jnp.full(U.shape[1:], 10.0), "h": jnp.zeros(U.shape[1:])}
    h_tot0, c_tot0 = _elements(U[0], x)
    M0, P0, E0 = U[0].sum(), U[1:4].sum((1, 2, 3)), U[4].sum()
    assert abs(h_tot0 / U[0].sum() - 1.0) < 1e-14
    dx = box / n
    for _ in range(3):
        W = np.asarray(cons_to_prim(jnp.asarray(U), GAMMA))
        cs = np.sqrt(GAMMA * W[4] / W[0])
        dt = 0.3 * dx / np.max(np.abs(W[1:4]) + cs)
        dU, dS = _advect(W, np.asarray(carry["S"]) / U[0], dt, inv=1.0 / dx)
        s_tot = np.asarray(carry["S"]).sum((1, 2, 3))
        U = U + dt * dU
        S = np.asarray(carry["S"]) + dt * dS
        np.testing.assert_allclose(S.sum((1, 2, 3)), s_tot, rtol=1e-13)
        Uc, carry = apply(jnp.asarray(U), {**carry, "S": jnp.asarray(S)}, dt)
        Uc = np.asarray(Uc)
        np.testing.assert_array_equal(Uc[:4], U[:4])                 # chemistry: energy only
        assert not np.any(np.asarray(carry["failed"])) and not np.any(np.asarray(carry["floored"]))
        U = Uc
        x = np.asarray(carry["S"]) / U[0]
        assert np.all(x >= 0.0) and np.all(x[0] <= 0.5) and np.all(x[2] <= ch.ABUNDC)
        assert np.all(2.0 * x[0] + x[1] <= 1.0)
        h_tot, c_tot = _elements(U[0], x)
        assert abs(h_tot / U[0].sum() - 1.0) < 1e-13
        assert abs(c_tot / (ch.ABUNDC * U[0].sum()) - 1.0) < 1e-13
    assert np.all(np.asarray(cons_to_prim(jnp.asarray(U), GAMMA))[4] > 0.0)
    # periodic: mass and momentum to round-off; energy not conserved by design (the network's heating/cooling)
    check("chemistry run |dM|/M", abs(U[0].sum() / M0 - 1.0), 1e-13)
    check("chemistry run |dP|/sqrt(M E)", float(np.abs(U[1:4].sum((1, 2, 3)) - P0).max() / np.sqrt(M0 * E0)), 1e-13)


@pytest.mark.skipif(not GPU, reason="the production step is a shard_map: GPU")
@pytest.mark.parametrize("ndev", [1, 2])
def test_the_production_scalar_step_on_gpus(ndev):
    if jax.device_count() < ndev:
        pytest.skip(f"needs {ndev} GPUs")
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from luna.time_integration.integrators import create_muscl_hancock_modular_step
    from luna.utils.halo_exchange_shardmap import make_halo_shardmap
    n = 64
    mesh = Mesh(np.array(jax.devices()[:ndev]).reshape(ndev, 1, 1), ("x", "y", "z"))
    sh = NamedSharding(mesh, P(None, "x", "y", "z"))
    dx = 1.0 / n
    W, X = _prims(n), _fractions(n)
    U0 = prim_to_cons(jnp.asarray(W), GAMMA)
    S0 = U0[0] * X
    dt = 0.2 / n / 3.0
    mk = lambda ns: create_muscl_hancock_modular_step(  # noqa: E731
        make_halo_shardmap(mesh, 3), GAMMA, 1, 0, dx, dx, dx, 1 / dx, 1 / dx, 1 / dx, 0, 3, 0.4,
        riemann_solver="hllc", use_shardmap=True, mesh=mesh, kernel="pallas", n_scalars=ns)[1]
    U1, S1 = mk(3)(jax.device_put(U0, sh), jax.device_put(S0, sh), dt)
    U_plain = mk(0)(jax.device_put(U0, sh), dt)
    np.testing.assert_array_equal(np.asarray(U1), np.asarray(U_plain))
    dU, dS = _advect(W, X, dt)
    np.testing.assert_allclose(np.asarray(U1), np.asarray(U0) + dt * dU, rtol=0, atol=1e-12 * np.abs(U0).max())
    np.testing.assert_allclose(np.asarray(S1), np.asarray(S0) + dt * dS, rtol=0, atol=1e-12 * np.abs(S0).max())
    for k in range(3):
        assert abs(float(np.sum(np.asarray(S1[k]) - np.asarray(S0[k])))) < 1e-13 * float(np.abs(S0[k]).sum())


def _co_states():
    """Every 5th output of the 168 DVODE trajectories, with their Params (divv = 0)."""
    from luna.ode import rodas3 as ig
    from tools.chem_integrator import validate_net5 as val
    names, cp, t, y0, ref = val.load()
    k = ref.shape[1] // 5
    y = jnp.asarray(ref[:, ::5, :4][:, :k].reshape(-1, 4))
    p = ig.broadcast_params(val.params(cp), cp.shape[0])
    p = jax.tree_util.tree_map(lambda a: jnp.repeat(a, k, axis=0), p)
    return y, p


def _rhs_lvg(y, p, lvg):
    """ydot, the 28 cooling terms and the gross rates (all outputs: the same program up to the lookup)."""
    from luna.chemistry import rates_net5

    def f(yy, pp):
        ydot, aux = rates_net5.rhs(yy, pp, lvg=lvg)
        return jnp.concatenate([ydot, aux["cool"], aux["gross"]])
    return np.asarray(jax.jit(jax.vmap(f))(y, p))


def test_one_co_lookup_without_lvg_is_bitwise():
    y, p = _co_states()
    co_on = np.asarray(y[:, 2]) > 1e-5 * ch.ABUNDO
    assert co_on.sum() > 100                      # CO cooling is exercised
    np.testing.assert_array_equal(_rhs_lvg(y, p, False), _rhs_lvg(y, p, True))


def test_teeth_one_co_lookup_without_lvg(monkeypatch):
    """Dropping the isotopologues in the single-lookup path must show."""
    from luna.chemistry import cooling
    real = cooling._co_terms

    def broken(*a):
        out = real(*a)
        if not a[-1]:
            out[21] = out[22] = 0.0 * out[21]
        return out
    monkeypatch.setattr(cooling, "_co_terms", broken)
    y, p = _co_states()
    assert not np.array_equal(_rhs_lvg(y, p, False), _rhs_lvg(y, p, True))
