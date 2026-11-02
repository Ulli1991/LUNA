"""luna/autodiff/stability.py against exact answers.

  1. the Chu-norm maps are inverse to each other and give the Chu energy
  2. subspace iteration on B^T B returns the top singular value of the dense
     matrix (and a vector that attains it), forward iteration its top
     |eigenvalue|, with one vector and with a block of four
  3. on a sharded mesh the singular value of the hydro rollout is the one
     device answer (the psum'd norms, the sharded random start)

  4. minmod's JVP is the derivative at a generic state and not at a flat one
  5. the opt-in smooth tangent: forward bitwise minmod, the central slope at
     the crease, minmod's derivative as e -> 0, its VJP its transpose

The physics validation -- Kelvin-Helmholtz against linear theory -- is
tools/kh_stability.py and docs/AUTODIFF.md, too long for the suite.
"""
import numpy as np
import pytest

import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402

from luna.autodiff import stability as st                      # noqa: E402
from luna.hydro.conversions import cons_to_prim               # noqa: E402
from luna.hydro.slopes import (SmoothTangentMinmod, minmod,     # noqa: E402
                               minmod_smooth_tangent)
from tests.test_autodiff_identities import GAMMA, N, _smooth_state   # noqa: E402
from tests.test_autodiff_sharded import (_mesh, _put, _rollout,     # noqa: E402
                                         generic_state)

pytestmark = pytest.mark.fast


def test_chu_maps_are_inverse_and_give_the_chu_energy():
    U = _smooth_state()
    dU = 1e-3 * st.random_like(U, 2)
    to_z, from_z = st.chu_norm(U, GAMMA)
    z = to_z(dU)
    back = float(jnp.max(jnp.abs(from_z(z) - dU)) / jnp.max(jnp.abs(dU)))
    assert back < 1e-13, f"from_z(to_z(dU)) != dU: {back:.2e}"
    # the energy from the primitive perturbation, independently
    W = cons_to_prim(U, GAMMA)
    dW = jax.jvp(lambda U: cons_to_prim(U, GAMMA), (U,), (dU,))[1]
    rho, p, drho, dp = W[0], W[-1], dW[0], dW[-1]
    chu2 = jnp.sum(rho * jnp.sum(dW[1:-1] ** 2, axis=0) + p * (drho / rho) ** 2
                   + p / (GAMMA - 1.0) * (dp / p - drho / rho) ** 2)
    rel = abs(float(jnp.sum(z * z) / chu2) - 1.0)
    assert rel < 1e-13, f"|z|^2 is not 2 x Chu energy: {rel:.2e}"


def _linear_prop(U0, seed=0):
    """prop(U) = U0 + A (U - U0) with a designed spectrum (a real dominant
    eigenvalue 3, then 2), and B = to_z A from_z densely."""
    n = U0.size
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.normal(size=(n, n)))[0]
    lam = np.concatenate([[3.0, 2.0], rng.uniform(-1.0, 1.0, n - 2)])
    A = jnp.asarray((Q * lam) @ Q.T + 0.3 * np.triu(rng.normal(size=(n, n)), 1) / np.sqrt(n))
    prop = lambda U: U0 + (A @ (U - U0).ravel()).reshape(U0.shape)
    to_z, from_z = st.chu_norm(U0, GAMMA)
    Bf = lambda z: to_z(prop(U0 + from_z(z.reshape(U0.shape)))).ravel()
    Bd = np.asarray(jax.jacfwd(Bf)(jnp.zeros(n)))
    return prop, Bd


def test_power_iterations_recover_the_dense_spectrum():
    x = (np.arange(4) + 0.5) / 4
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    rho = 1.0 + 0.2 * np.sin(2 * np.pi * X)
    U0 = jnp.asarray(np.stack([rho, 0.1 * rho * np.cos(2 * np.pi * Y), 0 * X, 0 * X,
                               1.5 + 0.1 * np.sin(2 * np.pi * Z)]))
    prop, Bd = _linear_prop(U0)
    s_dense = np.linalg.svd(Bd, compute_uv=False)
    lam_dense = np.max(np.abs(np.linalg.eigvals(Bd)))
    quiet = lambda s: None
    for n_vec in (1, 4):
        svd = st.optimal_perturbation(prop, U0, GAMMA, n_iter=400, rtol=1e-14,
                                      n_vec=n_vec, log=quiet)
        eig = st.leading_mode(prop, U0, GAMMA, n_iter=400, rtol=1e-14, n_vec=n_vec, log=quiet)
        assert abs(svd["s"] / s_dense[0] - 1) < 1e-10, (n_vec, svd["s"], s_dense[:3])
        assert abs(eig["lam"] / lam_dense - 1) < 1e-10, (n_vec, eig["lam"], lam_dense)
        # the returned input vector achieves the gain, under the DENSE matrix
        gain = np.linalg.norm(Bd @ np.asarray(svd["z_in"]).ravel())
        assert abs(gain / s_dense[0] - 1) < 1e-8, (n_vec, gain, s_dense[0])


def test_sharded_singular_value_is_the_one_device_one():
    """Hydro rollout, 10 power iterations from the same random start: the
    whole history must agree between (1,1,1) and (2,1,1)."""
    hist = {}
    for shape in ((1, 1, 1), (2, 1, 1)):
        mesh = _mesh(shape)
        U0 = _put(_smooth_state(), mesh)
        r = st.optimal_perturbation(_rollout(mesh), U0, GAMMA, n_iter=10, rtol=0.0,
                                    n_vec=2, log=lambda s: None)
        hist[shape] = np.array(r["history"])
    rel = float(np.max(np.abs(hist[(2, 1, 1)] / hist[(1, 1, 1)] - 1)))
    assert rel < 1e-12, f"2-device power iteration drifts from 1 device by {rel:.2e}"


def _shear_base(drift=0.0):
    """x-uniform shear: every x-slope, and every y-slope but u's, is exactly 0.
    `drift` moves it in y, off HLLC's contact branch point S_M = 0."""
    y = (np.arange(N) + 0.5) / N
    u = np.broadcast_to(0.5 * np.tanh((y - 0.5) / 0.1)[None, :, None], (N, N, N))
    rho, p, vy = np.ones_like(u), np.full_like(u, 2.0), np.full_like(u, drift)
    E = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + vy * vy)
    return jnp.asarray(np.stack([rho, rho * u, rho * vy, 0 * u, E]))


def _jvp_vs_difference(step, U, v, h):
    """|(S(U + h v) - S(U))/h - J v| / |J v|."""
    Jv = jax.jit(lambda U, v: jax.jvp(step, (U,), (v,))[1])(U, v)
    fd = (step(U + h * v) - step(U)) / h
    return float(jnp.linalg.norm(fd - Jv) / jnp.linalg.norm(Jv))


def test_minmod_tangent_is_the_derivative_only_off_the_crease():
    """4-step minmod rollout. At a generic state the JVP IS the derivative:
    a one-sided difference converges to it (MEASURED 4.8e-4, 1.2e-8 at
    h = 1e-6, 1e-8). At the flat shear it is not: minmod(a, b) at a = b = 0
    has the direction-dependent derivative minmod(da, db) and jnp.where takes
    the 0 branch, so the difference stays O(1e-2) at every h (MEASURED
    5.2e-2 at 1e-4, 1e-6, 1e-8). That flat state is why the KH study needs
    one of the two remedies in docs/AUTODIFF.md."""
    mesh = _mesh((1, 1, 1))
    run = _rollout(mesh)
    U = generic_state()
    v = st.random_like(U, 5)
    err = _jvp_vs_difference(run, U, v, 1e-8)
    assert err < 1e-6, f"generic state: JVP vs difference {err:.2e}"
    flat = [_jvp_vs_difference(run, _shear_base(), v, h) for h in (1e-6, 1e-8)]
    assert min(flat) > 1e-2, (f"flat shear: JVP matches the difference ({flat}); if minmod's "
                              "derivative at a = b = 0 changed, revisit docs/AUTODIFF.md")


# ---------------------------------------------------------------------------
# minmod with a smooth tangent (Config.limiter_tangent_eps)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 1, 1)])
def test_smooth_tangent_leaves_the_forward_bitwise_minmod(shape):
    mesh = _mesh(shape)
    for U in (generic_state(), _shear_base()):
        U = _put(U, mesh)
        a = np.asarray(jax.jit(_rollout(mesh))(U))
        b = np.asarray(jax.jit(_rollout(mesh, limiter=SmoothTangentMinmod(1e-3)))(U))
        assert np.array_equal(a, b), f"mesh {shape}: forward changed by {np.max(np.abs(a - b)):.1e}"


def test_smooth_tangent_is_the_central_slope_at_the_crease_and_minmod_off_it():
    """Unit: at a = b = 0 the tangent is (da + db)/2 for any e; with |a|,
    |b|, ||a|-|b|| >> e it is minmod's, to O((e/a)^2)."""
    rng = np.random.default_rng(3)
    da, db = jnp.asarray(rng.normal(size=100)), jnp.asarray(rng.normal(size=100))
    e = jnp.full(100, 1e-3)
    z = jnp.zeros(100)
    t = jax.jvp(minmod_smooth_tangent, (z, z, e), (da, db, z))[1]
    assert float(jnp.max(jnp.abs(t - 0.5 * (da + db)))) < 1e-14
    a = jnp.asarray(rng.choice([-1.0, 1.0], 100) * rng.uniform(0.1, 1.0, 100))
    b = a * jnp.asarray(rng.choice([-1.0, 1.0], 100) * rng.uniform(0.2, 0.8, 100))
    t = jax.jvp(minmod_smooth_tangent, (a, b, e), (da, db, z))[1]
    t0 = jax.jvp(minmod, (a, b), (da, db))[1]
    rel = float(jnp.max(jnp.abs(t - t0)))
    assert rel < 1e-3 ** 2 / 0.02 ** 2 * 10, f"away from the crease: {rel:.2e}"


def test_smooth_tangent_tends_to_the_minmod_derivative_at_a_generic_state():
    """|J_e v - J_minmod v| / |J_minmod v| over a 4-step rollout, e = 1e-2 ..
    1e-8: the smoothing is the only difference, so it must shrink with e."""
    mesh = _mesh((1, 1, 1))
    U = generic_state()
    v = st.random_like(U, 5)
    jv = lambda lim: jax.jit(lambda U, v: jax.jvp(_rollout(mesh, limiter=lim), (U,), (v,))[1])(U, v)
    J0 = jv(0)
    err = [float(jnp.linalg.norm(jv(SmoothTangentMinmod(e)) - J0) / jnp.linalg.norm(J0))
           for e in (1e-2, 1e-4, 1e-6, 1e-8)]
    assert all(b < 0.5 * a for a, b in zip(err, err[1:])), f"not shrinking with e: {err}"
    assert err[-1] < 1e-6, f"e = 1e-8 still {err[-1]:.2e} from minmod's derivative: {err}"


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 1, 1)])
def test_smooth_tangent_adjoint_is_its_transpose(shape):
    """The VJP comes from transposing the custom JVP rule; check it, sharded."""
    from tests.test_autodiff_identities import TOL, _dot_product_test
    mesh = _mesh(shape)
    for U in (generic_state(), _shear_base()):
        rel = _dot_product_test(_rollout(mesh, limiter=SmoothTangentMinmod(1e-3)), _put(U, mesh))
        assert rel < TOL, f"mesh {shape}: <Jv,w> vs <v,J^T w> disagree by {rel:.2e}"


def test_smooth_tangent_is_opt_in_and_autodiff_only():
    from dataclasses import replace
    from luna.template_config import Config
    for bad in (dict(limiter_tangent_eps=1e-3),
                dict(limiter_tangent_eps=1e-3, autodiff_mode=True, LIMITER=1)):
        with pytest.raises(Exception) as e:
            replace(Config(), **bad).validate()
        assert "limiter_tangent_eps" in str(e.value)
    replace(Config(), limiter_tangent_eps=1e-3, autodiff_mode=True).validate()
