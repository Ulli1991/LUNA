"""Exact identities the differentiated solver must satisfy.

These are not "run it and compare to finite differences" checks. Each one is
an identity that holds to round-off for a correct adjoint, so it fails loudly
and cheaply:

  1. the adjoint IS the transpose:   <J v, w> == <v, J^T w>
  2. conservation fixes the gradient: total mass is conserved whatever the
     state, so d(total mass)/d(rho cell) == 1 and d/d(momentum, energy) == 0
  3. a linear operator's Taylor remainder is exactly zero, and the
     nonlinear step's is first order -- measured, and a property of the
     scheme (see below)
  4. the chain rule holds across two steps

They run on one CPU device at 16^3 in float64, seconds each, and they cover
the production MUSCL-Hancock step -- the halo, the limiter, the Riemann
solver -- and the same step with the differentiable FMG self-gravity in it.

Written after a night when the only differentiable results in the code came
from scripts run by hand, and the headline sensitivity task had been
crashing for a day without anything noticing.
"""

import numpy as np
import pytest


import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402

pytestmark = pytest.mark.fast

N = 16
L = 1.0
GAMMA = 5.0 / 3.0
G_SG = 1.0
DT = 1.0e-3            # far below CFL: this is about derivatives, not accuracy
TOL = 1.0e-11          # round-off at 16^3 in float64, not a fitted bound


def _mesh():
    from jax.sharding import Mesh
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1, 1), ("x", "y", "z"))


def _smooth_state():
    """A smooth periodic state: no shock, so the limiter does not switch
    branch under a small perturbation and the derivative is the derivative."""
    x = (np.arange(N) + 0.5) / N
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    rho = 1.0 + 0.2 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    vx = 0.1 * np.sin(2 * np.pi * Z)
    vy = 0.05 * np.cos(2 * np.pi * X)
    vz = np.zeros_like(rho)
    p = 1.0 + 0.1 * np.cos(2 * np.pi * Y)
    E = p / (GAMMA - 1.0) + 0.5 * rho * (vx ** 2 + vy ** 2 + vz ** 2)
    return jnp.asarray(np.stack([rho, rho * vx, rho * vy, rho * vz, E]))


def _hydro_step(mesh, order=1, limiter=0, mass_flux=False):
    """The production step as a function of U alone (dt bound), one device,
    periodic. order=0 is the unlimited first-order scheme; order=1 with a
    limiter is what production runs; mass_flux: (U, face mass fluxes)."""
    from luna.time_integration.integrators import create_muscl_hancock_modular_step
    from luna.utils.halo_exchange_shardmap import make_halo_shardmap
    dx, inv = L / N, N / L
    step = create_muscl_hancock_modular_step(
        make_halo_shardmap(mesh, 3), GAMMA, order, limiter, dx, dx, dx,
        inv, inv, inv, 0, 3, 0.4, riemann_solver="hllc", use_shardmap=True,
        mesh=mesh, mass_flux=mass_flux)[1]
    return lambda U: step(U, DT)


def _gravity_step(mesh, energy="kick"):
    """The same step with the differentiable self-gravity kick around it (gravity_energy kick or flux)."""
    from dataclasses import replace
    from luna.setup.assemble import build_gravity
    from luna.setup.clock import ProperClock
    from luna.setup.mesh import DENSE
    from luna.setup.run_kind import GAS
    from luna.setup.state import State
    from luna.template_config import Config

    cfg = replace(Config(), SELF_GRAVITY=True, SPATIAL_DIMENSION=3,
                  Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, BOUNDARY=0,
                  GRAVITATIONAL_G=G_SG, autodiff_mode=True, gravity_energy=energy)
    grav = build_gravity(cfg, mesh, GAS, DENSE, ProperClock(0.0, 1.0), is_main=False)
    hydro = _hydro_step(mesh, mass_flux=energy == "flux")

    def step(U):
        st = State(U=U, t=jnp.asarray(0.0), a=jnp.asarray(1.0),
                   step=jnp.asarray(0, jnp.int32))
        st = grav.init(st)                  # the opening solve, as the runner does
        st = grav.before(st, 0.5 * DT, DT)
        if energy == "flux":
            U1, F = hydro(st.U)
            st = st.update(U=U1, mass_flux=F)
        else:
            st = st.update(U=hydro(st.U))
        return grav.after(st, 0.5 * DT, DT).U

    return step


def _random_like(U, seed):
    k = jax.random.PRNGKey(seed)
    return jax.random.normal(k, U.shape, dtype=U.dtype)


# ---------------------------------------------------------------------------
# 1. the adjoint is the transpose
# ---------------------------------------------------------------------------

def _dot_product_test(step, U, seed=0):
    """<J v, w> - <v, J^T w>, relative. Zero to round-off for any correct
    reverse-mode rule, whatever the physics in between."""
    v = _random_like(U, seed)
    w = _random_like(U, seed + 1)

    @jax.jit
    def both(U, v, w):
        _, Jv = jax.jvp(step, (U,), (v,))
        _, vjp = jax.vjp(step, U)
        (JTw,) = vjp(w)
        return jnp.vdot(Jv, w), jnp.vdot(v, JTw)

    a, b = both(U, v, w)
    return float(abs(a - b) / max(abs(a), abs(b), 1e-300))


def test_hydro_adjoint_is_the_transpose():
    rel = _dot_product_test(_hydro_step(_mesh()), _smooth_state())
    assert rel < TOL, f"<Jv,w> vs <v,J^T w> disagree by {rel:.2e}"


def test_unlimited_hydro_adjoint_is_the_transpose():
    """The same on the unlimited scheme, so a failure above can be blamed on
    the limiter rather than on the halo or the Riemann solver."""
    rel = _dot_product_test(_hydro_step(_mesh(), order=0), _smooth_state())
    assert rel < TOL, f"unlimited step: transpose mismatch {rel:.2e}"


def test_poisson_custom_vjp_is_the_transpose_of_the_solve():
    """The gravity adjoint, isolated from the kinks.

    The Poisson solve is linear, and LUNA has two differentiable routes
    through it: solve_poisson_fmg_diff, whose reverse rule is one adjoint FMG
    solve, and solve_poisson_fmg_bidiff, which goes through
    custom_linear_solve and supports forward mode. So the transpose identity
    can be tested exactly -- tangent from one, cotangent from the other:

        <J v, w> == <v, J^T w>

    Periodic AND Neumann. The Neumann case is the open item from audit
    sec.26: the adjoint gate there is off by 2.8e-3 and it is neither the
    stack nor the null-space gauge. If the two routes disagree here, the
    custom VJP is not the transpose of the solve it claims to invert; if they
    agree, the discrepancy lives in the operator itself and this test says so.
    Nothing in the surveyed literature cross-checks a custom VJP against an
    independent rule for the same solve.
    """
    from luna.gravity.poisson_diff import (solve_poisson_fmg_bidiff,
                                           solve_poisson_fmg_diff)
    rng = np.random.default_rng(5)
    dx = L / N

    def source(seed):
        r = jnp.asarray(rng.normal(size=(N, N, N)))
        return r - jnp.mean(r)          # solvable for both singular BCs

    rho, v, w = source(0), source(1), source(2)
    out = {}
    for bc, name in ((0, "periodic"), (1, "Neumann")):
        fwd = jax.jit(lambda r, bc=bc: solve_poisson_fmg_bidiff(r, dx, 1.0, bc))
        rev = jax.jit(lambda r, bc=bc: solve_poisson_fmg_diff(r, dx, 1.0, bc))
        _, Jv = jax.jvp(fwd, (rho,), (v,))
        (JTw,) = jax.vjp(rev, rho)[1](w)
        a, b = float(jnp.vdot(Jv, w)), float(jnp.vdot(v, JTw))
        out[name] = abs(a - b) / max(abs(a), abs(b), 1e-300)
    # MEASURED 2026-09-21: periodic 3.5e-16 -- the reverse-only rule is the
    # exact transpose there, to round-off. Neumann 1.5e-6: ten orders worse,
    # small in absolute terms but a real asymmetry, and the same boundary
    # whose adjoint gate sits at 2.8e-3 (audit sec.26). So the 2.8e-3 is NOT
    # a transpose error; the two rules agree far better than that. What is
    # left is the discrete Neumann operator itself at the wall.
    assert out["periodic"] < 1e-12, f"periodic transpose mismatch {out['periodic']:.2e}"
    assert out["Neumann"] < 1e-5, (
        f"Neumann transpose mismatch {out['Neumann']:.2e} (periodic is "
        f"{out['periodic']:.2e}): the reverse-only rule is not the transpose "
        "of the solve under this boundary")
    assert out["Neumann"] > 10 * out["periodic"], (
        "the Neumann transpose is now as tight as the periodic one "
        f"({out['Neumann']:.2e} vs {out['periodic']:.2e}) -- if the boundary "
        "adjoint was fixed, tighten this test instead of deleting it")


@pytest.mark.slow
@pytest.mark.parametrize("energy", ("kick", "flux"))
def test_mass_gradient_identity_survives_the_gravity_step(energy):
    """Gravity moves momentum and energy, not mass, so the identity of the
    mass test above must still hold with a Poisson solve and two half-kicks
    in the step. Exact, and immune to the limiter kinks that make a
    finite-difference check of this step useless (measured: the FD transpose
    floor is 1.3e-2 even where the true answer is 1e-12)."""
    step = _gravity_step(_mesh(), energy)
    U = _smooth_state()
    g = jax.jit(jax.grad(lambda U: jnp.sum(step(U)[0])))(U)
    assert float(jnp.max(jnp.abs(g[0] - 1.0))) < 1e-9
    assert float(jnp.max(jnp.abs(g[1:]))) < 1e-9


def test_forward_mode_through_the_reverse_only_solve_is_refused():
    """The differentiable FMG is reverse-mode only: its rule is one adjoint
    solve, and there is no jvp. jax refuses with "can't apply forward-mode
    autodiff (jvp) to a custom_vjp function"; the both-modes route is
    solve_poisson_fmg_bidiff (custom_linear_solve), which the linearity test
    above uses. Pinned so the refusal stays a clear error."""
    from luna.gravity.poisson_diff import solve_poisson_fmg_diff
    rho = jnp.asarray(np.random.default_rng(0).normal(size=(8, 8, 8)))
    rho = rho - jnp.mean(rho)
    with pytest.raises(Exception) as e:
        jax.jvp(lambda r: solve_poisson_fmg_diff(r, 0.125, 1.0, 0), (rho,), (rho,))
    assert "forward-mode" in str(e.value) and "custom_vjp" in str(e.value), str(e.value)


# ---------------------------------------------------------------------------
# 2. conservation fixes the gradient exactly
# ---------------------------------------------------------------------------

def test_total_mass_gradient_is_one_on_density_and_zero_elsewhere():
    """Total mass after a step equals total mass before, for every state, so
    its gradient is 1 on every density cell and 0 on every momentum and
    energy cell -- exactly, by conservation, with no analytic solution
    needed. A halo or seam bug in the adjoint breaks this immediately."""
    step = _hydro_step(_mesh())
    U = _smooth_state()
    m0 = float(jnp.sum(U[0]))
    m1 = float(jnp.sum(step(U)[0]))
    assert abs(m1 - m0) / m0 < TOL, f"the FORWARD does not conserve mass: {m1 - m0:.3e}"

    g = jax.jit(jax.grad(lambda U: jnp.sum(step(U)[0])))(U)
    err_rho = float(jnp.max(jnp.abs(g[0] - 1.0)))
    err_rest = float(jnp.max(jnp.abs(g[1:])))
    assert err_rho < TOL, f"d(mass)/d(rho) deviates from 1 by {err_rho:.2e}"
    assert err_rest < TOL, f"d(mass)/d(momentum,energy) is not 0: {err_rest:.2e}"


def test_total_energy_gradient_is_one_on_energy_and_zero_elsewhere():
    """The same identity for total energy, which the adiabatic step conserves."""
    step = _hydro_step(_mesh())
    U = _smooth_state()
    e0 = float(jnp.sum(U[4]))
    e1 = float(jnp.sum(step(U)[4]))
    assert abs(e1 - e0) / e0 < TOL, f"the FORWARD does not conserve energy: {e1 - e0:.3e}"

    g = jax.jit(jax.grad(lambda U: jnp.sum(step(U)[4])))(U)
    assert float(jnp.max(jnp.abs(g[4] - 1.0))) < TOL
    assert float(jnp.max(jnp.abs(g[:4]))) < TOL


# ---------------------------------------------------------------------------
# 3. the Taylor remainder is second order
# ---------------------------------------------------------------------------

def _taylor_slope(step, U, seed=7):
    """Convergence order of ||S(U+eps v) - S(U) - eps J v||."""
    v = _random_like(U, seed)
    S0 = step(U)
    _, Jv = jax.jvp(step, (U,), (v,))
    eps = np.array([1e-4, 1e-5, 1e-6])
    r = np.array([float(jnp.linalg.norm(step(U + e * v) - S0 - e * Jv))
                  for e in eps])
    assert np.all(r > 0), "remainder is exactly zero: the step is not perturbed"
    return float(np.polyfit(np.log(eps), np.log(r), 1)[0])


def test_the_poisson_solve_is_linear_and_its_remainder_vanishes():
    """The rate test needs an operator that is actually smooth. The Poisson
    solve is LINEAR in its source, so the first-order Taylor remainder is not
    merely second order, it is zero to round-off -- and the tangent equals the
    map applied to the perturbation. A wrong linearisation cannot hide."""
    from luna.gravity.poisson_diff import solve_poisson_fmg_bidiff
    rng = np.random.default_rng(0)
    rho = jnp.asarray(rng.normal(size=(N, N, N)))
    rho = rho - jnp.mean(rho)
    v = jnp.asarray(rng.normal(size=(N, N, N)))
    v = v - jnp.mean(v)
    solve = jax.jit(lambda r: solve_poisson_fmg_bidiff(r, L / N, 1.0, 0))
    S0 = solve(rho)
    _, Jv = jax.jvp(solve, (rho,), (v,))
    for eps in (1e-2, 1e-4):
        rem = float(jnp.linalg.norm(solve(rho + eps * v) - S0 - eps * Jv))
        scale = float(jnp.linalg.norm(S0))
        assert rem / scale < 1e-9, f"eps={eps}: remainder {rem / scale:.2e} is not zero"
    # and the tangent of a linear map is the map itself, to solver tolerance
    rel = float(jnp.linalg.norm(Jv - solve(v)) / jnp.linalg.norm(solve(v)))
    assert rel < 1e-9, f"J v differs from S(v) by {rel:.2e} for a linear solve"


@pytest.mark.parametrize("order", [0, 1])
def test_the_step_is_only_piecewise_differentiable(order):
    """MEASURED, and a property of the scheme rather than a bug: the Taylor
    remainder converges at FIRST order, with reconstruction (minmod) and
    WITHOUT it (order 0). So the kinks are not only the limiter -- the
    Riemann solver's wave-speed branches do it too. However small the
    perturbation, it flips a branch in some cells, so the map is piecewise
    linear and the gradient is correct on each piece. That is exactly why the
    transpose, conservation and chain-rule identities above still hold to
    round-off, and why a plain finite-difference check of a rollout wobbles.

    If someone makes the scheme smooth, this test should be moved, not
    deleted."""
    slope = _taylor_slope(_hydro_step(_mesh(), order=order, limiter=0),
                          _smooth_state())
    assert slope < 1.5, (
        f"order={order}: the step now converges at Taylor order {slope:.2f}; "
        "if the scheme was made smooth, assert > 1.8 instead")


# ---------------------------------------------------------------------------
# 4. the chain rule across two steps
# ---------------------------------------------------------------------------

def test_two_step_jacobian_is_the_composition():
    """J(S2 o S1) v == J2 (J1 v): the rollout's derivative is the product of
    the per-step derivatives. Catches a carry that is silently detached."""
    step = _hydro_step(_mesh())
    U = _smooth_state()
    v = _random_like(U, 11)

    @jax.jit
    def both(U, v):
        one = step
        two = lambda x: one(one(x))
        _, Jv_two = jax.jvp(two, (U,), (v,))
        U1, Jv_1 = jax.jvp(one, (U,), (v,))
        _, Jv_comp = jax.jvp(one, (U1,), (Jv_1,))
        return Jv_two, Jv_comp

    a, b = both(U, v)
    rel = float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))
    assert rel < TOL, f"two-step Jacobian differs from the composition by {rel:.2e}"
