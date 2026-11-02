"""Operator-split sources must not cap the scheme at 1st order.

The hydro is 2nd order, but applying a source ONCE after the step (Lie
splitting) contributes an O(dt) error that caps the whole scheme however
accurate the fluxes are. Strang -- half source, full hydro, half source --
makes it O(dt^2).

MEASURE IT AT FIXED CFL. MUSCL-Hancock is Lax-Wendroff-type: its 2nd-order
time term is built from SPATIAL derivatives, so refining dt at fixed dx
measures O(dt*dx^2) noise and reports 1st order for everything, splitting or
not (retracted claim, audit sec.19). Refining dx and dt together is the only
meaningful path. No analytic solution exists for the forced problem, so this
uses Richardson self-convergence: each resolution against the next finer one
coarsened back down.

RESOLUTIONS 16/32/64, measured against 32/64/128 before shrinking: the finer
triple gives 2.21 (Lie) and 3.83 (Strang), the coarse one 2.43 and 4.17 -- both
separate 1st from 2nd order unambiguously, and the coarse triple runs in 10 s
against 211 s on the CPU backend, which is the difference between a gate that
runs on every push and one marked `slow` that never runs at all. The Lie ratio
IS further from the ideal 2 on the coarse triple (0.43 off vs 0.21), so 16^3 is
marginally less asymptotic; the assertion windows were NOT widened to suit it.

SCOPE: tests the SCHEME. The loop's step (setup/evolve.py) mirrors every stage
by construction (tests/test_step_palindrome.py). The self-gravity case drives
the real gravity stage and multigrid solver, so it proves that the solve/kick
split composes to 2nd order at ONE solve per step.

Measured (1 MI300A): external accel 2.43 Lie / 4.17 Strang; self-gravity 2.55
Lie / 4.32 KDK.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp



from luna.hydro.conversions import prim_to_cons                                  # noqa: E402
from luna.time_integration.integrators import create_muscl_hancock_modular_step  # noqa: E402
from luna.utils.halo_exchange_shardmap import make_halo_shardmap                 # noqa: E402

GAMMA, L, EPS, AMP = 5.0 / 3.0, 1.0, 1e-3, 1.0
RHO0, P0 = 1.0, 1.0 / GAMMA
C = float(np.sqrt(GAMMA * P0 / RHO0))
T_END = 0.25                      # N steps at resolution N: CFL fixed, dt halves exactly


def _init(N):
    x = (np.arange(N) + 0.5) * (L / N)
    s = EPS * np.sin(2 * np.pi * x / L)
    o, z = np.ones((N, N)), np.zeros((N, N, N))
    return prim_to_cons(jnp.asarray(np.stack([
        (RHO0 + s)[:, None, None] * o, (s * C / RHO0)[:, None, None] * o,
        z, z, (P0 + s * C * C)[:, None, None] * o])), GAMMA)


def _accel(N):
    """Varies in space, so the source and the advection do not commute."""
    x = (np.arange(N) + 0.5) * (L / N)
    return jnp.asarray((AMP * np.sin(2 * np.pi * x / L))[:, None, None] * np.ones((N, N)))


def _source(U, dt, g):
    rho, mx = U[0], U[1]
    nmx = mx + dt * rho * g
    return jnp.stack([rho, nmx, U[2], U[3], U[4] + 0.5 * (mx + nmx) * g * dt])


def _run(N, strang):
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    dx, inv = L / N, N / L
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1, 1), ("x", "y", "z"))
    step = create_muscl_hancock_modular_step(
        make_halo_shardmap(mesh, 3), GAMMA, 1, 0, dx, dx, dx, inv, inv, inv,
        0, 3, 0.4, riemann_solver="hllc", use_shardmap=True, mesh=mesh)[1]
    U = jax.device_put(_init(N), NamedSharding(mesh, P(None, "x", "y", "z")))
    g, dt = _accel(N), T_END / N
    for _ in range(N):
        if strang:
            U = _source(U, 0.5 * dt, g)
            U = step(U, dt)
            U = _source(U, 0.5 * dt, g)
        else:
            U = step(U, dt)
            U = _source(U, dt, g)
    return np.asarray(U)


def _coarsen(U):
    n = U.shape[1] // 2
    return U.reshape(5, n, 2, n, 2, n, 2).mean(axis=(2, 4, 6))


@pytest.mark.parametrize("strang,lo,hi", [(False, 1.7, 2.8), (True, 3.3, 4.6)])
def test_splitting_order_at_fixed_cfl(strang, lo, hi):
    """Richardson ratio: ~2 for source-after-hydro, ~4 for Strang."""
    sols = {N: _run(N, strang) for N in (16, 32, 64)}
    e1 = float(np.mean(np.abs(sols[16] - _coarsen(sols[32]))))
    e2 = float(np.mean(np.abs(sols[32] - _coarsen(sols[64]))))
    ratio = e1 / max(e2, 1e-300)
    print(f"[split strang={strang}] |32-64|={e1:.3e} |64-128|={e2:.3e} "
          f"ratio={ratio:.2f} (expect ~{'4' if strang else '2'})")
    assert lo < ratio < hi, f"order ratio {ratio:.2f} outside [{lo}, {hi}]"


# --- the real self-gravity solver, through the same ordering ----------------
# A self-gravitating SOUND WAVE cannot separate the two orderings: the Jeans
# criterion is exactly the statement that gravity stays below pressure for a
# stable mode, so the splitting error never rises above the hydro's own. That
# version measured Lie 2.94 against KDK 3.19 -- errors 1% apart, useless as a
# gate. Instead: uniform pressure, gas at rest, density perturbed. There is no
# initial pressure force at all, so gravity drives the whole evolution.
# 4 pi G rho0 = 25 gives a growth time ~2 T_END: the perturbation grows by ~3x,
# staying smooth (no shock, no collapse) over the window.
G_SG, EPS_SG = 2.0, 0.1


def _init_sg(N):
    """Gas at rest at uniform pressure, density perturbed: gravity is the only
    force at t=0, so the splitting error is not buried under the hydro's."""
    x = (np.arange(N) + 0.5) * (L / N)
    s = EPS_SG * np.sin(2 * np.pi * x / L)
    o, z = np.ones((N, N)), np.zeros((N, N, N))
    return prim_to_cons(jnp.asarray(np.stack([
        (RHO0 + s)[:, None, None] * o, z, z, z,
        P0 * np.ones((N, N, N))])), GAMMA)


def _self_gravity(N, mesh):
    """The gravity stage on one device, traced (no host pulls) so a whole
    step compiles as one program; the differentiable FMG, as the adjoint
    runner uses."""
    from dataclasses import replace

    from luna.setup.assemble import build_gravity
    from luna.setup.clock import ProperClock
    from luna.setup.mesh import DENSE
    from luna.setup.run_kind import GAS
    from luna.template_config import Config

    cfg = replace(Config(), SELF_GRAVITY=True, SPATIAL_DIMENSION=3,
                  Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, BOUNDARY=0,
                  GRAVITATIONAL_G=G_SG, autodiff_mode=True)
    return build_gravity(cfg, mesh, GAS, DENSE, ProperClock(0.0, T_END), is_main=False)


def _run_sg(N, kdk):
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from luna.setup.state import State
    dx, inv = L / N, N / L
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1, 1), ("x", "y", "z"))
    step = create_muscl_hancock_modular_step(
        make_halo_shardmap(mesh, 3), GAMMA, 1, 0, dx, dx, dx, inv, inv, inv,
        0, 3, 0.4, riemann_solver="hllc", use_shardmap=True, mesh=mesh)[1]
    U = jax.device_put(_init_sg(N), NamedSharding(mesh, P(None, "x", "y", "z")))
    grav = _self_gravity(N, mesh)
    dt = T_END / N

    # ONE compiled program per step. Driving solve and kick as separate jitted
    # calls in a Python loop segfaults intermittently on rocm 7.2 / jax 0.10.0
    # (1 run in 3, one Viper MI300A node, inside the multigrid solve) -- the executable-mix
    # instability the DM blocks loop documents. Composing the step compiles it
    # as a unit: 3/3 clean afterwards. A gate that segfaults a third of the
    # time is worth the slower compile.
    @jax.jit
    def _one(state):
        if kdk:
            state = grav.before(state, 0.5 * dt, dt)    # carried from the last solve
            state = state.update(U=step(state.U, dt))
            return grav.after(state, 0.5 * dt, dt)      # ONE solve, as before
        state = state.update(U=step(state.U, dt))
        return grav.after(state, dt, 2.0 * dt)          # solve, then one full kick

    state = grav.init(State(U=U))
    for _ in range(N):
        state = _one(state)
    return np.asarray(state.U)


@pytest.mark.parametrize("kdk,lo,hi", [(False, 1.7, 2.8), (True, 3.3, 4.6)])
def test_self_gravity_splitting_order_at_fixed_cfl(kdk, lo, hi):
    """The real Poisson solve, Lie against KDK, at one solve per step either way."""
    sols = {N: _run_sg(N, kdk) for N in (16, 32, 64)}
    e1 = float(np.mean(np.abs(sols[16] - _coarsen(sols[32]))))
    e2 = float(np.mean(np.abs(sols[32] - _coarsen(sols[64]))))
    ratio = e1 / max(e2, 1e-300)
    print(f"[self-gravity kdk={kdk}] |16-32|={e1:.3e} |32-64|={e2:.3e} "
          f"ratio={ratio:.2f} (expect ~{'4' if kdk else '2'})")
    assert lo < ratio < hi, f"order ratio {ratio:.2f} outside [{lo}, {hi}]"
