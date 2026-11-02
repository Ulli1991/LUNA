"""Sink particles: the coupling order and mass conservation.

Sinks are a particle species of the gravity stage (species/sinks.py): the
gas potential from the run's one Poisson solve, the sinks' analytic softened
field added to the gas kick, the sink N-body kick-drift-kick around the
hydro like the gas. Two gates:

1. ORDER, Richardson at fixed CFL (docs/history/AUDIT_2026-09-16.md sec.19 for why
   fixed CFL): the KDK split must be ~4 (2nd order), everything-once-after-
   the-hydro ~2. One solve per step either way. Accretion and creation are
   switched off (threshold far above the density) so only the smooth
   operators are measured.
2. MASS: gas mass + sink mass is invariant under accretion. Accretion moves
   gas from the grid into the sink; a sign or volume slip loses mass silently,
   which is exactly how particles were lost before.

Sink softening and accretion radius are FIXED in physical units, not in
cells: a softening that scales with dx makes every resolution a different
continuum problem and Richardson then measures nothing (1.47 for an ordering
that must give ~2). GPU test.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp



from luna.hydro.conversions import prim_to_cons                                  # noqa: E402
from luna.time_integration.integrators import create_muscl_hancock_modular_step  # noqa: E402
from luna.utils.halo_exchange_shardmap import make_halo_shardmap                 # noqa: E402

GAMMA, L, EPS, G_SINK = 5.0 / 3.0, 1.0, 0.1, 2.0
RHO0, P0, T_END = 1.0, 1.0 / GAMMA, 0.25
M_SINK = 0.01                     # sink mass, against a gas mass of RHO0 * L^3
EPS_SOFT, R_ACC = 0.2 * L, 0.1 * L   # physical, resolution-independent


def _cfg(N, **kw):
    from dataclasses import replace

    from luna.template_config import Config
    base = dict(sinks_enabled=True, SPATIAL_DIMENSION=3,
                Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, BOUNDARY=0,
                GRAVITATIONAL_G=G_SINK, sink_max_count=1,
                sink_softening=EPS_SOFT, sink_accretion_radius=R_ACC,
                # far above max(rho): accretion AND creation off
                sink_creation_density=1e6)
    base.update(kw)
    return replace(Config(), **base)


def _init(N):
    """Gas at rest at uniform pressure, density perturbed: gravity is the only
    force at t=0, so the splitting error is not buried under the hydro's."""
    x = (np.arange(N) + 0.5) * (L / N)
    s = EPS * np.sin(2 * np.pi * x / L)
    o, z = np.ones((N, N)), np.zeros((N, N, N))
    return prim_to_cons(jnp.asarray(np.stack([
        (RHO0 + s)[:, None, None] * o, z, z, z,
        P0 * np.ones((N, N, N))])), GAMMA)


def _one_sink():
    """One active sink at the box centre, at rest."""
    from luna.setup.state import Particles
    return Particles(jnp.asarray([[0.5 * L, 0.5 * L, 0.5 * L]]), jnp.zeros((1, 3)),
                     jnp.asarray([M_SINK]), jnp.asarray([1.0]))


def _setup(cfg):
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from luna.setup.assemble import build_gravity
    from luna.setup.clock import ProperClock
    from luna.setup.mesh import DENSE
    from luna.setup.run_kind import GAS
    from luna.setup.state import State
    N = cfg.Nx
    dx, inv = L / N, N / L
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1, 1), ("x", "y", "z"))
    step = create_muscl_hancock_modular_step(
        make_halo_shardmap(mesh, 3), GAMMA, 1, 0, dx, dx, dx, inv, inv, inv,
        0, 3, 0.4, riemann_solver="hllc", use_shardmap=True, mesh=mesh)[1]
    U = jax.device_put(_init(N), NamedSharding(mesh, P(None, "x", "y", "z")))
    grav = build_gravity(cfg, mesh, GAS, DENSE, ProperClock(0.0, T_END), is_main=False)
    state = grav.init(State(U=U, species={"sinks": _one_sink()}))
    return state, step, grav


def _run(N, kdk):
    state, step, grav = _setup(_cfg(N))
    dt, h = T_END / N, 0.5 * T_END / N
    for _ in range(N):
        if kdk:
            state = grav.before(state, h, dt)
            state = state.update(U=step(state.U, dt))
            state = grav.after(state, h, dt)
        else:
            # everything once, after the hydro, at full dt: forward Euler
            # coupling on the start-of-step potential. One solve either way.
            state = state.update(U=step(state.U, dt))
            state = grav.before(state, dt, dt)
            state = grav.after(state, 0.0, 0.0)
    return np.asarray(state.U), state


def _coarsen(U):
    n = U.shape[1] // 2
    return U.reshape(5, n, 2, n, 2, n, 2).mean(axis=(2, 4, 6))


@pytest.mark.parametrize("kdk,lo,hi", [(False, 1.7, 2.8), (True, 3.3, 4.6)])
def test_sink_coupling_order_at_fixed_cfl(kdk, lo, hi):
    """Richardson ratio: ~2 for source-after-hydro, ~4 for KDK."""
    sols = {N: _run(N, kdk)[0] for N in (16, 32, 64)}
    e1 = float(np.mean(np.abs(sols[16] - _coarsen(sols[32]))))
    e2 = float(np.mean(np.abs(sols[32] - _coarsen(sols[64]))))
    ratio = e1 / max(e2, 1e-300)
    print(f"[sinks kdk={kdk}] |16-32|={e1:.3e} |32-64|={e2:.3e} "
          f"ratio={ratio:.2f} (expect ~{'4' if kdk else '2'})")
    assert lo < ratio < hi, f"order ratio {ratio:.2f} outside [{lo}, {hi}]"


def test_accretion_moves_mass_from_gas_to_sink_without_losing_any():
    N = 32
    dx = L / N
    # accretion ON this time: threshold under the peak density so the sink eats
    state, step, grav = _setup(_cfg(N, sink_creation_density=1.05 * RHO0,
                                    sink_creation_separation=10.0 * L))   # no second sink

    def total(st):
        s = st.species["sinks"]
        return float(jnp.sum(st.U[0])) * dx ** 3 + float(jnp.sum(s.mass * s.active))

    m0 = total(state)
    dt, h = T_END / N, 0.5 * T_END / N
    for _ in range(8):
        state = grav.before(state, h, dt)
        state = state.update(U=step(state.U, dt))
        state = grav.after(state, h, dt)
    m1 = total(state)
    s = state.species["sinks"]
    m_sink = float(jnp.sum(s.mass * s.active))
    print(f"[sinks mass] total {m0:.12e} -> {m1:.12e} "
          f"(rel {abs(m1/m0 - 1):.2e}), sink mass {M_SINK} -> {m_sink:.6e}")
    assert m_sink > M_SINK, "the sink accreted nothing; the test proves nothing"
    assert abs(m1 / m0 - 1.0) < 1e-12, f"mass not conserved: {m0:.12e} -> {m1:.12e}"
