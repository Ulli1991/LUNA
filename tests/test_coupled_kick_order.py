"""The coupled gas+DM step is second order in time.

The old coupled loops kicked the GAS once after the hydro at a_next (Lie
splitting), which capped the gas at first order however accurate the fluxes;
the DM was already KDK. Now both species half-kick on either side of the
hydro around ONE solve.

Richardson self-convergence at fixed CFL and fixed d ln a per cell (every
timestep limit halves with the cell; the particles' acceleration criterion
eta sqrt(eps/g) is made to scale that way by fixing the softening in physical
units and halving eta with the cell). The problem is the comoving analogue of
the self-gravity gate: gas at rest at a tiny uniform pressure with a density
perturbation, a dark-matter lattice at rest, no floor, EdS a = 0.3 -> 0.75.
Gravity is the only force at the start, so the splitting error is not buried
under the hydro's. (The cold Zel'dovich pancake cannot separate the two
orderings: a Mach-100 advection problem with a clamped pressure floor, both
orderings measured 1.95-1.97 with errors 2% apart.)

The field measured is the x MOMENTUM: the kick's time centering lives in the
velocity, and the density converges at second order under either ordering
(measured 4.19 Lie / 4.49 KDK in rho; the July probe on the pancake found the
same, "density unaffected"). Two things had to be right before the momentum
order could be measured at all: the dark matter is TWO interleaved lattices
at x offsets of a quarter cell, since one particle per cell displaced by a
fraction of a cell gives a CIC density whose divergence is a one-sided
difference, first order in dx (one lattice: KDK 2.57); and the limiter is
minmod, the project default, since MonCen at the velocity extrema left the
momentum first order (KDK 3.02). Measured: Lie 1.82, KDK 3.89.

Teeth: the same runs with the gas kick applied ONCE after the hydro at the
full dt (the old ordering, made by wrapping the gravity stage) must give ~2
in momentum, the KDK ~4. Same windows as the sibling gates. GPU test.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp



pytestmark = pytest.mark.slow

from luna.hydro.conversions import prim_to_cons                    # noqa: E402
from scripts.cosmo.run_pancake import build_cfg               # noqa: E402

EPS, P0 = 0.2, 1.0e-6


def _lie(gravity):
    """The old gas ordering on the same stage: no opening gas kick, one full
    gas kick after the solve. The species keep their KDK."""
    backend = gravity.backend
    kick = backend.kick_gas
    before, after = gravity.before, gravity.after

    def lie_before(state, h, dt):
        backend.kick_gas = lambda U, *a, **k: U
        try:
            return before(state, h, dt)
        finally:
            backend.kick_gas = kick

    def lie_after(state, h, dt):
        backend.kick_gas = lambda U, g, s, dtk, extra=None: kick(U, g, s, 2.0 * dtk, extra)
        try:
            return after(state, h, dt)
        finally:
            backend.kick_gas = kick

    gravity.before, gravity.after = lie_before, lie_after
    return gravity


def _init(N, cfg):
    """Gas rho~ = 1 + eps sin(kx) at rest at pressure P0; two DM lattices at
    the cell centres offset by -+ dx/4 in x, at rest, mass dx^3/2 each (mean
    rho~_dm = 1)."""
    L = cfg.Lx
    dx = L / N
    x = (np.arange(N) + 0.5) * dx
    rho = (1.0 + EPS * np.sin(2 * np.pi * x / L))[:, None, None] * np.ones((N, N))
    z = np.zeros((N, N, N))
    U = prim_to_cons(jnp.asarray(np.stack([rho, z, z, z, P0 * np.ones((N, N, N))])),
                     cfg.gamma)
    qx, qy, qz = np.meshgrid(x, x, x, indexing="ij")
    lattice = np.stack([qx, qy, qz], axis=-1).reshape(-1, 3)
    pos = np.mod(np.concatenate([lattice - [0.25 * dx, 0, 0],
                                 lattice + [0.25 * dx, 0, 0]]), L)
    return U, pos, np.zeros_like(pos), np.full(pos.shape[0], 0.5 * dx ** 3)


def _run(N, lie):
    from dataclasses import replace

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from luna.setup.assemble import assemble
    from luna.setup.evolve import evolve
    from luna.setup.ics import Initial
    from luna.setup.mesh import DENSE
    from luna.setup.run_kind import COUPLED
    from luna.setup.state import Particles
    cfg = replace(build_cfg(), Nx=N, Ny=N, Nz=N, dloga_max=0.04 * 16 / N,
                  dm_softening=1.0 / 16, dm_eta=0.025 * 16 / N,
                  a_end=0.75, cosmo_p_floor=0.0, LIMITER=0,
                  output_dir=f"snapshots/test_coupled_order_{N}",
                  write_snapshots=False, PLOT_AFTER_SIM=False,
                  hydro_fused=True)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1), ("x", "y", "z"))
    U0, pos, p, mass = _init(N, cfg)
    U0 = jax.device_put(U0, NamedSharding(mesh, P(None, "x", "y", "z")))
    init = Initial(cfg=cfg, U=U0, dm=Particles(pos, p, mass))
    physics = assemble(cfg, mesh, COUPLED, DENSE, init, is_main=False)
    if lie:
        _lie(physics.gravity)
    state = evolve(cfg, init, physics, mesh, is_main=False)
    print(f"[coupled lie={lie}] N={N}: {state.step} steps")   # must double per level
    U = np.asarray(state.U)
    # periodic, fused hydro: gas mass to round-off (gas + DM momentum is not checked: it needs the PM's units)
    from tests._helpers import check
    check(f"coupled lie={lie} N={N} gas |dM|/M", abs(U[0].sum() / float(np.sum(np.asarray(U0[0]))) - 1.0), 1e-13)
    return U


def _coarsen(U):
    n = U.shape[1] // 2
    return U.reshape(5, n, 2, n, 2, n, 2).mean(axis=(2, 4, 6))


@pytest.mark.parametrize("lie,lo,hi", [(True, 1.7, 2.8), (False, 3.3, 4.6)])
def test_coupled_step_order_at_fixed_cfl(lie, lo, hi):
    sols = {N: _run(N, lie) for N in (16, 32, 64)}
    ratios = {}
    for k, name in ((0, "rho"), (1, "mom_x")):
        e1 = float(np.mean(np.abs(sols[16][k] - _coarsen(sols[32])[k])))
        e2 = float(np.mean(np.abs(sols[32][k] - _coarsen(sols[64])[k])))
        ratios[name] = e1 / max(e2, 1e-300)
        print(f"[coupled lie={lie}] {name}: |16-32|={e1:.3e} |32-64|={e2:.3e} "
              f"ratio={ratios[name]:.2f}")
    ratio = ratios["mom_x"]
    assert lo < ratio < hi, f"momentum order ratio {ratio:.2f} outside [{lo}, {hi}]"
