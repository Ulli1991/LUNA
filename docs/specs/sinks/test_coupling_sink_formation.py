"""Does LUNA actually FORM a sink in a collapse? (Phase 5 / #1 honest check)

Sets up a dense, cold, converging, self-bound central clump and steps the real
sink module — a sink should spontaneously form (active 0 -> 1) per the SEREN /
Bate-Bonnell-Price criteria in sources/sinks.py::_maybe_create.
"""
import pytest  # noqa: E402

pytestmark = pytest.mark.skip(
    reason="Phase 2. These drive the pre-assemble/evolve `sources.sinks."
           "setup_sinks`, which main deleted in 4848abe -- sinks are now the "
           "`Sinks` class in species/sinks.py, built by assemble. Reviving "
           "them needs two things: the setup_sinks dict API mapped onto "
           "Sinks, and -- for the accretion moments -- the running "
           "accumulators (macc/sv/sv2/scs/sL) that Sinks._accrete does not "
           "keep, which is the same Phase 2 gap accreted_gas_moments "
           "documents. Kept rather than deleted: the assertions are the "
           "specification for that work.")

import os

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp  # noqa: E402

from tests.test_regressions import _build_cfg_for_test  # noqa: E402


def test_sink_forms_in_collapsing_clump():
    from luna.sources.sinks import setup_sinks  # noqa: F401
    N = 32
    L = 1.0
    cfg = _build_cfg_for_test(
        test_problem="Sod-Shock-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L, gamma=5.0 / 3.0,
        ORDER=1, LIMITER=1, GRAVITATIONAL_G=1.0, POISSON_BOUNDARY=0,
    )
    cfg.sinks_enabled = True
    cfg.sink_creation_density = 20.0          # central peak well above this
    cfg.sink_accretion_radius = 0.08
    cfg.sink_creation_separation = 0.15
    cfg.sink_max_count = 4
    ss = setup_sinks(cfg, sharding=None, is_main_process=False)
    apply, state0 = ss["apply"], ss["state0"]

    # coordinates
    dx = L / N
    xc = (np.arange(N) + 0.5) * dx
    X, Y, Z = np.meshgrid(xc, xc, xc, indexing="ij")
    c = 0.5
    r = np.sqrt((X - c) ** 2 + (Y - c) ** 2 + (Z - c) ** 2)

    # dense cold centrally-peaked clump + inward (converging) radial velocity
    rho = 1.0 + 200.0 * np.exp(-(r / 0.10) ** 2)     # peak ~201 >> rho_create
    vr = -2.0 * np.exp(-(r / 0.15) ** 2)             # inward flow -> ∇·v < 0
    rhat = np.stack([(X - c), (Y - c), (Z - c)], axis=0) / np.maximum(r, 1e-9)
    vx, vy, vz = vr * rhat[0], vr * rhat[1], vr * rhat[2]
    P = 0.05 * rho                                   # cold (low pressure) -> bound
    E = P / (cfg.gamma - 1.0) + 0.5 * rho * (vx**2 + vy**2 + vz**2)
    U = jnp.asarray(np.stack([rho, rho * vx, rho * vy, rho * vz, E]))

    carry = state0
    n_before = int(np.sum(np.asarray(carry[3])))
    assert n_before == 0

    formed = False
    for _ in range(4):
        U, carry = apply(U, carry, 2.0e-3)
        n = int(np.sum(np.asarray(carry[3])))
        if n >= 1:
            formed = True
            break
    assert formed, "no sink formed in the collapsing clump"

    # the formed sink has positive mass and accumulates accreted-gas moments
    mass = np.asarray(carry[2]); active = np.asarray(carry[3])
    i = int(np.argmax(active))
    assert mass[i] > 0.0
    assert float(np.asarray(carry[4]["macc"])[i]) >= 0.0
