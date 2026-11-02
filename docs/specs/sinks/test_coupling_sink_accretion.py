"""Exact accreted-gas moment tracking in LUNA's sink module (Phase 5 / #1).

Runs the real `sources.sinks` step (jitted `_apply` with the extended 5-tuple
carry) and checks that accretion accumulates the per-sink moments, which
`accreted_gas_moments` turns into mean c_s / velocity / dispersion / ang. mom.
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


def _sink_cfg(N=16):
    cfg = _build_cfg_for_test(
        test_problem="Sod-Shock-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0, gamma=5.0 / 3.0,
        ORDER=1, LIMITER=1, GRAVITATIONAL_G=1.0, POISSON_BOUNDARY=0,
    )
    # sink params (set on the instance; not all are Config fields)
    cfg.sinks_enabled = True
    cfg.sink_creation_density = 5.0          # high → no spontaneous creation here
    cfg.sink_accretion_radius = 0.2
    cfg.sink_max_count = 4
    return cfg


def test_extended_carry_runs_and_accretes():
    from luna.sources.sinks import setup_sinks, accreted_gas_moments  # noqa: F401
    N = 16
    cfg = _sink_cfg(N)
    ss = setup_sinks(cfg, sharding=None, is_main_process=False)
    assert ss is not None
    apply, state0 = ss["apply"], ss["state0"]
    Nmax = ss["n_max"]
    assert len(state0) == 5 and set(state0[4]) == {"macc", "sv", "sv2", "scs", "sL"}

    # uniform dense cold gas (rho=10 > rho_create), bulk flow, cs=1
    gamma = cfg.gamma
    rho0, vbulk, cs = 10.0, np.array([0.5, 0.0, 0.0]), 1.0
    P = cs**2 * rho0 / gamma
    sh = (N, N, N)
    rho = np.full(sh, rho0)
    mx = np.full(sh, rho0 * vbulk[0]); my = np.zeros(sh); mz = np.zeros(sh)
    E = P / (gamma - 1.0) + 0.5 * rho0 * vbulk[0] ** 2
    U = jnp.asarray(np.stack([rho, mx, my, mz, np.full(sh, E)]))

    # inject one sink at the box centre into the 5-tuple carry
    pos = np.zeros((Nmax, 3)); pos[0] = [0.5, 0.5, 0.5]
    mass = np.zeros(Nmax); mass[0] = 1.0
    active = np.zeros(Nmax); active[0] = 1.0
    carry = (jnp.asarray(pos), jnp.zeros((Nmax, 3)), jnp.asarray(mass),
             jnp.asarray(active), state0[4])

    U2, carry2 = apply(U, carry, 0.001)          # one real sink step (jitted)
    mom = carry2[4]
    macc0 = float(np.asarray(mom["macc"])[0])
    assert macc0 > 0.0                            # sink accreted the dense gas

    m = accreted_gas_moments(carry2)
    assert m["mean_cs"].shape == (1,)
    assert np.all(np.isfinite(m["mean_cs"]))
    # accreted-gas sound speed ≈ the imposed cs
    np.testing.assert_allclose(m["mean_cs"][0], cs, rtol=0.05)
    # mean velocity ≈ the bulk flow (gravity kick over dt=1e-3 is tiny)
    np.testing.assert_allclose(m["mean_velocity"][0], vbulk, atol=0.02)
    assert np.all(np.isfinite(m["var_velocity"])) and np.all(np.isfinite(m["angular_momentum"]))


def test_moments_accumulate_over_steps():
    from luna.sources.sinks import setup_sinks, accreted_gas_moments  # noqa: F401
    """macc grows monotonically as the sink keeps accreting over several steps."""
    N = 16
    cfg = _sink_cfg(N)
    ss = setup_sinks(cfg, sharding=None, is_main_process=False)
    apply, state0 = ss["apply"], ss["state0"]
    Nmax = ss["n_max"]
    gamma = cfg.gamma
    rho0 = 10.0; P = 1.0 * rho0 / gamma
    sh = (N, N, N)
    U = jnp.asarray(np.stack([np.full(sh, rho0), np.zeros(sh), np.zeros(sh),
                              np.zeros(sh), np.full(sh, P / (gamma - 1.0))]))
    pos = np.zeros((Nmax, 3)); pos[0] = [0.5, 0.5, 0.5]
    mass = np.zeros(Nmax); mass[0] = 1.0
    active = np.zeros(Nmax); active[0] = 1.0
    carry = (jnp.asarray(pos), jnp.zeros((Nmax, 3)), jnp.asarray(mass),
             jnp.asarray(active), state0[4])

    macc_hist = []
    for _ in range(3):
        U, carry = apply(U, carry, 0.001)
        macc_hist.append(float(np.asarray(carry[4]["macc"])[0]))
    assert macc_hist[0] > 0
    assert macc_hist[1] >= macc_hist[0] and macc_hist[2] >= macc_hist[1]
