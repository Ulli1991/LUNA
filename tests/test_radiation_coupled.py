"""The M1 field coupled to the gas: the conservation laws the coupling is
built to satisfy exactly, the same answer sharded as on one device, and one
run of the real step with the stage in it.

The coupling is written so these are round-off checks, not tolerance checks:
what the radiation loses the gas gains, cell by cell, in the same expression.
Anything looser would hide the bookkeeping error this file exists to catch.
"""

import numpy as np
import pytest


import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402

from luna.radiation.coupling import (absorb_emit, apply_force,  # noqa: E402
                                     substeps, total_energy)
from tests._helpers import check                              # noqa: E402

pytestmark = pytest.mark.fast

CHAT = 3.0
NDIM = 3


def _gas(n, seed=0):
    """A random 3D gas state, U = [rho, mom_x, mom_y, mom_z, E_tot]."""
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0.5, 1.5, size=(n, n, n))
    mom = rng.normal(0.0, 0.1, size=(3, n, n, n))
    e_int = rng.uniform(1.0, 2.0, size=(n, n, n))
    E = e_int + 0.5 * np.sum(mom ** 2, axis=0) / rho
    return jnp.asarray(np.concatenate([rho[None], mom, E[None]], axis=0))


def _rad(n, g=2, seed=1):
    """A random M1 state that is actually a state: |F| <= 0.9 c E.

    f = |F|/(cE) > 1 is outside the closure's domain, and `m1_floor` now
    rescales such a cell back onto the cone -- which silently removed
    radiation momentum and broke the conservation checks below when this
    helper drew E and F independently."""
    rng = np.random.default_rng(seed)
    E = rng.uniform(0.1, 1.0, size=(g, n, n, n))
    d = rng.normal(size=(g, NDIM, n, n, n))
    d /= np.sqrt((d ** 2).sum(axis=1))[:, None]
    f = rng.uniform(0.0, 0.9, size=(g, 1, n, n, n))
    return jnp.asarray(np.concatenate([E[:, None], CHAT * E[:, None] * f * d],
                                      axis=1))


def _internal(U):
    return U[-1] - 0.5 * jnp.sum(U[1:4] ** 2, axis=0) / U[0]


def _momentum(U, R, chat=CHAT):
    """Gas momentum plus radiation momentum F/c^2, summed over the box."""
    return jnp.sum(U[1:4], axis=(1, 2, 3)) + jnp.sum(R[:, 1:], axis=(0, 2, 3, 4)) / chat ** 2


def test_absorption_and_emission_conserve_total_energy():
    """Whatever leaves the groups lands in the gas internal energy, at any
    optical depth and any step length."""
    U, R = _gas(8), _rad(8)
    e0, tot0 = _internal(U), total_energy(U, R)
    for dt in (1e-6, 1e-2, 1.0, 1e3):
        U2, R2 = absorb_emit(U, R, dt, chat=CHAT, kappa_a=jnp.array([0.5, 40.0]),
                             source=jnp.array([0.2, 1.0]))
        rel = float(abs(total_energy(U2, R2) - tot0) / tot0)
        assert rel < 1e-14, f"dt={dt}: total energy moved by {rel:.2e}"
        # it is internal energy that changed, not kinetic: momentum untouched
        assert np.array_equal(np.asarray(U2[1:4]), np.asarray(U[1:4]))
        assert float(jnp.max(jnp.abs(_internal(U2) - e0))) > 0.0


def test_the_force_conserves_momentum_and_energy_and_leaves_heat_alone():
    """With c_true = c_hat there is no RSLA correction, so the momentum the
    flux gives up is exactly the momentum the gas takes on, the work is
    exactly what the radiation energy pays, and the internal energy of the
    gas does not move at all."""
    U, R = _gas(8), _rad(8)
    # the momentum reference is taken BEFORE the substeps: the flux the
    # damping destroyed is already out of R by the time they return, and it
    # is exactly that flux the force then hands to the gas. (The M1 transport
    # is conservative and the box is periodic, so the transport itself moves
    # no total momentum.)
    p0, e0 = _momentum(U, R), _internal(U)
    R2, dF = substeps(R, 0.02, 1.0 / 8, chat=CHAT, kappa=3.0, n_sub=2, boundary=0)
    tot0 = total_energy(U, R2)
    U3, R3 = apply_force(U, R2, dF, chat=CHAT, c_true=CHAT)
    dp = np.asarray(_momentum(U3, R3) - p0)
    assert float(np.max(np.abs(dp))) < 1e-12 * float(np.max(np.abs(np.asarray(p0)))) + 1e-14
    rel = float(abs(total_energy(U3, R3) - tot0) / tot0)
    assert rel < 1e-14, f"total energy moved by {rel:.2e}"
    assert float(jnp.max(jnp.abs(_internal(U3) - e0))) < 1e-12 * float(jnp.max(e0))
    assert float(jnp.max(jnp.abs(U3[1:4] - U[1:4]))) > 0.0      # it did push


def test_a_force_without_the_work_term_is_caught():
    """Teeth: the same kick with the kinetic energy left unpaid for. The gas
    heats or cools by the work, and the conservation check above catches it --
    which is why that check is at round-off and not at a tolerance."""
    U, R = _gas(8), _rad(8)
    R2, dF = substeps(R, 0.02, 1.0 / 8, chat=CHAT, kappa=3.0, n_sub=2, boundary=0)
    tot0, e0 = total_energy(U, R2), _internal(U)
    naive = U.at[1:4].add(jnp.sum(dF, axis=0) / CHAT ** 2)      # momentum only
    assert float(abs(total_energy(naive, R2) - tot0) / tot0) < 1e-14  # E_tot is fine
    # but the internal energy silently paid for the kick
    assert float(jnp.max(jnp.abs(_internal(naive) - e0))) > 1e-12 * float(jnp.max(e0))


def test_a_reduced_light_speed_weakens_the_force_by_c_hat_over_c():
    """The RSLA slows the transport, not the coupling: the force carries the
    true c, so raising it above c_hat scales the kick down by c_hat/c_true."""
    U, R = _gas(8), _rad(8)
    R2, dF = substeps(R, 0.02, 1.0 / 8, chat=CHAT, kappa=3.0, n_sub=2, boundary=0)
    a, _ = apply_force(U, R2, dF, chat=CHAT, c_true=CHAT)
    b, _ = apply_force(U, R2, dF, chat=CHAT, c_true=10.0 * CHAT)
    dp_a, dp_b = a[1:4] - U[1:4], b[1:4] - U[1:4]
    # dp is a difference of momenta two orders larger than itself, so the
    # comparison carries that cancellation: eps * |p| / |dp| ~ 1e-13.
    assert np.allclose(np.asarray(dp_b), np.asarray(dp_a) / 10.0, rtol=1e-9,
                       atol=1e-16 * float(np.max(np.abs(np.asarray(U[1:4])))))


def _mesh(shape):
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh
    n = int(np.prod(shape))
    return Mesh(mesh_utils.create_device_mesh(shape, jax.devices()[:n]),
                ("x", "y", "z"))


def _rad_cfg(n, **kw):
    from dataclasses import replace
    from luna.template_config import Config
    return replace(Config(), SPATIAL_DIMENSION=3, Nx=n, Ny=n, Nz=n,
                   Lx=1.0, Ly=1.0, Lz=1.0, BOUNDARY=0, enable_x64=True,
                   radiation_enabled=True, rad_groups=2, rad_chat=CHAT,
                   rad_E0=1.0, rad_kappa_a=0.7, rad_kappa_s=0.3, rad_source=0.4,
                   rad_substeps=2, **kw)


@pytest.mark.skipif(jax.device_count() < 8,
                    reason="the (2,2,2) mesh this compares needs 8 devices; "
                           "the suite's CPU backend has them, two GPUs do not")
@pytest.mark.parametrize("scheme", ["hancock_split", "rk2"])
def test_the_sharded_half_step_is_the_single_device_one(scheme):
    """The transport's ghost cells come from a ppermute halo when the grid is
    split and from a local pad when it is not. Same program, same answer: the
    halo is the only difference between the two paths. The split scheme asks
    the halo for one axis per sweep, which is its own code path."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    from luna.sources.radiation import setup_radiation
    n = 16
    cfg = _rad_cfg(n, rad_scheme=scheme)
    one = NamedSharding(_mesh((1, 1, 1)), P(None, "x", "y", "z"))
    many = NamedSharding(_mesh((2, 2, 2)), P(None, "x", "y", "z"))
    rad1 = setup_radiation(cfg, one, is_main=False)
    rad8 = setup_radiation(cfg, many, is_main=False)
    U = _gas(n)
    R = _rad(n)
    h = 1e-3
    U1, R1 = rad1["before"](U, R, h)
    U8, R8 = rad8["before"](jax.device_put(U, NamedSharding(many.mesh, P(None, "x", "y", "z"))),
                            jax.device_put(R, rad8["sharding"]), h)
    assert np.allclose(np.asarray(U1), np.asarray(U8), rtol=1e-12, atol=1e-14)
    assert np.allclose(np.asarray(R1), np.asarray(R8), rtol=1e-12, atol=1e-14)



@pytest.mark.skipif(jax.device_count() < 8,
                    reason="the (2,2,2) mesh this compares needs 8 devices")
@pytest.mark.parametrize("boundary", [0, 1])
def test_the_sharded_ionising_half_step_is_the_single_device_one(boundary):
    """The cloud's path: photons from a source, split transport, the implicit
    ionisation, the prescribed temperature -- on a (2,2,2) mesh against one
    device, with the source on a block corner so every face of it is a
    halo. Both boundaries: the zero-gradient ghosts are made per axis too."""
    from dataclasses import replace
    from jax.sharding import NamedSharding, PartitionSpec as P
    from luna.sources.radiation import setup_radiation
    n = 16
    cfg = replace(_rad_cfg(n), rad_groups=1, rad_ionization=True, rad_E0=0.0,
                  rad_kappa_a=0.0, rad_kappa_s=0.0, rad_source=0.0, BOUNDARY=boundary,
                  rad_boundary=boundary, rad_sigma=50.0, rad_alpha_B=0.3, rad_m_H=1.0,
                  rad_photon_rate=20.0, rad_cs2_neutral=0.01, rad_cs2_ionized=1.0,
                  rad_source_radius=0.1, gamma=5.0 / 3.0)
    one = NamedSharding(_mesh((1, 1, 1)), P(None, "x", "y", "z"))
    many = NamedSharding(_mesh((2, 2, 2)), P(None, "x", "y", "z"))
    rad1 = setup_radiation(cfg, one, is_main=False)
    rad8 = setup_radiation(cfg, many, is_main=False)
    U = _gas(n)
    c1 = rad1["R0"]
    c8 = rad8["R0"]
    U8 = jax.device_put(U, NamedSharding(many.mesh, P(None, "x", "y", "z")))
    h = 2e-2
    for _ in range(3):
        U, c1 = rad1["before"](U, c1, h)
        U8, c8 = rad8["before"](U8, c8, h)
    assert float(np.max(np.asarray(c1[1]))) > 0.5, "nothing was ionised"
    np.testing.assert_allclose(np.asarray(U8), np.asarray(U), rtol=1e-11, atol=1e-13)
    for a, b in zip(c1, c8):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-11, atol=1e-13)

def test_the_stage_runs_in_the_real_step_and_conserves_energy():
    """The integration gate: assemble a periodic run with radiation on and
    take real steps. The box is closed, so gas energy plus radiation energy
    is fixed -- through the hydro, the stage's two halves and the dt limit
    the fixed substep count imposes."""
    from luna.ic_generation.ic import resolve
    from luna.setup.assemble import assemble
    from luna.setup.evolve import initial_state, step, timestep
    from luna.setup.mesh import build_mesh, decomposition
    from luna.setup.run_kind import run_kind
    from luna.setup.ics import read_ics
    n = 16
    cfg = resolve(_rad_cfg(n, test_problem="Soundwave-3D", t_end=1.0))
    kind = run_kind(cfg)
    layout = decomposition(cfg, kind)
    mesh, _ = build_mesh(cfg, layout, kind)
    init = read_ics(cfg, mesh, kind, layout, False)
    physics = assemble(cfg, mesh, kind, layout, init, False)
    names = [getattr(s, "name", "") for s in physics.stages]
    assert "radiation" in names, names

    state = initial_state(init, physics)
    rad = physics.stages[names.index("radiation")]
    tot0 = float(total_energy(state.U, state.sources["radiation"]))
    m0, p0 = float(jnp.sum(state.U[0])), np.asarray(_momentum(state.U, state.sources["radiation"]))
    for _ in range(6):
        dt, t_new = timestep(state, physics, 1e30)
        # the radiation's fixed substep count is what bounds the step here
        assert dt <= rad.dt_max * (1.0 + 1e-12)
        state = step(state, dt, t_new, physics)
    R = state.sources["radiation"]
    assert R.shape == (2, 4, n, n, n) and bool(jnp.all(jnp.isfinite(R)))
    tot = float(total_energy(state.U, R))
    assert abs(tot - tot0) / abs(tot0) < 1e-12, f"total energy moved by {tot - tot0:.3e}"
    # periodic, c_true = c_hat (rad_c_true=0): gas mass, and gas + radiation momentum, to round-off
    check("rad real step |dM|/M", abs(float(jnp.sum(state.U[0])) / m0 - 1.0), 1e-13)
    check("rad real step |dP|/sqrt(M E)", float(np.abs(np.asarray(_momentum(state.U, R)) - p0).max())
          / np.sqrt(m0 * abs(tot0)), 1e-13)
    # and it actually did something. NOT that the field is spatially varying:
    # with a constant opacity and a constant source, E obeys an equation the
    # gas does not enter, so a uniform E stays uniform and is right to. What
    # has to have moved is its value, 1.0 -> about 0.49 here.
    assert abs(float(R[:, 0].mean()) - cfg.rad_E0) > 1e-3 * cfg.rad_E0


@pytest.mark.parametrize("ionization", [False, True])
def test_a_nan_radiation_field_stops_the_run(ionization):
    """Quiet is not correct: a NaN in the carry used to step on (an ionising
    run read it as fully ionised gas). The stage's step bound refuses it."""
    from dataclasses import replace
    from jax.sharding import NamedSharding, PartitionSpec as P
    from luna.setup.state import State
    from luna.sources.radiation import setup_radiation
    from luna.sources.stages import Radiation
    cfg = _rad_cfg(8)
    if ionization:
        cfg = replace(cfg, rad_groups=1, rad_ionization=True, rad_E0=0.0, rad_kappa_a=0.0,
                      rad_kappa_s=0.0, rad_source=0.0, rad_sigma=50.0, rad_alpha_B=0.3,
                      rad_m_H=1.0, rad_photon_rate=20.0, rad_cs2_neutral=0.01,
                      rad_cs2_ionized=1.0, rad_source_radius=0.1, gamma=5.0 / 3.0)
    stage = Radiation(setup_radiation(cfg, NamedSharding(_mesh((1, 1, 1)), P(None, "x", "y", "z")),
                                      is_main=False))
    c = stage.carry0
    state = State(U=_gas(8), sources={"radiation": c})
    assert np.isfinite(stage.dt_limit(state))
    bad = (c[0].at[0, 0, 1, 2, 3].set(jnp.nan),) + tuple(c[1:]) if ionization \
        else c.at[0, 0, 1, 2, 3].set(jnp.nan)
    with pytest.raises(RuntimeError, match="non-finite radiation"):
        stage.dt_limit(state.with_source("radiation", bad))


def test_the_config_refuses_the_combinations_that_would_lie():
    """An unsupported ghost rule would silently reflect nothing: an error,
    not a warning.

    Restart is NO LONGER one of these. It used to be refused here because a
    checkpoint carried U and the OU state but not the radiation field, so a
    resumed run would have begun again from rad_E0 without saying so. The
    field is written now (file_io/radiation_checkpoint.py), and the check
    moved to where it can tell the difference: setup_radiation raises if the
    field is genuinely absent, rather than the config refusing every restart.
    """
    from dataclasses import replace
    _rad_cfg(8, restart=True).validate()          # allowed, and must not raise
    # the silent one: two stages writing the same variable
    with pytest.raises(ValueError, match="cooling_enabled"):
        replace(_rad_cfg(8), rad_ionization=True, rad_sigma=1.0,
                rad_alpha_B=1e-3, rad_photon_rate=1.0,
                cooling_enabled=True).validate()
    with pytest.raises(ValueError, match="rad_chat"):
        replace(_rad_cfg(8), rad_chat=0.0).validate()
    with pytest.raises(ValueError, match="rad_boundary"):
        replace(_rad_cfg(8), rad_boundary=2).validate()
    with pytest.raises(ValueError, match="rad_kappa_a_groups"):
        replace(_rad_cfg(8), rad_kappa_a_groups=(1.0, 2.0, 3.0)).validate()


def test_the_coupled_half_step_is_differentiable():
    """The whole stage, gas and radiation together, under reverse mode: this
    is the shape the coupled adjoint will scan over."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    from luna.sources.radiation import setup_radiation
    n = 8
    rad = setup_radiation(_rad_cfg(n), NamedSharding(_mesh((1, 1, 1)),
                                                     P(None, "x", "y", "z")),
                          is_main=False)
    U, R = _gas(n), _rad(n)

    def loss(U, R):
        U2, R2 = rad["before"](U, R, 1e-3)
        return jnp.sum(U2[-1] ** 2) + jnp.sum(R2[:, 0] ** 2)

    gU, gR = jax.jit(jax.grad(loss, argnums=(0, 1)))(U, R)
    assert bool(jnp.all(jnp.isfinite(gU))) and bool(jnp.all(jnp.isfinite(gR)))
    assert float(jnp.max(jnp.abs(gU))) > 0.0 and float(jnp.max(jnp.abs(gR))) > 0.0
