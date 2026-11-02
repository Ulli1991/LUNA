"""The convergence-controlled warm solve (gravity_warm_tol) and the extrapolated seed (gravity_warm_extrapolate)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests import _gravity_warm as gw
from luna.gravity import poisson_multigrid as pm

N = 32


def _problem(seed=7):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((N, N, N))
    return jnp.asarray(f - f.mean())


def _rel_res(phi, f):
    return float(pm.rms(f - pm.laplacian_7pt(phi, 1.0 / N, pm.make_halo_3d(0))) / pm.rms(f))


@pytest.mark.parametrize("mixed", [False, True])
def test_cycles_stop_at_the_first_cycle_below_tolerance(mixed):
    f = _problem()
    halo, top, tol = pm.make_halo_3d(0), pm._max_levels(N) - 1, 1e-6
    phi, k, rel = pm.cycles_to_tol(jnp.zeros_like(f), f, 1.0 / N, halo, top, 1, 1, None, tol, 20,
                                   mixed=mixed)
    k = int(k)
    assert 1 < k < 20 and float(rel) <= tol
    assert abs(_rel_res(phi, f) - float(rel)) <= 1e-3 * float(rel)
    fewer, _, rel_fewer = pm.cycles_to_tol(jnp.zeros_like(f), f, 1.0 / N, halo, top, 1, 1, None,
                                           tol, k - 1, mixed=mixed)
    assert float(rel_fewer) > tol


def test_cycles_stop_at_the_cap():
    f = _problem()
    _, k, rel = pm.cycles_to_tol(jnp.zeros_like(f), f, 1.0 / N, pm.make_halo_3d(0),
                                 pm._max_levels(N) - 1, 1, 1, None, 1e-30, 3)
    assert int(k) == 3 and float(rel) > 1e-30


def _rel_res_top(phi, f, boundary):
    """|f - lap phi| / |f| under the solver's own top halo."""
    from luna.gravity.backend import _dense_halos
    f = jnp.asarray(f)
    top, _ = _dense_halos(f, 1.0 / N, boundary)
    return float(pm.rms(f - pm.laplacian_7pt(jnp.asarray(phi), 1.0 / N, top)) / pm.rms(f))


@pytest.mark.parametrize("agg_n", [32, 8])
@pytest.mark.parametrize("boundary", [0, 3])
@pytest.mark.parametrize("mixed", [False, True])
def test_sharded_adaptive_solve_matches_one_device(boundary, mixed, agg_n):
    """agg_n 32 gathers the whole 32^3 grid at once; 8 runs sharded levels down to an 8^3 root grid."""
    n = jax.device_count()
    if n < 2:
        pytest.skip("needs 2 or more devices")
    from luna.gravity.particle_mesh_blocks import make_block_mesh
    pdims = (2, 2, 2) if n >= 8 else (2, 1, 1)
    mesh = make_block_mesh(jax.devices()[:int(np.prod(pdims))], pdims)
    cfg = gw.config(N, boundary, gravity_warm_tol=1e-4, gravity_warm_extrapolate=True,
                    gravity_proper_mixed=mixed, gravity_mg_agg_n=agg_n)
    f = np.asarray(_problem()) + (3.0 if boundary == 3 else 0.0)
    seed, prev = 0.9 * gw.converged(f, boundary), 0.8 * gw.converged(f, boundary)
    out = []
    for be in (gw.backend(cfg), gw.backend(cfg, mesh)):
        psi = be.solve(gw.place(be, f), gw.place(be, seed), prev=gw.place(be, prev), q=1.5)
        out.append((np.asarray(psi), int(be.last_info[0]), float(be.last_info[1])))
    (d, kd, rd), (s, ks, rs) = out
    assert kd == ks and kd > 1 and rd <= 1e-4
    assert abs(rd - rs) <= (1e-5 if mixed else 1e-8) * rd
    assert np.sqrt(np.mean((d - s) ** 2) / np.mean(d ** 2)) < (1e-7 if mixed else 1e-10)
    assert abs(_rel_res_top(d, f, boundary) - rd) <= 1e-3 * rd


def test_extrapolated_seed():
    from luna.gravity.poisson_multigrid import extrapolated_seed
    a, b = jnp.full((2,), 3.0), jnp.full((2,), 1.0)
    assert np.allclose(extrapolated_seed(a, b, 0.5), 4.0)
    assert extrapolated_seed(a, None, 0.5) is a


def test_history_rules():
    """The seed is extrapolated only with two solves behind it, a dt ratio in range and no fresh event."""
    from luna.gravity.kdk import Gravity
    from luna.setup.state import State
    f = np.asarray(_problem())
    be = gw.backend(gw.config(N, 0, gravity_warm_extrapolate=True, gravity_warm_dt_ratio_max=4.0))
    stage = Gravity(be, None, [], True)
    psi = be.solve(jnp.asarray(f), None)
    state = State(psi=psi)
    _, state = stage._solve(state, jnp.asarray(f), 0.1)
    assert stage.history_valid is False and state.psi_prev is psi and state.dt_prev == 0.1
    state = state.update(psi=psi)
    _, state = stage._solve(state, jnp.asarray(f), 0.2)
    assert stage.history_valid is True
    _, state = stage._solve(state, jnp.asarray(f), 1.0)
    assert stage.history_valid is False          # dt ratio 5 > 4
    _, state = stage._solve(state, jnp.asarray(f), 1.0)
    assert stage.history_valid is True
    stage.events[0] = True
    _, state = stage._solve(state, jnp.asarray(f), 1.0)
    assert stage.history_valid is True           # the difference behind it predates the event
    _, state = stage._solve(state, jnp.asarray(f), 1.0)
    assert stage.history_valid is False          # now it spans the event
    _, state = stage._solve(state, jnp.asarray(f), 1.0)
    assert stage.history_valid is True
    for n in range(3):                           # events on three steps in a row
        stage.events[0] = True
        _, state = stage._solve(state, jnp.asarray(f), 1.0)
        assert stage.history_valid is (n == 0)
    _, state = stage._solve(state, jnp.asarray(f), 1.0)
    assert stage.history_valid is False
    _, state = stage._solve(state, jnp.asarray(f), 1.0)
    assert stage.history_valid is True


def test_autodiff_runs_fixed_cycles():
    """An autodiff run (adjoint backend, traced stage) solves with fixed cycles from the carried psi, in both passes."""
    from luna.gravity.backend import LocalGravity, PoissonOpts, PROPER
    from luna.gravity.kdk import Gravity
    from luna.gravity.particle_mesh import ParticleMeshGravity
    from luna.setup.cosmo_common import gravity_precision_for
    from luna.setup.state import State
    cfg = gw.config(N, 0)
    assert cfg.gravity_warm_tol > 0 and cfg.gravity_warm_extrapolate
    be = LocalGravity(ParticleMeshGravity(cfg), 1.0, PoissonOpts.from_config(cfg, cfg.gravity_warm_vcycles),
                      gravity_precision_for(cfg, False), PROPER, 1.0, adjoint=True)
    assert be.opts.warm_tol == 0.0
    f = jnp.asarray(_problem())
    stage = Gravity(be, None, [], True, traced=True)
    state = State(psi=be.solve(f, None), psi_prev=0.5 * be.solve(f, None), dt_prev=1.0)
    psi, _ = stage._solve(state, f, 1.0)
    assert be.last_info is None and stage.history_valid is None
    fixed = be.solve(f, state.psi)
    assert np.array_equal(np.asarray(psi), np.asarray(fixed))
    g = jax.grad(lambda s: jnp.sum(be.solve(s, state.psi) ** 2))(f)
    assert np.all(np.isfinite(np.asarray(g)))
