"""A negative pressure must make the CFL timestep non-finite on every path.

cs = sqrt(gamma P / rho) is NaN for P < 0. Inside one device the min
propagates the NaN, but the cross-device min (RCCL) drops NaN operands: the
128^3 slab A/B (2026-09-18) ran 87 steps with two ranks at cs = NaN and a
finite global dt, carrying 3e5 negative-pressure cells to the end. compute_dt
now maps NaN to -inf before any reduction so the runners' finite-dt guards
fire. Block AMR (driver.compute_dt / compute_dt_subcycle) had the same hole
one level up: nanmin over all slots, there to skip the NaN sentinel/padding
blocks, also skipped a P < 0 cell; it now masks inactive slots by coords and
maps NaN to -inf. The sharded halves need >= 2 devices (CPU-8); the AMR
runner case is GPU only."""

import functools

import numpy as np
import pytest

import jax
import jax.numpy as jnp



from luna.time_integration.integrators import _make_compute_dt_fn  # noqa: E402

GAMMA, CFL, DX = 5.0 / 3.0, 0.4, 0.1


def _state(negative_cell):
    N = 8
    rho = jnp.ones((N, N, N))
    mom = jnp.zeros((N, N, N))
    E = jnp.ones((N, N, N))                      # P = (gamma-1) E > 0 everywhere
    if negative_cell:
        E = E.at[1, 2, 3].set(-1.0)              # one cell with P < 0 -> cs = NaN
    return jnp.stack([rho, mom, mom, mom, E])


def test_negative_pressure_gives_minus_inf_single_device():
    f = _make_compute_dt_fn(GAMMA, CFL, 3, DX, DX, DX)
    assert np.isfinite(float(f(_state(False))))
    assert float(f(_state(True))) == -np.inf


def test_negative_pressure_survives_the_cross_device_min():
    devs = jax.devices()
    if len(devs) < 2:
        pytest.skip("needs >= 2 devices")
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    mesh = Mesh(np.array(devs[:2]).reshape(2, 1, 1), ("x", "y", "z"))
    f = _make_compute_dt_fn(GAMMA, CFL, 3, DX, DX, DX, mesh=mesh)
    sh = NamedSharding(mesh, P(None, "x", "y", "z"))
    assert np.isfinite(float(f(jax.device_put(_state(False), sh))))
    # the bad cell sits on device 0 only; device 1 reports a finite minimum
    assert float(f(jax.device_put(_state(True), sh))) == -np.inf


# --- block AMR: driver.compute_dt / compute_dt_subcycle, sharded and in the runner ---

AMR_GAMMA = 1.4


@functools.lru_cache(maxsize=None)
def _kh_hierarchy(pad_to=8):
    """Two-level periodic KH hierarchy (tests/test_amr_multilevel.py IC) with both levels refined."""
    from luna.amr import driver
    from luna.amr.config import AMRConfig
    from tests.test_amr_multilevel import _kh_ic
    cfg = AMRConfig(ndim=2, box=(1.0, 1.0), base_grid=(32, 32), block_shape=(8, 8), max_levels=2)
    h = driver.build_hierarchy(_kh_ic(cfg, AMR_GAMMA), cfg, thresholds=[0.05, 0.05], pad_to=pad_to)
    assert all(int(lvl.n_active) > 0 for lvl in h)
    return h, cfg


def _old_nanmin_dt(levels, cfg, CFL, subcycle):
    """The pre-fix rule (driver.py nanmin over every slot), jitted like the runner's dt_fn: the bitwise reference."""
    return jax.jit(lambda lv: _old_nanmin_dt_body(lv, cfg, CFL, subcycle))(levels)


def _old_nanmin_dt_body(levels, cfg, CFL, subcycle):
    from luna.amr.hydro import cons_to_prim_blocks
    from luna.time_integration.integrators import cell_dt
    dt = jnp.inf
    for L, lvl in enumerate(levels):
        W = cons_to_prim_blocks(lvl.data, AMR_GAMMA)
        cs = jnp.sqrt(AMR_GAMMA * W[:, -1] / W[:, 0])
        local = cell_dt([W[:, 1 + d] for d in range(cfg.ndim)], cs, cfg.level_dx(L)[:cfg.ndim], 1.0)
        cand = ((cfg.refinement_ratio ** L) if subcycle else 1) * CFL * jnp.nanmin(local)
        dt = jnp.minimum(dt, jnp.where(jnp.isnan(cand), jnp.inf, cand))
    return dt


def _amr_dt(levels, cfg, subcycle):
    from luna.amr import driver
    fn = driver.compute_dt_subcycle if subcycle else driver.compute_dt
    return jax.jit(lambda lv: fn(lv, cfg, gamma=AMR_GAMMA, CFL=CFL))(levels)


def _with_negative_pressure(levels, L, block):
    """Set E = -1 (rho > 0, so P < 0, cs = NaN) in cell (0, 0) of active block `block` on level L."""
    lvl = levels[L]
    assert int(lvl.coords[block, 0]) >= 0
    data = lvl.data.at[block, 3, 0, 0].set(-1.0)
    return tuple(lv._replace(data=data) if i == L else lv for i, lv in enumerate(levels))


@pytest.mark.parametrize("subcycle", [False, True])
def test_amr_valid_hierarchy_dt_is_bitwise_the_old_dt(subcycle):
    h, cfg = _kh_hierarchy()
    assert all(bool(jnp.all(jnp.isnan(lvl.data[-1]))) for lvl in h)   # NaN sentinel slots present
    new, old = float(_amr_dt(h, cfg, subcycle)), float(_old_nanmin_dt(h, cfg, CFL, subcycle))
    assert np.isfinite(new) and new > 0
    assert new == old


@pytest.mark.parametrize("subcycle", [False, True])
@pytest.mark.parametrize("level", [0, 1, 2])
def test_amr_negative_pressure_gives_minus_inf(level, subcycle):
    h, cfg = _kh_hierarchy()
    bad = _with_negative_pressure(h, level, int(h[level].n_active) - 1)
    assert np.isfinite(float(_old_nanmin_dt(bad, cfg, CFL, subcycle)))   # the bug: nanmin dropped it
    assert float(_amr_dt(bad, cfg, subcycle)) == -np.inf


@pytest.mark.parametrize("subcycle", [False, True])
def test_amr_negative_pressure_survives_block_sharding(subcycle):
    devs = jax.devices()
    if len(devs) < 2:
        pytest.skip("needs >= 2 devices")
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    D = len(devs)
    h, cfg = _kh_hierarchy(pad_to=8 if 8 % D == 0 else D)
    mesh = Mesh(np.asarray(devs), ("blocks",))
    blk = NamedSharding(mesh, P("blocks"))

    def shard(levels):
        return tuple(lv._replace(data=jax.device_put(lv.data, blk)) for lv in levels)

    L = 1
    block = int(h[L].n_active) - 1
    assert block // (h[L].data.shape[0] // D) > 0          # the bad block is not on device 0
    assert float(_amr_dt(shard(h), cfg, subcycle)) == float(_amr_dt(h, cfg, subcycle))
    assert float(_amr_dt(shard(_with_negative_pressure(h, L, block)), cfg, subcycle)) == -np.inf


@pytest.mark.skipif(jax.default_backend() == "cpu",
                    reason="GPU only: the runner compile alone is ~9 s on CPU, over the file's CPU budget")
def test_amr_runner_stops_on_negative_pressure(tmp_path, monkeypatch):
    """main.run on a 2-level AMR KH whose IC has one P < 0 cell must raise, not finish."""
    import luna.ic_generation.ic as ic
    from luna.main import run
    from tests._helpers import _build_cfg_for_test

    orig = ic.initial_conditions

    def bad_ic(cfg):
        U = np.array(orig(cfg))
        U[3, 5, 8] = -1.0                                   # E < 0 on the shear layer: P < 0
        return U

    monkeypatch.setattr(ic, "initial_conditions", bad_ic)
    cfg = _build_cfg_for_test(
        test_problem="Kelvin-Helmholtz", SPATIAL_DIMENSION=2, Nx=32, Ny=32, Lx=1.0, Ly=1.0,
        BOUNDARY=0, t_end=0.05, time_bet_snaps=0.05, AMR=True, amr_block_shape=8,
        amr_levels=1, amr_thresholds=[0.05], output_dir=str(tmp_path / "out"),
        clear_output_dir=True)
    with pytest.raises(RuntimeError, match="non-finite time"):
        run(cfg)
