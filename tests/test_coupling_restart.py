"""A coupled run comes back from its checkpoint: the gas, the M1 carry, the
sinks, every star with what Torch parked on it, the LUNA-side IMF queues and
their random stream, and the file numbering Torch keys its own files off.
"""
import json
import os
from dataclasses import replace

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

import jax.numpy as jnp  # noqa: E402

from tests._helpers import _build_cfg_for_test  # noqa: E402
from amuse_luna.particles import PARKED_ATTRS  # noqa: E402
from amuse_luna.worker import LunaWorker, SINK_SPECIES  # noqa: E402


def _cfg(out, **kw):
    base = dict(
        test_problem="Soundwave-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=16, Ny=16, Nz=16, Lx=1.0, Ly=1.0, Lz=1.0, gamma=1.4,
        ORDER=1, LIMITER=1, GRAVITATIONAL_G=1.0, POISSON_BOUNDARY=0,
        SELF_GRAVITY=True, sinks_enabled=True, sink_max_count=4,
        sink_creation_density=1e30,            # none form on their own
        radiation_enabled=True, rad_chat=10.0, rad_E0=1e-3, rad_substeps=1,
        write_snapshots=True, time_bet_snaps=1e-3, output_dir=str(out))
    return _build_cfg_for_test(**{**base, **kw})


def _run(out):
    """A worker with a few steps behind it, two stars, one sink, a queue."""
    w = LunaWorker(_cfg(out))
    w.initialize_code(build_ic=True)
    w.commit_parameters()
    w.evolve_model(2e-3)
    tags = w.new_particle(np.array([0.2, 0.7]), np.array([0.3, 0.6]), np.array([0.4, 0.5]))
    w.set_particle_mass(tags, np.array([3.0, 5.0]))
    w.set_particle_velocity(tags, np.array([1.0, -1.0]), np.array([0.5, 0.5]), np.array([0.0, 2.0]))
    for i, a in enumerate(PARKED_ATTRS):
        w.set_particle_attr(a, tags, np.array([10.0 + i, 20.0 + i]))
    w.clear_new_tags()
    # a sink in slot 2 of the carry, with the moments a star maker reads
    s = w.state._s
    part = s.species[SINK_SPECIES]
    part = replace(part, pos=part.pos.at[2].set(jnp.array([0.5, 0.5, 0.5])),
                   p=part.p.at[2].set(jnp.array([0.1, 0.2, 0.3])),
                   mass=part.mass.at[2].set(7.0), active=part.active.at[2].set(1.0))
    s.species[SINK_SPECIES] = part
    w._sf_queues = {2: np.array([0.5, 1.5, 2.5])}
    w._sf_rng.random(5)
    w._fb["n_sn"] = 3
    return w


def _particles(w):
    tags = w.get_particle_tags()
    return {a: np.asarray(w.get_particle_attr(a, tags)) for a in PARKED_ATTRS} | {
        "tags": np.asarray(tags), "mass": np.asarray(w.get_particle_mass(tags)),
        "pos": np.asarray(w.get_particle_position(tags)),
        "vel": np.asarray(w.get_particle_velocity(tags))}


def test_a_sink_keeps_the_tag_of_its_slot(tmp_path):
    w = _run(tmp_path)
    w.set_particle_pointers("sink")
    assert w.get_particle_tags() == [2]
    w.set_particle_pointers("mass")


def test_the_coupled_state_comes_back(tmp_path):
    w = _run(tmp_path)
    # Torch's output check at a new tick: LUNA writes, reports the next number
    n = w.IO_out("chk")
    assert w._chk_written == 0 and n == 1      # numbered 0, 1, 2 ... as written
    d = os.path.join(str(tmp_path), "checkpoints")
    step = w.state.step
    assert os.path.isfile(os.path.join(d, f"coupled_{step:06d}.npz"))
    with open(os.path.join(d, f"coupled_{step:06d}.json")) as fh:
        assert json.load(fh)["chk"] == n - 1
    before = _particles(w)
    U, t, R = np.asarray(w.state.U), w.state.t, w.state._s.sources["radiation"]

    r = LunaWorker(_cfg(tmp_path, restart=True))
    r.initialize_code()
    r.commit_parameters()
    assert r.state.step == step and r.state.t == pytest.approx(t)
    np.testing.assert_array_equal(np.asarray(r.state.U), U)
    for a, b in zip(jax.tree_util.tree_leaves(R),
                    jax.tree_util.tree_leaves(r.state._s.sources["radiation"])):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    after = _particles(r)
    for k in before:
        np.testing.assert_array_equal(after[k], before[k], err_msg=k)
    r.set_particle_pointers("sink")
    assert r.get_particle_tags() == [2]
    assert float(np.ravel(r.get_particle_mass(2))[0]) == pytest.approx(7.0)
    r.set_particle_pointers("mass")
    assert list(r._sf_queues) == [2]
    np.testing.assert_array_equal(r._sf_queues[2], w._sf_queues[2])
    assert r._sf_rng.random(3).tolist() == w._sf_rng.random(3).tolist()
    assert r._fb["n_sn"] == 3
    # the numbering continues: Torch's next chk/plt is the one after the last
    assert r.IO_num("chk") == n
    assert r.IO_out("chk") == n            # same tick: nothing written again
    # and a later tick writes the next one. The round-trip values above are
    # arbitrary, and as winds (a code mass per bridge interval at ~20 c_s into
    # an adiabatic 16^3 box) they break the gas with or without heat -- off
    # for the step, which is only here for the numbering.
    r.set_particle_attr("wind_mass", r.get_particle_tags(), np.zeros(len(r.get_particle_tags())))
    r.evolve_model(r.state.t + 2e-3)
    assert r.IO_out("chk") > n


def test_torch_and_luna_must_restart_together(tmp_path, monkeypatch):
    w = _run(tmp_path)
    w.IO_out("chk")
    monkeypatch.chdir(tmp_path)
    with open("flash.par", "w") as fh:
        fh.write("restart = .true.\ncheckpointFileNumber = 7\nplotFileNumber = 8\n")
    with pytest.raises(ValueError, match="restart together"):
        LunaWorker(_cfg(tmp_path)).initialize_code()
    with pytest.raises(ValueError, match="checkpointFileNumber"):
        LunaWorker(_cfg(tmp_path, restart=True)).initialize_code()


def test_the_wall_clock_writes_checkpoints_between_ticks_and_old_sets_go(tmp_path):
    """At a tiny step the snapshot tick can be a day apart; the wall-clock
    interval still checkpoints, numbered on from the last, and only the newest
    max_checkpoints sets stay."""
    w = LunaWorker(_cfg(tmp_path, enable_checkpointing=True, checkpoint_wall_time_interval=1e-9, max_checkpoints=2,
                        time_bet_snaps=1e3))
    w.initialize_code(build_ic=True)
    w.commit_parameters()
    nums = []
    for _ in range(4):
        w.evolve_model(w.state.t + 1e-3)
        nums.append(w.IO_out("chk"))
    assert nums == [1, 2, 3, 4]          # the first on the tick, then every call on the clock
    d = os.path.join(str(tmp_path), "checkpoints")
    assert len([f for f in os.listdir(d) if f.startswith("coupled_") and f.endswith(".json")]) == 2
    r = LunaWorker(_cfg(tmp_path, restart=True, time_bet_snaps=1e3))
    r.initialize_code()
    assert r.IO_num("chk") == nums[-1]
