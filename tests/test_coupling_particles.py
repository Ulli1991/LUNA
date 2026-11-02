"""Tests for the fixed-capacity particle store (Torch/AMUSE coupling, Phase 2).

Two things matter here:
1. Correctness of the host-side tag↔slot bookkeeping (add / remove / get / set).
2. The non-negotiable performance invariant: a jitted kernel over the store's
   device arrays must compile **exactly once** across add/remove/accrete cycles
   — i.e. the particle dimension never changes shape. See
   `docs/TORCH.md` (Phase 2).
"""

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from amuse_luna.particles import (  # noqa: E402
    ParticleCapacityError,
    ParticleStore,
    default_sink_schema,
    default_star_schema,
)


def _star_store(capacity=16):
    return ParticleStore(capacity, default_star_schema(), name="stars")


def test_add_assigns_tags_and_scatters_attributes():
    s = _star_store()
    pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mass = np.array([10.0, 20.0])
    tags = s.add(position=pos, mass=mass)

    assert tags.tolist() == [0, 1]
    assert s.n_active == 2
    np.testing.assert_allclose(s.get("position", tags), pos)
    np.testing.assert_allclose(s.get("mass", tags), mass)
    # Unspecified attributes default to zero.
    np.testing.assert_allclose(s.get("velocity", tags), np.zeros((2, 3)))


def test_get_set_roundtrip_by_tag():
    s = _star_store()
    tags = s.add(mass=np.array([1.0, 2.0, 3.0]))
    s.set("velocity", tags[1], np.array([[7.0, 8.0, 9.0]]))
    np.testing.assert_allclose(s.get("velocity", tags[1:2]), [[7.0, 8.0, 9.0]])
    # Other particles untouched.
    np.testing.assert_allclose(s.get("velocity", tags[[0, 2]]), np.zeros((2, 3)))


def test_remove_frees_slot_keeps_other_tags_valid():
    s = _star_store()
    tags = s.add(mass=np.array([1.0, 2.0, 3.0]))   # tags 0,1,2 in slots 0,1,2
    s.remove(tags[1])                               # free slot 1
    assert s.n_active == 2
    assert s.active_tags().tolist() == [0, 2]
    # Surviving particles keep their data addressable by the same tags.
    np.testing.assert_allclose(s.get("mass", tags[[0, 2]]), [1.0, 3.0])
    # Removed slot is reused by the next add (capacity is conserved).
    new = s.add(mass=np.array([99.0]))
    assert s.slots_for(new)[0] == 1
    np.testing.assert_allclose(s.get("mass", new), [99.0])


def test_capacity_is_a_hard_ceiling():
    s = _star_store(capacity=2)
    s.add(mass=np.array([1.0, 2.0]))
    with pytest.raises(ParticleCapacityError):
        s.add(mass=np.array([3.0]))


def test_inactive_slots_have_zero_mass_after_removal():
    s = _star_store()
    tags = s.add(mass=np.array([5.0, 6.0]))
    s.remove(tags[0])
    # The raw device array slot must be zeroed so masked gravity sees nothing.
    slot0 = 0
    assert float(s.arrays["mass"][slot0]) == 0.0
    assert bool(s.arrays["active"][slot0]) is False


def test_sink_schema_has_accreted_gas_moments():
    s = ParticleStore(8, default_sink_schema(), name="sinks")
    for attr in ("mean_cs", "mean_velocity", "var_velocity", "angular_momentum"):
        assert attr in s.schema
    tags = s.add(mass=np.array([1.0]))
    s.set("mean_cs", tags, np.array([0.3]))
    np.testing.assert_allclose(s.get("mean_cs", tags), [0.3])


def test_no_recompile_across_add_remove_cycles():
    """A jitted kernel over the static arrays must trace exactly ONCE, no matter
    how many particles are added/removed — the whole point of fixed capacity."""
    s = _star_store(capacity=32)

    trace_count = {"n": 0}

    @jax.jit
    def total_active_mass(arrays):
        trace_count["n"] += 1  # runs once per trace (== per compile)
        return jnp.sum(jnp.where(arrays["active"], arrays["mass"], 0.0))

    # Exercise it across a sequence of mutations.
    masses_seen = []
    for k in range(5):
        tags = s.add(mass=np.array([float(k + 1)] * (k + 1)))
        masses_seen.append(float(total_active_mass(s.arrays)))
        if k % 2 == 0:
            s.remove(tags[0])
            masses_seen.append(float(total_active_mass(s.arrays)))

    # Sanity: the kernel actually tracked the changing active mass.
    assert masses_seen[0] == pytest.approx(1.0)        # one particle, mass 1
    assert masses_seen[1] == pytest.approx(0.0)        # then removed
    # The invariant: ONE compile despite all the add/remove churn.
    assert trace_count["n"] == 1


def test_jit_op_can_update_values_in_place():
    """A jitted op may change attribute values (e.g. a kick); update_arrays
    accepts the result as long as shapes/keys are preserved."""
    s = _star_store()
    tags = s.add(velocity=np.zeros((2, 3)), mass=np.array([1.0, 1.0]))

    @jax.jit
    def kick(arrays, dv):
        v = arrays["velocity"] + arrays["active"][:, None] * dv
        return {**arrays, "velocity": v}

    s.update_arrays(kick(s.arrays, jnp.array([1.0, 0.0, 0.0])))
    np.testing.assert_allclose(s.get("velocity", tags),
                               np.tile([1.0, 0.0, 0.0], (2, 1)))
