"""The sinks' field on the gas (species.sinks._sink_field_on_grid): the sum
over the slots, SLOT_CHUNK at a time, is the direct sum -- with a slot count
that is not a multiple of the chunk and inactive slots -- and the program's
scratch stays a few grids, not one per slot (33 GiB at 512^3 with 32 slots
did not fit a fragmented pool at 1024^3 on 8 GPUs)."""
import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from luna.species.sinks import Sinks  # noqa: E402


def _sinks(N, n_max):
    return Sinks(N3=(N, N, N), box3=(1.0, 1.0, 1.0), G=1.0, boundary=0, n_max=n_max,
                 softening=0.05, accretion_radius=0.0, creation_density=0.0,
                 creation_separation=0.0, create_frac=0.0, is_main=False)


def test_chunked_field_is_the_direct_sum():
    N, n_max = 12, 7                                   # 7 slots: one padded chunk
    s = _sinks(N, n_max)
    rng = np.random.default_rng(3)
    pos = rng.random((n_max, 3))
    mass = rng.random(n_max) + 0.5
    active = np.array([1, 1, 0, 1, 1, 0, 1], float)
    got = [np.asarray(g) for g in jax.jit(s._sink_field_on_grid)(
        jnp.asarray(pos), jnp.asarray(mass), jnp.asarray(active))]
    c = (np.arange(N) + 0.5) / N
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    want = [np.zeros((N, N, N)) for _ in range(3)]
    for p, m, a in zip(pos, mass, active):
        d = (X - p[0], Y - p[1], Z - p[2])
        if s.wrap:                                     # periodic box: nearest image
            d = tuple(x - np.round(x) for x in d)
        inv_r3 = a * m * (d[0] ** 2 + d[1] ** 2 + d[2] ** 2 + 0.05 ** 2) ** -1.5
        for k in range(3):
            want[k] -= d[k] * inv_r3
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, rtol=1e-12, atol=1e-12)


def test_scratch_is_a_few_grids_not_one_per_slot():
    N, n_max = 32, 32
    s = _sinks(N, n_max)
    args = (jnp.zeros((n_max, 3)), jnp.ones(n_max), jnp.ones(n_max))
    mem = jax.jit(s._sink_field_on_grid).lower(*args).compile().memory_analysis()
    if mem is None:
        pytest.skip("this backend reports no memory analysis")
    grids = mem.temp_size_in_bytes / (N ** 3 * 8)
    assert grids <= 8, f"scratch of {grids:.1f} grids for {n_max} slots"


def test_skipping_empty_chunks_changes_no_bit(monkeypatch):
    """Slots 0-3 empty, 4-7 one live sink, the rest empty: the chunks with no
    live sink are skipped, and the field is bitwise the one that sums them."""
    import luna.species.sinks as sk
    N, n_max = 12, 10
    s = _sinks(N, n_max)
    rng = np.random.default_rng(5)
    pos = jnp.asarray(rng.random((n_max, 3)))
    mass = jnp.asarray(rng.random(n_max) + 0.5)
    active = jnp.asarray(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], float))
    out = {}
    for skip in (True, False):
        monkeypatch.setattr(sk, "SKIP_EMPTY", skip)
        out[skip] = [np.asarray(g) for g in jax.jit(s._sink_field_on_grid)(pos, mass, active)]
    assert np.abs(out[True][0]).max() > 0.0
    for a, b in zip(out[True], out[False]):
        np.testing.assert_array_equal(a, b)
