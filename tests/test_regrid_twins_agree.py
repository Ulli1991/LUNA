"""The replicated and sharded regrids must produce the same hierarchy.

They were two copies of one algorithm -- same flag, build, match, preserve,
pad, neighbours, parent-slots sequence -- differing only in the child-builder
and preserve kernels. `amr.shard.regrid_sharded` now delegates to
`amr.driver.regrid` with those two swapped in, so agreement is the property the
deduplication has to guarantee, and a misordered adapter would show here.

On one device the two see identical data, so topology must match EXACTLY
(coords, neighbours, parent slots, active counts are integers -- no tolerance
is appropriate) and the block data to round-off.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp



from luna.amr import driver                                   # noqa: E402
from luna.amr.config import AMRConfig                         # noqa: E402

N, BOX = 32, 1.0


def _sine_U(n):
    x = (jnp.arange(n) + 0.5) * (BOX / n)
    k = 2 * jnp.pi / BOX
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    U = jnp.zeros((5, n, n, n))
    U = U.at[0].set(1.0 + 0.5 * jnp.sin(k * X) * jnp.sin(k * Y) * jnp.sin(k * Z))
    return U.at[4].set(1.0)


@pytest.mark.parametrize("periodic,boundary", [(True, 0), (False, 1)])
def test_sharded_regrid_matches_replicated(periodic, boundary):
    from luna.amr import shard
    from jax.sharding import Mesh
    cfg = AMRConfig(ndim=3, box=(BOX,) * 3, base_grid=(N,) * 3,
                    block_shape=(16,) * 3, max_levels=1, boundary=boundary)
    h0 = driver.build_hierarchy(_sine_U(N), cfg, thresholds=[1e-9],
                                criterion="density", periodic=periodic)
    if len(h0) < 2:
        pytest.skip("hierarchy did not refine")
    # the AMR sharded path shards over BLOCKS (runner_amr.py:160), not the
    # hydro grid's x/y/z -- a ("x","y","z") mesh fails in_specs on 'blocks'
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("blocks",))

    rep = driver.regrid(h0, cfg, thresholds=[1e-9], periodic=periodic,
                        criterion="density")
    shd = shard.regrid_sharded(h0, cfg, mesh, thresholds=[1e-9],
                               periodic=periodic, criterion="density")

    assert len(rep) == len(shd)
    for L, (a, b) in enumerate(zip(rep, shd)):
        assert int(a.n_active) == int(b.n_active), f"level {L} n_active"
        # topology is integer: exact, no tolerance
        assert np.array_equal(np.asarray(a.coords), np.asarray(b.coords)), f"L{L} coords"
        assert np.array_equal(np.asarray(a.neighbors),
                              np.asarray(b.neighbors)), f"L{L} neighbors"
        assert np.array_equal(np.asarray(a.parent_slots),
                              np.asarray(b.parent_slots)), f"L{L} parent_slots"
        da, db = np.asarray(a.data), np.asarray(b.data)
        assert da.shape == db.shape, f"L{L} data shape"
        assert np.allclose(da, db, rtol=0, atol=0, equal_nan=True), f"L{L} data"
