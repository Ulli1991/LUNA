"""The sharded point scatter (utils.grid_scatter) the worker's winds and
ionising sources go through: on a (2,2,2) mesh each device adds only the
cells in its own block, and the result is the single-device one -- with
windows straddling block edges and corners, and periodic wrap."""
import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from luna.radiation.ionization import deposit_sources, deposit_updates  # noqa: E402
from luna.sources.feedback import make_wind_injector  # noqa: E402

N = 16
DX = 1.0 / N

pytestmark = pytest.mark.skipif(jax.device_count() < 8, reason="a (2,2,2) mesh needs 8 devices")


def _mesh():
    from jax.experimental import mesh_utils
    return Mesh(mesh_utils.create_device_mesh((2, 2, 2), jax.devices()[:8]), ("x", "y", "z"))


def _stars():
    # on a block corner (0.5, 0.5, 0.5), on a face, and across the periodic wrap
    pos = np.array([[0.5, 0.5, 0.5], [0.5, 0.26, 0.74], [0.02, 0.98, 0.5], [0.3, 0.3, 0.3]])
    vel = np.array([[0.1, 0.0, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.3], [0.0, 0.0, 0.0]])
    dm = np.array([0.02, 0.03, 0.01, 0.0])                 # the last one does not blow
    vt = np.array([5.0, 8.0, 3.0, 4.0])
    return pos, vel, dm, vt, np.ones(4, bool)


def _gas():
    rng = np.random.default_rng(0)
    U = np.zeros((5, N, N, N))
    U[0] = 1.0 + 0.1 * rng.random((N, N, N))
    U[1:4] = 0.05 * rng.standard_normal((3, N, N, N))
    U[4] = 2.0 + 0.5 * (U[1:4] ** 2).sum(0) / U[0]
    return U


@pytest.mark.parametrize("add_heat", [False, True])
def test_sharded_winds_are_the_single_device_ones(add_heat):
    sh = NamedSharding(_mesh(), P(None, "x", "y", "z"))
    one = make_wind_injector((N, N, N), (DX, DX, DX), periodic=True, add_heat=add_heat)
    many = make_wind_injector((N, N, N), (DX, DX, DX), periodic=True, add_heat=add_heat,
                              sharding=sh)
    U = _gas()
    a, na, dma, Ea = one(jnp.asarray(U), *_stars())
    b, nb, dmb, Eb = many(jax.device_put(jnp.asarray(U), sh), *_stars())
    assert b.sharding == sh
    np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-13, atol=1e-15)
    assert na == nb == 3
    want = (0.06, 0.5 * (0.02 * 25 + 0.03 * 64 + 0.01 * 9))
    assert (dma, Ea) == pytest.approx(want, rel=1e-14) and (dmb, Eb) == pytest.approx(want, rel=1e-14)
    np.testing.assert_allclose(float(jnp.sum(b[0] - U[0])) * DX ** 3, 0.06, rtol=1e-12)


def test_sharded_deposit_is_the_single_device_one():
    sh = NamedSharding(_mesh(), P("x", "y", "z"))
    from luna.utils.grid_scatter import make_grid_scatter
    pos, _, _, _, _ = _stars()
    rates = np.array([3.0, 0.0, 2.0, 5.0])
    want = np.asarray(deposit_sources((N, N, N), (DX, DX, DX), jnp.asarray(pos),
                                      jnp.asarray(rates), boundary=0))
    idx, vals = deposit_updates((N, N, N), (DX, DX, DX), pos, rates, boundary=0)
    got = make_grid_scatter(sh)(jax.device_put(jnp.zeros((N, N, N)), sh), idx, vals)
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(float(jnp.sum(got)) * DX ** 3, 10.0, rtol=1e-13)


def test_short_spec_on_a_slab_mesh():
    """A 2-GPU run lays the grid out as P("x") (trailing axes left out); the
    scatter pads the spec (it raised IndexError on the first ionising source)."""
    from luna.utils.grid_scatter import make_grid_scatter
    mesh = Mesh(np.array(jax.devices()[:2]).reshape(2, 1, 1), ("x", "y", "z"))
    pos, _, _, _, _ = _stars()
    rates = np.array([3.0, 0.0, 2.0, 5.0])
    want = np.asarray(deposit_sources((N, N, N), (DX, DX, DX), jnp.asarray(pos),
                                      jnp.asarray(rates), boundary=0))
    idx, vals = deposit_updates((N, N, N), (DX, DX, DX), pos, rates, boundary=0)
    sh = NamedSharding(mesh, P("x"))
    got = make_grid_scatter(sh)(jax.device_put(jnp.zeros((N, N, N)), sh), idx, vals)
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-13, atol=1e-15)
    shU = NamedSharding(mesh, P(None, "x"))
    U = _gas()
    a = make_wind_injector((N, N, N), (DX, DX, DX), periodic=True)(jnp.asarray(U), *_stars())[0]
    b = make_wind_injector((N, N, N), (DX, DX, DX), periodic=True, sharding=shU)(
        jax.device_put(jnp.asarray(U), shU), *_stars())[0]
    np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-13, atol=1e-15)


def _whole_block_reference(U, idx, vals):
    """The hook as it was: add, then E_new = E_old + ke(new) - ke(old) over the
    whole grid (numpy, one device)."""
    from luna.sources.feedback import _keep_internal
    new = U.copy()
    np.add.at(new, (slice(None), idx[:, 0], idx[:, 1], idx[:, 2]), vals)
    return np.asarray(_keep_internal(jnp.asarray(U), jnp.asarray(new)))


@pytest.mark.parametrize("mesh", [None, (2, 2, 2)])
def test_the_hook_touches_only_the_updated_cells(mesh):
    """The keep-internal hook runs on the touched cells alone (a whole-block
    pass rewrote every cell's energy on 1024^3 blocks, fb_1024). Same values as
    the whole-grid formula there -- including a cell hit twice, and cells hit
    from across a block corner -- and every other cell bitwise unchanged."""
    from luna.sources.feedback import _keep_internal
    from luna.utils.grid_scatter import make_grid_scatter
    U = _gas()
    rng = np.random.default_rng(3)
    idx = np.array([[7, 7, 7], [8, 8, 8], [7, 8, 7], [8, 8, 8], [0, 15, 3], [15, 0, 0]])
    vals = rng.normal(0.0, 0.05, (5, len(idx)))
    vals[0] = np.abs(vals[0])
    want = _whole_block_reference(U, idx, vals)
    sh = None if mesh is None else NamedSharding(_mesh(), P(None, "x", "y", "z"))
    field = jnp.asarray(U) if sh is None else jax.device_put(jnp.asarray(U), sh)
    got = np.asarray(make_grid_scatter(sh, n_comp=5, after=_keep_internal)(field, idx, vals))
    touched = np.zeros((N, N, N), bool)
    touched[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    np.testing.assert_allclose(got[:, touched], want[:, touched], rtol=1e-13, atol=1e-15)
    np.testing.assert_array_equal(got[:, ~touched], U[:, ~touched])
