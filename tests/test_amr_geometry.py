"""Tests for the dimension-generic AMR block-geometry core (amr/geometry.py).

Two layers:
  * intrinsic 2D+3D unit tests (conservation, round-trips, topology);
  * 2D parity vs upstream JAX-AMR's prolong/restrict outputs (vendored, reduced, in
    luna-reference-solutions), to prove the generic re-implementation reproduces them.
"""

import jax.numpy as jnp
import numpy as np
import pytest


from luna.amr.config import AMRConfig
from luna.amr import geometry as g



# --------------------------------------------------------------------------
# prolong / restrict
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ndim", [2, 3])
def test_restrict_prolong_roundtrip_is_identity(ndim):
    # prolong copies a coarse value into 2**ndim children; restrict averages
    # them back -> exact identity.
    rng = np.random.default_rng(0)
    shape = (3, 2) + (4,) * ndim  # (n_blocks, n_vars, *block_shape)
    x = jnp.asarray(rng.standard_normal(shape))
    back = g.restrict(g.prolong(x, ndim), ndim)
    np.testing.assert_allclose(np.asarray(back), np.asarray(x), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("ndim", [2, 3])
def test_prolong_shape_and_constant_children(ndim):
    shape = (2, 1) + (3,) * ndim
    x = jnp.asarray(np.arange(np.prod(shape)).reshape(shape).astype(float))
    p = g.prolong(x, ndim)
    assert p.shape == shape[:2] + tuple(2 * s for s in shape[2:])
    # Each coarse cell's 2**ndim children are identical.
    r = g.restrict(p, ndim)
    np.testing.assert_allclose(np.asarray(r), np.asarray(x), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("ndim", [2, 3])
def test_restrict_conserves_sum_with_volume_weight(ndim):
    # Sum over fine cells * fine-volume == sum over coarse * coarse-volume.
    rng = np.random.default_rng(1)
    fine = jnp.asarray(rng.random((1, 1) + (6,) * ndim))
    coarse = g.restrict(fine, ndim)
    # coarse cell volume = 2**ndim * fine cell volume (unit fine volume here)
    fine_mass = np.asarray(fine).sum()
    coarse_mass = np.asarray(coarse).sum() * (2 ** ndim)
    np.testing.assert_allclose(coarse_mass, fine_mass, rtol=1e-12)


# --------------------------------------------------------------------------
# Morton codes
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ndim", [2, 3])
def test_morton_roundtrip(ndim):
    rng = np.random.default_rng(2)
    coords = jnp.asarray(rng.integers(0, 2000, size=(50, ndim)))
    code = g.morton_encode(coords)
    back = g.morton_decode(code, ndim)
    np.testing.assert_array_equal(np.asarray(back), np.asarray(coords))


@pytest.mark.parametrize("ndim", [2, 3])
def test_morton_is_unique(ndim):
    rng = np.random.default_rng(3)
    coords = np.unique(rng.integers(0, 100, size=(500, ndim)), axis=0)
    code = np.asarray(g.morton_encode(jnp.asarray(coords)))
    assert len(np.unique(code)) == len(coords)


@pytest.mark.parametrize("ndim", [2, 3])
def test_morton_uint32_path_no_overflow(ndim):
    # Explicitly exercise the float32-mode code dtype (production normally runs
    # fp64/uint64, but the float32 path must not silently collide). 16 bits/dim
    # in 2D, 10 in 3D -> coords must stay below those budgets.
    import jax.numpy as _jnp
    bits = 16 if ndim == 2 else 10
    hi = (1 << bits) - 1
    rng = np.random.default_rng(4)
    coords = np.unique(rng.integers(0, hi, size=(800, ndim)), axis=0)
    code = g.morton_encode(jnp.asarray(coords), dtype=_jnp.uint32, bits=bits)
    assert code.dtype == _jnp.uint32
    assert len(np.unique(np.asarray(code))) == len(coords)
    back = g.morton_decode(code, ndim, dtype=_jnp.uint32, bits=bits)
    np.testing.assert_array_equal(np.asarray(back), coords)


# --------------------------------------------------------------------------
# Config + block coordinates
# --------------------------------------------------------------------------
def test_config_derived_geometry_2d():
    cfg = AMRConfig(ndim=2, box=(1.0, 1.0), base_grid=(64, 64), block_shape=(16, 16),
                    max_levels=2)
    assert cfg.base_blocks == (4, 4)
    assert cfg.children_per_block == 4
    np.testing.assert_allclose(cfg.level_dx(0), (1 / 64, 1 / 64))
    np.testing.assert_allclose(cfg.level_dx(1), (1 / 128, 1 / 128))


def test_config_rejects_indivisible():
    with pytest.raises(ValueError):
        AMRConfig(ndim=2, box=(1.0, 1.0), base_grid=(60, 64), block_shape=(16, 16))


@pytest.mark.parametrize("ndim", [2, 3])
def test_block_cell_centers_in_block(ndim):
    cfg = AMRConfig(ndim=ndim, box=(1.0,) * ndim, base_grid=(32,) * ndim,
                    block_shape=(8,) * ndim, max_levels=1)
    coords = jnp.array([1] * ndim)
    centers = g.block_cell_centers(coords, level=0, cfg=cfg)
    assert centers.shape == (ndim,) + (8,) * ndim
    dx = cfg.level_dx(0)[0]
    block_w = dx * 8
    # block (1,1,..) spans [block_w, 2*block_w); first center at block_w+dx/2.
    for d in range(ndim):
        lo = np.asarray(centers[d]).min()
        hi = np.asarray(centers[d]).max()
        np.testing.assert_allclose(lo, block_w + 0.5 * dx, rtol=1e-12)
        np.testing.assert_allclose(hi, 2 * block_w - 0.5 * dx, rtol=1e-12)


# --------------------------------------------------------------------------
# Refinement criterion
# --------------------------------------------------------------------------
def test_refinement_mask_flags_and_buffers():
    cfg = AMRConfig(ndim=2, box=(1.0, 1.0), base_grid=(32, 32), block_shape=(8, 8),
                    max_levels=1, n_ghost=1)
    blk = jnp.zeros((2, 1, 8, 8))
    blk = blk.at[0, 0, 4, 4].set(10.0)  # one hot cell in block 0
    mask = g.refinement_grid_mask(blk, cfg, threshold=1.0, var_index=0, buffer=1,
                                  contrast_mode="value")
    m0 = np.asarray(mask[0])
    assert m0[4, 4] == 1.0
    # buffer=1 is a BOX (Moore) dilation: the hot cell grows to a full 3x3,
    # including diagonals (deliberately stronger than upstream's cross kernel,
    # so refined patches keep margin at corners too).
    assert m0[3:6, 3:6].sum() == 9.0
    assert m0.sum() == 9.0
    assert np.asarray(mask[1]).sum() == 0.0  # untouched block


def test_refinement_mask_nan_block_flags_nothing():
    cfg = AMRConfig(ndim=2, box=(1.0, 1.0), base_grid=(32, 32), block_shape=(8, 8))
    blk = jnp.full((2, 1, 8, 8), jnp.nan)
    mask = g.refinement_grid_mask(blk, cfg, threshold=0.0, buffer=0,
                                  contrast_mode="value")
    assert np.asarray(mask).sum() == 0.0  # NaN > t is False everywhere


# 2D parity vs upstream JAX-AMR (Wen & Luo, https://github.com/JA4S/JAX-AMR, MIT: src/luna/amr/LICENSE-JAX-AMR)
#
# The upstream outputs on fixed inputs are vendored, reduced, in luna-reference-solutions (PARITY): written
# by upstream e8bbcf0 interpolate_coarse_to_fine (amr.py:289) and the restriction of interpolate_fine_to_coarse
# (amr.py:303-305), CPU FP64, provenance in the npz key 'provenance'. No upstream import at test time.
PARITY = "jaxamr/amr_parity_2d/parity_2d.npz"


def test_parity_prolong_restrict_2d():
    from luna.reference_data import reference_path
    ref = np.load(reference_path(PARITY))
    # prolong: upstream is jnp.kron with ones((2, 2)), a copy of each coarse value: bitwise
    ours_p = np.asarray(g.prolong(jnp.asarray(ref["coarse"]), ndim=2))
    np.testing.assert_array_equal(ours_p, ref["prolonged"])
    # restrict: the mean of 2 x 2 fine cells; summed in another order each sum of 4 is within 3 u sum|x| of the
    # exact one (u = 2^-53, Higham 2002 eq. 4.4), so two orders differ by at most 6 u sum|x| / 4 per cell
    fine = ref["fine"]
    ours_r = np.asarray(g.restrict(jnp.asarray(fine), ndim=2))
    s = fine.shape
    bound = 6 * 2.0 ** -53 * np.abs(fine).reshape(s[0], s[1], s[2] // 2, 2, s[3] // 2, 2).sum(axis=(3, 5)) / 4
    assert np.all(np.abs(ours_r - ref["restricted"]) <= bound)
    # teeth: a restriction that takes one fine cell instead of the mean is far outside the bound
    assert not np.all(np.abs(fine[:, :, ::2, ::2] - ref["restricted"]) <= bound)
