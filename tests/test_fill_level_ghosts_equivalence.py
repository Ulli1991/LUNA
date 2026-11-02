"""`fill_level_ghosts` must reproduce the three inline sequences BITWISE.

It replaced the same-level -> coarse -> physical sequence written out in
amr/hybrid (hydro; since removed), amr/driver's subcycled ghost_prim, and amr/poisson
(potential). That refactor claims to preserve behaviour exactly, and "the AMR
suites still pass" does not establish that -- it establishes that whatever the
suites cover still works.

So each variant is checked against the ORIGINAL sequence reproduced verbatim
below, asserting bit-for-bit equality. If someone later changes the shared
helper, these fail regardless of whether any downstream test happens to
exercise that path.

The subcycled variant deliberately lacks the coords/level physical fill that
the hydro one has (audit item 6). That asymmetry is asserted here too, so
fixing it is a visible change rather than a silent one.
"""

import numpy as np
import pytest

import jax.numpy as jnp



from luna.amr import driver                                            # noqa: E402
from luna.amr.config import AMRConfig                                  # noqa: E402
from luna.amr.halo import (fill_ghost_from_coarse, fill_ghost_from_coarse_sparse,  # noqa: E402
                      fill_ghost_physical, fill_ghost_same_level,
                      fill_level_ghosts)

N, BOX = 32, 1.0


def _sine_U(n):
    x = (jnp.arange(n) + 0.5) * (BOX / n)
    k = 2 * jnp.pi / BOX
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    U = jnp.zeros((5, n, n, n))
    U = U.at[0].set(1.0 + 0.5 * jnp.sin(k * X) * jnp.sin(k * Y) * jnp.sin(k * Z))
    return U.at[4].set(1.0)


@pytest.fixture(scope="module")
def hier():
    # thresholds is ONE entry per level (driver.py:415 indexes thresholds[L]),
    # so max_levels=1 + one low threshold gives the two levels this needs.
    cfg = AMRConfig(ndim=3, box=(BOX,) * 3, base_grid=(N,) * 3,
                    block_shape=(16,) * 3, max_levels=1, boundary=0)
    h = driver.build_hierarchy(_sine_U(N), cfg, thresholds=[1e-9],
                               criterion="density", periodic=True)
    if len(h) < 2:
        pytest.skip("hierarchy did not refine; nothing to compare")
    return h, cfg


def _field(lvl):
    """Any smooth per-block field of the right shape works for an equality test."""
    return jnp.asarray(lvl.data[:, 0:1]) * 1.0 + 0.25


def test_potential_variant_is_bitwise(hier):
    """amr/poisson._ghost_phi: same-level -> sparse coarse (linear) -> physical."""
    h, cfg = hier
    lvl, parent = h[1], None
    f = _field(lvl)
    want = fill_ghost_same_level(f, lvl.neighbors, cfg)
    want = fill_ghost_physical(want, f, cfg)
    got = fill_level_ghosts(f, lvl, cfg, parent=parent, interp="linear")
    assert np.array_equal(np.asarray(want), np.asarray(got), equal_nan=True)


def test_subcycled_variant_is_bitwise(hier):
    """amr/driver.ghost_prim: same-level -> wall-band physical -> coarse
    (piecewise-constant, so C/F fluxes telescope against restriction) ->
    physical. The wall-band fill was added by audit item 6."""
    h, cfg = hier
    lvl = h[1]
    f = _field(lvl)
    parent = fill_ghost_same_level(_field(h[0]), h[0].neighbors, cfg)
    want = fill_ghost_same_level(f, lvl.neighbors, cfg)
    want = fill_ghost_physical(want, f, cfg, coords=lvl.coords, level=1)
    want = fill_ghost_from_coarse_sparse(want, lvl.coords, lvl.parent_slots,
                                         parent, cfg)
    want = fill_ghost_physical(want, f, cfg)
    got = fill_level_ghosts(f, lvl, cfg, level=1, parent=parent,
                            physical_first=True)
    assert np.array_equal(np.asarray(want), np.asarray(got), equal_nan=True)


def test_subcycled_still_differs_from_hydro_in_interp(hier):
    """The two remain distinct: the subcycle keeps piecewise-constant C/F
    ghosts on purpose, the hydro path prolongs linearly."""
    h, cfg = hier
    lvl = h[1]
    f = _field(lvl)
    parent = fill_ghost_same_level(_field(h[0]), h[0].neighbors, cfg)
    sub = fill_level_ghosts(f, lvl, cfg, level=1, parent=parent, physical_first=True)
    hyd = fill_level_ghosts(f, lvl, cfg, level=1, parent=parent,
                            interp="linear", physical_first=True)
    assert not np.array_equal(np.asarray(sub), np.asarray(hyd), equal_nan=True)


def test_hydro_variant_is_bitwise(hier):
    """The hydro variant (was amr/hybrid): the extra pre-coarse physical fill WITH coords and level."""
    h, cfg = hier
    lvl = h[1]
    f = _field(lvl)
    parent = fill_ghost_same_level(_field(h[0]), h[0].neighbors, cfg)
    want = fill_ghost_same_level(f, lvl.neighbors, cfg)
    want = fill_ghost_physical(want, f, cfg, coords=lvl.coords, level=1)
    want = fill_ghost_from_coarse_sparse(want, lvl.coords, lvl.parent_slots,
                                         parent, cfg, interp="linear")
    want = fill_ghost_physical(want, f, cfg)
    got = fill_level_ghosts(f, lvl, cfg, level=1, parent=parent,
                            interp="linear", physical_first=True)
    assert np.array_equal(np.asarray(want), np.asarray(got), equal_nan=True)


def test_dense_parent_variant_is_bitwise(hier):
    """Level 1 with a dense parent (was amr/hybrid) via fill_ghost_from_coarse.

    parent_dense is the DENSE base-level field (the removed hybrid.py
    passed cons_to_prim(U0)), not the block-structured one the sparse path takes.
    """
    h, cfg = hier
    lvl = h[1]
    f = _field(lvl)
    W0 = _sine_U(N)[0:1]
    want = fill_ghost_same_level(f, lvl.neighbors, cfg)
    want = fill_ghost_physical(want, f, cfg, coords=lvl.coords, level=1)
    want = fill_ghost_from_coarse(want, lvl.coords, W0, 1, cfg, boundary=cfg.boundary)
    want = fill_ghost_physical(want, f, cfg)
    got = fill_level_ghosts(f, lvl, cfg, level=1, parent_dense=W0,
                            boundary=cfg.boundary, physical_first=True)
    assert np.array_equal(np.asarray(want), np.asarray(got), equal_nan=True)


def test_subcycled_and_hydro_variants_really_differ(hier):
    """Teeth: if these ever agree, the parameters have stopped doing anything."""
    h, cfg = hier
    lvl = h[1]
    f = _field(lvl)
    parent = fill_ghost_same_level(_field(h[0]), h[0].neighbors, cfg)
    sub = fill_level_ghosts(f, lvl, cfg, level=1, parent=parent)
    hyd = fill_level_ghosts(f, lvl, cfg, level=1, parent=parent,
                            interp="linear", physical_first=True)
    assert not np.array_equal(np.asarray(sub), np.asarray(hyd), equal_nan=True)
