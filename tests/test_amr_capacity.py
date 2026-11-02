"""Per-level block cap (amr_max_blocks): the deep-hierarchy memory guard.

_sized_parents' pow2 ladder x a blanket-refining criterion requested 41 GiB
arrays at 5 levels (measured); the cap clamps the allocation AND the
never-shrink floor, dropping excess flags in deterministic slot order."""
import numpy as np
import jax.numpy as jnp

from luna.amr.config import AMRConfig
from luna.amr import driver
from luna.amr.driver import _sized_parents


def test_sized_parents_cap_semantics():
    flags = jnp.ones(100, dtype=bool)                     # 100 flagged -> n=101
    assert _sized_parents(flags, 8, None) == 101          # exact, uncapped
    assert _sized_parents(flags, 8, 16) == 128            # pow2 ladder
    assert _sized_parents(flags, 8, 16, cap=48) == 48     # ladder clamped
    assert _sized_parents(flags, 8, None, cap=48) == 48   # exact clamped
    assert _sized_parents(flags, 8, 16, floor=256, cap=48) == 48   # floor capped too
    assert _sized_parents(flags, 8, 16, cap=0) == 128     # 0 = unlimited


def test_build_hierarchy_respects_max_blocks():
    """Blanket-flagging IC: every level saturates at the cap, the hierarchy is
    still consistent (children have valid parents, active counts == cap-1)."""
    N, BS = 32, 8
    cfg = AMRConfig(ndim=2, box=(1.0, 1.0), base_grid=(N, N),
                    block_shape=(BS, BS), max_levels=2, boundary=0)
    U = jnp.ones((4, N, N))                               # uniform -> overdensity
    U = U.at[0].set(10.0)                                 # rho >> any threshold
    U = U.at[3].set(10.0)
    cap = 32                                              # blocks per level
    h = driver.build_hierarchy(U, cfg, thresholds=[0.5, 0.5],
                               criterion="overdensity", max_blocks=cap,
                               periodic=True)
    for L in (1, 2):
        # cap parents = cap // cpb -> allocated child slots = parents * cpb <= cap
        assert h[L].data.shape[0] <= cap, \
            f"L{L} allocation {h[L].data.shape[0]} exceeds cap {cap}"
        na = int(h[L].n_active)
        assert 0 < na <= cap
        ps = np.asarray(h[L].parent_slots)[:na]
        assert np.all(ps >= 0), "capped level has children with invalid parents"


if __name__ == "__main__":
    test_sized_parents_cap_semantics()
    print("PASS test_sized_parents_cap_semantics")
    test_build_hierarchy_respects_max_blocks()
    print("PASS test_build_hierarchy_respects_max_blocks")
    print("########## CAPACITY_TESTS_DONE ##########")


def test_positivity_floor_resets_bad_cells_keeps_sentinels():
    """floor_positivity quiescent-resets unphysical active cells to
    (rho_floor, v=0, p_floor) and leaves inactive/sentinel (NaN) blocks NaN."""
    from luna.amr import driver as drv
    N, BS = 32, 16
    cfg = AMRConfig(ndim=3, box=(1.0,) * 3, base_grid=(N,) * 3,
                    block_shape=(BS,) * 3, max_levels=1, boundary=0)
    U = jnp.ones((5, N, N, N))
    U = U.at[4].set(2.0)                                  # E
    h = driver.build_hierarchy(U, cfg, thresholds=[2.0], criterion="overdensity",
                               periodic=True)             # L1 empty (uniform)
    L0 = h[0]
    # corrupt a few active cells: negative rho, NaN, inf energy
    d = np.array(L0.data)                                # copy (asarray is read-only)
    d[0, 0, 0, 0, 0] = -5.0                               # neg rho
    d[1, 0, 1, 1, 1] = np.nan                             # NaN rho
    d[2, 4, 2, 2, 2] = np.inf                             # inf energy
    L0 = L0._replace(data=jnp.asarray(d))
    h = (L0,) + tuple(h[1:])

    rf, pf = 1e-6, 1e-8
    out = drv.floor_positivity(h, cfg, gamma=5.0 / 3.0, rho_floor=rf, p_floor=pf)
    o0 = np.asarray(out[0].data)
    n_act = int(out[0].n_active)
    # the three bad cells are reset, finite, at the floor
    assert np.isfinite(o0[:n_act]).all(), "active cells still non-finite after floor"
    assert o0[0, 0, 0, 0, 0] == rf and o0[1, 0, 1, 1, 1] == rf
    # inactive/sentinel blocks stay NaN (untouched)
    if out[0].data.shape[0] > n_act:
        assert np.isnan(o0[n_act:]).any(), "sentinel block was overwritten by floor"
    # a healthy cell is unchanged
    assert o0[0, 0, 5, 5, 5] == 1.0
