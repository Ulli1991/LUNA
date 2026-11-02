"""utils/sorting.stable_argsort_uint must be a stable permutation on tied
integer keys -- checked against numpy, on whatever backend runs the tests
(ROCm jax 0.10.0's own sorts are not: audit follow-up 2026-09-18)."""
import numpy as np


import jax.numpy as jnp

from luna.utils.sorting import bits_for, stable_argsort_uint


def test_stable_argsort_uint_matches_numpy_on_tied_keys():
    rng = np.random.default_rng(3)
    n = 393216
    key = rng.integers(0, 32 * 34 * 34, size=n).astype(np.int32)
    key[rng.random(n) < 0.33] = 2 ** 20 - 1                 # 'inactive' tail keys
    o = np.asarray(stable_argsort_uint(jnp.asarray(key), bits_for(2 ** 20 - 1)))
    assert np.array_equal(np.sort(o), np.arange(n))          # a permutation
    np.testing.assert_array_equal(o, np.argsort(key, kind="stable"))


def test_native_self_check_and_dispatch_agree_with_numpy():
    """Whatever the build, argsort_uint must return a stable permutation: the
    native kernel when native_argsort_ok() passes, the radix otherwise."""
    from luna.utils.sorting import argsort_uint, native_argsort_ok
    ok = native_argsort_ok()
    print(f"native argsort ok on this build: {ok}")
    rng = np.random.default_rng(5)
    n = 300000
    key = rng.integers(0, 50000, size=n).astype(np.int32)
    o = np.asarray(argsort_uint(jnp.asarray(key), bits_for(50000), ok))
    np.testing.assert_array_equal(o, np.argsort(key, kind="stable"))


def test_stable_argsort_uint_small_and_all_equal():
    for key in (np.array([3, 1, 2, 1, 0], np.int32), np.zeros(7, np.int32),
                np.arange(10, dtype=np.int32)[::-1].copy()):
        o = np.asarray(stable_argsort_uint(jnp.asarray(key), 4))
        np.testing.assert_array_equal(o, np.argsort(key, kind="stable"))
