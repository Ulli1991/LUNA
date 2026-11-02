"""Sort-free stable argsort (cumsum LSD radix) for bounded non-negative integer keys.

Some ROCm jax builds return wrong device sorts; this needs only cumsum.
"""
import jax.numpy as jnp


def _stable_partition(order, bit):
    """Stably partition the current order by one key bit (zeros first)."""
    ones = bit.astype(jnp.int32)
    n = order.shape[0]
    n_zero = n - jnp.sum(ones)
    pos_zero = jnp.cumsum(1 - ones) - 1                # rank among the zeros
    pos_one = n_zero + jnp.cumsum(ones) - 1            # rank among the ones
    dest = jnp.where(bit, pos_one, pos_zero)
    return jnp.zeros_like(order).at[dest].set(order)


def stable_argsort_uint(key, n_bits):
    """Return indices that stably sort ``key`` (non-negative ints < 2**n_bits)."""
    key = key.astype(jnp.int32)
    order = jnp.arange(key.shape[0], dtype=jnp.int32)
    # unrolled: a scan carry that becomes shard-varying trips shard_map's typing
    for b in range(int(n_bits)):
        k = key[order]
        order = _stable_partition(order, (k >> b) & 1)
    return order


def bits_for(max_value):
    """Return the number of bits needed for keys in [0, max_value]."""
    return max(int(max_value).bit_length(), 1)


_NATIVE_OK = None


def native_argsort_ok(n=1 << 18):
    """Return whether this build's device argsort is a permutation on n tied keys (cached).

    Must run eagerly, outside any trace.
    """
    global _NATIVE_OK
    if _NATIVE_OK is None:
        import numpy as np
        rng = np.random.default_rng(0)
        key = jnp.asarray(rng.integers(0, 40000, size=n).astype(np.int32))
        o = np.asarray(jnp.argsort(key, stable=True))
        _NATIVE_OK = bool(np.array_equal(np.sort(o), np.arange(n)))
        if not _NATIVE_OK:
            print(f"⚠ device argsort is not a permutation on this jax/ROCm build "
                  f"(n={n}); using the cumsum radix argsort (20x slower per sort)",
                  flush=True)
    return _NATIVE_OK


def argsort_uint(key, n_bits, native_ok):
    """Stable argsort: native when ``native_ok`` (from native_argsort_ok()), else the radix fallback."""
    if native_ok:
        return jnp.argsort(key, stable=True)
    return stable_argsort_uint(key, n_bits)
