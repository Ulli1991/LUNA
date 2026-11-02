"""Learned 3-D M2 closure: an MLP predicts the seven free components of trace-free m3.

m3 = (1/5)(d_ij m1_k + d_jk m1_i + d_ik m1_j) + Qt_ijk; the trace part is imposed.
See docs/RADIATION.md.
"""
import jax
import jax.numpy as jnp

from luna.radiation.m2_3d import density_scale, targets

WIDTHS = (64, 64)

# the seven independent index triples of a totally symmetric trace-free 3-tensor
_FREE = ((0, 0, 0), (1, 1, 1), (0, 0, 1), (0, 1, 1),
         (0, 0, 2), (1, 1, 2), (0, 1, 2))


def init(key, widths=WIDTHS, scale=0.5):
    """Return MLP parameters 8 -> widths -> 7, output layer zero."""
    params, prev = [], 8
    for w in widths:
        key, k = jax.random.split(key)
        params.append((jax.random.normal(k, (w, prev)) * scale / prev ** 0.5,
                       jnp.zeros((w,))))
        prev = w
    params.append((jnp.zeros((7, prev)), jnp.zeros((7,))))   # start at zero
    return params


BOUND = 2.0
NET_DTYPE = jnp.float32


def free_components(params, u, dtype=NET_DTYPE):
    """Evaluate the MLP in `dtype` (None: u's dtype) and return in u's dtype."""
    if dtype is None:
        return _free_components(params, u)
    out = _free_components(jax.tree.map(lambda x: x.astype(dtype), params),
                           u.astype(dtype))
    return out.astype(u.dtype)


def _free_components(params, u):
    """Return the seven free components, squashed into (-BOUND, BOUND); u: (N, 8).

    The bound keeps the flux subluminal; an unbounded head diverges in long rollouts.
    """
    h = u
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W.T + b)
    W, b = params[-1]
    return BOUND * jnp.tanh((h @ W.T + b) / BOUND)


def _rebuild(free, m1):
    """Return m3 (N,3,3,3) from the seven free components and the m1 trace part.

    Stacked, not scattered: the `.at[].set` version segfaults on ROCm.
    """
    v = {t: free[:, i] for i, t in enumerate(_FREE)}
    v[(0, 2, 2)] = -(v[(0, 0, 0)] + v[(0, 1, 1)])
    v[(1, 2, 2)] = -(v[(0, 0, 1)] + v[(1, 1, 1)])
    v[(2, 2, 2)] = -(v[(0, 0, 2)] + v[(1, 1, 2)])
    Q = jnp.stack([jnp.stack([jnp.stack(
        [v[tuple(sorted((i, j, k)))] for k in range(3)])
        for j in range(3)]) for i in range(3)])           # (3,3,3,N)
    Q = jnp.moveaxis(Q, 3, 0)                             # (N,3,3,3)
    d = jnp.eye(3)
    trace = (jnp.einsum("ij,nk->nijk", d, m1)
             + jnp.einsum("jk,ni->nijk", d, m1)
             + jnp.einsum("ik,nj->nijk", d, m1)) / 5.0
    return Q + trace


BLOCK = 65536
MAX_BLOCKS = 16       # a cap on the loop trip count, not a tuning knob


def third_moment(params, U, block=BLOCK, dtype=NET_DTYPE):
    """Return m3_ijk for a field by one forward pass.

    `block`: lax.map over cell blocks, at most MAX_BLOCKS of them (ROCm segfault
    workaround); block=0 inlines.
    """
    m0 = density_scale(U)                     # m0 where realizable, never ~0 next to large moments
    shape = m0.shape
    u = targets(U).reshape(8, -1).T                      # (N, 8)
    m1 = (U[1:4] / m0).reshape(3, -1).T                  # (N, 3)
    n = u.shape[0]
    if block and n > block:
        nb = min(-(-n // block), MAX_BLOCKS)
        size = -(-n // nb)
        pad = nb * size - n
        if pad:
            u = jnp.concatenate([u, jnp.zeros((pad, 8))])
            m1 = jnp.concatenate([m1, jnp.zeros((pad, 3))])
        Q = jax.lax.map(lambda a: _rebuild(
            free_components(params, a[0], dtype), a[1]),
                        (u.reshape(nb, size, 8), m1.reshape(nb, size, 3)))
        Q = Q.reshape(-1, 3, 3, 3)[:n]
    else:
        Q = _rebuild(free_components(params, u, dtype), m1)
    return jnp.moveaxis(Q, 0, 3).reshape(3, 3, 3, *shape) * m0


def closure_fn(params, block=BLOCK, dtype=NET_DTYPE):
    """Return a closure for `m2_3d.step(..., closure_fn=...)`."""
    return lambda U: third_moment(params, U, block, dtype)


def loss(params, u, m1, target_free):
    """Return the MSE on the seven free components, in the state's precision."""
    return jnp.mean((free_components(params, u, None) - target_free) ** 2)


def free_of(Q):
    """Return the seven free components of a (N,3,3,3) trace-free part."""
    return jnp.stack([Q[:, i, j, k] for i, j, k in _FREE], axis=1)


def trace_free_part(m3, m1):
    """Return m3 minus the trace part fixed by m1."""
    d = jnp.eye(3)
    trace = (jnp.einsum("ij,nk->nijk", d, m1)
             + jnp.einsum("jk,ni->nijk", d, m1)
             + jnp.einsum("ik,nj->nijk", d, m1)) / 5.0
    return m3 - trace
