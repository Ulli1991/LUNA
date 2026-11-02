"""Slab M2 closure from an input-convex network for the entropy dual h(f, g) (Amos+2017).

a = grad h_theta, m3 read off exp(a.v), so the closure is realizable for any weights.
See docs/RADIATION.md; training in scripts/validation/runs/m2_learned_train.py.
"""
import jax
import jax.numpy as jnp

from luna.radiation.m2 import _MU, _W, clamp, normalised_moments

WIDTHS = (32, 32, 32)


def init(key, widths=WIDTHS, scale=0.1):
    """Return input-convex network parameters; hidden-path weights are raw (softplus applied)."""
    # first layer is a 2-tuple: a zero-sized weight array segfaults ROCm gradients
    params = []
    prev = 0
    for w in widths:
        key, k1, k2 = jax.random.split(key, 3)
        Wy = jax.random.normal(k2, (w, 2)) * scale
        if prev == 0:
            params.append((Wy, jnp.zeros((w,))))
        else:
            params.append((jax.random.normal(k1, (w, prev)) * scale, Wy,
                           jnp.zeros((w,))))
        prev = w
    # output layer starts "off" (a = 0, isotropic): a random start saturates the closure
    params.append((jnp.full((1, prev), -8.0), jnp.zeros((1, 2)),
                   jnp.zeros((1,))))
    return params


def potential(params, fg):
    """Return h_theta(f, g), convex in (f, g) by construction."""
    z = None
    for i, layer in enumerate(params):
        if len(layer) == 2:                          # input-only first layer
            Wy, b = layer
            a = Wy @ fg + b
        else:
            Wz, Wy, b = layer
            a = Wy @ fg + b + jax.nn.softplus(Wz) @ z    # non-negative path
        z = a if i == len(params) - 1 else jax.nn.softplus(a)
    return z[0]


def multipliers(params, fg):
    """Return a = grad h_theta by forward Jacobian propagation (no nested jax.grad)."""
    J = None
    z = None
    for i, layer in enumerate(params):
        if len(layer) == 2:
            Wy, b = layer
            a, JW = Wy @ fg + b, Wy
        else:
            Wz, Wy, b = layer
            W = jax.nn.softplus(Wz)
            a, JW = Wy @ fg + b + W @ z, W @ J + Wy
        if i == len(params) - 1:
            return JW[0]
        s = jax.nn.sigmoid(a)                      # d softplus
        z, J = jax.nn.softplus(a), s[:, None] * JW
    raise ValueError("an input-convex network needs at least one layer")


def closure(params, fg):
    """Return m3/m0 of exp(a.v) with a from the network."""
    return normalised_moments(multipliers(params, fg))[0][2]


def multipliers_batched(params, fg):
    """Return a = grad h_theta for a field, fg (N, 2), as matmuls over N (not a vmap)."""
    J = z = None
    for i, layer in enumerate(params):
        if len(layer) == 2:
            Wy, b = layer
            a = fg @ Wy.T + b                              # (N, w)
            JW = jnp.broadcast_to(Wy, (fg.shape[0],) + Wy.shape)
        else:
            Wz, Wy, b = layer
            W = jax.nn.softplus(Wz)
            a = fg @ Wy.T + b + z @ W.T
            JW = jnp.einsum("ij,njk->nik", W, J) + Wy      # (N, w, 2)
        if i == len(params) - 1:
            return JW[:, 0, :]
        sig = jax.nn.sigmoid(a)
        z, J = jax.nn.softplus(a), sig[:, :, None] * JW
    raise ValueError("an input-convex network needs at least one layer")


def closure_batched(params, fg):
    """Return m3/m0 for a field of states, fg (N, 2)."""
    ab = multipliers_batched(params, fg)                   # (N, 2)
    zq = ab[:, :1] * _MU[None, :] + ab[:, 1:] * (_MU ** 2)[None, :]
    zq = zq - jnp.max(zq, axis=1, keepdims=True)
    w = jnp.exp(zq) * _W[None, :]
    return (w @ (_MU ** 3)) / jnp.sum(w, axis=1)


def third_moment(params, U, batched=True):
    """Return m3 for a slab state; drop-in for `m2.third_moment`."""
    m0 = jnp.maximum(U[0], 1e-300)
    f, g = clamp(U[1] / m0, U[2] / m0)
    fg = jnp.stack([f.reshape(-1), g.reshape(-1)], axis=1)
    h = (closure_batched(params, fg) if batched
         else jax.vmap(lambda p: closure(params, p))(fg))
    return m0 * h.reshape(f.shape)


def closure_fn(params, batched=True):
    """Return a closure for `m2.step(..., closure_fn=...)`."""
    return lambda U: third_moment(params, U, batched)


def sample_states(key, n, margin=0.02):
    """Sample n realizable (f, g) at least `margin` inside the boundary (multipliers diverge there)."""
    k1, k2 = jax.random.split(key)
    f = jax.random.uniform(k1, (n,), minval=-1.0 + margin, maxval=1.0 - margin)
    t = jax.random.uniform(k2, (n,), minval=margin, maxval=1.0 - margin)
    g = f * f + t * (1.0 - f * f)
    return jnp.stack([f, g], axis=1)


def sobolev_loss(params, fg, h_star, a_star, weight=1.0):
    """Return the Sobolev loss on h and grad h against the exact solve."""
    h = jax.vmap(lambda p: potential(params, p))(fg)
    a = jax.vmap(lambda p: multipliers(params, p))(fg)
    return (jnp.mean((h - h_star) ** 2)
            + weight * jnp.mean(jnp.sum((a - a_star) ** 2, axis=1)))
