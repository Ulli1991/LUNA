"""Train the 3-D learned M2 closure (`m2_learned_3d`) against the entropy
closure (`m2_3d`), the way the shipped `luna/radiation/m2_closure_w32.npz`
was made: 12000 realizable states (positive mixtures of 1-3 beams and an
isotropic part), their entropy m3, 3000 Adam steps on the seven free
components, held-out error on 2000 more.

    JAX_PLATFORMS=cpu python -m scripts.validation.runs.m2_learned_3d_train [WIDTH] [OUT.npz]

On the CPU backend: parameter gradients segfault on the ROCm stack.
"""
import sys
import time

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from luna.radiation import m2_3d as M, m2_learned_3d as L  # noqa: E402
from luna.sources.ml_corrector import adam_init, adam_step  # noqa: E402

N_TRAIN, N_HELD, STEPS = 12000, 2000, 3000


def sample(rng, n):
    """(m1, m2) of positive mixtures of beams plus an isotropic part, m0 = 1."""
    m1 = np.zeros((n, 3))
    m2 = np.zeros((n, 3, 3))
    for i in range(n):
        k = rng.integers(1, 4)
        d = rng.normal(size=(k, 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        w = rng.uniform(0.1, 1.0, size=k)
        w /= w.sum()
        iso = rng.uniform(0.02, 0.9)
        m1[i] = (1 - iso) * (w[:, None] * d).sum(0)
        m2[i] = (1 - iso) * np.einsum("k,ki,kj->ij", w, d, d) + iso * np.eye(3) / 3
    return jnp.asarray(m1), jnp.asarray(m2)


def build(rng, n, chunk=2000):
    """Targets u, m1, entropy m3 and its seven free components."""
    m1, m2 = sample(rng, n)
    U = jax.vmap(lambda a, b: M.pack(jnp.array(1.0), a, b))(m1, m2)
    u = M.targets(jnp.moveaxis(U, 0, 1)).T
    solve = jax.jit(jax.vmap(M.closure_one))
    m3 = jnp.concatenate([solve(u[i:i + chunk]) for i in range(0, n, chunk)])
    return u, m1, m3, L.free_of(L.trace_free_part(m3, m1))


def main():
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    out = sys.argv[2] if len(sys.argv) > 2 else f"m2_closure_w{width}.npz"
    t0 = time.perf_counter()
    u, m1, _, free = build(np.random.default_rng(0), N_TRAIN)
    ut, m1t, m3t, _ = build(np.random.default_rng(99), N_HELD)
    print(f"data in {time.perf_counter() - t0:.0f} s", flush=True)
    params = L.init(jax.random.PRNGKey(0), widths=(width, width))
    state = adam_init(params)
    lg = jax.jit(jax.value_and_grad(L.loss))
    for i in range(STEPS):
        lr = 3e-3 * (0.1 if i > 0.8 * STEPS else 0.3 if i > 0.5 * STEPS else 1.0)
        _, g = lg(params, u, m1, free)
        params, state = adam_step(g, params, state, lr)
    err = np.asarray(L._rebuild(L.free_components(params, ut, None), m1t) - m3t)
    print(f"{width}-{width}: held-out m3 error rms {np.sqrt(np.mean(err ** 2)):.3e} "
          f"max {np.abs(err).max():.3e}", flush=True)
    (W1, b1), (W2, b2), (W3, b3) = params
    np.savez(out, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
