"""Solver-in-the-loop ML flux corrector: a small CNN emits face fluxes whose
discrete divergence corrects the coarse finite-volume update each step
(conservative by construction). The network WEIGHTS are the differentiated
theta for the autodiff "corrector_training" task. 2D only; state is
(4, N, N) conserved [rho, rho vx, rho vy, E].
"""
import jax
import jax.numpy as jnp

# last layer: 8 = 4 conserved fields x 2 face-flux directions; the correction
# is the discrete divergence of those fluxes -> conservative per channel.
CHANNELS = [4, 64, 96, 96, 64, 8]


def pool_factor(U, f):
    """Conservative f x f average pool of a (C, fN, fN) field -> (C, N, N).
    Downsamples the fine production solution onto the coarse grid."""
    C, H, W = U.shape
    return U.reshape(C, H // f, f, W // f, f).mean(axis=(2, 4))


def adam_init(p):
    """Zero-initialized Adam state dict {m, v, t} matching the param pytree."""
    z = jax.tree.map(jnp.zeros_like, p)
    return dict(m=z, v=jax.tree.map(jnp.zeros_like, p), t=0)


def adam_step(g, p, s, lr, b1=0.9, b2=0.999, eps=1e-8):
    """One Adam update; returns (new_params, new_state)."""
    t = s["t"] + 1
    m = jax.tree.map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, s["m"], g)
    v = jax.tree.map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ * g_, s["v"], g)
    mh = jax.tree.map(lambda m_: m_ / (1 - b1 ** t), m)
    vh = jax.tree.map(lambda v_: v_ / (1 - b2 ** t), v)
    p = jax.tree.map(lambda p_, mh_, vh_: p_ - lr * mh_ / (jnp.sqrt(vh_) + eps),
                     p, mh, vh)
    return p, dict(m=m, v=v, t=t)


def cnn_init(key):
    """He-init CNN params [(W, b), ...]; last layer zero-init (starts as pure solver)."""
    params = []
    for i, (cin, cout) in enumerate(zip(CHANNELS[:-1], CHANNELS[1:])):
        key, k = jax.random.split(key)
        last = i == len(CHANNELS) - 2
        scale = 0.0 if last else jnp.sqrt(2.0 / (cin * 9))   # zero-init last -> starts as pure solver
        W = scale * jax.random.normal(k, (cout, cin, 3, 3), dtype=jnp.float64)
        b = jnp.zeros((cout,), dtype=jnp.float64)
        params.append((W, b))
    return params


def cnn_correction(params, U, norm):
    """Conservative flux-divergence correction for a (4,N,N) state."""
    h = (U / norm[:, None, None])[None]
    for i, (W, b) in enumerate(params):
        h = jnp.pad(h, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="wrap")
        h = jax.lax.conv_general_dilated(
            h, W, window_strides=(1, 1), padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"))
        h = h + b[None, :, None, None]
        if i < len(params) - 1:
            h = jax.nn.gelu(h)
    f = h[0]
    fx, fy = f[:4], f[4:]
    div = (fx - jnp.roll(fx, 1, axis=1)) + (fy - jnp.roll(fy, 1, axis=2))
    return 0.02 * div * norm[:, None, None]
