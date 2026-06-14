"""Solver-in-the-loop ML flux corrector as an in-loop source term.

A small CNN emits face fluxes whose discrete divergence corrects the coarse
finite-volume update each step (conservative by construction). The network
WEIGHTS are the differentiated parameter (theta) for the autodiff
"corrector_training" task — the orchestrator trains them through the
production solver, reverse-mode, with gradients flowing through the unrolled
rollout (Um et al. 2020, solver-in-the-loop).

2D (Kelvin-Helmholtz). State is (4, N, N) conserved [rho, rho vx, rho vy, E].
"""
import jax
import jax.numpy as jnp

# last layer: 8 = 4 conserved fields x 2 face-flux directions; the correction
# is the discrete divergence of those fluxes -> conservative per channel.
CHANNELS = [4, 48, 48, 48, 8]


def pool_factor(U, f):
    """Conservative f x f average pool of a (C, fN, fN) field -> (C, N, N).
    Downsamples the fine production solution onto the coarse grid."""
    C, H, W = U.shape
    return U.reshape(C, H // f, f, W // f, f).mean(axis=(2, 4))


def adam_init(p):
    z = jax.tree.map(jnp.zeros_like, p)
    return dict(m=z, v=jax.tree.map(jnp.zeros_like, p), t=0)


def adam_step(g, p, s, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = s["t"] + 1
    m = jax.tree.map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, s["m"], g)
    v = jax.tree.map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ * g_, s["v"], g)
    mh = jax.tree.map(lambda m_: m_ / (1 - b1 ** t), m)
    vh = jax.tree.map(lambda v_: v_ / (1 - b2 ** t), v)
    p = jax.tree.map(lambda p_, mh_, vh_: p_ - lr * mh_ / (jnp.sqrt(vh_) + eps),
                     p, mh, vh)
    return p, dict(m=m, v=v, t=t)


def cnn_init(key):
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


def setup_corrector(cfg, U0):
    """Return {"init": params, "apply": fn(U, weights), "norm": norm}.

    norm (per-field scale) is frozen from the IC so the correction is
    scale-aware but the normalization itself isn't a trained quantity."""
    norm = jnp.std(U0, axis=(1, 2)) + 1e-12
    key = jax.random.PRNGKey(int(getattr(cfg, "corrector_seed", 0)))
    params0 = cnn_init(key)

    def apply(U, weights):
        return U + cnn_correction(weights, U, norm)

    return {"init": params0, "apply": apply, "norm": norm}
