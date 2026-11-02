"""Solver-in-the-loop ML flux corrector: a small CNN emits face fluxes whose
discrete divergence corrects the coarse finite-volume update each step
(conservative by construction). The network WEIGHTS are the differentiated
theta for the autodiff "corrector_training" task (2D, (4, N, N) state) and
for the 3D solver-in-the-loop corrector (`make_corrector3d`, (5, N, N, N)
state on the sharded block mesh; tools/sil3d.py trains it).
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


# ---------------------------------------------------------------------------
# 3D: face fluxes along x, y, z for the 5 conserved fields, on a sharded grid.
# docs/AUTODIFF.md, "Solver in the loop at scale".
# ---------------------------------------------------------------------------

# flux scale per field: rho_face * c^n, n = 1 (mass), 2 (momentum), 3 (energy)
_FLUX_POW = (1, 2, 2, 2, 3)


def cnn3d_init(key, width=16, depth=3, n_in=5, n_out=15, dtype=jnp.float64, kind="sep"):
    """`depth` layers, He init, the last one zero so the corrector starts as
    exactly the bare solver. kind "dense": full 3^3 convolutions (W, b);
    kind "sep": a per-channel 3^3 stencil then a 1x1 channel mix (D, P, b)."""
    chans = [n_in] + [width] * (depth - 1) + [n_out]
    params = []
    for i, (cin, cout) in enumerate(zip(chans[:-1], chans[1:])):
        key, k1, k2 = jax.random.split(key, 3)
        last = i == depth - 1
        if kind == "dense":
            scale = 0.0 if last else (2.0 / (cin * 27)) ** 0.5
            params.append((scale * jax.random.normal(k1, (cout, cin, 3, 3, 3), dtype=dtype),
                           jnp.zeros((cout,), dtype=dtype)))
        else:
            D = jax.random.normal(k1, (cin, 3, 3, 3), dtype=dtype) / 27.0 ** 0.5
            scale = 0.0 if last else (2.0 / cin) ** 0.5
            params.append((D, scale * jax.random.normal(k2, (cout, cin), dtype=dtype),
                           jnp.zeros((cout,), dtype=dtype)))
    return params


def _layer(h, layer):
    """One VALID layer: dense (W, b) or separable (D, P, b)."""
    if len(layer) == 2:
        return _conv3(h, *layer)
    D, P, b = layer
    X, Y, Z = (s - 2 for s in h.shape[1:])
    d = sum(D[:, i, j, k, None, None, None] * h[:, i:i + X, j:j + Y, k:k + Z]
            for i in range(3) for j in range(3) for k in range(3))
    return jnp.einsum("oc,cxyz->oxyz", P, d,
                      precision=jax.lax.Precision.HIGHEST) + b[:, None, None, None]


def param_count(params):
    return int(sum(x.size for x in jax.tree.leaves(params)))


def features3d(U, rho0, c, gamma):
    """(5, ...) network input: ln(rho/rho0), v/c, ln(p/(rho0 c^2))."""
    rho = U[0]
    v = U[1:4] / rho
    p = (gamma - 1.0) * (U[4] - 0.5 * rho * jnp.sum(v * v, axis=0))
    p = jnp.maximum(p, 1.0e-12 * rho0 * c * c)
    return jnp.concatenate([jnp.log(rho / rho0)[None], v / c,
                            jnp.log(p / (rho0 * c * c))[None]])


def _conv3(h, W, b):
    """VALID 3^3 convolution of (Cin, X, Y, Z) as 9 matmuls over the three
    z-shifts stacked along the channels. ROCm has no FP64 convolution
    (MIOpen: "Unsupported DNN data type: tf.float64"); FP64 GEMM it has."""
    cout, cin = W.shape[:2]
    X, Y, Z = (s - 2 for s in h.shape[1:])
    h3 = jnp.concatenate([h[:, :, :, k:k + Z] for k in range(3)])     # (3 Cin, X+2, Y+2, Z)
    W3 = jnp.transpose(W, (2, 3, 0, 4, 1)).reshape(3, 3, cout, 3 * cin)  # [i, j, o, (k, c)]
    out = b[:, None, None, None]
    for i in range(3):
        for j in range(3):
            out = out + jnp.einsum("oc,cxyz->oxyz", W3[i, j], h3[:, i:i + X, j:j + Y],
                                   precision=jax.lax.Precision.HIGHEST)
    return out


def correction3d(params, Ug, dt_dx, rho0, c, gamma, amp):
    """dU on the interior of a block with a halo of len(params) + 1 cells.

    The net emits o = amp tanh(raw / amp) per face and field; the flux through
    face i+1/2 along d is min(rho_i, rho_i+1) c^n o, and dU is minus its
    discrete divergence times dt/dx, so every flux leaves one cell and enters
    its neighbour (conservative), and a cell loses at most 6 amp c dt/dx of
    its mass."""
    h = features3d(Ug, rho0, c, gamma)
    for i, layer in enumerate(params):
        h = _layer(h, layer)
        if i < len(params) - 1:
            h = jax.nn.gelu(h)
    o = amp * jnp.tanh(h / amp)                     # halo 1 left: (15, n+2, n+2, n+2)
    rho = Ug[0, len(params):-len(params), len(params):-len(params),
             len(params):-len(params)]              # halo 1 as well
    n = [s - 2 for s in rho.shape]
    scale = jnp.asarray([c ** k for k in _FLUX_POW], dtype=Ug.dtype)[:, None, None, None]
    dU = jnp.zeros((5,) + tuple(n), dtype=Ug.dtype)
    inner = [slice(1, m + 1) for m in n]
    for d in range(3):
        # faces i+1/2 for i = -1 .. n-1 along d; interior along the others
        lo = list(inner); lo[d] = slice(0, n[d] + 1)
        hi = list(inner); hi[d] = slice(1, n[d] + 2)
        rho_f = jnp.minimum(rho[tuple(lo)], rho[tuple(hi)])
        F = rho_f[None] * scale * o[(slice(5 * d, 5 * d + 5),) + tuple(lo)]
        a = [slice(None)] * 4; a[d + 1] = slice(1, None)
        b = [slice(None)] * 4; b[d + 1] = slice(0, -1)
        dU = dU - dt_dx * (F[tuple(a)] - F[tuple(b)])
    return dU


def make_corrector3d(mesh, *, dx, rho0, c, gamma, depth, amp=1.0):
    """apply(params, U, dt) -> U + dU on the (5, Nx, Ny, Nz) grid sharded as
    P(None, "x", "y", "z"); params replicated (their gradient comes back
    summed over the shards by the transpose of the shard_map)."""
    from jax.sharding import PartitionSpec as P
    from luna.utils.halo_exchange_shardmap import make_halo_shardmap
    from luna.utils.shmap import shard_map
    halo = make_halo_shardmap(mesh, 3)
    R = int(depth) + 1

    def local(params, U, dt):
        Ug = halo(U, 0, 3, width=R)
        return U + correction3d(params, Ug, dt / dx, rho0, c, gamma, amp)

    spec = P(None, "x", "y", "z")
    # jitted (the step calls it outside any jit); the weights stay arguments:
    # closed over, ROCm's FP64 GEMM autotuner fails on them
    return jax.jit(shard_map(local, mesh=mesh, in_specs=(P(), spec, P()), out_specs=spec))


def save_corrector3d(path, params, *, rho0, c, amp):
    import numpy as np
    arrs = {f"L{i}_{j}": np.asarray(x) for i, layer in enumerate(params)
            for j, x in enumerate(layer)}
    np.savez(path, rho0=rho0, c=c, amp=amp, depth=len(params),
             nparts=len(params[0]), **arrs)


def load_corrector3d(path):
    """(params, dict(rho0, c, amp)) from save_corrector3d's .npz."""
    import numpy as np
    z = np.load(path)
    params = [tuple(jnp.asarray(z[f"L{i}_{j}"]) for j in range(int(z["nparts"])))
              for i in range(int(z["depth"]))]
    return params, {k: float(z[k]) for k in ("rho0", "c", "amp")}
