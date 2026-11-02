"""Fused Pallas (Triton) kernels for the 3-D M2 Hancock sweep and the learned-closure MLP.

Trace closure or the learned closure frozen over the step. See docs/RADIATION.md.
"""
import functools

import jax
import jax.numpy as jnp

from luna.radiation import m2_split
from luna.radiation.m1_pallas import _largest_pow2_divisor, _strong, select_triton

TILE = (1, 4, 64)          # the trace closure
TILE_FROZEN = (1, 4, 64)   # the frozen learned closure
TILE_NVIDIA = (1, 1, 128)  # both, on NVIDIA: one cell per thread, 4 warps (A100, CODE_NOTES.md)


def tile_for(shape, cap=TILE):
    """Return the largest power-of-two tile dividing `shape`, capped by `cap`."""
    return tuple(_largest_pow2_divisor(shape[a], cap[a]) for a in range(3))


def _tile(shape, d, clamp, cap):
    """`tile_for`, with one plane along d for a clamped sweep (the y sweep's tile is cy x 1 x cz)."""
    if clamp and d == 1:
        return (_largest_pow2_divisor(shape[0], cap[1]), 1, _largest_pow2_divisor(shape[2], cap[2]))
    return tile_for(shape, cap)


@functools.lru_cache(maxsize=None)
def _sweep_call(shape, d, chat, glf, interpret, mat, frozen, cap, clamp=False, boundary=1, dn=None):
    """Fused sweep along d; input clamped (unpadded) or with 2 ghosts along d; output padded along dn."""
    from jax.experimental import pallas as pl
    from luna.radiation.m1_pallas import _launch, _nvidia, _padded
    if not interpret:
        select_triton()
    sizes = _tile(shape, d, clamp, cap)
    grid = tuple(shape[a] // sizes[a] for a in range(3))
    n = shape[d]

    def kernel(k_ref, up_ref, *refs):
        vp_ref, o_ref = refs if frozen else (None, refs[0])
        half, lam = k_ref[0], k_ref[1]
        base = [pl.program_id(a) * sizes[a] for a in range(3)]

        def at_u(s):
            st = list(base)
            p = st[d] + s
            if not clamp:       # input with 2 ghosts along d
                st[d] = p + 2
            else:               # unpadded input: the ghost plane is an edge or wrapped index
                st[d] = (p + n) % n if boundary == 0 else jnp.minimum(jnp.maximum(p, 0), n - 1)
            return tuple(pl.ds(st[a], sizes[a]) for a in range(3))

        def at_v(s):
            st = [b + 1 for b in base]
            st[d] = st[d] + s
            return tuple(pl.ds(st[a], sizes[a]) for a in range(3))

        Us = {s: [up_ref[(c,) + at_u(s)] for c in range(10)] for s in range(-2, 3)}
        if frozen:
            Qs = {s: m2_split.free_row([vp_ref[(c,) + at_v(s)].astype(jnp.float64)
                                        for c in range(7)], d) for s in range(-1, 2)}

        def flux(c, q):
            if q is None:
                return m2_split.flux_cells(c, d, chat)
            return m2_split.frozen_flux(c, q, d, chat)

        def edges(X, s, n):
            sl = [m2_split._minmod(X[s][c] - X[s - 1][c], X[s + 1][c] - X[s][c])
                  for c in range(n)]
            return ([X[s][c] - 0.5 * sl[c] for c in range(n)],
                    [X[s][c] + 0.5 * sl[c] for c in range(n)])

        def predicted(s):
            Um, Uq = edges(Us, s, 10)
            qm = qq = Qs[s] if frozen else None
            Fq, Fm = flux(Uq, qq), flux(Um, qm)
            inc = [half * (a - b) for a, b in zip(Fq, Fm)]
            return ([a - i for a, i in zip(Um, inc)], qm), ([a - i for a, i in zip(Uq, inc)], qq)

        P = {s: predicted(s) for s in (-1, 0, 1)}

        def face(sl_, sr_):
            (UL, qL), (UR, qR) = P[sl_][1], P[sr_][0]
            fL, fR = flux(UL, qL), flux(UR, qR)
            return [0.5 * (a + b) - 0.5 * glf * (r - l) for a, b, l, r in zip(fL, fR, UL, UR)]

        fm, fp = face(-1, 0), face(0, 1)
        out = m2_split.floor_cells([Us[0][c] - lam * (fp[c] - fm[c]) for c in range(10)])
        st = list(base)
        if dn is not None:
            st[dn] = st[dn] + 2
        idx = tuple(pl.ds(st[a], sizes[a]) for a in range(3))
        for c in range(10):
            o_ref[(c,) + idx] = out[c]

    out = jax.ShapeDtypeStruct((10,) + _padded(shape, dn), jnp.float64, manual_axis_type=mat)
    return pl.pallas_call(kernel, out_shape=out, grid=grid, interpret=interpret,
                          compiler_params=_launch(sizes, _nvidia(interpret)))


def _cap(frozen, interpret):
    """The tile cap: one cell per thread on NVIDIA, else TILE / TILE_FROZEN."""
    from luna.radiation.m1_pallas import _nvidia
    if _nvidia(interpret):
        return TILE_NVIDIA
    return TILE_FROZEN if frozen else TILE


def sweep(Up, dt, dx, d, *, chat, diffusion=1.0, Vp=None, interpret=False):
    """Floored Hancock sweep along d of a 3-D M2 state; Up is (10, N..) with 2 ghosts along d.

    Vp: the learned closure's seven free components with 1 ghost on every axis; None = trace closure.
    """
    Up = _strong(jnp.asarray(Up, jnp.float64))
    shape = list(Up.shape[1:])
    shape[d] -= 4
    k = jnp.stack([0.5 * dt / dx, dt / dx]).astype(jnp.float64)
    mat = getattr(jax.typeof(Up), "mat", None)
    frozen = Vp is not None
    call = _sweep_call(tuple(shape), d, float(chat), float(diffusion * chat), bool(interpret),
                       mat, frozen, _cap(frozen, interpret))
    if frozen:
        Vp = _vp(Vp)
        return call(k, Up, Vp)
    return call(k, Up)


def _vp(Vp):
    """Return the frozen components as passed to the kernel: FP32 as is, else strongly typed FP64."""
    if Vp.dtype != jnp.float32:
        return _strong(jnp.asarray(Vp, jnp.float64))
    return Vp


def split_step(U, dt, dx, *, chat, boundary, pad=None, local=(0, 1, 2), diffusion=1.0, Vp=None,
               interpret=False):
    """The fused sweeps of `m2_split.split_step` for U (10, N..), without the XLA pad on local axes.

    As `m1_pallas.split_step`: x and y sweeps on a local axis read the unpadded state with an edge or
    wrapped plane index, the sweep before a z sweep writes z ghosts filled in place, and sharded axes
    take theirs from `pad(A, 2, axes=(d,))`. Bitwise the padded sweeps.
    """
    from luna.radiation.m1_pallas import SWEEPS, _ghost_call, _padded
    shape = tuple(U.shape[1:])
    h = 0.5 * dt
    k = jnp.stack([0.5 * h / dx, h / dx]).astype(jnp.float64)
    frozen = Vp is not None
    cap = _cap(frozen, interpret)
    if frozen:
        Vp = _vp(Vp)
    cur = _strong(jnp.asarray(U, jnp.float64))
    cur_pad = None                      # the axis along which `cur` carries 2 + 2 ghosts
    for i, d in enumerate(SWEEPS):
        clamp = d in local and d != 2
        if d not in local:
            cur = pad(cur, 2, axes=(d,))
        elif not clamp and cur_pad != d:
            cur = m2_split._pad_axis_local(cur, 2, boundary, d)
        nxt = SWEEPS[i + 1] if i + 1 < len(SWEEPS) else None
        dn = nxt if (nxt == 2 and 2 in local) else None
        mat = getattr(jax.typeof(cur), "mat", None)
        call = _sweep_call(shape, d, float(chat), float(diffusion * chat), bool(interpret), mat,
                           frozen, cap, clamp, int(boundary), dn)
        cur = call(k, cur, Vp) if frozen else call(k, cur)
        if dn is not None:
            cur = _ghost_call(_padded(shape, dn), dn, int(boundary), bool(interpret), mat, 10)(cur)
        cur_pad = dn
    return cur


TILE_NET = (1, 4, 64)


def _const_weights(params):
    """Return the weights as nested tuples of FP32-rounded floats (kernel constants and cache key)."""
    import numpy as np
    return tuple(tuple(map(tuple, np.asarray(W, np.float32).astype(float).tolist()))
                 + (tuple(np.asarray(b, np.float32).astype(float).tolist()),)
                 for W, b in params)


@functools.lru_cache(maxsize=None)
def _net_call(shape, weights, interpret, mat, cap):
    from jax.experimental import pallas as pl
    from luna.radiation.m2_learned_3d import BOUND
    from luna.radiation.m2_3d import normalised
    if not interpret:
        select_triton()
    sizes = tile_for(shape, cap)
    grid = tuple(shape[a] // sizes[a] for a in range(3))
    layers = [(w[:-1], w[-1]) for w in weights]

    def kernel(u_ref, o_ref):
        idx = tuple(pl.ds(pl.program_id(a) * sizes[a], sizes[a]) for a in range(3))
        c = [u_ref[(k,) + idx] for k in range(10)]
        h = [v.astype(jnp.float32) for v in normalised(c)]
        for li, (W, b) in enumerate(layers):
            nxt = []
            for j, row in enumerate(W):
                acc = h[0] * row[0] + b[j]
                for k in range(1, len(row)):
                    acc = acc + h[k] * row[k]
                nxt.append(jnp.tanh(acc) if li < len(layers) - 1
                           else BOUND * jnp.tanh(acc / BOUND))
            h = nxt
        for m in range(len(h)):
            o_ref[(m,) + idx] = h[m]

    out = jax.ShapeDtypeStruct((7,) + tuple(shape), jnp.float32, manual_axis_type=mat)
    return pl.pallas_call(kernel, out_shape=out, grid=grid, interpret=interpret)


def free_components(U, params, interpret=False):
    """`m2_split.free_components` as one kernel: (7, N...) FP32 from the FP64 state."""
    U = _strong(jnp.asarray(U, jnp.float64))
    mat = getattr(jax.typeof(U), "mat", None)
    return _net_call(tuple(U.shape[1:]), _const_weights(params), bool(interpret), mat,
                     TILE_NET)(U)


def use_pallas(U, closure_fn, kernel="auto"):
    """Return whether the fused sweep applies: built-in closure, 3-D FP64, and a GPU or kernel="pallas"."""
    if kernel == "pallas" and closure_fn is not None:
        raise ValueError("rad_kernel='pallas': the fused M2 sweep has the trace and the frozen learned "
                         "closure only; a closure at every face (rad_closure='m2_entropy') runs on XLA "
                         "-- rad_kernel='xla' or 'auto'")
    if kernel == "xla" or closure_fn is not None or U.ndim != 4 or U.shape[0] != 10 \
            or U.dtype != jnp.float64:
        return False
    if kernel == "pallas":
        return True
    return jax.default_backend() in ("gpu", "rocm", "cuda")


def interpret_mode():
    return jax.default_backend() == "cpu"
