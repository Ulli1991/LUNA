"""Fused Pallas (Triton) kernels for the M1 Hancock sweep and the implicit ionization.

Grey, 3-D, one group per call; the sweep is bitwise the XLA path, the ionization agrees to
round-off (R within 1.1e-16 on an A100). See docs/RADIATION.md.
"""
import functools

import jax
import jax.numpy as jnp

from luna.radiation.m1 import E_FLOOR, _closure

KERNEL = "auto"     # "auto" (Pallas on a GPU), "pallas", "xla"; setup_radiation sets it


def _strong(x):
    """Return x as a strongly typed FP64 array (no-op cast)."""
    from jax._src.lax.lax import _convert_element_type
    return _convert_element_type(x, jnp.dtype("float64"), weak_type=False)


def _minmod(a, b):
    return jnp.where(a * b > 0, jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)


def _flux(R, d, chat):
    """Physical flux along d of R = [E, F0, F1, F2], as `m1._phys_flux_axis`."""
    E, F = R[0], R[1:]
    s, inv = _closure(E, [F[0], F[1], F[2]], chat)
    n = [Fi * inv for Fi in F]
    iso = (s - 1.0) * (1.0 / 3.0)
    ani = 2.0 - s
    P = [E * (iso * (1.0 if j == d else 0.0) + ani * n[d] * n[j]) for j in range(3)]
    return [F[d]] + [chat * chat * p for p in P]


def _largest_pow2_divisor(n, cap):
    t = 1
    while t * 2 <= cap and n % (t * 2) == 0:
        t *= 2
    return t


def _nvidia(interpret):
    return not interpret and jax.devices()[0].device_kind.upper().startswith("NVIDIA")


def tile_for(shape, nvidia=False):
    """Return a power-of-two tile dividing the block, up to 1 x 8 x 64 (1 x 4 x 32 on NVIDIA)."""
    ty, tz = (4, 32) if nvidia else (8, 64)
    return (1, _largest_pow2_divisor(shape[1], ty), _largest_pow2_divisor(shape[2], tz))


def _launch(tile, nvidia):
    """NVIDIA: one cell per thread; Triton's 4 warps on 1 x 8 x 64 spill (A100 1.4x, CODE_NOTES.md)."""
    if not nvidia:
        return None
    from jax.experimental.pallas import triton as plt
    return plt.CompilerParams(num_warps=max(1, tile[0] * tile[1] * tile[2] // 32))


def _tile(shape, d, clamp, nvidia):
    """`tile_for`, with one plane along d for a clamped sweep (the y sweep's tile is ty x 1 x tz)."""
    t = tile_for(shape, nvidia)
    if clamp and d == 1:
        return (_largest_pow2_divisor(shape[0], t[1] if nvidia else 8), 1, t[2])
    return t


def _predicted(Rl, Rc, Rr, half, d, chat):
    """Return the Hancock-predicted (minus, plus) edge states of the cell Rc."""
    dl = [Rc[c] - Rl[c] for c in range(4)]
    dr = [Rr[c] - Rc[c] for c in range(4)]
    sl = [_minmod(a, b) for a, b in zip(dl, dr)]
    Rm = [Rc[c] - 0.5 * sl[c] for c in range(4)]
    Rq = [Rc[c] + 0.5 * sl[c] for c in range(4)]
    Fq, Fm = _flux(Rq, d, chat), _flux(Rm, d, chat)
    inc = [half * (a - b) for a, b in zip(Fq, Fm)]
    return [a - i for a, i in zip(Rm, inc)], [a - i for a, i in zip(Rq, inc)]


def _face(RL, RR, d, chat):
    """Return the GLF flux between the face states RL and RR."""
    fL, fR = _flux(RL, d, chat), _flux(RR, d, chat)
    return [0.5 * (a + b) - 0.5 * chat * (r - l) for a, b, l, r in zip(fL, fR, RL, RR)]


def _floored(R0, fm, fp, lam, chat):
    """Return the conservative update of R0 by the faces fm, fp, floored as `m1.m1_floor`."""
    out = [R0[c] - lam * (fp[c] - fm[c]) for c in range(4)]
    E = jnp.maximum(out[0], 0.0)
    Fmag = jnp.sqrt(out[1] * out[1] + out[2] * out[2] + out[3] * out[3] + E_FLOOR * E_FLOOR)
    scale = jnp.minimum(1.0, chat * E / Fmag)
    return [E] + [o * scale for o in out[1:]]


def _loader(pl, rp_ref, d, sizes, n, clamp, boundary):
    """Return load(base, s): the four components of the tile at base shifted s cells along d."""
    def load(base, s):
        st = list(base)
        if clamp:       # unpadded input, one plane along d: the ghost plane is an edge or wrapped index
            p = st[d] + s
            st[d] = (p + n) % n if boundary == 0 else jnp.minimum(jnp.maximum(p, 0), n - 1)
        else:           # input with 2 ghosts along d
            st[d] = st[d] + 2 + s
        idx = tuple(pl.ds(st[a], sizes[a]) for a in range(3))
        return [rp_ref[(c,) + idx] for c in range(4)]
    return load


def _padded(shape, dn):
    """Return shape with 2 + 2 ghosts along dn (None: as is)."""
    o = list(shape)
    if dn is not None:
        o[dn] += 4
    return tuple(o)


@functools.lru_cache(maxsize=None)
def _sweep_call(shape, d, chat, interpret, mat, clamp=False, boundary=1, dn=None):
    """Fused sweep along d; input clamped (unpadded) or with 2 ghosts along d; output padded along dn."""
    from jax.experimental import pallas as pl
    if not interpret:
        select_triton()
    nvidia = _nvidia(interpret)
    sizes = _tile(shape, d, clamp, nvidia)

    def kernel(k_ref, rp_ref, o_ref):
        half, lam = k_ref[0], k_ref[1]
        base = [pl.program_id(a) * sizes[a] for a in range(3)]
        load = _loader(pl, rp_ref, d, sizes, shape[d], clamp, boundary)
        Rs = {s: load(base, s) for s in range(-2, 3)}
        P = {s: _predicted(Rs[s - 1], Rs[s], Rs[s + 1], half, d, chat) for s in (-1, 0, 1)}
        fm, fp = _face(P[-1][1], P[0][0], d, chat), _face(P[0][1], P[1][0], d, chat)
        out = _floored(Rs[0], fm, fp, lam, chat)
        st = list(base)
        if dn is not None:
            st[dn] = st[dn] + 2
        idx = tuple(pl.ds(st[a], sizes[a]) for a in range(3))
        for c in range(4):
            o_ref[(c,) + idx] = out[c]

    # inside a shard_map the output varies over the mesh axes as the input does
    out = jax.ShapeDtypeStruct((4,) + _padded(shape, dn), jnp.float64, manual_axis_type=mat)
    return pl.pallas_call(kernel, out_shape=out,
                          grid=tuple(shape[a] // sizes[a] for a in range(3)), interpret=interpret,
                          compiler_params=_launch(sizes, nvidia))


@functools.lru_cache(maxsize=None)
def _ghost_call(oshape, dn, boundary, interpret, mat, nc=4):
    """Fill the 2 + 2 ghost planes along dn of an (nc, oshape) array in place (edge or periodic)."""
    from jax.experimental import pallas as pl
    if not interpret:
        select_triton()
    nvidia = _nvidia(interpret)
    n = oshape[dn] - 4
    others = [a for a in range(3) if a != dn]
    sizes = [1, 1, 1]
    sizes[others[1]] = _largest_pow2_divisor(oshape[others[1]], 128)
    src = {0: n, 1: n + 1, n + 2: 2, n + 3: 3} if boundary == 0 else \
        {0: 2, 1: 2, n + 2: n + 1, n + 3: n + 1}

    def kernel(a_ref, o_ref):
        base = [pl.program_id(a) * sizes[a] for a in range(3)]
        for g, s_ in src.items():
            si, gi = list(base), list(base)
            si[dn], gi[dn] = s_, g
            sidx = tuple(pl.ds(si[a], sizes[a]) for a in range(3))
            gidx = tuple(pl.ds(gi[a], sizes[a]) for a in range(3))
            for c in range(nc):
                o_ref[(c,) + gidx] = a_ref[(c,) + sidx]

    grid = tuple(1 if a == dn else oshape[a] // sizes[a] for a in range(3))
    out = jax.ShapeDtypeStruct((nc,) + tuple(oshape), jnp.float64, manual_axis_type=mat)
    return pl.pallas_call(kernel, out_shape=out, grid=grid, input_output_aliases={0: 0},
                          interpret=interpret, compiler_params=_launch(tuple(sizes), nvidia))


@functools.lru_cache(maxsize=None)
def _predict_call(shape, d, chat, interpret, mat, clamp, boundary):
    """Pass one of a two-kernel sweep: the predicted (minus, plus) states of cells -1..n along d, (8, ..n+2..)."""
    from jax.experimental import pallas as pl
    if not interpret:
        select_triton()
    nvidia = _nvidia(interpret)
    sizes = _tile(shape, d, clamp, nvidia)
    M = list(shape)
    M[d] += 2
    alloc = list(M)
    if d == 2:          # whole tiles along the contiguous axis
        alloc[2] = -(-M[2] // sizes[2]) * sizes[2]

    def kernel(k_ref, rp_ref, o_ref):
        half = k_ref[0]
        # the last tile along d is shifted back to end at n+2: it recomputes (and rewrites) a few states
        base = [jnp.minimum(pl.program_id(a) * sizes[a], M[a] - sizes[a]) for a in range(3)]
        load = _loader(pl, rp_ref, d, sizes, shape[d], clamp, boundary)
        cb = list(base)
        cb[d] = cb[d] - 1          # output index j holds cell j-1
        Rs = {s: load(cb, s) for s in (-1, 0, 1)}
        Pm, Pq = _predicted(Rs[-1], Rs[0], Rs[1], half, d, chat)
        idx = tuple(pl.ds(base[a], sizes[a]) for a in range(3))
        for c, v in enumerate(Pm + Pq):
            o_ref[(c,) + idx] = v

    out = jax.ShapeDtypeStruct((8,) + tuple(alloc), jnp.float64, manual_axis_type=mat)
    grid = tuple(-(-M[a] // sizes[a]) for a in range(3))
    return pl.pallas_call(kernel, out_shape=out, grid=grid, interpret=interpret,
                          compiler_params=_launch(sizes, nvidia))


@functools.lru_cache(maxsize=None)
def _faces_call(shape, d, chat, interpret, mat, clamp, boundary, dn):
    """Pass two: both faces of each cell from pass one, the update and the floor; output padded along dn."""
    from jax.experimental import pallas as pl
    if not interpret:
        select_triton()
    nvidia = _nvidia(interpret)
    sizes = _tile(shape, d, clamp, nvidia)

    def kernel(k_ref, rp_ref, p_ref, o_ref):
        lam = k_ref[1]
        base = [pl.program_id(a) * sizes[a] for a in range(3)]
        R0 = _loader(pl, rp_ref, d, sizes, shape[d], clamp, boundary)(base, 0)

        def q(j):
            st = list(base)
            st[d] = st[d] + j
            idx = tuple(pl.ds(st[a], sizes[a]) for a in range(3))
            return [p_ref[(c,) + idx] for c in range(8)]
        Q = {j: q(j) for j in (0, 1, 2)}
        fm, fp = _face(Q[0][4:], Q[1][:4], d, chat), _face(Q[1][4:], Q[2][:4], d, chat)
        out = _floored(R0, fm, fp, lam, chat)
        st = list(base)
        if dn is not None:
            st[dn] = st[dn] + 2
        idx = tuple(pl.ds(st[a], sizes[a]) for a in range(3))
        for c in range(4):
            o_ref[(c,) + idx] = out[c]

    out = jax.ShapeDtypeStruct((4,) + _padded(shape, dn), jnp.float64, manual_axis_type=mat)
    return pl.pallas_call(kernel, out_shape=out,
                          grid=tuple(shape[a] // sizes[a] for a in range(3)), interpret=interpret,
                          compiler_params=_launch(sizes, nvidia))


def sweep(Rp, dt, dx, d, *, chat, interpret=False):
    """Floored Hancock sweep along d of one grey 3-D group; Rp is (4, N..) with 2 ghosts along d."""
    # weakly typed input lowers to mixed FP32/FP64 tiles in Triton
    Rp = _strong(jnp.asarray(Rp, jnp.float64))
    shape = list(Rp.shape[1:])
    shape[d] -= 4
    # formed as the XLA sweep does, for bitwise agreement
    k = jnp.stack([0.5 * dt / dx, dt / dx]).astype(jnp.float64)
    mat = getattr(jax.typeof(Rp), "mat", None)
    return _sweep_call(tuple(shape), d, float(chat), bool(interpret), mat)(k, Rp)


SWEEPS = (0, 1, 2, 2, 1, 0)
# Predictor once per cell (two kernels) up to this many cells per group and device: it keeps an
# (8, N..) buffer of predicted states, +1.6 GiB at 256^3; above it one fused kernel per sweep.
PREDICT_ONCE_MAX_CELLS = 2 ** 25


def split_step(R, dt, dx, *, chat, boundary, pad=None, local=(0, 1, 2), interpret=False):
    """`m1.m1_split_step` of grey 3-D groups R (G, 4, N..) as fused sweeps, without the XLA pad on local axes.

    On an axis in `local` (whole on this device) x and y sweeps read the unpadded state with an edge or
    wrapped plane index, and the sweep before a z sweep writes its output with z ghosts, filled in place.
    Other axes take their ghosts from `pad(A, 2, axes=(d,))`, the halo exchange. Up to
    PREDICT_ONCE_MAX_CELLS a sweep is two kernels, predicting each cell's edge states once instead of
    three times. Bitwise the padded fused sweeps either way.
    """
    G = R.shape[0]
    shape = tuple(R.shape[2:])
    once = shape[0] * shape[1] * shape[2] <= PREDICT_ONCE_MAX_CELLS
    h = 0.5 * dt
    k = jnp.stack([0.5 * h / dx, h / dx]).astype(jnp.float64)
    cur = [_strong(jnp.asarray(R[g], jnp.float64)) for g in range(G)]
    cur_pad = None                      # the axis along which `cur` carries 2 + 2 ghosts
    for i, d in enumerate(SWEEPS):
        clamp = d in local and d != 2
        if d not in local:              # a sharded axis: the halo exchange, for all groups at once
            flat = pad(jnp.concatenate(cur), 2, axes=(d,))
            cur = [flat[4 * g:4 * (g + 1)] for g in range(G)]
        elif not clamp and cur_pad != d:
            cur = [jnp.pad(c, [(0, 0)] + [(2, 2) if a == d else (0, 0) for a in range(3)],
                           mode="wrap" if boundary == 0 else "edge") for c in cur]
        nxt = SWEEPS[i + 1] if i + 1 < len(SWEEPS) else None
        dn = nxt if (nxt == 2 and 2 in local) else None
        mat = getattr(jax.typeof(cur[0]), "mat", None)
        if once:
            pred = _predict_call(shape, d, float(chat), bool(interpret), mat, clamp, int(boundary))
            call = _faces_call(shape, d, float(chat), bool(interpret), mat, clamp, int(boundary), dn)
            cur = [call(k, c, pred(k, c)) for c in cur]
        else:
            call = _sweep_call(shape, d, float(chat), bool(interpret), mat, clamp, int(boundary), dn)
            cur = [call(k, c) for c in cur]
        if dn is not None:
            ghost = _ghost_call(_padded(shape, dn), dn, int(boundary), bool(interpret), mat)
            cur = [ghost(c) for c in cur]
        cur_pad = dn
    return jnp.stack(cur)


def use_pallas(R):
    """Return whether the Pallas kernels apply: 3-D FP64 grey state and a GPU or KERNEL="pallas"."""
    if KERNEL == "xla" or R.ndim != 5 or R.shape[1] != 4 or R.dtype != jnp.float64:
        return False
    if KERNEL == "pallas":
        return True
    return jax.default_backend() in ("gpu", "rocm", "cuda")


def interpret_mode():
    return jax.default_backend() == "cpu"


def select_triton():
    """Lower Pallas through Triton instead of Mosaic (CUDA-only)."""
    from jax.experimental import pallas  # noqa: F401  (defines the option)
    jax.config.update("jax_pallas_use_mosaic_gpu", False)


@functools.lru_cache(maxsize=None)
def _ionize_call(shape, chat, sigma, alpha_B, m_H, iters, early, interpret, mat):
    """Build `ionization.ionize_implicit` for one grey group as a kernel.

    `early`: a tile stops once no cell's z or bracket moved (bitwise the full `iters`).
    """
    from jax.experimental import pallas as pl
    if not interpret:
        select_triton()
    nvidia = _nvidia(interpret)
    TX, TY, TZ = tile_for(shape, nvidia)
    Nx, Ny, Nz = shape
    sizes = (TX, TY, TZ)
    Z_MIN = 1e-300

    def kernel(k_ref, r_ref, x_ref, rho_ref, ro_ref, xo_ref, it_ref):
        dt = k_ref[0]
        idx = tuple(pl.ds(pl.program_id(a) * sizes[a], sizes[a]) for a in range(3))
        rho = rho_ref[idx]
        n_H = rho / m_H
        N0 = r_ref[(0,) + idx]
        P = N0 / jnp.maximum(n_H, 1e-300)
        a = chat * sigma * n_H * dt
        b = alpha_B * n_H * dt
        x0 = jnp.clip(x_ref[idx], 0.0, 1.0)

        def step(z, lo, hi):
            u = a * z
            y = 1.0 - z
            r = 1.0 / (1.0 + u)                    # one division for g and g', as ionize_implicit
            Pr = P * r
            gz = y - x0 - Pr * u + b * y * y
            dg = -1.0 - Pr * a * r - 2.0 * b * y
            lo = jnp.where(gz > 0.0, z, lo)
            hi = jnp.where(gz <= 0.0, z, hi)
            zn = z - gz / dg
            zn = jnp.where((zn > lo) & (zn < hi), zn, jnp.sqrt(lo * hi))
            return jnp.where(gz == 0.0, z, zn), lo, hi

        z0 = jnp.clip(1.0 - x0, Z_MIN, 1.0)
        c0 = (jnp.int32(0), z0, jnp.full(sizes, Z_MIN, jnp.float64),
              jnp.ones(sizes, jnp.float64), jnp.bool_(True))
        if early:
            def cond(c):
                return (c[0] < iters) & c[4]

            def body(c):
                it, z, lo, hi, _ = c
                zn, ln, hn = step(z, lo, hi)
                # no `any` in the Triton lowering: the max of 0/1 flags
                ch = (zn != z) | (ln != lo) | (hn != hi)
                moved = jnp.max(ch.astype(jnp.int32)) > 0
                return it + 1, zn, ln, hn, moved
            it, z, lo, hi, _ = jax.lax.while_loop(cond, body, c0)
        else:
            def body(i, c):
                z, lo, hi = step(*c)
                return z, lo, hi
            z, lo, hi = jax.lax.fori_loop(0, iters, body, c0[1:4])
            it = jnp.int32(iters)
        N1 = N0 / (1.0 + a * z)
        keep = N1 / jnp.maximum(N0, 1e-300)
        ro_ref[(0,) + idx] = N1
        for c in (1, 2, 3):
            ro_ref[(c,) + idx] = r_ref[(c,) + idx] * keep
        xo_ref[idx] = jnp.where(jnp.abs(N0) < jnp.inf, 1.0 - jnp.clip(z, 0.0, 1.0), jnp.nan)
        it_ref[idx] = jnp.full(sizes, it, jnp.int32)

    outs = (jax.ShapeDtypeStruct((4, Nx, Ny, Nz), jnp.float64, manual_axis_type=mat),
            jax.ShapeDtypeStruct((Nx, Ny, Nz), jnp.float64, manual_axis_type=mat),
            jax.ShapeDtypeStruct((Nx, Ny, Nz), jnp.int32, manual_axis_type=mat))
    return pl.pallas_call(kernel, out_shape=outs, grid=(Nx // TX, Ny // TY, Nz // TZ),
                          interpret=interpret, compiler_params=_launch(sizes, nvidia))


def ionize(R, x, rho, dt, *, chat, sigma, alpha_B, m_H, iters=60, early=True,
           return_iters=False):
    """`ionization.ionize_implicit` for a grey 3-D state R (1, 4, N..) as one kernel.

    `return_iters` also returns each cell's tile iteration count.
    """
    R4 = _strong(jnp.asarray(R[0], jnp.float64))
    x = _strong(jnp.asarray(x, jnp.float64))
    rho = _strong(jnp.asarray(rho, jnp.float64))
    k = jnp.stack([jnp.asarray(dt, jnp.float64), jnp.zeros((), jnp.float64)])
    shape = tuple(int(n) for n in x.shape)
    mat = getattr(jax.typeof(x), "mat", None)
    call = _ionize_call(shape, float(chat), float(sigma), float(alpha_B), float(m_H), int(iters),
                        bool(early), interpret_mode(), mat)
    Rn, xn, it = call(k, R4, x, rho)
    out = (Rn[None], xn)
    return out + (it,) if return_iters else out
