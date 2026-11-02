"""3-D M2 transport as split x y z z y x MUSCL-Hancock sweeps, floored after each.

Per-cell pieces take lists of arrays so the fused kernel (`m2_pallas`) runs the same lines.
See docs/RADIATION.md ("M2 as a stage").
"""
import math

import jax
import jax.numpy as jnp

from luna.radiation.m2_3d import EPS, _SYM, normalised

_ROW = {(i, j): k for k, (i, j) in enumerate(_SYM)}     # (i<=j) -> m2 slot


def _sym(i, j):
    return _ROW[(min(i, j), max(i, j))]


def trace_m3(c, d, i, j):
    """Return m3_dij of the trace closure (d_ij m1_k + perms) / 5 from ten components."""
    terms = ([c[1 + j]] if d == i else []) + ([c[1 + d]] if i == j else []) \
        + ([c[1 + i]] if d == j else [])
    if not terms:
        return 0.0 * c[0]
    s = terms[0]
    for t in terms[1:]:
        s = s + t
    return s * 0.2


def flux_cells(c, d, chat, m3_row=None):
    """Return the ten fluxes along d as a list; `m3_row` (six m3_dij, i<=j) or None for the trace closure."""
    if m3_row is None:
        m3_row = [trace_m3(c, d, i, j) for i, j in _SYM]
    return ([chat * c[1 + d]] + [chat * c[4 + _sym(d, j)] for j in range(3)]
            + [chat * m for m in m3_row])


T_MAX = 1e100       # |m2_ij| / tr(m2) beyond which floor_cells resets m2 to isotropic (realizable: <= 1)


def floor_cells(c):
    """Floor to m0 >= 0, |m1| <= m0, m2/m0 PSD with unit trace (mix with I/3 by the smallest eigenvalue).

    The eigenvalue is closed-form (trigonometric cubic), so a fused kernel can run it.
    """
    m0 = jnp.maximum(c[0], 0.0)
    isafe = 1.0 / jnp.maximum(m0, EPS)                    # one division for the seven by m0
    n1 = jnp.sqrt(c[1] * c[1] + c[2] * c[2] + c[3] * c[3] + EPS) * isafe
    s1 = jnp.minimum(1.0, 1.0 / jnp.maximum(n1, EPS))
    m1 = [c[1 + i] * s1 for i in range(3)]
    t = [c[4 + k] * isafe for k in range(6)]              # xx yy zz xy xz yz
    tr = t[0] + t[1] + t[2]
    good = tr > 0.0
    inv = 1.0 / jnp.where(good, tr, 1.0)
    iso = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0, 0.0, 0.0)
    t = [jnp.where(good, t[k] * inv, iso[k]) for k in range(6)]
    # m0 ~ 0 next to large m2 overflows m2/m0 (inf - inf = NaN below): such a cell gets I/3 times its m0
    sane = jnp.abs(t[0]) < T_MAX
    for k in range(1, 6):
        sane = sane & (jnp.abs(t[k]) < T_MAX)
    t = [jnp.where(sane, t[k], iso[k]) for k in range(6)]
    q = 1.0 / 3.0
    a, b, cz = t[0] - q, t[1] - q, t[2] - q
    xy, xz, yz = t[3], t[4], t[5]
    p = jnp.sqrt((a * a + b * b + cz * cz + 2.0 * (xy * xy + xz * xz + yz * yz)) / 6.0)
    ip = 1.0 / jnp.maximum(p, 1e-150)
    det = a * (b * cz - yz * yz) - xy * (xy * cz - yz * xz) + xz * (xy * yz - b * xz)
    r = jnp.clip(0.5 * det * ip * ip * ip, -1.0, 1.0)
    lmin = q + 2.0 * p * jnp.cos(jnp.arccos(r) / 3.0 + 2.0 * math.pi / 3.0)
    lmin = jnp.where(p > 1e-150, lmin, q)
    al = jnp.where(lmin < 0.0, -lmin / (q - jnp.minimum(lmin, 0.0)), 0.0)
    t = [(1.0 - al) * t[k] + al * iso[k] for k in range(6)]
    return [m0] + m1 + [tk * m0 for tk in t]


def floor(U):
    """Apply `floor_cells` to a (10, N...) state."""
    return jnp.stack(floor_cells([U[k] for k in range(10)]))


def _minmod(a, b):
    return jnp.where(a * b > 0, jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)


NET_BLOCK = 2 ** 24     # cells per network evaluation: XLA materialises its hidden layers


def free_components(U, params, dtype=jnp.float32):
    """Return the learned closure's seven free components (7, N...) for a (10, N...) state.

    The network runs in `dtype` (None keeps FP64) on slabs of at most NET_BLOCK cells.
    """
    nb = 1
    while U[0].size // nb > NET_BLOCK and U.shape[1] % (2 * nb) == 0:
        nb *= 2
    if nb == 1:
        return _free_components(U, params, dtype)
    s = U.shape[1] // nb
    out = jax.lax.map(lambda i: _free_components(
        jax.lax.dynamic_slice_in_dim(U, i * s, s, axis=1), params, dtype), jnp.arange(nb))
    return jnp.moveaxis(out, 0, 1).reshape((7,) + U.shape[1:])


def _free_components(U, params, dtype):
    from luna.radiation.m2_learned_3d import free_components as mlp
    x = jnp.stack(normalised([U[k] for k in range(10)]), axis=-1)
    f = mlp(params, x.reshape(-1, 8), dtype).reshape(x.shape[:-1] + (7,))
    return jnp.moveaxis(f, -1, 0)


def free_row(V, d):
    """Return row d (six m3_dij, i<=j) of normalised trace-free m3 from the seven components V."""
    from luna.radiation.m2_learned_3d import _FREE
    v = dict(zip(_FREE, [V[k] for k in range(7)]))
    v[(0, 2, 2)] = -(v[(0, 0, 0)] + v[(0, 1, 1)])
    v[(1, 2, 2)] = -(v[(0, 0, 1)] + v[(1, 1, 1)])
    v[(2, 2, 2)] = -(v[(0, 0, 2)] + v[(1, 1, 2)])
    return [v[tuple(sorted((d, i, j)))] for i, j in _SYM]


def frozen_flux(c, q, d, chat):
    """Return the ten fluxes along d with m3 = trace part of c's m1 + q times c's m0."""
    m0 = jnp.maximum(c[0], EPS)
    return flux_cells(c, d, chat, [trace_m3(c, d, i, j) + qk * m0
                                   for (i, j), qk in zip(_SYM, q)])


def _flux(U, d, chat, closure_fn):
    c = [U[k] for k in range(10)]
    row = None
    if closure_fn is not None:
        m3 = closure_fn(U)
        row = [m3[d, i, j] for i, j in _SYM]
    return jnp.stack(flux_cells(c, d, chat, row))


def _slicer(ndim, ax):
    def s(a, b):
        return tuple(slice(a, b if b != 0 else None) if k == ax else slice(None)
                     for k in range(ndim))
    return s


def hancock_sweep(U, Up, dt, dx, d, *, chat, closure_fn=None, diffusion=1.0):
    """One MUSCL-Hancock GLF sweep along d on ten moments; Up has 2 ghosts along d."""
    s = _slicer(U.ndim, 1 + d)
    slope = _minmod(Up[s(1, -1)] - Up[s(0, -2)], Up[s(2, 0)] - Up[s(1, -1)])
    Uc = Up[s(1, -1)]
    Um, Uq = Uc - 0.5 * slope, Uc + 0.5 * slope
    # no barrier on these fluxes: 6 % at 128^3 on an A100, same memory (CODE_NOTES.md)
    Fq, Fm = _flux(Uq, d, chat, closure_fn), _flux(Um, d, chat, closure_fn)
    inc = (0.5 * dt / dx) * (Fq - Fm)
    UL = (Uq - inc)[s(0, -1)]
    UR = (Um - inc)[s(1, 0)]
    UL, UR = jax.lax.optimization_barrier((UL, UR))
    fL, fR = jax.lax.optimization_barrier((_flux(UL, d, chat, closure_fn),
                                           _flux(UR, d, chat, closure_fn)))
    face = 0.5 * (fL + fR) - 0.5 * (diffusion * chat) * (UR - UL)
    return U - dt / dx * (face[s(1, 0)] - face[s(0, -1)])


def frozen_sweep(U, Up, Vp, dt, dx, d, *, chat, diffusion=1.0):
    """`hancock_sweep` with the learned closure frozen per cell; Vp has 1 ghost on every axis."""
    s = _slicer(U.ndim, 1 + d)
    inner = tuple(slice(None) if k in (0, 1 + d) else slice(1, -1) for k in range(U.ndim))
    Qc = jnp.stack(free_row(Vp[inner].astype(jnp.float64), d))     # cells -1..n along d

    def fl(V, Q):
        return jnp.stack(frozen_flux([V[k] for k in range(10)], [Q[k] for k in range(6)], d, chat))

    slope = _minmod(Up[s(1, -1)] - Up[s(0, -2)], Up[s(2, 0)] - Up[s(1, -1)])
    Uc = Up[s(1, -1)]
    Um, Uq = Uc - 0.5 * slope, Uc + 0.5 * slope
    Fq, Fm = fl(Uq, Qc), fl(Um, Qc)
    inc = (0.5 * dt / dx) * (Fq - Fm)
    UL, qL = (Uq - inc)[s(0, -1)], Qc[s(0, -1)]
    UR, qR = (Um - inc)[s(1, 0)], Qc[s(1, 0)]
    UL, UR = jax.lax.optimization_barrier((UL, UR))
    fL, fR = jax.lax.optimization_barrier((fl(UL, qL), fl(UR, qR)))
    face = 0.5 * (fL + fR) - 0.5 * (diffusion * chat) * (UR - UL)
    return U - dt / dx * (face[s(1, 0)] - face[s(0, -1)])


def _pad_local(U, g, boundary, axes):
    mode = "wrap" if boundary == 0 else "edge"
    width = [(0, 0)] * U.ndim
    for d in axes:
        width[1 + d] = (g, g)
    return jnp.pad(U, width, mode=mode)


def _pad_axis_local(U, g, boundary, d):
    return _pad_local(U, g, boundary, (d,))


def split_step(U, dt, dx, *, chat, boundary=0, closure_fn=None, params=None, pad=None,
               diffusion=1.0, kernel="auto"):
    """One transport step of U (10, N, N, N): x y z z y x sweeps at dt/2, floored after each.

    Closure: none (trace), `closure_fn(U) -> m3` at every face, or learned `params` frozen per step.
    `pad(A, g, axes)` supplies ghosts (None: local); `kernel`: "auto", "pallas" or "xla".
    """
    from luna.radiation import m2_pallas
    fused = m2_pallas.use_pallas(U, closure_fn, kernel)
    Vp = None
    if params is not None:
        if fused:
            V = m2_pallas.free_components(U, params, interpret=m2_pallas.interpret_mode())
        else:
            V = free_components(U, params)
        Vp = _pad_local(V, 1, boundary, (0, 1, 2)) if pad is None else pad(V, 1, axes=(0, 1, 2))
    if fused:
        # no XLA pad on axes this device holds whole (m2_pallas.split_step)
        local = (0, 1, 2) if pad is None else getattr(pad, "local_axes", ())
        return m2_pallas.split_step(U, dt, dx, chat=chat, boundary=boundary, pad=pad, local=local,
                                    diffusion=diffusion, Vp=Vp, interpret=m2_pallas.interpret_mode())
    for d in (0, 1, 2, 2, 1, 0):
        Up = _pad_axis_local(U, 2, boundary, d) if pad is None else pad(U, 2, axes=(d,))
        if Vp is not None:
            U = floor(frozen_sweep(U, Up, Vp, 0.5 * dt, dx, d, chat=chat, diffusion=diffusion))
        else:
            U = floor(hancock_sweep(U, Up, 0.5 * dt, dx, d, chat=chat, closure_fn=closure_fn,
                                    diffusion=diffusion))
        U = jax.lax.optimization_barrier(U)
    return U


def substep_dt(dx, *, chat, cfl=0.4):
    """Return the longest stable split step, 2 cfl dx / c_hat."""
    return 2.0 * cfl * dx / chat
