"""The wide-halo MUSCL-Hancock divergence as Pallas (Triton) GPU kernels in 1-D, 2-D and 3-D: one
fused kernel, or on NVIDIA a predictor pass and a faces pass. minmod, MC or van Leer
(`slopes.slope_limiter`), HLLC or HLL (`fluxes` cores), FP64, FP32 and (3-D) recon32;
see docs/HYDRO_KERNEL_PATHS.md.
"""
import functools

import jax
import jax.numpy as jnp

from luna.hydro.fluxes import _hll_core, _hllc_core
from luna.hydro.jacobian import (primitive_jacobian_rows_1d, primitive_jacobian_rows_2d,
                                 primitive_jacobian_rows_3d_axis)
from luna.hydro.slopes import face_rule, slope_limiter

_CORES = {"hllc": _hllc_core, "hll": _hll_core}
LIMITERS = (0, 1, 2)                      # minmod, MC, van Leer: slopes.slope_limiter


def _strong(x, dtype):
    """Return x as ``dtype`` with weak_type=False."""
    from jax._src.lax.lax import _convert_element_type
    return _convert_element_type(x, jnp.dtype(dtype), weak_type=False)


def _slope(Wc, Wm, Wp, limiter, nc):
    """slopes.slope_limiter(Wc - Wm, Wp - Wc) on lists of tiles, as slopes.compute_slopes_from_padded."""
    return [slope_limiter(Wc[c] - Wm[c], Wp[c] - Wc[c], limiter) for c in range(nc)]


def _jac(Wc, dW, d, gamma, nc):
    """jacobian.primitive_jacobian_rows_3d_axis on lists of tiles; scalars ride on vn."""
    return primitive_jacobian_rows_3d_axis(Wc, dW, gamma, d) + [Wc[1 + d] * dW[c]
                                                                for c in range(5, nc)]


def _predict(Wc, slopes, inv, h, gamma, nc):
    """The half-step predictor W - h A dW/dx (the XLA step's jacobian rows); slopes[d] per axis."""
    ndim = len(slopes)
    if ndim == 3:                         # x then y then z, as muscl_hancock._muscl_hancock_3d
        A = _jac(Wc, [s * inv[0] for s in slopes[0]], 0, gamma, nc)
        for d in (1, 2):
            B = _jac(Wc, [s * inv[d] for s in slopes[d]], d, gamma, nc)
            A = [a + b for a, b in zip(A, B)]
    elif ndim == 2:
        dx = [s * inv[0] for s in slopes[0]]
        dy = [s * inv[1] for s in slopes[1]]
        A = primitive_jacobian_rows_2d(Wc[:4], dx[:4], dy[:4], gamma) + [
            Wc[1] * dx[c] + Wc[2] * dy[c] for c in range(4, nc)]
    else:
        dx = [s * inv[0] for s in slopes[0]]
        A = primitive_jacobian_rows_1d(Wc[:3], dx[:3], gamma) + [
            Wc[1] * dx[c] for c in range(3, nc)]
    return [Wc[c] - h * A[c] for c in range(nc)]


def _face(whL, sL, whR, sR, d, gamma, nc, cast, solver="hllc", ndim=3, wcL=None, wcR=None, guard=False):
    """(flux rows, left mask, right mask): HLLC or HLL on axis d from the left cell's (W_half, slope, centre) and the
    right cell's; cast: flux dtype or None; guard: slopes.face_guard_fn spec; a mask is (first order, floored or None)."""
    fix = face_rule(guard)
    L, *mL = fix([a + 0.5 * s for a, s in zip(whL, sL)], wcL, ndim + 1)
    R, *mR = fix([a - 0.5 * s for a, s in zip(whR, sR)], wcR, ndim + 1)
    if cast is not None:
        L, R = [x.astype(cast) for x in L], [x.astype(cast) for x in R]
    o = [k for k in range(ndim) if k != d]
    tL, tR = [L[1 + k] for k in o], [R[1 + k] for k in o]
    if ndim < 3:                          # missing transverse velocities are zero (fluxes.rotate_primitive_pair)
        z = jnp.zeros_like(L[0])
        tL, tR = tL + [z] * (2 - len(o)), tR + [z] * (2 - len(o))
    iP = ndim + 1
    F_rho, F_mn, F_t1, F_t2, F_E = _CORES[solver](
        L[0], R[0], L[1 + d], R[1 + d], L[iP], R[iP], tL[0], tR[0], tL[1], tR[1], gamma)
    mom = {d: F_mn, **dict(zip(o, (F_t1, F_t2)))}
    # a passive scalar rides on the mass flux, upwind of the contact (exact HLLC)
    up = F_rho >= 0.0
    return [F_rho] + [mom[k] for k in range(ndim)] + [F_E] + [
        F_rho * jnp.where(up, L[c], R[c]) for c in range(ndim + 2, nc)], mL, mR


def _own_faces(own):
    """Per cell: the bitmask of its first-order face states, bit 2 d the +d face and 2 d + 1 the -d face (as XLA's
    report); with floors (int16) the floored ones << 8. own: (mask, mask or None) for -d, +d per axis d."""
    floors = own[0][1] is not None
    t = jnp.int16 if floors else jnp.int8
    bit = [2 * (j // 2) + (1 - j % 2) for j in range(len(own))]          # own[2d] is -d, own[2d + 1] is +d
    n = sum(m[0].astype(t) << b for m, b in zip(own, bit))
    if floors:
        n = n + sum(m[1].astype(t) << (b + 8) for m, b in zip(own, bit))
    return n


def _report_out(shape, mat, guard):
    """The per-cell face report output (_own_faces): int8, int16 with floors."""
    from luna.hydro.slopes import _parse_spec
    t = "int16" if _parse_spec(guard)[1] is not None else "int8"
    return jax.ShapeDtypeStruct(tuple(shape), jnp.dtype(t), manual_axis_type=mat)


def _pow2_divisor(n, cap):
    t = 1
    while t * 2 <= cap and n % (t * 2) == 0:
        t *= 2
    return t


def _pow2_floor(n, cap):
    t = 1
    while t * 2 <= min(n, cap):
        t *= 2
    return t


# Tile caps per dimension, one cell per thread on NVIDIA; 1-D/2-D from the A100 sweep of
# 2026-10-14 (CODE_NOTES.md): every tile is bitwise the same, the spread is 10-30 %.
FUSED_TILE = {1: (64,), 2: (4, 32), 3: (1, 4, 64)}
# Split path: (predictor caps, faces caps, faces caps in FP32/recon32).
SPLIT_TILE = {1: ((256,), (256,), (256,)),
              2: ((2, 64), (2, 64), (4, 32)),
              3: ((1, 4, 32), (1, 2, 64), (1, 4, 64))}


def tile_for(shape):
    """The fused kernel's tile: FUSED_TILE[ndim] shrunk to powers of two that divide the block."""
    return tuple(_pow2_divisor(n, c) for n, c in zip(shape, FUSED_TILE[len(shape)]))


def split_tiles(shape, mixed_or_fp32):
    """(predictor tile, faces tile) of the split path, one cell per thread (A100, CODE_NOTES.md)."""
    capA, capB, capB32 = SPLIT_TILE[len(shape)]
    tA = tuple(_pow2_floor(n + 2, c) for n, c in zip(shape, capA))
    tB = tuple(_pow2_divisor(n, c) for n, c in zip(shape, capB32 if mixed_or_fp32 else capB))
    return tA, tB


def unsupported(*, spatial_dim, order, limiter, riemann_solver, boundary, fused=False,
                recon32=False, autodiff=False, n_scalars=0):
    """What of this configuration the Pallas kernels cannot run, or None (fused, recon32: as requested)."""
    from luna.time_integration.integrators import _hydro_paths
    if autodiff:
        return "autodiff_mode (the kernels have no VJP; autodiff differentiates XLA)"
    if spatial_dim not in (1, 2, 3):
        return f"SPATIAL_DIMENSION={spatial_dim}"
    if order < 1:
        return "ORDER=0 (first-order Godunov is XLA only)"
    if fused:
        return "hydro_fused=True (that switch selects XLA's fused kernel)"
    if not isinstance(limiter, int) or limiter not in LIMITERS:
        return (f"limiter {limiter!r} (LIMITER 0/1/2 only; limiter_tangent_eps is an autodiff "
                "tangent)")
    if riemann_solver not in _CORES:
        return f"riemann_solver={riemann_solver!r} (hllc or hll)"
    if recon32 and spatial_dim != 3:
        return f"hydro_recon32 in {spatial_dim}-D (recon32 is 3-D only, as in XLA)"
    if not _hydro_paths(order, spatial_dim, boundary, False, False)[2]:
        return (f"BOUNDARY={boundary} in {spatial_dim}-D (the narrow one-ghost path; Pallas is "
                "the wide two-ghost step)")
    if n_scalars and riemann_solver != "hllc":
        return ("passive scalars with HLL (the scalar flux is upwind of the contact, which HLL "
                "has none of)")
    return None


def select(kernel, reason=None):
    """True if the Pallas kernels run: "pallas" always (raising with ``reason`` if it cannot), "auto" on a GPU when reason is None."""
    if kernel not in ("auto", "xla", "pallas"):
        raise ValueError(f"hydro_kernel={kernel!r}: 'auto', 'xla' or 'pallas'")
    if kernel == "xla":
        return False
    if kernel == "pallas":
        if reason is not None:
            raise NotImplementedError(f"hydro_kernel='pallas' cannot run {reason}; use "
                                      "hydro_kernel='xla', or 'auto' (XLA where Pallas cannot run)")
        return True
    return reason is None and jax.default_backend() in ("gpu", "rocm", "cuda")


def _launch(nvidia, interpret, cells):
    if nvidia and not interpret:
        from jax.experimental.pallas import triton as plt
        return plt.CompilerParams(num_warps=max(1, cells // 32))
    return None


def _cells(tile):
    n = 1
    for t in tile:
        n *= t
    return n


def _sh(p, d, s):
    q = list(p)
    q[d] += s
    return tuple(q)


def _faces_out(shape, dtype, mat):
    """The face mass flux outputs: N_d + 1 faces along axis d."""
    return tuple(jax.ShapeDtypeStruct(tuple(n + (a == d) for a, n in enumerate(shape)), jnp.dtype(dtype),
                                      manual_axis_type=mat) for d in range(len(shape)))


def _store_mass_flux(f_refs, d, base, tile, Fm, Fp):
    """Each cell writes its lower face i and its upper face i + 1: inner faces are written twice with the same bits
    (both neighbours compute them from the same inputs). A `pl.when` for the last tile alone cost +45 % (A100, FP64
    256^3); the second store costs +4 %."""
    from jax.experimental import pallas as pl
    ndim = len(tile)
    f_refs[d][tuple(pl.ds(base[a], tile[a]) for a in range(ndim))] = Fm[0]
    f_refs[d][tuple(pl.ds(base[a] + (a == d), tile[a]) for a in range(ndim))] = Fp[0]


@functools.lru_cache(maxsize=None)
def _call(shape, gamma, inv, interpret, mat, ns=0, dtype="float64", rdtype="float64",
          nvidia=False, solver="hllc", limiter=0, mass_flux=False, guard=False, report=False):
    """The fused kernel: slopes, 2 ndim + 1 predictors, 2 ndim faces and dU per cell (mass_flux: and F_rho; report: and
    _own_faces); guard: slopes.face_guard_fn spec."""
    from jax.experimental import pallas as pl
    if not interpret:
        jax.config.update("jax_pallas_use_mosaic_gpu", False)   # ROCm: Triton only
    ndim = len(shape)
    sizes = tile_for(shape)
    nc = ndim + 2 + ns                    # gas rows, then the passive scalars
    mixed = rdtype != dtype               # recon32: reconstruct in rdtype, fluxes in dtype

    grid = tuple(shape[a] // sizes[a] for a in range(ndim))

    def kernel(k_ref, w_ref, o_ref, *f_refs):
        f_refs, r_ref = (f_refs[:-1], f_refs[-1]) if report else (f_refs, None)
        dt = k_ref[0]
        dt_r = dt.astype(rdtype) if mixed else dt
        base = tuple(pl.program_id(a) * sizes[a] for a in range(ndim))
        Wc_, sl_, wh_ = {}, {}, {}

        def W(p):
            if p not in Wc_:
                idx = tuple(pl.ds(base[a] + 2 + p[a], sizes[a]) for a in range(ndim))
                Wc_[p] = [w_ref[(c,) + idx] for c in range(nc)]
            return Wc_[p]

        def slope(p, d):
            if (p, d) not in sl_:
                sl_[(p, d)] = _slope(W(p), W(_sh(p, d, -1)), W(_sh(p, d, 1)), limiter, nc)
            return sl_[(p, d)]

        def whalf(p):                     # the predictor at p, 2 ndim + 1 times per cell
            if p not in wh_:
                wh_[p] = _predict(W(p), [slope(p, d) for d in range(ndim)], inv, 0.5 * dt_r,
                                  gamma, nc)
            return wh_[p]

        def face(p, d):                   # the Riemann flux between p and p + e_d
            q = _sh(p, d, 1)
            return _face(whalf(p), slope(p, d), whalf(q), slope(q, d), d, gamma, nc,
                         dtype if mixed else None, solver, ndim, W(p), W(q), guard)

        o = (0,) * ndim
        dU, own = None, []
        for d in range(ndim):
            (Fm, _, mR), (Fp, mL, _) = face(_sh(o, d, -1), d), face(o, d)
            own += [mR, mL]                 # this cell's two face states on axis d
            if mass_flux:
                _store_mass_flux(f_refs, d, base, sizes, Fm, Fp)
            contrib = [(a - b) * inv[d] for a, b in zip(Fm, Fp)]
            dU = contrib if dU is None else [x + y for x, y in zip(dU, contrib)]
        idx = tuple(pl.ds(base[a], sizes[a]) for a in range(ndim))
        for c in range(nc):
            o_ref[(c,) + idx] = dU[c]
        if report:
            r_ref[idx] = _own_faces(own)

    out = jax.ShapeDtypeStruct((nc,) + tuple(shape), jnp.dtype(dtype), manual_axis_type=mat)
    if mass_flux or report:
        out = ((out,) + (_faces_out(shape, dtype, mat) if mass_flux else ())
               + ((_report_out(shape, mat, guard),) if report else ()))
    # NVIDIA: one cell per thread; Triton's 4 warps spill (A100 FP64 2.9x slower, CODE_NOTES.md).
    return pl.pallas_call(kernel, out_shape=out, grid=grid,
                          interpret=interpret,
                          compiler_params=_launch(nvidia, interpret, _cells(sizes)))


@functools.lru_cache(maxsize=None)
def _predict_call(shape, gamma, inv, interpret, mat, dtype, rdtype, nvidia, limiter=0):
    """Pass A: W_half on cells -1 .. N per axis, shape (ndim + 2, N+2, ...), once per cell."""
    from jax.experimental import pallas as pl
    if not interpret:
        jax.config.update("jax_pallas_use_mosaic_gpu", False)   # ROCm: Triton only
    ndim = len(shape)
    M = tuple(n + 2 for n in shape)
    tile, _ = split_tiles(shape, rdtype != dtype or dtype == "float32")
    nc, mixed = ndim + 2, rdtype != dtype
    grid = tuple(-(-M[a] // tile[a]) for a in range(ndim))

    def kernel(k_ref, w_ref, o_ref):
        dt = k_ref[0]
        dt_r = dt.astype(rdtype) if mixed else dt
        # the last tile is shifted back to end at M: in bounds, a few cells written twice, same bits
        base = tuple(jnp.minimum(pl.program_id(a) * tile[a], M[a] - tile[a]) for a in range(ndim))
        Wc_ = {}

        def W(p):                         # cell i of W_half is W_pad index i + 1
            if p not in Wc_:
                idx = tuple(pl.ds(base[a] + 1 + p[a], tile[a]) for a in range(ndim))
                Wc_[p] = [w_ref[(c,) + idx] for c in range(nc)]
            return Wc_[p]

        o = (0,) * ndim
        slopes = [_slope(W(o), W(_sh(o, d, -1)), W(_sh(o, d, 1)), limiter, nc)
                  for d in range(ndim)]
        wh = _predict(W(o), slopes, inv, 0.5 * dt_r, gamma, nc)
        idx = tuple(pl.ds(base[a], tile[a]) for a in range(ndim))
        for c in range(nc):
            o_ref[(c,) + idx] = wh[c]

    out = jax.ShapeDtypeStruct((nc,) + M, jnp.dtype(rdtype), manual_axis_type=mat)
    return pl.pallas_call(kernel, out_shape=out, grid=grid, interpret=interpret,
                          compiler_params=_launch(nvidia, interpret, _cells(tile)))


@functools.lru_cache(maxsize=None)
def _faces_call(shape, gamma, inv, interpret, mat, dtype, rdtype, nvidia, solver="hllc",
                limiter=0, mass_flux=False, guard=False, report=False):
    """Pass B: slopes (recomputed, cheap) and W_half (read) -> Riemann faces -> dU (mass_flux: and F_rho; report: and
    _own_faces); guard: slopes.face_guard_fn spec."""
    from jax.experimental import pallas as pl
    ndim = len(shape)
    _, tile = split_tiles(shape, rdtype != dtype or dtype == "float32")
    nc, mixed = ndim + 2, rdtype != dtype
    grid = tuple(shape[a] // tile[a] for a in range(ndim))

    def kernel(w_ref, wh_ref, o_ref, *f_refs):
        f_refs, r_ref = (f_refs[:-1], f_refs[-1]) if report else (f_refs, None)
        base = tuple(pl.program_id(a) * tile[a] for a in range(ndim))
        Wc_, wh_, sl_ = {}, {}, {}

        def W(p):                         # interior cell p is W_pad index p + 2
            if p not in Wc_:
                idx = tuple(pl.ds(base[a] + 2 + p[a], tile[a]) for a in range(ndim))
                Wc_[p] = [w_ref[(c,) + idx] for c in range(nc)]
            return Wc_[p]

        def WH(p):                        # and W_half index p + 1
            if p not in wh_:
                idx = tuple(pl.ds(base[a] + 1 + p[a], tile[a]) for a in range(ndim))
                wh_[p] = [wh_ref[(c,) + idx] for c in range(nc)]
            return wh_[p]

        def slope(p, d):
            if (p, d) not in sl_:
                sl_[(p, d)] = _slope(W(p), W(_sh(p, d, -1)), W(_sh(p, d, 1)), limiter, nc)
            return sl_[(p, d)]

        def face(p, d):
            q = _sh(p, d, 1)
            return _face(WH(p), slope(p, d), WH(q), slope(q, d), d, gamma, nc,
                         dtype if mixed else None, solver, ndim, W(p), W(q), guard)

        o = (0,) * ndim
        dU, own = None, []
        for d in range(ndim):
            (Fm, _, mR), (Fp, mL, _) = face(_sh(o, d, -1), d), face(o, d)
            own += [mR, mL]                 # this cell's two face states on axis d
            if mass_flux:
                _store_mass_flux(f_refs, d, base, tile, Fm, Fp)
            contrib = [(a - b) * inv[d] for a, b in zip(Fm, Fp)]
            dU = contrib if dU is None else [x + y for x, y in zip(dU, contrib)]
        idx = tuple(pl.ds(base[a], tile[a]) for a in range(ndim))
        for c in range(nc):
            o_ref[(c,) + idx] = dU[c]
        if report:
            r_ref[idx] = _own_faces(own)

    out = jax.ShapeDtypeStruct((nc,) + tuple(shape), jnp.dtype(dtype), manual_axis_type=mat)
    if mass_flux or report:
        out = ((out,) + (_faces_out(shape, dtype, mat) if mass_flux else ())
               + ((_report_out(shape, mat, guard),) if report else ()))
    return pl.pallas_call(kernel, out_shape=out, grid=grid, interpret=interpret,
                          compiler_params=_launch(nvidia, interpret, _cells(tile)))


def divergence(W_pad, dt, *, gamma, inv, n_scalars=0, recon32=False, split=None,
               riemann_solver="hllc", limiter=0, mass_flux=False, face_guard=False, report=False):
    """dU of the wide step from the 2-ghost primitive pad (ndim + 2 + n_scalars, N+4, ...); inv = 1/dx per axis;
    mass_flux: (dU, (F_rho on the N_d + 1 faces along d for each d)); face_guard: slopes.face_guard_fn spec;
    report: also the per-cell face report (integrators._report_from_pallas: first-order bitmask, floored << 8), last."""
    ndim = W_pad.ndim - 1
    if ndim not in (1, 2, 3):
        raise ValueError(f"hydro_kernel=pallas: a 1-, 2- or 3-D pad, got shape {W_pad.shape}")
    if riemann_solver not in _CORES:
        raise ValueError(f"hydro_kernel=pallas: riemann_solver={riemann_solver!r}, expected hllc or hll")
    if not isinstance(limiter, int) or limiter not in LIMITERS:
        raise NotImplementedError(f"hydro_kernel=pallas: limiter {limiter!r}, expected 0 (minmod), "
                                  "1 (MC) or 2 (van Leer)")
    if recon32 and ndim != 3:
        raise NotImplementedError(f"hydro_kernel=pallas: recon32 in {ndim}-D (3-D only, as in XLA)")
    if n_scalars and riemann_solver != "hllc":
        raise NotImplementedError("hydro Pallas passive scalars: HLLC only (the scalar flux is "
                                  "upwind of the contact, which HLL has none of)")
    dtype = jnp.dtype(W_pad.dtype)
    if dtype not in (jnp.dtype("float32"), jnp.dtype("float64")):
        raise TypeError(f"hydro_kernel=pallas: state dtype {dtype}, expected float32 or float64")
    # recon32 on an FP32 state is plain FP32, as in muscl_hancock._muscl_hancock_3d.
    rdtype = jnp.dtype("float32") if recon32 else dtype
    # A weakly typed state lowers to mixed FP32/FP64 tiles in Triton: make it strong.
    W_pad = _strong(W_pad, rdtype)
    shape = tuple(int(n) - 4 for n in W_pad.shape[1:])
    k = jnp.stack([jnp.asarray(dt, dtype), jnp.zeros((), dtype)])
    mat = getattr(jax.typeof(W_pad), "mat", None)
    interpret = jax.default_backend() == "cpu"
    nvidia = not interpret and jax.devices()[0].device_kind.upper().startswith("NVIDIA")
    g, invt = float(gamma), tuple(float(v) for v in inv)
    lim = int(limiter)
    # split: the predictor once per cell (pass A), then the faces (pass B); +ndim+2 fields of
    # (N+2)^ndim memory, faster on an A100 in 2-D and 3-D (1-D: 0.6-0.9x, CODE_NOTES.md).
    # Fused in 1-D, on AMD and with scalars.
    if split is None:
        split = nvidia and not n_scalars and ndim > 1
    if split:
        if n_scalars:
            raise NotImplementedError("hydro Pallas split path: no passive scalars (use split=False)")
        W_half = _predict_call(shape, g, invt, interpret, mat, dtype.name, rdtype.name, nvidia,
                               lim)(k, W_pad)
        out = _faces_call(shape, g, invt, interpret, mat, dtype.name, rdtype.name, nvidia,
                          riemann_solver, lim, bool(mass_flux), face_guard or False,
                          bool(report))(W_pad, W_half)
    else:
        out = _call(shape, g, invt, interpret, mat, int(n_scalars), dtype.name, rdtype.name,
                    nvidia, riemann_solver, lim, bool(mass_flux), face_guard or False,
                    bool(report))(k, W_pad)
    if not report:
        return (out[0], tuple(out[1:])) if mass_flux else out
    return ((out[0], tuple(out[1:-1])) if mass_flux else out[0]), out[-1]
