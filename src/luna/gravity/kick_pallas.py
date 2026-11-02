"""The gas kick as Pallas (Triton) kernels, 3-D: plain (any float dtype), flux-mode (energy by a given dE), or plus the live sinks' softened field (FP64).

Same terms, same order as `backend.kick_gas_body` (+ `Sinks._field_at`), so they match the XLA path bitwise / to round-off.
"""
import functools

import jax
import jax.numpy as jnp


def _strong(x):
    from jax._src.lax.lax import _convert_element_type
    return _convert_element_type(x, jnp.dtype("float64"), weak_type=False)


def _pow2_divisor(n, cap):
    t = 1
    while t * 2 <= cap and n % (t * 2) == 0:
        t *= 2
    return t


def _pow2_at_least(n):
    t = 1
    while t < n:
        t *= 2
    return t


def tile_for(shape):
    return (1, _pow2_divisor(shape[1], 4), _pow2_divisor(shape[2], 64))


def sink_args(extra):
    """(sinks, pos, mass, active) when `extra` is exactly the sinks' field, else None."""
    if not extra or len(extra) != 1:
        return None
    fn, args = extra[0]
    owner = getattr(fn, "__self__", None)
    if getattr(fn, "__name__", "") != "_sink_field_on_grid" or owner is None:
        return None
    return (owner, *args)


def select(kernel, autodiff=False):
    """True if the Pallas kick should run ("auto" means on a GPU); autodiff always gets XLA (no VJP)."""
    if kernel not in ("auto", "xla", "pallas"):
        raise ValueError(f"gravity_kernel={kernel!r}: 'auto', 'xla' or 'pallas'")
    if kernel == "xla" or autodiff:
        return False
    return kernel == "pallas" or jax.default_backend() in ("gpu", "rocm", "cuda")


@functools.lru_cache(maxsize=None)
def _call(shape, n_slots, dxyz, G, eps2, interpret, mat, box=None, nvidia=False):
    from jax.experimental import pallas as pl
    if not interpret:
        jax.config.update("jax_pallas_use_mosaic_gpu", False)   # ROCm: Triton only
    TX, TY, TZ = tile_for(shape)
    Nx, Ny, Nz = shape
    sizes = (TX, TY, TZ)
    inv_box = None if box is None else tuple(1.0 / L for L in box)

    def kernel(k_ref, org_ref, px_ref, py_ref, pz_ref, a_ref, m_ref,
               u_ref, gx_ref, gy_ref, gz_ref, o_ref):
        s, dt = k_ref[0], k_ref[1]
        n_act = org_ref[3]
        base = (pl.program_id(0) * TX, pl.program_id(1) * TY, pl.program_id(2) * TZ)
        idx = tuple(pl.ds(base[a], sizes[a]) for a in range(3))
        # cell centres as Sinks forms them: ((origin + index) + 0.5) * dx
        coords = []
        for ax in range(3):
            i = jax.lax.broadcasted_iota(jnp.int32, sizes, ax) + base[ax] + org_ref[ax]
            coords.append((i.astype(jnp.float64) + 0.5) * dxyz[ax])
        X, Y, Z = coords
        zero = jnp.zeros(sizes, jnp.float64)

        def slot(j, g):
            fx, fy, fz = g
            ddx = X - px_ref[j]; ddy = Y - py_ref[j]; ddz = Z - pz_ref[j]
            if box is not None:             # nearest image, as Sinks._image; floor makes 1/L bitwise
                ddx = ddx - box[0] * jnp.floor(ddx * inv_box[0] + 0.5)
                ddy = ddy - box[1] * jnp.floor(ddy * inv_box[1] + 0.5)
                ddz = ddz - box[2] * jnp.floor(ddz * inv_box[2] + 0.5)
            r = jax.lax.rsqrt(ddx * ddx + ddy * ddy + ddz * ddz + eps2)
            inv_r3 = a_ref[j] * G * m_ref[j] * r * r * r
            return fx - ddx * inv_r3, fy - ddy * inv_r3, fz - ddz * inv_r3

        fx, fy, fz = jax.lax.fori_loop(0, n_act, slot, (zero, zero, zero))
        # backend.kick_gas_body with extra = the sinks' field
        rho = u_ref[(0,) + idx]
        mx, my, mz = (u_ref[(c,) + idx] for c in (1, 2, 3))
        E = u_ref[(4,) + idx]
        gx = gx_ref[idx] * s + fx
        gy = gy_ref[idx] * s + fy
        gz = gz_ref[idx] * s + fz
        nmx = mx + dt * rho * gx
        nmy = my + dt * rho * gy
        nmz = mz + dt * rho * gz
        E = E + dt * (0.5 * (mx + nmx) * gx + 0.5 * (my + nmy) * gy + 0.5 * (mz + nmz) * gz)
        for c, v in enumerate((rho, nmx, nmy, nmz, E)):
            o_ref[(c,) + idx] = v

    out = jax.ShapeDtypeStruct((5, Nx, Ny, Nz), jnp.float64, manual_axis_type=mat)
    cp = None
    if nvidia and not interpret:
        # a light loop over slots: several cells per thread beat one (CODE_NOTES, gravity)
        from jax.experimental.pallas import triton as plt
        cp = plt.CompilerParams(num_warps=1)
    return pl.pallas_call(kernel, out_shape=out, grid=(Nx // TX, Ny // TY, Nz // TZ),
                          interpret=interpret, compiler_params=cp)


def kick(U, g, s, dt, pos, mass, active, *, origin, dxyz, G, eps, box=None):
    """U after the kick by g * s plus the live sinks' field.

    origin: the block's global cell offset (zeros on one device); pos (n, 3), mass (n,), active (n,): sink slots.
    """
    U = _strong(U)
    g = [_strong(q) for q in g]
    n = int(mass.shape[0])
    n2 = _pow2_at_least(max(n, 1))
    live = active != 0.0
    order = jnp.argsort(~live, stable=True)            # live slots first, in slot order

    def col(v):
        v = _strong(jnp.asarray(v, jnp.float64)[order])
        return jnp.concatenate([v, jnp.zeros(n2 - n, jnp.float64)]) if n2 > n else v
    px, py, pz = (col(pos[:, k]) for k in range(3))
    a, m = col(active), col(mass)
    org = jnp.stack([jnp.asarray(o, jnp.int32) for o in origin]
                    + [jnp.sum(live).astype(jnp.int32)])
    k = jnp.stack([jnp.asarray(s, jnp.float64), jnp.asarray(dt, jnp.float64)])
    shape = tuple(int(v) for v in U.shape[1:])
    mat = getattr(jax.typeof(U), "mat", None)
    interpret = jax.default_backend() == "cpu"
    nvidia = not interpret and jax.devices()[0].device_kind.upper().startswith("NVIDIA")
    call = _call(shape, n2, tuple(float(v) for v in dxyz), float(G), float(eps) ** 2,
                 interpret, mat, None if box is None else tuple(float(v) for v in box), nvidia)
    return call(k, org, px, py, pz, a, m, U, g[0], g[1], g[2])


@functools.lru_cache(maxsize=None)
def _plain_call(shape, dtype, interpret, mat):
    from jax.experimental import pallas as pl
    if not interpret:
        jax.config.update("jax_pallas_use_mosaic_gpu", False)   # ROCm: Triton only
    TX, TY, TZ = sizes = tile_for(shape)
    Nx, Ny, Nz = shape

    def kernel(k_ref, u_ref, gx_ref, gy_ref, gz_ref, o_ref):
        s, dt = k_ref[0], k_ref[1]
        base = (pl.program_id(0) * TX, pl.program_id(1) * TY, pl.program_id(2) * TZ)
        idx = tuple(pl.ds(base[a], sizes[a]) for a in range(3))
        rho = u_ref[(0,) + idx]
        mx, my, mz = (u_ref[(c,) + idx] for c in (1, 2, 3))
        gx, gy, gz = gx_ref[idx] * s, gy_ref[idx] * s, gz_ref[idx] * s
        nmx = mx + dt * rho * gx
        nmy = my + dt * rho * gy
        nmz = mz + dt * rho * gz
        E = u_ref[(4,) + idx] + dt * (0.5 * (mx + nmx) * gx + 0.5 * (my + nmy) * gy
                                      + 0.5 * (mz + nmz) * gz)
        for c, v in enumerate((rho, nmx, nmy, nmz, E)):
            o_ref[(c,) + idx] = v

    out = jax.ShapeDtypeStruct((5, Nx, Ny, Nz), jnp.dtype(dtype), manual_axis_type=mat)
    return pl.pallas_call(kernel, out_shape=out, grid=(Nx // TX, Ny // TY, Nz // TZ),
                          interpret=interpret)


def applies_plain(U, g):
    """The plain kick's preconditions: 5 gas rows (energy last), 3-D, g in U's float dtype."""
    return (getattr(U, "ndim", 0) == 4 and U.shape[0] == 5 and jnp.issubdtype(U.dtype, jnp.floating)
            and all(q.dtype == U.dtype and q.shape == U.shape[1:] for q in g))


def kick_plain(U, g, s, dt):
    """`backend.kick_gas_body(U, g, s, dt)` as one Pallas kernel, bitwise equal."""
    k = jnp.stack([jnp.asarray(s, U.dtype), jnp.asarray(dt, U.dtype)])
    shape = tuple(int(v) for v in U.shape[1:])
    mat = getattr(jax.typeof(U), "mat", None)
    interpret = jax.default_backend() == "cpu"
    return _plain_call(shape, U.dtype.name, interpret, mat)(k, U, g[0], g[1], g[2])


@functools.lru_cache(maxsize=None)
def _flux_call(shape, dtype, interpret, mat):
    from jax.experimental import pallas as pl
    if not interpret:
        jax.config.update("jax_pallas_use_mosaic_gpu", False)   # ROCm: Triton only
    TX, TY, TZ = sizes = tile_for(shape)
    Nx, Ny, Nz = shape

    def kernel(k_ref, u_ref, gx_ref, gy_ref, gz_ref, de_ref, o_ref):
        s, dt = k_ref[0], k_ref[1]
        base = (pl.program_id(0) * TX, pl.program_id(1) * TY, pl.program_id(2) * TZ)
        idx = tuple(pl.ds(base[a], sizes[a]) for a in range(3))
        rho = u_ref[(0,) + idx]
        gx, gy, gz = gx_ref[idx] * s, gy_ref[idx] * s, gz_ref[idx] * s
        nmx = u_ref[(1,) + idx] + dt * rho * gx
        nmy = u_ref[(2,) + idx] + dt * rho * gy
        nmz = u_ref[(3,) + idx] + dt * rho * gz
        E = u_ref[(4,) + idx] + de_ref[idx]
        for c, v in enumerate((rho, nmx, nmy, nmz, E)):
            o_ref[(c,) + idx] = v

    out = jax.ShapeDtypeStruct((5, Nx, Ny, Nz), jnp.dtype(dtype), manual_axis_type=mat)
    return pl.pallas_call(kernel, out_shape=out, grid=(Nx // TX, Ny // TY, Nz // TZ),
                          interpret=interpret)


def kick_flux(U, g, s, dt, dE):
    """`backend.kick_gas_flux_body(U, g, s, dt, dE)` as one Pallas kernel, bitwise equal."""
    k = jnp.stack([jnp.asarray(s, U.dtype), jnp.asarray(dt, U.dtype)])
    shape = tuple(int(v) for v in U.shape[1:])
    mat = getattr(jax.typeof(U), "mat", None)
    interpret = jax.default_backend() == "cpu"
    return _flux_call(shape, U.dtype.name, interpret, mat)(k, U, g[0], g[1], g[2], dE)
