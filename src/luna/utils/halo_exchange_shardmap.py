"""Per-shard halo exchange via jax.lax.ppermute (1D/2D/3D); call the halo inside a shard_map body.

Boundary codes: 0 periodic, 1 zero-gradient, 2 reflective, 4 diode. See docs/HYDRO_KERNEL_PATHS.md.
"""

import jax
import jax.numpy as jnp

_AXIS_NAMES = {1: ("x",), 2: ("x", "y"), 3: ("x", "y", "z")}


def make_halo_shardmap(mesh, spatial_dim):
    """Return `halo(U_local, boundary, sdim, width=1, axes=None)` for use inside a shard_map body."""
    if spatial_dim not in _AXIS_NAMES:
        raise ValueError(f"spatial_dim must be 1, 2 or 3; got {spatial_dim}")
    axis_names = _AXIS_NAMES[spatial_dim]
    n_proc = {a: int(mesh.shape[a]) for a in axis_names}
    array_axes = list(range(1, spatial_dim + 1))  # after the leading Nvar axis

    def halo(U_local, boundary, sdim, width=1, axes=None):
        """Pad U_local with `width` ghosts; `axes` (spatial indices) limits them to those axes."""
        if sdim != spatial_dim:
            raise ValueError(
                f"halo built for spatial_dim={spatial_dim}, called with {sdim}")
        if boundary not in (0, 1, 2, 4):
            raise NotImplementedError(
                f"shard_map halo supports boundary 0/1/2/4; got {boundary}")
        periodic = boundary == 0
        if axes is not None:
            if boundary not in (0, 1):
                raise NotImplementedError("per-axis halo: boundary 0/1 only")
            U = U_local
            for d in axes:
                axis, name = array_axes[d], axis_names[d]
                left, right = _axis_ghosts(U, axis=axis, axis_name=name,
                                           n_proc=n_proc[name], periodic=periodic,
                                           width=width)
                U = jnp.concatenate([left, U, right], axis=axis)
            return U
        # fill ghosts in place x -> y -> z, reading earlier axes' ghosts, so corners are filled
        w = width
        U = jnp.pad(U_local, [(0, 0)] + [(w, w)] * spatial_dim)
        for k, (axis, name) in enumerate(zip(array_axes, axis_names)):
            view = [slice(None)] * U.ndim
            for later in array_axes[k:]:
                view[later] = slice(w, -w)
            left, right = _axis_ghosts(U[tuple(view)], axis=axis, axis_name=name,
                                       n_proc=n_proc[name], periodic=periodic, width=w,
                                       mirror=boundary == 2)
            # barrier: else XLA copies the whole buffer to read an edge slab while writing into it
            left, right = jax.lax.optimization_barrier((left, right))
            view[axis] = slice(0, w)
            U = U.at[tuple(view)].set(left)
            view[axis] = slice(-w, None)
            U = U.at[tuple(view)].set(right)
        # reflective / diode: correct the outer ghost on boundary shards after edge-filling
        if boundary == 2:
            for axis, name in zip(array_axes, axis_names):
                U = _fix_face(U, axis=axis, axis_name=name, n_proc=n_proc[name],
                              comp=axis, op="reflect", width=w)
        elif boundary == 4:
            for axis, name in zip(array_axes, axis_names):
                U = _fix_face(U, axis=axis, axis_name=name, n_proc=n_proc[name],
                              comp=axis, op="diode", width=w)
        return U

    return halo


def _slab(U, axis, lo, width=1):
    """Return the outer `width` layers (low end if lo, else high) along `axis`, keepdims."""
    sl = [slice(None)] * U.ndim
    sl[axis] = slice(0, width) if lo else slice(-width, None)
    return U[tuple(sl)]


def _set_slab(U, axis, lo, val, width=1):
    sl = [slice(None)] * U.ndim
    sl[axis] = slice(0, width) if lo else slice(-width, None)
    return U.at[tuple(sl)].set(val)


def _pad_axis(U, *, axis, axis_name, n_proc, periodic, width=1):
    """Concatenate a `width`-cell halo along array `axis` (1..3); `axis_name` is the mesh axis."""
    left_ghost, right_ghost = _axis_ghosts(U, axis=axis, axis_name=axis_name,
                                           n_proc=n_proc, periodic=periodic, width=width)
    return jnp.concatenate([left_ghost, U, right_ghost], axis=axis)


def _axis_ghosts(U, *, axis, axis_name, n_proc, periodic, width=1, mirror=False):
    """Return the (left, right) `width`-cell ghost slabs of `U` along `axis`; `mirror`: boundary
    shards mirror their edge cells (reflective) instead of repeating the edge cell."""
    sl_lo = [slice(None)] * U.ndim
    sl_hi = [slice(None)] * U.ndim
    sl_lo[axis] = slice(0, width)
    sl_hi[axis] = slice(-width, None)
    left_slab = U[tuple(sl_lo)]
    right_slab = U[tuple(sl_hi)]

    # perm = [(src, dst)]: my left ghost is the right slab of i-1, sent i -> i+1
    perm_to_right = [(i, (i + 1) % n_proc) for i in range(n_proc)]
    perm_to_left = [(i, (i - 1) % n_proc) for i in range(n_proc)]
    left_ghost_periodic = jax.lax.ppermute(right_slab, axis_name, perm_to_right)
    right_ghost_periodic = jax.lax.ppermute(left_slab, axis_name, perm_to_left)

    if periodic:
        left_ghost = left_ghost_periodic
        right_ghost = right_ghost_periodic
    else:  # zero-gradient / reflective: boundary shards replicate / mirror their own edge cells
        idx = jax.lax.axis_index(axis_name)
        if width == 1:
            left_edge, right_edge = left_slab, right_slab
        elif mirror:
            left_edge, right_edge = jnp.flip(left_slab, axis), jnp.flip(right_slab, axis)
        else:
            e_lo = [slice(None)] * U.ndim
            e_hi = [slice(None)] * U.ndim
            e_lo[axis] = slice(0, 1)
            e_hi[axis] = slice(-1, None)
            reps = [1] * U.ndim
            reps[axis] = width
            left_edge = jnp.tile(U[tuple(e_lo)], reps)
            right_edge = jnp.tile(U[tuple(e_hi)], reps)
        left_ghost = jnp.where(idx == 0, left_edge, left_ghost_periodic)
        right_ghost = jnp.where(idx == n_proc - 1, right_edge, right_ghost_periodic)

    return left_ghost, right_ghost


def _fix_face(U, *, axis, axis_name, n_proc, comp, op, width=1):
    """On boundary shards, reflect (negate) or diode-clamp normal momentum `comp` in the outer ghosts."""
    idx = jax.lax.axis_index(axis_name)
    lo = _slab(U, axis, lo=True, width=width)
    hi = _slab(U, axis, lo=False, width=width)
    if op == "reflect":
        lo_new = lo.at[comp].set(-lo[comp])
        hi_new = hi.at[comp].set(-hi[comp])
    else:  # diode: low face -> min(v,0), high face -> max(v,0)
        lo_new = lo.at[comp].set(jnp.minimum(lo[comp], 0.0))
        hi_new = hi.at[comp].set(jnp.maximum(hi[comp], 0.0))
    lo_final = jnp.where(idx == 0, lo_new, lo)
    hi_final = jnp.where(idx == n_proc - 1, hi_new, hi)
    U = _set_slab(U, axis, lo=True, val=lo_final, width=width)
    U = _set_slab(U, axis, lo=False, val=hi_final, width=width)
    return U
