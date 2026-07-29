"""Explicit per-shard halo exchange via jax.lax.ppermute — the single
multi-GPU halo path for LUNA (1D/2D/3D). Returned halo functions MUST be
called inside a shard_map body. Boundary codes: 0=periodic, 1=zero-gradient,
2=reflective (negate normal momentum in outer ghost), 4=diode (clamp
outer-ghost normal velocity against inflow). Variable layout
[density, mom_x, mom_y, mom_z, energy]: component `axis` is the normal
momentum/velocity along array axis `axis`.
"""

import jax
import jax.numpy as jnp

_AXIS_NAMES = {1: ("x",), 2: ("x", "y"), 3: ("x", "y", "z")}


def make_halo_shardmap(mesh, spatial_dim):
    """Factory: closure-capture the mesh axis sizes, return a halo function
    for use *inside* a shard_map body.

    The returned function has signature `halo(U_local, boundary, spatial_dim)`
    and is called from inside the integrator's shard_map body.
    """
    if spatial_dim not in _AXIS_NAMES:
        raise ValueError(f"spatial_dim must be 1, 2 or 3; got {spatial_dim}")
    axis_names = _AXIS_NAMES[spatial_dim]
    n_proc = {a: int(mesh.shape[a]) for a in axis_names}
    array_axes = list(range(1, spatial_dim + 1))  # leading Nvar dim -> axes 1..D

    def halo(U_local, boundary, sdim, width=1):
        if sdim != spatial_dim:
            raise ValueError(
                f"halo built for spatial_dim={spatial_dim}, called with {sdim}")
        if boundary not in (0, 1, 2, 4):
            raise NotImplementedError(
                f"shard_map halo supports boundary 0/1/2/4; got {boundary}")
        if width != 1 and boundary in (2, 4):
            raise NotImplementedError(
                "width>1 halo only supports boundary 0/1 (the _fix_face outer-"
                "ghost correction is single-layer)")
        periodic = boundary == 0
        # x -> y -> z order; later axes use the already-padded array so corner
        # cells fill in correctly.
        U = U_local
        for axis, name in zip(array_axes, axis_names):
            U = _pad_axis(U, axis=axis, axis_name=name, n_proc=n_proc[name],
                          periodic=periodic, width=width)
        # Reflective / diode: fix the OUTER ghost on the boundary shards after
        # all axes are padded (edge-fill first, then correct the outer ghost).
        if boundary == 2:
            for axis, name in zip(array_axes, axis_names):
                U = _fix_face(U, axis=axis, axis_name=name, n_proc=n_proc[name],
                              comp=axis, op="reflect")
        elif boundary == 4:
            for axis, name in zip(array_axes, axis_names):
                U = _fix_face(U, axis=axis, axis_name=name, n_proc=n_proc[name],
                              comp=axis, op="diode")
        return U

    return halo


def make_halo_3d_shardmap(mesh):
    """Backwards-compatible 3D factory (thin wrapper over make_halo_shardmap)."""
    return make_halo_shardmap(mesh, 3)


def _slab(U, axis, lo):
    """Boundary slab (index 0 if lo else -1) along `axis`, keepdims, any rank."""
    sl = [slice(None)] * U.ndim
    sl[axis] = slice(0, 1) if lo else slice(-1, None)
    return U[tuple(sl)]


def _set_slab(U, axis, lo, val):
    sl = [slice(None)] * U.ndim
    sl[axis] = slice(0, 1) if lo else slice(-1, None)
    return U.at[tuple(sl)].set(val)


def _pad_axis(U, *, axis, axis_name, n_proc, periodic, width=1):
    """Add a `width`-cell halo along `axis` via ppermute then concat.

    `axis` is the array axis (1, 2 or 3 for x, y, z given the leading Nvar
    dim). `axis_name` is the mesh-axis name ('x','y','z').
    """
    sl_lo = [slice(None)] * U.ndim
    sl_hi = [slice(None)] * U.ndim
    sl_lo[axis] = slice(0, width)
    sl_hi[axis] = slice(-width, None)
    left_slab = U[tuple(sl_lo)]
    right_slab = U[tuple(sl_hi)]

    # Convention: perm = [(src, dst), ...] — device `src` sends its slab to
    # device `dst`. To receive my left ghost from neighbour (i-1)%N, device
    # (i-1)%N sends its right slab to me, i.e. perm sends from each i to (i+1).
    perm_to_right = [(i, (i + 1) % n_proc) for i in range(n_proc)]
    perm_to_left = [(i, (i - 1) % n_proc) for i in range(n_proc)]
    left_ghost_periodic = jax.lax.ppermute(right_slab, axis_name, perm_to_right)
    right_ghost_periodic = jax.lax.ppermute(left_slab, axis_name, perm_to_left)

    if periodic:
        left_ghost = left_ghost_periodic
        right_ghost = right_ghost_periodic
    else:  # zero-gradient: boundary shards replicate their own EDGE CELL
        idx = jax.lax.axis_index(axis_name)
        if width == 1:
            left_edge, right_edge = left_slab, right_slab
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

    return jnp.concatenate([left_ghost, U, right_ghost], axis=axis)


def _fix_face(U, *, axis, axis_name, n_proc, comp, op):
    """Correct the outer ghost layer along `axis` on the boundary shards.

    `op='reflect'` negates the normal momentum component `comp`; `op='diode'`
    clamps the normal velocity against inflow. Interior shards are untouched.
    """
    idx = jax.lax.axis_index(axis_name)
    lo = _slab(U, axis, lo=True)
    hi = _slab(U, axis, lo=False)
    if op == "reflect":
        lo_new = lo.at[comp].set(-lo[comp])
        hi_new = hi.at[comp].set(-hi[comp])
    else:  # diode: low face -> min(v,0), high face -> max(v,0)
        lo_new = lo.at[comp].set(jnp.minimum(lo[comp], 0.0))
        hi_new = hi.at[comp].set(jnp.maximum(hi[comp], 0.0))
    lo_final = jnp.where(idx == 0, lo_new, lo)
    hi_final = jnp.where(idx == n_proc - 1, hi_new, hi)
    U = _set_slab(U, axis, lo=True, val=lo_final)
    U = _set_slab(U, axis, lo=False, val=hi_final)
    return U
