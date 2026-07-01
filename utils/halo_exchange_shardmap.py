"""Shard_map halo exchange — explicit per-shard halo via jax.lax.ppermute.

Why explicit halos: letting the compiler infer the communication from a
NamedSharding (via `jnp.pad(mode='wrap')`) materialises ~4 extra full-shard
scratch arrays per halo call and refuses to drop below ~71 GB working set at
1024^3 fp32 / 8 GPUs (rematerialization warnings confirm). The shard_map path
exchanges only thin slabs (~5 MB each at 2x2x2 mesh) and gives us explicit
per-shard footprint control.

For each spatial axis we ppermute the +/-1 boundary slab to/from the
neighbour and concatenate into the local body. Axis-by-axis chaining
populates corner ghost cells correctly via the standard pad order. This
is the single multi-GPU halo path for LUNA (1D/2D/3D, every boundary).

Boundary types (variable layout being
[density, mom/vel_x, mom/vel_y, mom/vel_z, energy] so component `axis` is
the normal momentum/velocity along array axis `axis`):
  0 = periodic    — ppermute wrap on all neighbours.
  1 = zero-grad   — interior shards ppermute; boundary shards reuse own
                    edge as ghost (via `jax.lax.axis_index` + `jnp.where`).
  2 = reflective  — zero-grad ghost, then negate the normal momentum
                    component in the outer ghost on the boundary shards.
  4 = diode       — zero-grad ghost, then clamp the outer-ghost normal
                    velocity against inflow (gas may leave, none enters):
                    low face -> min(v, 0), high face -> max(v, 0).
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

    def halo(U_local, boundary, sdim):
        if sdim != spatial_dim:
            raise ValueError(
                f"halo built for spatial_dim={spatial_dim}, called with {sdim}")
        if boundary not in (0, 1, 2, 4):
            raise NotImplementedError(
                f"shard_map halo supports boundary 0/1/2/4; got {boundary}")
        periodic = boundary == 0
        # x -> y -> z order; later axes use the already-padded array so corner
        # cells fill in correctly.
        U = U_local
        for axis, name in zip(array_axes, axis_names):
            U = _pad_axis(U, axis=axis, axis_name=name, n_proc=n_proc[name],
                          periodic=periodic)
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


def _pad_axis(U, *, axis, axis_name, n_proc, periodic):
    """Add a 1-cell halo along `axis` via ppermute then concat.

    `axis` is the array axis (1, 2 or 3 for x, y, z given the leading Nvar
    dim). `axis_name` is the mesh-axis name ('x','y','z').
    """
    left_slab = _slab(U, axis, lo=True)
    right_slab = _slab(U, axis, lo=False)

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
    else:  # zero-gradient: boundary shards reuse their own edge as ghost
        idx = jax.lax.axis_index(axis_name)
        left_ghost = jnp.where(idx == 0, left_slab, left_ghost_periodic)
        right_ghost = jnp.where(idx == n_proc - 1, right_slab, right_ghost_periodic)

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
