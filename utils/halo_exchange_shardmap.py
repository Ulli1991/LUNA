"""Shard_map halo exchange — explicit per-shard halo via jax.lax.ppermute.

For the auto-sharded `pad_with_halo_auto` path, XLA's lowering of
`jnp.pad(mode='wrap')` over a NamedSharding mesh materialises ~4 extra
full-shard scratch arrays per halo call and refuses to drop below
~71 GB working set at 1024^3 fp32 / 8 GPUs (rematerialization warnings
confirm). The shard_map path exchanges only thin slabs (~5 MB each at
2x2x2 mesh) and gives us explicit per-shard footprint control.

For each spatial axis we ppermute the +/-1 boundary slab to/from the
neighbour and concatenate into the local body. Axis-by-axis chaining
populates corner ghost cells correctly via the standard pad order.

Boundary types supported in this module:
  0 = periodic    — ppermute wrap on all neighbours
  1 = zero-grad   — interior shards ppermute; boundary shards reuse own
                     edge as ghost (via `jax.lax.axis_index` + `jnp.where`)
  2 = reflective  — TODO (phase 2): like zero-grad but negate the normal
                     momentum component on boundary shards.
"""

import jax
import jax.numpy as jnp


def make_halo_3d_shardmap(mesh):
    """Factory: closure-capture the 3D mesh axis sizes, return a halo
    function suitable for use *inside* a shard_map body.

    Returned function has signature `halo(U_local, boundary, spatial_dim)`
    matching the auto-sharded `pad_with_halo_auto` so the integrator
    factory can swap one for the other transparently.
    """
    nx = int(mesh.shape["x"])
    ny = int(mesh.shape["y"])
    nz = int(mesh.shape["z"])

    def halo(U_local, boundary, spatial_dim):
        if spatial_dim != 3:
            raise NotImplementedError(
                "shard_map halo only implemented for 3D in phase 1; "
                "1D/2D paths still use the auto-sharded jnp.pad."
            )
        if boundary not in (0, 1):
            raise NotImplementedError(
                f"shard_map halo only supports periodic (0) and "
                f"zero-gradient (1) in phase 1; got boundary={boundary}"
            )

        # x → y → z order; later axes use the already-padded array so
        # corner cells fill in correctly.
        U_padded = _pad_axis(U_local, axis=1, axis_name="x", n_proc=nx, boundary=boundary)
        U_padded = _pad_axis(U_padded, axis=2, axis_name="y", n_proc=ny, boundary=boundary)
        U_padded = _pad_axis(U_padded, axis=3, axis_name="z", n_proc=nz, boundary=boundary)
        return U_padded

    return halo


def _pad_axis(U, *, axis, axis_name, n_proc, boundary):
    """Add a 1-cell halo along `axis` via ppermute then concat.

    `axis` is the array axis (1, 2 or 3 for x, y, z given the leading
    Nvar dim). `axis_name` is the mesh-axis name ('x','y','z').
    """
    if axis == 1:
        left_slab = U[:, 0:1, :, :]
        right_slab = U[:, -1:, :, :]
    elif axis == 2:
        left_slab = U[:, :, 0:1, :]
        right_slab = U[:, :, -1:, :]
    elif axis == 3:
        left_slab = U[:, :, :, 0:1]
        right_slab = U[:, :, :, -1:]
    else:
        raise ValueError(f"axis must be 1, 2 or 3; got {axis}")

    # Periodic transports for both ghosts. For boundary == 1 (zero-grad)
    # we overwrite the result on boundary shards with their own edge slab.
    # Convention: perm = [(src, dst), ...] — device `src` sends its slab
    # to device `dst`. To receive my left ghost from neighbour (i-1) %N, I
    # have device (i-1)%N send its right slab to me, i.e. perm sends from
    # each device i to (i+1) % n_proc.
    perm_to_right = [(i, (i + 1) % n_proc) for i in range(n_proc)]
    perm_to_left = [(i, (i - 1) % n_proc) for i in range(n_proc)]
    left_ghost_periodic = jax.lax.ppermute(right_slab, axis_name, perm_to_right)
    right_ghost_periodic = jax.lax.ppermute(left_slab, axis_name, perm_to_left)

    if boundary == 0:
        left_ghost = left_ghost_periodic
        right_ghost = right_ghost_periodic
    else:  # boundary == 1, zero-gradient
        idx = jax.lax.axis_index(axis_name)
        # On the lower-edge shard (idx == 0) the left ghost should mirror
        # the local left edge (zero gradient). Same for the upper edge.
        left_ghost = jnp.where(idx == 0, left_slab, left_ghost_periodic)
        right_ghost = jnp.where(idx == n_proc - 1, right_slab, right_ghost_periodic)

    return jnp.concatenate([left_ghost, U, right_ghost], axis=axis)
