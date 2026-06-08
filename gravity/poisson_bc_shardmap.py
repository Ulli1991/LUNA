"""shard_map halo for the scalar Poisson field (Φ, RHS, residual, etc.).

Mirrors `utils/halo_exchange_shardmap.py` (the hydro 3D halo) but on a
scalar field of shape (Nx_local, Ny_local, Nz_local) instead of the
hydro (Nvar, Nx, Ny, Nz). Same ppermute-then-concat pattern,
axis-by-axis, with `jax.lax.axis_index` + `jnp.where` for boundary
shards under non-periodic BCs.
"""

import jax
import jax.numpy as jnp


def make_scalar_halo_3d_shardmap(mesh):
    """Factory: closure-capture the 3D mesh axis sizes, return a halo
    function suitable for use inside a shard_map body.

    Returned function has signature `halo(phi)` where phi is a 3D scalar
    shard. Boundary type is captured at factory time.
    """
    nx = int(mesh.shape["x"])
    ny = int(mesh.shape["y"])
    nz = int(mesh.shape["z"])

    def make_halo_for_boundary(boundary):
        if boundary not in (0, 1, 2):
            raise NotImplementedError(
                f"Phase 2 shard_map halo only supports periodic (0) and "
                f"Neumann (1/2); got boundary={boundary}"
            )

        def halo(phi):
            # x → y → z axis-by-axis; chaining populates corners.
            phi_padded = _pad_axis(phi, axis=0, axis_name="x", n_proc=nx, boundary=boundary)
            phi_padded = _pad_axis(phi_padded, axis=1, axis_name="y", n_proc=ny, boundary=boundary)
            phi_padded = _pad_axis(phi_padded, axis=2, axis_name="z", n_proc=nz, boundary=boundary)
            return phi_padded

        return halo

    return make_halo_for_boundary


def _pad_axis(U, *, axis, axis_name, n_proc, boundary):
    """1-cell halo along `axis` via ppermute then concat."""
    if axis == 0:
        left_slab = U[0:1, :, :]
        right_slab = U[-1:, :, :]
    elif axis == 1:
        left_slab = U[:, 0:1, :]
        right_slab = U[:, -1:, :]
    elif axis == 2:
        left_slab = U[:, :, 0:1]
        right_slab = U[:, :, -1:]
    else:
        raise ValueError(f"axis must be 0, 1 or 2; got {axis}")

    perm_to_right = [(i, (i + 1) % n_proc) for i in range(n_proc)]
    perm_to_left = [(i, (i - 1) % n_proc) for i in range(n_proc)]
    left_ghost_periodic = jax.lax.ppermute(right_slab, axis_name, perm_to_right)
    right_ghost_periodic = jax.lax.ppermute(left_slab, axis_name, perm_to_left)

    if boundary == 0:
        left_ghost = left_ghost_periodic
        right_ghost = right_ghost_periodic
    else:  # boundary in (1, 2): Neumann / edge mirror on the boundary shards
        idx = jax.lax.axis_index(axis_name)
        left_ghost = jnp.where(idx == 0, left_slab, left_ghost_periodic)
        right_ghost = jnp.where(idx == n_proc - 1, right_slab, right_ghost_periodic)

    return jnp.concatenate([left_ghost, U, right_ghost], axis=axis)
