"""shard_map halo for the scalar Poisson field (Φ, RHS, residual, etc.).

Reuses `utils/halo_exchange_shardmap._pad_axis` (the hydro halo's
axis-pad kernel, rank-generic) at width=1 on a scalar field of shape
(Nx_local, Ny_local, Nz_local): ppermute-then-concat axis-by-axis, with
`jax.lax.axis_index` + `jnp.where` edge-replication on boundary shards
under non-periodic BCs.
"""

from utils.halo_exchange_shardmap import _pad_axis


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
        periodic = boundary == 0

        def halo(phi, width=1):
            # x → y → z axis-by-axis; chaining populates corners.
            phi_padded = _pad_axis(phi, axis=0, axis_name="x", n_proc=nx, periodic=periodic, width=width)
            phi_padded = _pad_axis(phi_padded, axis=1, axis_name="y", n_proc=ny, periodic=periodic, width=width)
            phi_padded = _pad_axis(phi_padded, axis=2, axis_name="z", n_proc=nz, periodic=periodic, width=width)
            return phi_padded

        return halo

    return make_halo_for_boundary
