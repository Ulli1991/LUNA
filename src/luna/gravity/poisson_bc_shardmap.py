"""shard_map halo exchange for scalar Poisson fields (Φ, RHS, residual): ghosts written in place, one padded buffer."""

from luna.gravity.poisson_bc import edge_ghosts, pad_inplace, wrap_ghosts
from luna.utils.halo_exchange_shardmap import _axis_ghosts

AXES = ("x", "y", "z")


def sharded_ghosts(mesh, periodic, width):
    """`ghosts(core, axis)` for `pad_inplace`: ppermuted slabs, or a local wrap/edge on a mesh axis of size 1."""
    n = [int(mesh.shape[a]) for a in AXES]

    def ghosts(core, axis):
        if n[axis] == 1:
            return (wrap_ghosts if periodic else edge_ghosts)(core, axis, width)
        return _axis_ghosts(core, axis=axis, axis_name=AXES[axis], n_proc=n[axis],
                            periodic=periodic, width=width)
    return ghosts


def make_scalar_halo_3d_shardmap(mesh):
    """Return `make_halo_for_boundary(boundary)`, which builds `halo(phi, width=1)` for a shard_map body."""

    def make_halo_for_boundary(boundary):
        if boundary not in (0, 1, 2):
            raise NotImplementedError(
                f"Phase 2 shard_map halo only supports periodic (0) and "
                f"Neumann (1/2); got boundary={boundary}"
            )
        periodic = boundary == 0

        def halo(phi, width=1):
            return pad_inplace(phi, width, sharded_ghosts(mesh, periodic, width))

        return halo

    return make_halo_for_boundary
