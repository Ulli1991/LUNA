"""Boundary helpers for the Poisson solver.

Each `make_*_halo_3d` factory returns a function `halo(phi)` that takes
a 3D scalar field of shape (Nx, Ny, Nz) and returns a padded array of
shape (Nx+2, Ny+2, Nz+2) with the requested 1-cell ghost layer.

Phase 1 ships single-process versions (just jnp.pad). The shard_map
variants (using jax.lax.ppermute) come in Phase 2 and will live in
gravity/poisson_bc_shardmap.py.

Boundary types follow the hydro convention:
  0 = periodic
  1 = Neumann (zero-gradient, ∂Φ/∂n = 0 → ghost = nearest boundary cell)
  2 = reflective (Φ does not change sign at reflective walls → same as Neumann)
"""

import jax.numpy as jnp


def make_periodic_halo_3d():
    """Periodic wrap on all three axes."""
    def halo(phi):
        return jnp.pad(phi, ((1, 1), (1, 1), (1, 1)), mode="wrap")
    return halo


def make_neumann_halo_3d():
    """Zero-gradient (Neumann) BC: ghost cells equal the adjacent edge cell."""
    def halo(phi):
        return jnp.pad(phi, ((1, 1), (1, 1), (1, 1)), mode="edge")
    return halo


def make_halo_3d(boundary):
    """Dispatch on the boundary code used by LUNA's hydro path."""
    if boundary == 0:
        return make_periodic_halo_3d()
    if boundary in (1, 2):
        return make_neumann_halo_3d()
    raise ValueError(f"Unknown Poisson boundary code: {boundary}")
