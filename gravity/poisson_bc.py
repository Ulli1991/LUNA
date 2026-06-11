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


def make_zero_dirichlet_halo_3d():
    """Φ = 0 at all faces. Correct BC for the error-correction sub-problems
    of the multigrid V-cycle (the top-level solution already carries the
    Dirichlet BC; coarse levels solve ∇²e = r with e=0 at faces)."""
    def halo(phi):
        return jnp.pad(phi, ((1, 1), (1, 1), (1, 1)), mode="constant",
                       constant_values=0.0)
    return halo


def make_fixed_dirichlet_halo_3d(halo_values):
    """Halo with PRE-COMPUTED Dirichlet values (e.g. from a multipole
    expansion of the interior mass). `halo_values` has shape (Nx+2, Ny+2,
    Nz+2); only the face cells matter — the interior is overwritten by
    the input phi inside the halo function.
    """
    def halo(phi):
        out = halo_values
        Nx, Ny, Nz = phi.shape
        return out.at[1:Nx+1, 1:Ny+1, 1:Nz+1].set(phi)
    return halo


def multipole_dirichlet_halo_values(rho, dx, fourpiG, l_max=0):
    """Compute Dirichlet halo values from a multipole expansion of the
    interior mass distribution. Returns a (Nx+2, Ny+2, Nz+2) array; only
    the face cells carry meaningful values — interior cells are zeros.

    l_max = 0: monopole only,  Φ(r) = −G·M_tot / |r − r_com|.
    Quadrupole-and-higher could be added but for compact mass distributions
    in boxes ≳ 4·R the monopole is already accurate at the box face.
    """
    Nx, Ny, Nz = rho.shape
    G = fourpiG / (4.0 * jnp.pi)

    # Total mass and centre of mass.
    M_total = jnp.sum(rho) * (dx ** 3)
    x = (jnp.arange(Nx, dtype=jnp.float64) + 0.5) * dx
    y = (jnp.arange(Ny, dtype=jnp.float64) + 0.5) * dx
    z = (jnp.arange(Nz, dtype=jnp.float64) + 0.5) * dx
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    inv_mass = 1.0 / (jnp.sum(rho) + 1e-30)
    cx = jnp.sum(rho * X) * inv_mass
    cy = jnp.sum(rho * Y) * inv_mass
    cz = jnp.sum(rho * Z) * inv_mass

    # Halo cell coordinates: -0.5*dx, 0.5*dx, ..., (Nx-0.5)*dx, (Nx+0.5)*dx.
    xh = (jnp.arange(-1, Nx + 1, dtype=jnp.float64) + 0.5) * dx
    yh = (jnp.arange(-1, Ny + 1, dtype=jnp.float64) + 0.5) * dx
    zh = (jnp.arange(-1, Nz + 1, dtype=jnp.float64) + 0.5) * dx
    Xh, Yh, Zh = jnp.meshgrid(xh, yh, zh, indexing="ij")
    r = jnp.sqrt((Xh - cx) ** 2 + (Yh - cy) ** 2 + (Zh - cz) ** 2)
    phi_far = -G * M_total / jnp.maximum(r, 0.5 * dx)
    return phi_far


def make_halo_3d(boundary):
    """Dispatch on the boundary code used by LUNA's hydro path.

    Boundary codes:
      0 = periodic        (Jeans-swindle multigrid)
      1 = Neumann         (∂Φ/∂n = 0)
      2 = reflective      (same as Neumann for Φ)
      3 = isolated        (Dirichlet from multipole expansion — see
                           multipole_dirichlet_halo_values + the
                           make_fixed_dirichlet_halo_3d factory; the
                           solver constructs the halo factory itself
                           because it needs ρ)
    """
    if boundary == 0:
        return make_periodic_halo_3d()
    if boundary in (1, 2):
        return make_neumann_halo_3d()
    if boundary == 3:
        # The solver builds the actual halo factory itself once ρ is known;
        # return a sentinel so the V-cycle knows to route via the isolated
        # code path. Same value as zero-dirichlet for the coarse subproblem.
        return make_zero_dirichlet_halo_3d()
    raise ValueError(f"Unknown Poisson boundary code: {boundary}")
