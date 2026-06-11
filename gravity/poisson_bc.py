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
    """LEGACY: samples Φ at ghost-cell centres (-dx/2 outside the box).

    Kept for backwards compatibility. The improved Guillet-Teyssier 2011
    recipe uses `multipole_dirichlet_face_values` + the linear-extrap halo
    factory instead — the boundary then sits at the cell *face* (x=0)
    rather than drifting toward the ghost (x=-dx/2). Use the *_face_*
    variant for production runs.
    """
    Nx, Ny, Nz = rho.shape
    G = fourpiG / (4.0 * jnp.pi)

    M_total = jnp.sum(rho) * (dx ** 3)
    x = (jnp.arange(Nx, dtype=jnp.float64) + 0.5) * dx
    y = (jnp.arange(Ny, dtype=jnp.float64) + 0.5) * dx
    z = (jnp.arange(Nz, dtype=jnp.float64) + 0.5) * dx
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    inv_mass = 1.0 / (jnp.sum(rho) + 1e-30)
    cx = jnp.sum(rho * X) * inv_mass
    cy = jnp.sum(rho * Y) * inv_mass
    cz = jnp.sum(rho * Z) * inv_mass

    xh = (jnp.arange(-1, Nx + 1, dtype=jnp.float64) + 0.5) * dx
    yh = (jnp.arange(-1, Ny + 1, dtype=jnp.float64) + 0.5) * dx
    zh = (jnp.arange(-1, Nz + 1, dtype=jnp.float64) + 0.5) * dx
    Xh, Yh, Zh = jnp.meshgrid(xh, yh, zh, indexing="ij")
    r = jnp.sqrt((Xh - cx) ** 2 + (Yh - cy) ** 2 + (Zh - cz) ** 2)
    phi_far = -G * M_total / jnp.maximum(r, 0.5 * dx)
    return phi_far


def multipole_dirichlet_face_values(rho, dx, fourpiG, l_max=0):
    """Guillet-Teyssier 2011 §3 face-centred BC: Φ evaluated at the box
    face positions (not at ghost cell centres). Pairs with
    `make_linear_extrap_dirichlet_halo_3d` so the discrete BC lands
    exactly on the face — which is what gives 2nd-order accuracy.

    Returned array is shape (Nx+2, Ny+2, Nz+2); only the six face
    slices carry meaningful values (interior/edges/corners are zero
    and never read by the 7-point stencil).
    """
    Nx, Ny, Nz = rho.shape
    G = fourpiG / (4.0 * jnp.pi)

    # Total mass + COM (positive box coordinates).
    M_total = jnp.sum(rho) * (dx ** 3)
    inv_mass = 1.0 / (jnp.sum(rho) + 1e-30)
    x = (jnp.arange(Nx, dtype=jnp.float64) + 0.5) * dx
    y = (jnp.arange(Ny, dtype=jnp.float64) + 0.5) * dx
    z = (jnp.arange(Nz, dtype=jnp.float64) + 0.5) * dx
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    cx = jnp.sum(rho * X) * inv_mass
    cy = jnp.sum(rho * Y) * inv_mass
    cz = jnp.sum(rho * Z) * inv_mass

    Lx, Ly, Lz = Nx * dx, Ny * dx, Nz * dx

    def _phi(rx, ry, rz):
        r = jnp.sqrt(rx * rx + ry * ry + rz * rz)
        return -G * M_total / jnp.maximum(r, 0.5 * dx)

    out = jnp.zeros((Nx + 2, Ny + 2, Nz + 2), dtype=jnp.float64)

    # −x and +x faces: Φ at (0, y_j, z_k) and (Lx, y_j, z_k).
    Yf, Zf = jnp.meshgrid(y, z, indexing="ij")
    out = out.at[0,      1:Ny + 1, 1:Nz + 1].set(_phi(0.0 - cx, Yf - cy, Zf - cz))
    out = out.at[Nx + 1, 1:Ny + 1, 1:Nz + 1].set(_phi(Lx  - cx, Yf - cy, Zf - cz))

    # −y and +y faces.
    Xf, Zf2 = jnp.meshgrid(x, z, indexing="ij")
    out = out.at[1:Nx + 1, 0,      1:Nz + 1].set(_phi(Xf - cx, 0.0 - cy, Zf2 - cz))
    out = out.at[1:Nx + 1, Ny + 1, 1:Nz + 1].set(_phi(Xf - cx, Ly  - cy, Zf2 - cz))

    # −z and +z faces.
    Xf2, Yf2 = jnp.meshgrid(x, y, indexing="ij")
    out = out.at[1:Nx + 1, 1:Ny + 1, 0     ].set(_phi(Xf2 - cx, Yf2 - cy, 0.0 - cz))
    out = out.at[1:Nx + 1, 1:Ny + 1, Nz + 1].set(_phi(Xf2 - cx, Yf2 - cy, Lz  - cz))

    return out


def make_linear_extrap_dirichlet_halo_3d(phi_face_values):
    """Halo factory implementing the Guillet-Teyssier 2011 §3 linear-
    extrapolation Dirichlet BC:

        GHOST = 2 · Φ_face − Φ_first_interior

    `phi_face_values` is the (Nx+2, Ny+2, Nz+2) array produced by
    `multipole_dirichlet_face_values`. The corners and edges of the
    halo array are never read by the 7-point stencil so they remain
    zero. The ghost depends on the current `phi` (the first interior
    cell along the normal direction), so the halo is *not* fixed —
    it's recomputed each smoother sweep.
    """
    def halo(phi):
        Nx, Ny, Nz = phi.shape
        padded = jnp.pad(phi, ((1, 1), (1, 1), (1, 1)),
                          mode="constant", constant_values=0.0)
        # 2 · Φ_face − Φ_first_interior, per face.
        padded = padded.at[0,      1:Ny + 1, 1:Nz + 1].set(
            2.0 * phi_face_values[0,      1:Ny + 1, 1:Nz + 1] - phi[0,    :, :])
        padded = padded.at[Nx + 1, 1:Ny + 1, 1:Nz + 1].set(
            2.0 * phi_face_values[Nx + 1, 1:Ny + 1, 1:Nz + 1] - phi[Nx-1, :, :])
        padded = padded.at[1:Nx + 1, 0,      1:Nz + 1].set(
            2.0 * phi_face_values[1:Nx + 1, 0,      1:Nz + 1] - phi[:, 0,    :])
        padded = padded.at[1:Nx + 1, Ny + 1, 1:Nz + 1].set(
            2.0 * phi_face_values[1:Nx + 1, Ny + 1, 1:Nz + 1] - phi[:, Ny-1, :])
        padded = padded.at[1:Nx + 1, 1:Ny + 1, 0     ].set(
            2.0 * phi_face_values[1:Nx + 1, 1:Ny + 1, 0     ] - phi[:, :, 0   ])
        padded = padded.at[1:Nx + 1, 1:Ny + 1, Nz + 1].set(
            2.0 * phi_face_values[1:Nx + 1, 1:Ny + 1, Nz + 1] - phi[:, :, Nz-1])
        return padded
    return halo


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
