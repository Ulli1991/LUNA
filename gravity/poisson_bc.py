"""Boundary helpers for the Poisson solver: each `make_*_halo_3d` factory
returns `halo(phi)` mapping (Nx, Ny, Nz) -> (Nx+2, Ny+2, Nz+2) with a 1-cell
ghost layer. Boundary codes follow the hydro convention: 0 = periodic,
1 = Neumann (∂Φ/∂n = 0), 2 = reflective (same as Neumann for Φ).
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


def multipole_dirichlet_face_values(rho, dx, fourpiG, l_max=2):
    """Guillet-Teyssier 2011 §3 face-centred Dirichlet BC: Φ evaluated at the
    box FACE positions (pairs with `make_linear_extrap_dirichlet_halo_3d` for
    2nd-order accuracy). l_max=0 -> monopole Φ = -G M/r; l_max>=2 adds the
    traceless quadrupole ½ Q_ij r_i r_j / r^5. Returns (Nx+2, Ny+2, Nz+2);
    only the six face slices carry meaningful values."""
    Nx, Ny, Nz = rho.shape
    G = fourpiG / (4.0 * jnp.pi)

    # Total mass + COM (positive box coordinates).
    w = rho * (dx ** 3)
    M_total = jnp.sum(w)
    inv_mass = 1.0 / (jnp.sum(rho) + 1e-30)
    x = (jnp.arange(Nx, dtype=jnp.float64) + 0.5) * dx
    y = (jnp.arange(Ny, dtype=jnp.float64) + 0.5) * dx
    z = (jnp.arange(Nz, dtype=jnp.float64) + 0.5) * dx
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    cx = jnp.sum(rho * X) * inv_mass
    cy = jnp.sum(rho * Y) * inv_mass
    cz = jnp.sum(rho * Z) * inv_mass

    # Traceless quadrupole moment tensor about the COM, Q_ij = Σ ρ(3 r_i r_j − r²δ).
    use_quad = l_max >= 2
    if use_quad:
        dXc, dYc, dZc = X - cx, Y - cy, Z - cz
        r2c = dXc * dXc + dYc * dYc + dZc * dZc
        Qxx = jnp.sum(w * (3.0 * dXc * dXc - r2c))
        Qyy = jnp.sum(w * (3.0 * dYc * dYc - r2c))
        Qzz = jnp.sum(w * (3.0 * dZc * dZc - r2c))
        Qxy = jnp.sum(w * (3.0 * dXc * dYc))
        Qxz = jnp.sum(w * (3.0 * dXc * dZc))
        Qyz = jnp.sum(w * (3.0 * dYc * dZc))

    Lx, Ly, Lz = Nx * dx, Ny * dx, Nz * dx

    def _phi(rx, ry, rz):
        r = jnp.maximum(jnp.sqrt(rx * rx + ry * ry + rz * rz), 0.5 * dx)
        phi = -G * M_total / r
        if use_quad:
            quad = (Qxx * rx * rx + Qyy * ry * ry + Qzz * rz * rz
                    + 2.0 * Qxy * rx * ry + 2.0 * Qxz * rx * rz + 2.0 * Qyz * ry * rz)
            phi = phi - G * 0.5 * quad / r ** 5
        return phi

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
    """Halo factory for the Guillet-Teyssier linear-extrapolation Dirichlet
    BC: GHOST = 2·Φ_face − Φ_first_interior. The ghost depends on the current
    phi, so the halo is recomputed each smoother sweep."""
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
    """Dispatch on the LUNA boundary code: 0 periodic, 1 Neumann,
    2 reflective (= Neumann for Φ), 3 isolated (returns the zero-Dirichlet
    halo as a sentinel; the solver builds the real multipole halo once ρ is
    known)."""
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
