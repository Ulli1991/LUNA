"""Poisson boundary halos: each `make_*_halo_3d` returns `halo(phi)`, (Nx, Ny, Nz) -> (Nx+2, Ny+2, Nz+2).

Boundary codes as in hydro: 0 periodic, 1 Neumann, 2 reflective (= Neumann for Φ), 3 isolated.
"""

import jax.numpy as jnp


def _slab(a, axis, sl):
    idx = [slice(None)] * a.ndim
    idx[axis] = sl
    return a[tuple(idx)]


def pad_inplace(phi, width, ghosts):
    """`phi` with `width` ghosts written x -> y -> z into one padded buffer; `ghosts(core, axis)` gives (lo, hi)."""
    w = width
    U = jnp.pad(phi, w)
    for ax in range(phi.ndim):
        view = [slice(None)] * phi.ndim
        for later in range(ax, phi.ndim):
            view[later] = slice(w, -w)
        lo, hi = ghosts(U[tuple(view)], ax)
        view[ax] = slice(0, w)
        U = U.at[tuple(view)].set(lo)
        view[ax] = slice(-w, None)
        U = U.at[tuple(view)].set(hi)
    return U


def wrap_ghosts(core, axis, width=1):
    """Periodic ghosts of a whole axis."""
    return _slab(core, axis, slice(-width, None)), _slab(core, axis, slice(0, width))


def edge_ghosts(core, axis, width=1):
    """Ghosts repeating the edge cell (zero gradient)."""
    lo, hi = _slab(core, axis, slice(0, 1)), _slab(core, axis, slice(-1, None))
    if width == 1:
        return lo, hi
    reps = [1] * core.ndim
    reps[axis] = width
    return jnp.tile(lo, reps), jnp.tile(hi, reps)


def odd_ghosts(core, axis):
    """Ghost = -edge: Φ = 0 on the face."""
    return -_slab(core, axis, slice(0, 1)), -_slab(core, axis, slice(-1, None))


def make_periodic_halo_3d(inplace=True):
    """Periodic wrap; `whole_periodic` marks the whole grid; inplace False: one `jnp.pad`, the same values."""
    def halo(phi):
        if not inplace:
            return jnp.pad(phi, 1, mode="wrap")
        return pad_inplace(phi, 1, wrap_ghosts)
    halo.whole_periodic = True
    return halo


def make_neumann_halo_3d(inplace=True):
    """Zero-gradient (Neumann) BC: ghost cells equal the adjacent edge cell; `inplace` as for periodic."""
    def halo(phi):
        if not inplace:
            return jnp.pad(phi, 1, mode="edge")
        return pad_inplace(phi, 1, edge_ghosts)
    return halo


def make_zero_dirichlet_halo_3d():
    """Ghost Φ = 0: the BC of the V-cycle's coarse error-correction problems."""
    def halo(phi):
        return jnp.pad(phi, ((1, 1), (1, 1), (1, 1)), mode="constant",
                       constant_values=0.0)
    return halo


def make_odd_dirichlet_halo_3d(inplace=True):
    """Φ = 0 on the faces (ghost = -edge): the homogeneous BC of the isolated V-cycle's corrections."""
    def halo(phi):
        if not inplace:
            for ax in range(3):
                lo, hi = odd_ghosts(phi, ax)
                phi = jnp.concatenate([lo, phi, hi], axis=ax)
            return phi
        return pad_inplace(phi, 1, odd_ghosts)
    return halo


def multipole_dirichlet_face_values(rho, dx, fourpiG, l_max=2):
    """Multipole Φ at the box faces (Guillet & Teyssier 2011 §3); l_max=0 monopole, >=2 adds the quadrupole.

    Returns (Nx+2, Ny+2, Nz+2); only the six face slices are meaningful.
    """
    Nx, Ny, Nz = rho.shape
    G = fourpiG / (4.0 * jnp.pi)

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

    # Traceless quadrupole about the COM, Q_ij = Σ m (3 r_i r_j − r²δ_ij).
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

    Yf, Zf = jnp.meshgrid(y, z, indexing="ij")
    out = out.at[0,      1:Ny + 1, 1:Nz + 1].set(_phi(0.0 - cx, Yf - cy, Zf - cz))
    out = out.at[Nx + 1, 1:Ny + 1, 1:Nz + 1].set(_phi(Lx  - cx, Yf - cy, Zf - cz))

    Xf, Zf2 = jnp.meshgrid(x, z, indexing="ij")
    out = out.at[1:Nx + 1, 0,      1:Nz + 1].set(_phi(Xf - cx, 0.0 - cy, Zf2 - cz))
    out = out.at[1:Nx + 1, Ny + 1, 1:Nz + 1].set(_phi(Xf - cx, Ly  - cy, Zf2 - cz))

    Xf2, Yf2 = jnp.meshgrid(x, y, indexing="ij")
    out = out.at[1:Nx + 1, 1:Ny + 1, 0     ].set(_phi(Xf2 - cx, Yf2 - cy, 0.0 - cz))
    out = out.at[1:Nx + 1, 1:Ny + 1, Nz + 1].set(_phi(Xf2 - cx, Yf2 - cy, Lz  - cz))

    return out


def make_linear_extrap_dirichlet_halo_3d(phi_face_values):
    """Face-centred Dirichlet halo: ghost = 2·Φ_face − Φ_first_interior (Guillet & Teyssier 2011)."""
    def halo(phi):
        Nx, Ny, Nz = phi.shape
        padded = jnp.pad(phi, ((1, 1), (1, 1), (1, 1)),
                          mode="constant", constant_values=0.0)
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


def make_halo_3d(boundary, inplace=True):
    """Halo for a boundary code; isolated (3) returns the zero-Dirichlet halo for the coarse levels."""
    if boundary == 0:
        return make_periodic_halo_3d(inplace)
    if boundary in (1, 2):
        return make_neumann_halo_3d(inplace)
    if boundary == 3:
        # The solver builds the multipole halo itself once ρ is known.
        return make_zero_dirichlet_halo_3d()
    raise ValueError(f"Unknown Poisson boundary code: {boundary}")
