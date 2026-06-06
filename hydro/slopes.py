import jax.numpy as jnp


# ------------------- Slope limiters -------------------
def minmod(a, b):
    return jnp.where(a * b > 0, jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)


def minmod3(a, b, c):
    s = jnp.sign(a)
    return jnp.where(
        (a * b > 0) & (a * c > 0),
        s * jnp.minimum(jnp.abs(a), jnp.minimum(jnp.abs(b), jnp.abs(c))),
        0.0,
    )


def moncen(dl, dr):
    return minmod3(2 * dl, 2 * dr, 0.5 * (dl + dr))


def vanleer(dl, dr):
    return jnp.where(dl * dr > 0, 2 * dl * dr / (dl + dr), 0.0)


# ------------------- Slope selection -------------------
def slope_limiter(dl, dr, limiter):
    """Apply slope limiter based on (static) limiter type.

    `limiter` is a Python int closure-captured by the JIT factory, so a
    plain `if`-chain compiles only the selected branch. The previous
    lax.switch version forced XLA to trace all 3 limiter branches each
    time the slope was computed, materialising redundant intermediates.
    """
    if limiter == 0:
        return minmod(dl, dr)
    if limiter == 1:
        return moncen(dl, dr)
    if limiter == 2:
        return vanleer(dl, dr)
    raise ValueError(f"Unknown limiter: {limiter}")


# ------------------- Compute slopes -------------------
from hydro.padding import pad


def compute_slopes_from_padded(U_pad, limiter):
    """
    Compute slopes from an already-padded array.
    More efficient when padding was already done externally.

    U_pad: (Nvar, Nx+2) for 1D, (Nvar, Nx+2, Ny+2) for 2D, (Nvar, Nx+2, Ny+2, Nz+2) for 3D
    Returns slopes for interior cells only.
    """
    ndim = U_pad.ndim - 1  # number of spatial dimensions

    # 1D case
    if ndim == 1:
        dx_L = U_pad[:, 1:-1] - U_pad[:, :-2]
        dx_R = U_pad[:, 2:] - U_pad[:, 1:-1]
        dUx = slope_limiter(dx_L, dx_R, limiter)
        dUy = None
        dUz = None
        return dUx, dUy, dUz

    # 2D case
    if ndim == 2:
        # Extract center cells once (was being extracted 4 times!)
        Qc = U_pad[:, 1:-1, 1:-1]

        dx_L = Qc - U_pad[:, :-2, 1:-1]
        dx_R = U_pad[:, 2:, 1:-1] - Qc
        dy_D = Qc - U_pad[:, 1:-1, :-2]
        dy_U = U_pad[:, 1:-1, 2:] - Qc

        dUx = slope_limiter(dx_L, dx_R, limiter)
        dUy = slope_limiter(dy_D, dy_U, limiter)
        dUz = None
        return dUx, dUy, dUz

    # 3D case
    # Extract center cells once (was being extracted 6 times!)
    Qc = U_pad[:, 1:-1, 1:-1, 1:-1]

    dx_L = Qc - U_pad[:, :-2, 1:-1, 1:-1]
    dx_R = U_pad[:, 2:, 1:-1, 1:-1] - Qc
    dy_D = Qc - U_pad[:, 1:-1, :-2, 1:-1]
    dy_U = U_pad[:, 1:-1, 2:, 1:-1] - Qc
    dz_B = Qc - U_pad[:, 1:-1, 1:-1, :-2]
    dz_F = U_pad[:, 1:-1, 1:-1, 2:] - Qc

    dUx = slope_limiter(dx_L, dx_R, limiter)
    dUy = slope_limiter(dy_D, dy_U, limiter)
    dUz = slope_limiter(dz_B, dz_F, limiter)

    return dUx, dUy, dUz


def compute_slope_x_3d_from_padded(U_pad, limiter):
    """Single-direction x-slope from a 3D padded array. Used by the
    memory-conscious MUSCL-Hancock 3D path that builds the predictor
    Jacobian product direction by direction so only one slope array
    is alive at a time (cuts ~40 GB of working set at 1024^3 fp32)."""
    Qc = U_pad[:, 1:-1, 1:-1, 1:-1]
    dx_L = Qc - U_pad[:, :-2, 1:-1, 1:-1]
    dx_R = U_pad[:, 2:, 1:-1, 1:-1] - Qc
    return slope_limiter(dx_L, dx_R, limiter)


def compute_slope_y_3d_from_padded(U_pad, limiter):
    Qc = U_pad[:, 1:-1, 1:-1, 1:-1]
    dy_D = Qc - U_pad[:, 1:-1, :-2, 1:-1]
    dy_U = U_pad[:, 1:-1, 2:, 1:-1] - Qc
    return slope_limiter(dy_D, dy_U, limiter)


def compute_slope_z_3d_from_padded(U_pad, limiter):
    Qc = U_pad[:, 1:-1, 1:-1, 1:-1]
    dz_B = Qc - U_pad[:, 1:-1, 1:-1, :-2]
    dz_F = U_pad[:, 1:-1, 1:-1, 2:] - Qc
    return slope_limiter(dz_B, dz_F, limiter)


def compute_slopes(U, limiter, boundary):
    """
    Compute slopes in 1D, 2D, or 3D depending on U shape.
    Pads U first using the selected boundary condition, then computes slopes.

    U: (Nvar, Nx) or (Nvar, Nx, Ny) or (Nvar, Nx, Ny, Nz)
    Returns slopes for interior cells only.
    """
    U_pad = pad(U, boundary)
    return compute_slopes_from_padded(U_pad, limiter)
