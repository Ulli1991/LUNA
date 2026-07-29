"""Slope limiters and limited-slope computation from ghost-padded arrays
for the MUSCL-Hancock reconstruction."""

import jax.numpy as jnp


# ------------------- Slope limiters -------------------
def minmod(a, b):
    """Minmod limiter: smaller-magnitude of a, b when same sign, else 0."""
    return jnp.where(a * b > 0, jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)


def minmod3(a, b, c):
    """Three-argument minmod: smallest magnitude when all share a sign, else 0."""
    s = jnp.sign(a)
    return jnp.where(
        (a * b > 0) & (a * c > 0),
        s * jnp.minimum(jnp.abs(a), jnp.minimum(jnp.abs(b), jnp.abs(c))),
        0.0,
    )


def moncen(dl, dr):
    """Monotonized-central limiter: minmod3(2 dl, 2 dr, (dl+dr)/2)."""
    return minmod3(2 * dl, 2 * dr, 0.5 * (dl + dr))


def vanleer(dl, dr):
    """Van Leer harmonic limiter: 2 dl dr / (dl + dr) when dl*dr > 0, else 0."""
    # Double-where: the denominator must be made safe BEFORE the division,
    # otherwise 0/0 at dl+dr=0 (flat regions) is NaN in the unselected
    # branch and the VJP of jnp.where propagates it into the gradient
    # even though the forward pass is masked correctly.
    # The threshold must be the smallest NORMAL float, not 0: for subnormal
    # slopes (~1e-160) the backward-pass ratio 2*dr^2/(dl+dr)^2 underflows
    # to 0/0 = NaN even though it is analytically bounded by 2. Requiring
    # dl*dr above the normal floor keeps (dl+dr)^2 from underflowing on the
    # taken branch; slopes that tiny are physically zero anyway.
    tiny = jnp.finfo(jnp.result_type(dl, dr)).tiny
    same_sign = dl * dr > tiny
    denom = jnp.where(same_sign, dl + dr, 1.0)
    return jnp.where(same_sign, 2 * dl * dr / denom, 0.0)


# ------------------- Slope selection -------------------
def slope_limiter(dl, dr, limiter):
    """Apply the limiter selected by the STATIC Python int ``limiter``
    (0 = minmod, 1 = moncen, 2 = vanleer); only that branch compiles."""
    if limiter == 0:
        return minmod(dl, dr)
    if limiter == 1:
        return moncen(dl, dr)
    if limiter == 2:
        return vanleer(dl, dr)
    raise ValueError(f"Unknown limiter: {limiter}")


# ------------------- Compute slopes -------------------


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


def _compute_slope_3d_axis(U_pad, limiter, d):
    """Single-axis limited slope (``d``: 0=x, 1=y, 2=z) from a 3D padded
    array. Shared body of the three per-direction wrappers below."""
    Qc = U_pad[:, 1:-1, 1:-1, 1:-1]
    lo = [slice(1, -1)] * 3
    hi = [slice(1, -1)] * 3
    lo[d] = slice(None, -2)
    hi[d] = slice(2, None)
    d_L = Qc - U_pad[(slice(None), *lo)]
    d_R = U_pad[(slice(None), *hi)] - Qc
    return slope_limiter(d_L, d_R, limiter)


def compute_slope_x_3d_from_padded(U_pad, limiter):
    """Single-direction x-slope from a 3D padded array. Used by the
    memory-conscious MUSCL-Hancock 3D path that builds the predictor
    Jacobian product direction by direction so only one slope array
    is alive at a time (cuts ~40 GB of working set at 1024^3 fp32)."""
    return _compute_slope_3d_axis(U_pad, limiter, 0)


def compute_slope_y_3d_from_padded(U_pad, limiter):
    """Single-direction y-slope (see the x variant)."""
    return _compute_slope_3d_axis(U_pad, limiter, 1)


def compute_slope_z_3d_from_padded(U_pad, limiter):
    """Single-direction z-slope (see the x variant)."""
    return _compute_slope_3d_axis(U_pad, limiter, 2)
