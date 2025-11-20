import jax.numpy as jnp
from jax import jit, lax
from functools import partial

# ------------------- Slope limiters -------------------
@jit
def minmod(a, b):
    return jnp.where(a * b > 0, jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)

@jit
def minmod3(a, b, c):
    s = jnp.sign(a)
    return jnp.where((a*b > 0) & (a*c > 0),
                     s * jnp.minimum(jnp.abs(a),
                                     jnp.minimum(jnp.abs(b), jnp.abs(c))),
                     0.0)

@jit
def moncen(dl, dr):
    return minmod3(2*dl, 2*dr, 0.5*(dl+dr))

@jit
def vanleer(dl, dr):
    return jnp.where(dl*dr > 0, 2*dl*dr/(dl+dr), 0.0)

# ------------------- Slope selection -------------------
@partial(jit, static_argnums=(2,))
def slope_limiter(dl, dr, limiter):
    """
    Apply slope limiter based on limiter type.
    Uses lax.switch for efficient 3-way branching.
    """
    # Wrap limiters for lax.switch
    def apply_minmod(_): return minmod(dl, dr)
    def apply_moncen(_): return moncen(dl, dr)
    def apply_vanleer(_): return vanleer(dl, dr)

    return lax.switch(limiter, [apply_minmod, apply_moncen, apply_vanleer], None)

# ------------------- Compute slopes -------------------
from hydro.padding import pad

@partial(jit, static_argnums=(1,))
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
        dx_R = U_pad[:, 2:  ] - U_pad[:, 1:-1]
        dUx = slope_limiter(dx_L, dx_R, limiter)
        dUy = None
        dUz = None
        return dUx, dUy, dUz

    # 2D case
    if ndim == 2:
        # Extract center cells once (was being extracted 4 times!)
        Qc = U_pad[:, 1:-1, 1:-1]

        dx_L = Qc - U_pad[:, :-2, 1:-1]
        dx_R = U_pad[:, 2:  , 1:-1] - Qc
        dy_D = Qc - U_pad[:, 1:-1, :-2]
        dy_U = U_pad[:, 1:-1, 2:  ] - Qc

        dUx = slope_limiter(dx_L, dx_R, limiter)
        dUy = slope_limiter(dy_D, dy_U, limiter)
        dUz = None
        return dUx, dUy, dUz

    # 3D case
    # Extract center cells once (was being extracted 6 times!)
    Qc = U_pad[:, 1:-1, 1:-1, 1:-1]

    dx_L = Qc - U_pad[:, :-2, 1:-1, 1:-1]
    dx_R = U_pad[:, 2:  , 1:-1, 1:-1] - Qc
    dy_D = Qc - U_pad[:, 1:-1, :-2, 1:-1]
    dy_U = U_pad[:, 1:-1, 2:  , 1:-1] - Qc
    dz_B = Qc - U_pad[:, 1:-1, 1:-1, :-2]
    dz_F = U_pad[:, 1:-1, 1:-1, 2:  ] - Qc

    dUx = slope_limiter(dx_L, dx_R, limiter)
    dUy = slope_limiter(dy_D, dy_U, limiter)
    dUz = slope_limiter(dz_B, dz_F, limiter)

    return dUx, dUy, dUz

@partial(jit, static_argnums=(1, 2))
def compute_slopes(U, limiter, boundary):
    """
    Compute slopes in 1D, 2D, or 3D depending on U shape.
    Pads U first using the selected boundary condition, then computes slopes.

    U: (Nvar, Nx) or (Nvar, Nx, Ny) or (Nvar, Nx, Ny, Nz)
    Returns slopes for interior cells only.
    """
    U_pad = pad(U, boundary)
    return compute_slopes_from_padded(U_pad, limiter)

