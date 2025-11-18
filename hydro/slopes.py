import jax.numpy as jnp
from jax import jit, lax

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
def slope_limiter(dl, dr, limiter):
    # Wrap limiters for lax.cond
    def apply_minmod(_): return minmod(dl, dr)
    def apply_moncen(_): return moncen(dl, dr)
    def apply_vanleer(_): return vanleer(dl, dr)

    return lax.cond(limiter == 0,
                    apply_minmod,
                    lambda _: lax.cond(limiter == 1, apply_moncen, apply_vanleer, None),
                    None)

# ------------------- Compute slopes -------------------
from utils.padding import pad

@jit
def compute_slopes(U, limiter, boundary):
    """
    Compute slopes in 1D or 2D depending on U shape.
    Pads U first using the selected boundary condition.

    U: (Nvar, Nx) or (Nvar, Nx, Ny)
    Returns slopes for interior cells only.
    """
    U_pad = pad(U, boundary)  # <-- now uses general BC
    ndim = U.ndim - 1  # number of spatial dimensions

    # 1D case
    if ndim == 1:
        dx_L = U_pad[:, 1:-1] - U_pad[:, :-2]
        dx_R = U_pad[:, 2:  ] - U_pad[:, 1:-1]
        dUx = slope_limiter(dx_L, dx_R, limiter)
        dUy = None
        return dUx, dUy

    # 2D case
    dx_L = U_pad[:, 1:-1, 1:-1] - U_pad[:, :-2, 1:-1]
    dx_R = U_pad[:, 2:  , 1:-1] - U_pad[:, 1:-1, 1:-1]
    dy_D = U_pad[:, 1:-1, 1:-1] - U_pad[:, 1:-1, :-2]
    dy_U = U_pad[:, 1:-1, 2:  ] - U_pad[:, 1:-1, 1:-1]

    dUx = slope_limiter(dx_L, dx_R, limiter)
    dUy = slope_limiter(dy_D, dy_U, limiter)

    return dUx, dUy

