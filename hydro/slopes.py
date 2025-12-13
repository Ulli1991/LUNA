import jax.numpy as jnp
from jax import jit, lax
from functools import partial

# ------------------- Slope limiters -------------------
def minmod(a, b):
    return jnp.where(a * b > 0, jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)

def minmod3(a, b, c):
    s = jnp.sign(a)
    return jnp.where((a*b > 0) & (a*c > 0),
                     s * jnp.minimum(jnp.abs(a),
                                     jnp.minimum(jnp.abs(b), jnp.abs(c))),
                     0.0)

def moncen(dl, dr):
    return minmod3(2*dl, 2*dr, 0.5*(dl+dr))

def vanleer(dl, dr):
    return jnp.where(dl*dr > 0, 2*dl*dr/(dl+dr), 0.0)

# ------------------- Slope selection -------------------
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

def compute_slopes(U, limiter, boundary):
    """
    Compute slopes in 1D, 2D, or 3D depending on U shape.
    Pads U first using the selected boundary condition, then computes slopes.

    U: (Nvar, Nx) or (Nvar, Nx, Ny) or (Nvar, Nx, Ny, Nz)
    Returns slopes for interior cells only.
    """
    U_pad = pad(U, boundary)
    return compute_slopes_from_padded(U_pad, limiter)

# ===============================================================================
# DIFFERENTIABLE (SMOOTH) LIMITERS FOR AUTODIFF
# ===============================================================================
# These are smooth approximations of the limiters above, designed for use with
# automatic differentiation. They use smooth functions (tanh, logaddexp) instead
# of non-differentiable operations (sign, min, max, where).
#
# The 'sharpness' parameter controls the tradeoff:
#   - Higher values (e.g., 50-100): closer to original limiters, sharper transitions
#   - Lower values (e.g., 5-10): smoother, better for optimization
#   - For smooth test problems (soundwave, advection), sharpness=10 works well
# ===============================================================================

def smooth_sign(x, sharpness=10.0):
    """
    Smooth approximation of sign function using tanh.

    For x >> 0: returns ~1
    For x << 0: returns ~-1
    Gradient: d/dx smooth_sign(x) = sharpness * sech^2(sharpness*x)
    """
    return jnp.tanh(sharpness * x)

def smooth_min(a, b, sharpness=10.0):
    """
    Smooth approximation of minimum using LogSumExp trick.

    min(a,b) ≈ -log(exp(-a*k) + exp(-b*k)) / k where k = sharpness
    As sharpness → ∞, converges to true minimum.
    """
    k = sharpness
    return -(jnp.logaddexp(-k*a, -k*b)) / k

def smooth_heaviside(x, sharpness=10.0):
    """
    Smooth approximation of Heaviside step function.

    Returns ~0 for x<0, ~1 for x>0, smooth transition at x=0.
    H(x) ≈ 0.5 * (1 + tanh(sharpness * x))
    """
    return 0.5 * (1.0 + jnp.tanh(sharpness * x))

def minmod_smooth(a, b, sharpness=10.0):
    """
    Differentiable approximation of minmod limiter.

    For smooth problems, behaves nearly identically to minmod
    but has well-defined gradients everywhere.
    """
    # Smooth version of: a*b > 0 (same sign check)
    same_sign = smooth_heaviside(a * b, sharpness)

    # Smooth minimum of absolute values
    min_abs = smooth_min(jnp.abs(a), jnp.abs(b), sharpness)

    # Smooth sign
    s = smooth_sign(a, sharpness)

    # Combine: if same sign, return sign(a) * min(|a|, |b|), else 0
    return same_sign * s * min_abs

def minmod3_smooth(a, b, c, sharpness=10.0):
    """
    Differentiable approximation of 3-argument minmod limiter.
    """
    # Check if all have same sign (smooth version)
    same_sign_ab = smooth_heaviside(a * b, sharpness)
    same_sign_ac = smooth_heaviside(a * c, sharpness)
    all_same_sign = same_sign_ab * same_sign_ac  # product acts like smooth AND

    # Smooth minimum of three absolute values
    min_ab = smooth_min(jnp.abs(a), jnp.abs(b), sharpness)
    min_abc = smooth_min(min_ab, jnp.abs(c), sharpness)

    # Smooth sign
    s = smooth_sign(a, sharpness)

    return all_same_sign * s * min_abc

def moncen_smooth(dl, dr, sharpness=10.0):
    """
    Differentiable approximation of Monotonized Central limiter.
    """
    return minmod3_smooth(2*dl, 2*dr, 0.5*(dl+dr), sharpness)

def vanleer_smooth(dl, dr, sharpness=10.0):
    """
    Differentiable approximation of van Leer limiter.

    Van Leer is already relatively smooth, but we smooth the sign check.
    """
    same_sign = smooth_heaviside(dl * dr, sharpness)
    # Add small epsilon to avoid division by zero
    eps = 1e-10
    return same_sign * 2*dl*dr / (dl + dr + eps)

# ------------------- Smooth slope selection (for autodiff) -------------------
def slope_limiter_smooth(dl, dr, limiter, sharpness=10.0):
    """
    Apply SMOOTH slope limiter based on limiter type.
    For use with autodiff - all limiters are differentiable.

    Parameters
    ----------
    dl, dr : array
        Left and right slopes
    limiter : int (static)
        0 = smooth minmod
        1 = smooth moncen
        2 = smooth vanleer
    sharpness : float (static)
        Controls smoothness (higher = closer to original limiters)
    """
    # Wrap smooth limiters for lax.switch
    def apply_minmod_smooth(_): return minmod_smooth(dl, dr, sharpness)
    def apply_moncen_smooth(_): return moncen_smooth(dl, dr, sharpness)
    def apply_vanleer_smooth(_): return vanleer_smooth(dl, dr, sharpness)

    return lax.switch(limiter, [apply_minmod_smooth, apply_moncen_smooth, apply_vanleer_smooth], None)

# ------------------- Compute slopes with smooth limiters (for autodiff) -------------------
def compute_slopes_from_padded_smooth(U_pad, limiter, sharpness=10.0):
    """
    Compute slopes from an already-padded array using SMOOTH limiters.
    For use with autodiff - all operations are differentiable.

    U_pad: (Nvar, Nx+2) for 1D, (Nvar, Nx+2, Ny+2) for 2D, (Nvar, Nx+2, Ny+2, Nz+2) for 3D
    Returns slopes for interior cells only.
    """
    ndim = U_pad.ndim - 1  # number of spatial dimensions

    # 1D case
    if ndim == 1:
        dx_L = U_pad[:, 1:-1] - U_pad[:, :-2]
        dx_R = U_pad[:, 2:  ] - U_pad[:, 1:-1]
        dUx = slope_limiter_smooth(dx_L, dx_R, limiter, sharpness)
        dUy = None
        dUz = None
        return dUx, dUy, dUz

    # 2D case
    if ndim == 2:
        Qc = U_pad[:, 1:-1, 1:-1]

        dx_L = Qc - U_pad[:, :-2, 1:-1]
        dx_R = U_pad[:, 2:  , 1:-1] - Qc
        dy_D = Qc - U_pad[:, 1:-1, :-2]
        dy_U = U_pad[:, 1:-1, 2:  ] - Qc

        dUx = slope_limiter_smooth(dx_L, dx_R, limiter, sharpness)
        dUy = slope_limiter_smooth(dy_D, dy_U, limiter, sharpness)
        dUz = None
        return dUx, dUy, dUz

    # 3D case
    Qc = U_pad[:, 1:-1, 1:-1, 1:-1]

    dx_L = Qc - U_pad[:, :-2, 1:-1, 1:-1]
    dx_R = U_pad[:, 2:  , 1:-1, 1:-1] - Qc
    dy_D = Qc - U_pad[:, 1:-1, :-2, 1:-1]
    dy_U = U_pad[:, 1:-1, 2:  , 1:-1] - Qc
    dz_B = Qc - U_pad[:, 1:-1, 1:-1, :-2]
    dz_F = U_pad[:, 1:-1, 1:-1, 2:  ] - Qc

    dUx = slope_limiter_smooth(dx_L, dx_R, limiter, sharpness)
    dUy = slope_limiter_smooth(dy_D, dy_U, limiter, sharpness)
    dUz = slope_limiter_smooth(dz_B, dz_F, limiter, sharpness)

    return dUx, dUy, dUz

def compute_slopes_smooth(U, limiter, boundary, sharpness=10.0):
    """
    Compute slopes in 1D, 2D, or 3D using SMOOTH limiters.
    For use with autodiff - all operations are differentiable.

    U: (Nvar, Nx) or (Nvar, Nx, Ny) or (Nvar, Nx, Ny, Nz)
    Returns slopes for interior cells only.
    """
    U_pad = pad(U, boundary)
    return compute_slopes_from_padded_smooth(U_pad, limiter, sharpness)

