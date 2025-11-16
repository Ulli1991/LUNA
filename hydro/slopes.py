import jax.numpy as jnp
from jax import jit, lax

@jit
def minmod(a, b):
    return jnp.where(a*b > 0, jnp.sign(a)*jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)

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

def slope_limiter(dl, dr, limiter):
    # Create functions for each limiter
    def apply_minmod(dl, dr):
        return minmod(dl, dr)
    
    def apply_moncen(dl, dr):
        return moncen(dl, dr)
    
    def apply_vanleer(dl, dr):
        return vanleer(dl, dr)
    
    # Use lax.cond to apply the correct limiter
    return lax.cond(limiter == 0, 
                    lambda _: apply_minmod(dl, dr),  # Ensure dl, dr are passed
                    lambda _: lax.cond(limiter == 1, 
                                       lambda _: apply_moncen(dl, dr),  # Ensure dl, dr are passed
                                       lambda _: apply_vanleer(dl, dr),  # Ensure dl, dr are passed
                                       None),  # The third argument must be a valid function signature
                    None)

@jit
def compute_slopes(U_pad, limiter):
    dx_L = U_pad[:, 1:-1, 1:-1] - U_pad[:, :-2, 1:-1]
    dx_R = U_pad[:, 2:, 1:-1]   - U_pad[:, 1:-1, 1:-1]
    dy_D = U_pad[:, 1:-1, 1:-1] - U_pad[:, 1:-1, :-2]
    dy_U = U_pad[:, 1:-1, 2:]   - U_pad[:, 1:-1, 1:-1]

    # Use the slope_limiter function
    dUx = slope_limiter(dx_L, dx_R, limiter)
    dUy = slope_limiter(dy_D, dy_U, limiter)

    return dUx, dUy

