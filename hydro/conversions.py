"""
Conversion between conservative and primitive variables.

Conservative: U = [rho, rho*vx, rho*vy, E]  (2D)
              U = [rho, rho*vx, rho*vy, rho*vz, E]  (3D)

Primitive: W = [rho, vx, vy, P]  (2D)
           W = [rho, vx, vy, vz, P]  (3D)

where E = total energy density, P = pressure
"""

import jax.numpy as jnp
from jax import jit
from functools import partial


@partial(jit, static_argnums=(1,))
def cons_to_prim(U, gamma):
    """
    Convert conservative to primitive variables.

    Parameters
    ----------
    U : array
        Conservative variables
        1D: (3, Nx) = [rho, rho*vx, E]
        2D: (4, Nx, Ny) = [rho, rho*vx, rho*vy, E]
        3D: (5, Nx, Ny, Nz) = [rho, rho*vx, rho*vy, rho*vz, E]
    gamma : float (static)
        Adiabatic index

    Returns
    -------
    W : array
        Primitive variables (same shape as U)
        1D: [rho, vx, P]
        2D: [rho, vx, vy, P]
        3D: [rho, vx, vy, vz, P]
    """
    ndim = U.ndim - 1
    Nvar = U.shape[0]

    rho = U[0]
    inv_rho = 1.0 / rho
    gm1 = gamma - 1.0

    if ndim == 1:  # 1D
        vx = U[1] * inv_rho
        E = U[2]
        ke = 0.5 * rho * vx * vx
        P = gm1 * (E - ke)
        return jnp.stack([rho, vx, P])

    elif ndim == 2:  # 2D
        vx = U[1] * inv_rho
        vy = U[2] * inv_rho
        E = U[3]
        ke = 0.5 * rho * (vx*vx + vy*vy)
        P = gm1 * (E - ke)
        return jnp.stack([rho, vx, vy, P])

    else:  # 3D
        vx = U[1] * inv_rho
        vy = U[2] * inv_rho
        vz = U[3] * inv_rho
        E = U[4]
        ke = 0.5 * rho * (vx*vx + vy*vy + vz*vz)
        P = gm1 * (E - ke)
        return jnp.stack([rho, vx, vy, vz, P])


@partial(jit, static_argnums=(1,))
def prim_to_cons(W, gamma):
    """
    Convert primitive to conservative variables.

    Parameters
    ----------
    W : array
        Primitive variables
        1D: (3, Nx) = [rho, vx, P]
        2D: (4, Nx, Ny) = [rho, vx, vy, P]
        3D: (5, Nx, Ny, Nz) = [rho, vx, vy, vz, P]
    gamma : float (static)
        Adiabatic index

    Returns
    -------
    U : array
        Conservative variables (same shape as W)
        1D: [rho, rho*vx, E]
        2D: [rho, rho*vx, rho*vy, E]
        3D: [rho, rho*vx, rho*vy, rho*vz, E]
    """
    ndim = W.ndim - 1
    Nvar = W.shape[0]

    rho = W[0]
    gm1 = gamma - 1.0

    if ndim == 1:  # 1D
        vx = W[1]
        P = W[2]
        rho_vx = rho * vx
        ke = 0.5 * rho * vx * vx
        E = P / gm1 + ke
        return jnp.stack([rho, rho_vx, E])

    elif ndim == 2:  # 2D
        vx = W[1]
        vy = W[2]
        P = W[3]
        rho_vx = rho * vx
        rho_vy = rho * vy
        ke = 0.5 * rho * (vx*vx + vy*vy)
        E = P / gm1 + ke
        return jnp.stack([rho, rho_vx, rho_vy, E])

    else:  # 3D
        vx = W[1]
        vy = W[2]
        vz = W[3]
        P = W[4]
        rho_vx = rho * vx
        rho_vy = rho * vy
        rho_vz = rho * vz
        ke = 0.5 * rho * (vx*vx + vy*vy + vz*vz)
        E = P / gm1 + ke
        return jnp.stack([rho, rho_vx, rho_vy, rho_vz, E])
