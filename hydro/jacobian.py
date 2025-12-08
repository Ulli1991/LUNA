"""
Primitive variable Jacobian matrix multiplication for MUSCL-Hancock predictor.

For the Euler equations, the primitive Jacobian A_p = ∂F/∂W where W = [ρ, vx, vy, P].
This computes A_p * (∂W/∂x) efficiently without forming the full matrix.
"""

import jax.numpy as jnp
from jax import jit
from functools import partial


@partial(jit, static_argnums=(2,))
def apply_primitive_jacobian_1d(W, dW, gamma):
    """
    Compute A_x * dW for 1D Euler equations in primitive variables.

    W = [ρ, vx, P]
    A_x * dW advances the solution in time using characteristic information.

    Parameters
    ----------
    W : array, shape (3, ...)
        Primitive variables [ρ, vx, P]
    dW : array, shape (3, ...)
        Slopes [dρ, dvx, dP]
    gamma : float
        Adiabatic index

    Returns
    -------
    A_x * dW : array, shape (3, ...)
    """
    rho, vx, P = W
    drho, dvx, dP = dW

    # Sound speed
    c = jnp.sqrt(gamma * P / rho)
    c2 = c * c

    # A_x * dW for primitive variables
    # A_x = [vx   ρ    0  ]
    #       [0    vx  1/ρ ]
    #       [0   ρc²  vx  ]

    result_rho = vx * drho + rho * dvx
    result_vx = vx * dvx + (1.0 / rho) * dP
    result_P = rho * c2 * dvx + vx * dP

    return jnp.stack([result_rho, result_vx, result_P])


@partial(jit, static_argnums=(3,))
def apply_primitive_jacobian_2d(W, dWx, dWy, gamma):
    """
    Compute A_x * dWx + A_y * dWy for 2D Euler equations in primitive variables.

    W = [ρ, vx, vy, P]
    This combines contributions from both spatial directions for the predictor step.

    Parameters
    ----------
    W : array, shape (4, Nx, Ny)
        Primitive variables [ρ, vx, vy, P]
    dWx : array, shape (4, Nx, Ny)
        X-direction slopes [dρ/dx, dvx/dx, dvy/dx, dP/dx]
    dWy : array, shape (4, Nx, Ny)
        Y-direction slopes [dρ/dy, dvx/dy, dvy/dy, dP/dy]
    gamma : float
        Adiabatic index

    Returns
    -------
    A_x * dWx + A_y * dWy : array, shape (4, Nx, Ny)
    """
    rho, vx, vy, P = W
    drho_dx, dvx_dx, dvy_dx, dP_dx = dWx
    drho_dy, dvx_dy, dvy_dy, dP_dy = dWy

    # Sound speed
    c = jnp.sqrt(gamma * P / rho)
    c2 = c * c
    inv_rho = 1.0 / rho

    # A_x * dWx where A_x = [vx   ρ    0    0  ]
    #                        [0    vx   0   1/ρ ]
    #                        [0    0    vx   0  ]
    #                        [0   ρc²   0    vx ]

    Ax_dWx_rho = vx * drho_dx + rho * dvx_dx
    Ax_dWx_vx = vx * dvx_dx + inv_rho * dP_dx
    Ax_dWx_vy = vx * dvy_dx
    Ax_dWx_P = rho * c2 * dvx_dx + vx * dP_dx

    # A_y * dWy where A_y = [vy   0    ρ    0  ]
    #                        [0    vy   0    0  ]
    #                        [0    0    vy  1/ρ ]
    #                        [0    0   ρc²   vy ]

    Ay_dWy_rho = vy * drho_dy + rho * dvy_dy
    Ay_dWy_vx = vy * dvx_dy
    Ay_dWy_vy = vy * dvy_dy + inv_rho * dP_dy
    Ay_dWy_P = rho * c2 * dvy_dy + vy * dP_dy

    # Sum contributions from both directions
    result_rho = Ax_dWx_rho + Ay_dWy_rho
    result_vx = Ax_dWx_vx + Ay_dWy_vx
    result_vy = Ax_dWx_vy + Ay_dWy_vy
    result_P = Ax_dWx_P + Ay_dWy_P

    return jnp.stack([result_rho, result_vx, result_vy, result_P])


@partial(jit, static_argnums=(4,))
def apply_primitive_jacobian_3d(W, dWx, dWy, dWz, gamma):
    """
    Compute A_x * dWx + A_y * dWy + A_z * dWz for 3D Euler equations.

    W = [ρ, vx, vy, vz, P]

    Parameters
    ----------
    W : array, shape (5, Nx, Ny, Nz)
        Primitive variables [ρ, vx, vy, vz, P]
    dWx, dWy, dWz : arrays, shape (5, Nx, Ny, Nz)
        Slopes in each direction
    gamma : float
        Adiabatic index

    Returns
    -------
    A_x * dWx + A_y * dWy + A_z * dWz : array, shape (5, Nx, Ny, Nz)
    """
    rho, vx, vy, vz, P = W
    drho_dx, dvx_dx, dvy_dx, dvz_dx, dP_dx = dWx
    drho_dy, dvx_dy, dvy_dy, dvz_dy, dP_dy = dWy
    drho_dz, dvx_dz, dvy_dz, dvz_dz, dP_dz = dWz

    # Sound speed
    c = jnp.sqrt(gamma * P / rho)
    c2 = c * c
    inv_rho = 1.0 / rho

    # X-direction contribution
    Ax_dWx_rho = vx * drho_dx + rho * dvx_dx
    Ax_dWx_vx = vx * dvx_dx + inv_rho * dP_dx
    Ax_dWx_vy = vx * dvy_dx
    Ax_dWx_vz = vx * dvz_dx
    Ax_dWx_P = rho * c2 * dvx_dx + vx * dP_dx

    # Y-direction contribution
    Ay_dWy_rho = vy * drho_dy + rho * dvy_dy
    Ay_dWy_vx = vy * dvx_dy
    Ay_dWy_vy = vy * dvy_dy + inv_rho * dP_dy
    Ay_dWy_vz = vy * dvz_dy
    Ay_dWy_P = rho * c2 * dvy_dy + vy * dP_dy

    # Z-direction contribution
    Az_dWz_rho = vz * drho_dz + rho * dvz_dz
    Az_dWz_vx = vz * dvx_dz
    Az_dWz_vy = vz * dvy_dz
    Az_dWz_vz = vz * dvz_dz + inv_rho * dP_dz
    Az_dWz_P = rho * c2 * dvz_dz + vz * dP_dz

    # Sum all contributions
    result_rho = Ax_dWx_rho + Ay_dWy_rho + Az_dWz_rho
    result_vx = Ax_dWx_vx + Ay_dWy_vx + Az_dWz_vx
    result_vy = Ax_dWx_vy + Ay_dWy_vy + Az_dWz_vy
    result_vz = Ax_dWx_vz + Ay_dWy_vz + Az_dWz_vz
    result_P = Ax_dWx_P + Ay_dWy_P + Az_dWz_P

    return jnp.stack([result_rho, result_vx, result_vy, result_vz, result_P])
